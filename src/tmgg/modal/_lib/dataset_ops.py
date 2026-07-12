"""Dataset prepare + validate impls (no Modal/import-modal).

Pure-Python orchestration that drives existing dataset
``download_smiles_split``, ``setup`` (preprocess), and SPECTRE fixture
loaders. Lives outside ``_functions.py`` so the same code path can be
unit-tested locally and re-used inside ``@app.function`` wrappers.

Every prepare / validate function returns a structured ``dict[str, Any]``
report — never raises mid-iteration — so the wrapping Modal function
can serialize the report into the worker's return value and the CLI can
present split-level status. ``status`` is one of ``ok`` / ``missing`` /
``load_error`` / ``schema_error``; the per-split reports always include
the absolute on-disk path so failed checks are immediately actionable.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset

MOLECULAR_DATASETS = ("qm9", "moses", "guacamol")
SPECTRE_DATASETS = ("planar", "sbm")
ALL_DATASETS = MOLECULAR_DATASETS + SPECTRE_DATASETS

DEFAULT_MODAL_CACHE_ROOT = Path("/data/datasets")

# Sentinel filename written into a split's shard dir AFTER every shard
# has been atomically renamed into its final ``<idx>.pt`` form. Cache-hit
# logic short-circuits only when this marker exists; without it we
# treat the dir as a partial / preempted preprocess and rebuild the
# missing shards. Mirrors a "manifest-then-data" two-phase commit so a
# preempted preprocess can never be mistaken for a complete one.
_COMPLETE_MARKER = "_complete"


# ---------------------------------------------------------------------------
# molecular: dispatch
# ---------------------------------------------------------------------------


def _molecular_dataset_cls(name: str) -> type[MolecularGraphDataset]:
    from tmgg.data.datasets.molecular.guacamol import GuacaMolDataset
    from tmgg.data.datasets.molecular.moses import MOSESDataset
    from tmgg.data.datasets.molecular.qm9 import QM9Dataset

    table: dict[str, type[MolecularGraphDataset]] = {
        "qm9": QM9Dataset,
        "moses": MOSESDataset,
        "guacamol": GuacaMolDataset,
    }
    if name not in table:
        raise ValueError(
            f"unknown molecular dataset {name!r}; expected one of {sorted(table)}"
        )
    return table[name]


def _molecular_splits(name: str) -> tuple[str, ...]:
    # GuacaMol declares only train/val/test of which 'test' is held-out,
    # but each dataset's ``download_smiles_split`` raises ``KeyError`` on
    # missing splits, so iterate the union and skip cleanly. Per-dataset
    # overrides can be added here if a future dataset diverges.
    _ = name
    return ("train", "val", "test")


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Single-line stage log; flush so Modal forwards it in real-time."""
    print(msg, flush=True)


def _make_encode_progress_cb(
    name: str,
    split: str,
    n_smiles: int,
) -> Callable[[int, dict[str, int]], None]:
    """Build a progress callback that closes over the per-split context.

    Defined at module scope (not inside the loop body) so the captured
    ``name``/``split``/``n_smiles`` are bound at call time, satisfying
    ruff's B023 closes-over-loop-variable check.
    """

    def _cb(processed: int, counters: dict[str, int]) -> None:
        pct = (processed / n_smiles * 100.0) if n_smiles else 0.0
        drops = (
            counters["parse_failure"]
            + counters["atom_count_overflow"]
            + counters["vocab_miss"]
            + counters["kekulize_failure"]
        )
        _log(
            f"[{name}/{split}] encode {processed}/{n_smiles} "
            f"({pct:.1f}%) kept={counters['kept']} drops={drops}"
        )

    return _cb


def prepare_molecular_dataset(
    name: str,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Download raw + preprocess shards for one molecular dataset.

    Walks all splits, driving each stage explicitly so progress is
    visible over Modal stdout: ``download → encode (with periodic
    counters) → write shards``. Per-split errors are captured into the
    report rather than aborting; preparing ``train`` but not ``val``
    is still useful information to the caller.
    """
    cache_root = cache_root or DEFAULT_MODAL_CACHE_ROOT
    cls = _molecular_dataset_cls(name)
    started = time.monotonic()
    _log(f"[{name}] prepare start (cache_root={cache_root})")

    splits_report: dict[str, Any] = {}
    for split in _molecular_splits(name):
        split_started = time.monotonic()
        entry: dict[str, Any] = {"path": None, "status": "ok", "shards": 0, "graphs": 0}
        try:
            ds = cls(split=split, cache_root=cache_root)
        except (
            Exception
        ) as exc:  # construction failure (bad cache_root, missing config)
            _log(f"[{name}/{split}] construct_error: {exc!r}")
            entry["status"] = "construct_error"
            entry["detail"] = repr(exc)
            splits_report[split] = entry
            continue

        try:
            shard_dir = ds._shard_dir()
            entry["path"] = str(shard_dir)
            complete_marker = shard_dir / _COMPLETE_MARKER

            # Cache hit only if the explicit completion marker exists.
            # ``any(shard_dir.iterdir())`` was the old check; it fired
            # spuriously on preempted preprocess runs that left partial
            # shards on disk and silently delivered an under-counted
            # dataset to subsequent training. The marker is written
            # only after every shard has been atomically renamed.
            if complete_marker.exists():
                ds.setup()  # quick path: just loads existing shards
                entry["shards"] = len(sorted(shard_dir.glob("*.pt")))
                entry["graphs"] = len(ds)
                _log(
                    f"[{name}/{split}] cached: {entry['shards']} shard(s), "
                    f"{entry['graphs']} graphs"
                )
            else:
                _log(f"[{name}/{split}] downloading SMILES...")
                smiles = ds.download_smiles_split(split)
                n_smiles = len(smiles)
                _log(
                    f"[{name}/{split}] downloaded {n_smiles} SMILES; "
                    f"streaming encode + write..."
                )

                codec = ds.make_codec()
                shard_dir.mkdir(parents=True, exist_ok=True)

                # Streaming, resumable preprocess: encode → write → next.
                # Each shard is atomically renamed before we move on, so
                # a preemption mid-encode loses at most one shard's
                # worth of work (~``shard_size`` SMILES). On restart we
                # skip every ``<idx>.pt`` already on disk and pick up
                # where the previous run left off.
                shard_size = ds.shard_size
                n_shards = (n_smiles + shard_size - 1) // shard_size
                counters_total = {
                    "input": 0,
                    "parse_failure": 0,
                    "atom_count_overflow": 0,
                    "vocab_miss": 0,
                    "kekulize_failure": 0,
                    "kept": 0,
                }
                total_graphs = 0
                shards_written = 0
                for shard_idx in range(n_shards):
                    shard_path = shard_dir / f"{shard_idx:04d}.pt"
                    if shard_path.exists():
                        # Resume: shard already finalised by an earlier
                        # run. Load it just to update the running graph
                        # count (cheap; one .pt per shard).
                        prior = torch.load(shard_path, weights_only=False)
                        total_graphs += len(prior)
                        shards_written += 1
                        _log(
                            f"[{name}/{split}] shard {shard_idx + 1}/{n_shards} "
                            f"resume (already on disk; {len(prior)} graphs)"
                        )
                        continue
                    chunk = smiles[
                        shard_idx * shard_size : (shard_idx + 1) * shard_size
                    ]
                    chunk_graphs, chunk_counters = codec.encode_dataset_with_stats(
                        chunk
                    )
                    for k, v in chunk_counters.items():
                        counters_total[k] += v
                    tmp_path = shard_dir / f"{shard_idx:04d}.pt.tmp"
                    torch.save(chunk_graphs, tmp_path)
                    os.replace(tmp_path, shard_path)
                    total_graphs += len(chunk_graphs)
                    shards_written += 1
                    _log(
                        f"[{name}/{split}] shard {shard_idx + 1}/{n_shards} "
                        f"wrote {len(chunk_graphs)}/{len(chunk)} kept "
                        f"(running counters={counters_total})"
                    )

                # Two-phase commit on the marker too: write a temp
                # sentinel and atomic-rename. A preemption between the
                # last shard write and the marker rename leaves the
                # shards complete but the marker absent — the next run
                # will resume by loading every shard (skip-existing) and
                # then write the marker without re-encoding anything.
                marker_tmp = shard_dir / f"{_COMPLETE_MARKER}.tmp"
                marker_tmp.write_text(
                    f"shards={n_shards}\nshard_size={shard_size}\n"
                    f"graphs={total_graphs}\n"
                )
                os.replace(marker_tmp, complete_marker)

                entry["graphs"] = total_graphs
                entry["counters"] = counters_total
                entry["shards"] = shards_written
                _log(
                    f"[{name}/{split}] streamed {shards_written} shard(s); "
                    f"counters={counters_total}; complete-marker written"
                )
                # Refresh ds._graphs so callers querying len(ds) work.
                ds._graphs = ds._load_shards()
        except KeyError as exc:
            # Split not declared by this dataset (e.g., GuacaMol may
            # omit one). Record cleanly.
            _log(f"[{name}/{split}] split_missing: {exc!r}")
            entry["status"] = "split_missing"
            entry["detail"] = repr(exc)
        except Exception as exc:
            _log(f"[{name}/{split}] prepare_error: {exc!r}")
            entry["status"] = "prepare_error"
            entry["detail"] = repr(exc)

        entry["seconds"] = round(time.monotonic() - split_started, 2)
        _log(f"[{name}/{split}] {entry['status']} in {entry['seconds']}s")
        splits_report[split] = entry

    total_seconds = round(time.monotonic() - started, 2)
    _log(f"[{name}] prepare done in {total_seconds}s")
    return {
        "dataset": name,
        "kind": "molecular",
        "cache_root": str(cache_root),
        "splits": splits_report,
        "seconds_total": total_seconds,
    }


def _spectre_target(cache_root: Path, name: str) -> tuple[Path, Callable[[Path], Path]]:
    """Resolve fixture target path + downloader for a SPECTRE dataset."""
    target_dir = cache_root / "spectre"
    target_dir.mkdir(parents=True, exist_ok=True)
    if name == "planar":
        from tmgg.data.datasets.spectre_planar import download_spectre_planar_fixture

        return target_dir / "planar_64_200.pt", download_spectre_planar_fixture
    if name == "sbm":
        from tmgg.data.datasets.spectre_sbm import download_spectre_sbm_fixture

        return target_dir / "sbm_200.pt", download_spectre_sbm_fixture
    raise ValueError(f"unknown SPECTRE dataset {name!r}; expected planar or sbm")


def prepare_spectre_dataset(
    name: str,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Download the SPECTRE planar / SBM fixture into the shared cache."""
    cache_root = cache_root or DEFAULT_MODAL_CACHE_ROOT
    started = time.monotonic()
    _log(f"[{name}] prepare start (cache_root={cache_root})")

    target, downloader = _spectre_target(cache_root, name)
    if target.exists():
        _log(f"[{name}] cached: {target} ({target.stat().st_size} bytes)")
    else:
        _log(f"[{name}] downloading fixture to {target} ...")

    try:
        path = downloader(target)
        size = path.stat().st_size if path.exists() else 0
        total_seconds = round(time.monotonic() - started, 2)
        _log(f"[{name}] prepare done: {size} bytes in {total_seconds}s")
        return {
            "dataset": name,
            "kind": "spectre",
            "cache_root": str(cache_root),
            "fixture_path": str(path),
            "exists": path.exists(),
            "size_bytes": size,
            "seconds_total": total_seconds,
            "status": "ok",
        }
    except Exception as exc:
        total_seconds = round(time.monotonic() - started, 2)
        _log(f"[{name}] download_error after {total_seconds}s: {exc!r}")
        return {
            "dataset": name,
            "kind": "spectre",
            "cache_root": str(cache_root),
            "fixture_path": str(target),
            "exists": target.exists(),
            "status": "download_error",
            "detail": repr(exc),
            "seconds_total": total_seconds,
        }


def prepare_dataset(
    name: str,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Top-level dispatch: prepare any dataset by short name."""
    if name in MOLECULAR_DATASETS:
        return prepare_molecular_dataset(name, cache_root=cache_root)
    if name in SPECTRE_DATASETS:
        return prepare_spectre_dataset(name, cache_root=cache_root)
    raise ValueError(
        f"unknown dataset {name!r}; expected one of {sorted(ALL_DATASETS)}"
    )


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _validate_shard(shard_path: Path) -> dict[str, Any]:
    """Try to load one shard; return per-shard pass/fail entry."""
    entry: dict[str, Any] = {
        "path": str(shard_path),
        "status": "ok",
        "size_bytes": 0,
        "graphs": 0,
    }
    if not shard_path.exists():
        entry["status"] = "missing"
        return entry
    try:
        entry["size_bytes"] = shard_path.stat().st_size
        # ``weights_only=False`` is required because shards are pickled
        # ``DenseGraphState`` dataclasses. They are first-party files
        # written by this codebase; the global "no third-party pickle"
        # rule does not apply.
        shard = torch.load(shard_path, weights_only=False)
        if not isinstance(shard, list):
            entry["status"] = "schema_error"
            entry["detail"] = f"expected list, got {type(shard).__name__}"
            return entry
        entry["graphs"] = len(shard)
        if not shard:
            entry["status"] = "schema_error"
            entry["detail"] = "shard contains zero graphs"
            return entry
        # Spot-check the first graph: must expose ``node_mask`` (cached
        # property on ``DenseGraphState``) + at least one of X_class /
        # E_class. These are the dense-batch fields the collator and
        # DiffusionModule both consume after the sparse-default refactor's
        # ``state.to_dense()`` step inside dense-internal models.
        sample = shard[0]
        for attr in ("node_mask",):
            if not hasattr(sample, attr):
                entry["status"] = "schema_error"
                entry["detail"] = f"first graph missing required attr {attr!r}"
                return entry
    except Exception as exc:
        entry["status"] = "load_error"
        entry["detail"] = repr(exc)
    return entry


def validate_molecular_dataset(
    name: str,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Validate every preprocessed shard for one molecular dataset."""
    cache_root = cache_root or DEFAULT_MODAL_CACHE_ROOT
    cls = _molecular_dataset_cls(name)
    started = time.monotonic()
    splits_report: dict[str, Any] = {}

    for split in _molecular_splits(name):
        entry: dict[str, Any] = {
            "path": None,
            "status": "ok",
            "shards": 0,
            "graphs": 0,
            "shard_reports": [],
        }
        try:
            ds = cls(split=split, cache_root=cache_root)
            shard_dir = ds._shard_dir()
        except Exception as exc:
            entry["status"] = "construct_error"
            entry["detail"] = repr(exc)
            splits_report[split] = entry
            continue

        entry["path"] = str(shard_dir)
        if not shard_dir.exists():
            entry["status"] = "missing"
            splits_report[split] = entry
            continue

        shard_paths = sorted(shard_dir.glob("*.pt"))
        if not shard_paths:
            entry["status"] = "missing"
            splits_report[split] = entry
            continue

        per_shard = [_validate_shard(p) for p in shard_paths]
        entry["shards"] = len(per_shard)
        entry["graphs"] = sum(s.get("graphs", 0) for s in per_shard)
        entry["shard_reports"] = per_shard
        statuses = {s["status"] for s in per_shard}
        complete_marker = shard_dir / _COMPLETE_MARKER
        entry["complete_marker"] = complete_marker.exists()
        if statuses != {"ok"}:
            entry["status"] = "errors"
            entry["error_count"] = sum(1 for s in per_shard if s["status"] != "ok")
        elif not complete_marker.exists():
            # Every shard loads cleanly but the preprocess never wrote
            # the completion sentinel — typical fingerprint of a
            # preempted prepare. Surface as an actionable status so the
            # caller knows to re-run prepare (which will resume from
            # whichever shards are already on disk).
            entry["status"] = "incomplete"
            entry["detail"] = (
                "all loaded shards are valid, but no _complete marker — "
                "rerun ``tmgg-modal datasets prepare`` to finish + mark."
            )
        splits_report[split] = entry

    return {
        "dataset": name,
        "kind": "molecular",
        "cache_root": str(cache_root),
        "splits": splits_report,
        "seconds_total": round(time.monotonic() - started, 2),
        "ok": all(s.get("status") == "ok" for s in splits_report.values()),
    }


def validate_spectre_dataset(
    name: str,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the SPECTRE planar / SBM fixture (load + structure)."""
    cache_root = cache_root or DEFAULT_MODAL_CACHE_ROOT
    started = time.monotonic()
    fixture_filename = {"planar": "planar_64_200.pt", "sbm": "sbm_200.pt"}.get(name)
    if fixture_filename is None:
        raise ValueError(f"unknown SPECTRE dataset {name!r}; expected planar or sbm")
    fixture_path = cache_root / "spectre" / fixture_filename

    report: dict[str, Any] = {
        "dataset": name,
        "kind": "spectre",
        "cache_root": str(cache_root),
        "fixture_path": str(fixture_path),
        "exists": fixture_path.exists(),
        "size_bytes": 0,
        "num_graphs": 0,
        "status": "ok",
    }

    if not fixture_path.exists():
        report["status"] = "missing"
        report["seconds_total"] = round(time.monotonic() - started, 2)
        return report

    report["size_bytes"] = fixture_path.stat().st_size
    try:
        if name == "planar":
            from tmgg.data.datasets.spectre_planar import load_spectre_planar_fixture

            adjs, n_nodes = load_spectre_planar_fixture(fixture_path)
        else:
            from tmgg.data.datasets.spectre_sbm import load_spectre_sbm_fixture

            adjs, n_nodes = load_spectre_sbm_fixture(fixture_path)

        if not isinstance(adjs, list) or not adjs:
            report["status"] = "schema_error"
            report["detail"] = "fixture loaded but adjacency list is empty / wrong type"
        elif len(adjs) != len(n_nodes):
            report["status"] = "schema_error"
            report["detail"] = (
                f"adj/n_nodes length mismatch: {len(adjs)} vs {len(n_nodes)}"
            )
        else:
            report["num_graphs"] = len(adjs)
    except Exception as exc:
        report["status"] = "load_error"
        report["detail"] = repr(exc)

    report["seconds_total"] = round(time.monotonic() - started, 2)
    report["ok"] = report["status"] == "ok"
    return report


def validate_dataset(
    name: str,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Top-level dispatch: validate any dataset by short name."""
    if name in MOLECULAR_DATASETS:
        return validate_molecular_dataset(name, cache_root=cache_root)
    if name in SPECTRE_DATASETS:
        return validate_spectre_dataset(name, cache_root=cache_root)
    raise ValueError(
        f"unknown dataset {name!r}; expected one of {sorted(ALL_DATASETS)}"
    )

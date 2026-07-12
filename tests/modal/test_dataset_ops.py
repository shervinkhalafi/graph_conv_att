"""Tests for ``tmgg.modal._lib.dataset_ops`` impl functions.

These tests exercise the pure-Python prepare/validate path against
synthetic shard files written into a ``tmp_path`` cache. The Modal
``@app.function`` wrappers are not invoked.

Testing Strategy
----------------
- Build a fake ``preprocessed/<hash>/<split>/<shard>.pt`` tree under a
  ``tmp_path`` rooted ``cache_root``, then call the validate impl
  directly. Avoids heavyweight downloads while still exercising the
  full schema-check path on real ``.pt`` files.
- Unknown-dataset names must raise ``ValueError`` (loud failure as per
  the project's no-graceful-fallback rule).

Key Invariants
--------------
- A clean ``preprocessed/<hash>/<split>/0000.pt`` validates ``ok``.
- A truncated shard file marks the split as ``errors`` with
  ``load_error`` on the affected shard.
- A missing shard directory marks the split as ``missing``.
- The dispatcher rejects names outside ``ALL_DATASETS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tmgg.modal._lib.dataset_ops import (
    _COMPLETE_MARKER,
    ALL_DATASETS,
    _validate_shard,
    prepare_dataset,
    prepare_molecular_dataset,
    validate_dataset,
)


@dataclass
class _FakeGraph:
    node_mask: torch.Tensor
    X_class: torch.Tensor


def _write_clean_shard(path: Path, n: int = 4) -> None:
    """Materialise ``n`` synthetic GraphData-like dataclass instances."""
    path.parent.mkdir(parents=True, exist_ok=True)
    graphs = [
        _FakeGraph(
            node_mask=torch.ones(9, dtype=torch.bool),
            X_class=torch.zeros(9, 4),
        )
        for _ in range(n)
    ]
    torch.save(graphs, path)


# ---------------------------------------------------------------------------
# Per-shard validator
# ---------------------------------------------------------------------------


class TestValidateShard:
    def test_clean_shard_returns_ok(self, tmp_path: Path) -> None:
        shard = tmp_path / "0000.pt"
        _write_clean_shard(shard, n=3)
        report = _validate_shard(shard)
        assert report["status"] == "ok"
        assert report["graphs"] == 3
        assert report["size_bytes"] > 0

    def test_missing_file_returns_missing(self, tmp_path: Path) -> None:
        shard = tmp_path / "absent.pt"
        report = _validate_shard(shard)
        assert report["status"] == "missing"

    def test_truncated_shard_returns_load_error(self, tmp_path: Path) -> None:
        shard = tmp_path / "0000.pt"
        _write_clean_shard(shard)
        # Truncate to half its length — guaranteed to corrupt the
        # internal zip directory and trigger PytorchStreamReader.
        size = shard.stat().st_size
        with open(shard, "r+b") as f:
            f.truncate(size // 2)
        report = _validate_shard(shard)
        assert report["status"] == "load_error"
        assert "detail" in report

    def test_non_list_payload_returns_schema_error(self, tmp_path: Path) -> None:
        shard = tmp_path / "0000.pt"
        torch.save({"not": "a list"}, shard)
        report = _validate_shard(shard)
        assert report["status"] == "schema_error"

    def test_empty_list_returns_schema_error(self, tmp_path: Path) -> None:
        shard = tmp_path / "0000.pt"
        torch.save([], shard)
        report = _validate_shard(shard)
        assert report["status"] == "schema_error"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_unknown_name_raises_validate(self) -> None:
        with pytest.raises(ValueError, match="unknown dataset"):
            validate_dataset("not-a-dataset")

    def test_unknown_name_raises_prepare(self) -> None:
        with pytest.raises(ValueError, match="unknown dataset"):
            prepare_dataset("not-a-dataset")

    def test_all_datasets_listed(self) -> None:
        # Sanity: catches typos in the constant if a contributor adds a
        # new dataset to one helper but not the other.
        assert set(ALL_DATASETS) >= {"qm9", "moses", "guacamol", "planar", "sbm"}


# ---------------------------------------------------------------------------
# Spectre validate path (synthesise a fixture, no network)
# ---------------------------------------------------------------------------


class TestStreamingPreprocess:
    """``prepare_molecular_dataset`` must be preemption-robust.

    The cache-hit short-circuit must only fire when the explicit
    ``_complete`` marker exists. A partial shard layout (some final
    ``<idx>.pt`` files present, marker absent) must trigger a resume —
    not a silent "everything is fine" report.
    """

    def _patch_dataset_cls(self, monkeypatch, smiles, max_atoms=5, shard_size=3):
        from tmgg.data.datasets.molecular import dataset as ds_mod
        from tmgg.data.datasets.molecular.codec import SMILESCodec
        from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
        from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary
        from tmgg.modal._lib import dataset_ops

        class _TinyQM9(MolecularGraphDataset):
            DATASET_NAME = "qm9"
            DEFAULT_MAX_ATOMS = max_atoms
            SAMPLE_SMILES = smiles

            def __init__(self_inner, *, split: str, cache_root=None):  # type: ignore[no-untyped-def]
                super().__init__(
                    split=split,
                    cache_root=cache_root,
                    shard_size=shard_size,
                )

            @classmethod
            def make_codec(cls) -> SMILESCodec:
                return SMILESCodec(
                    vocab=AtomBondVocabulary.qm9(),
                    max_atoms=cls.DEFAULT_MAX_ATOMS,
                )

            def download_smiles_split(self, split: str) -> list[str]:
                return list(self.SAMPLE_SMILES)

        # Redirect dispatcher to point at the tiny in-memory dataset for ``qm9``.
        monkeypatch.setattr(
            dataset_ops, "_molecular_dataset_cls", lambda name: _TinyQM9
        )
        monkeypatch.setattr(dataset_ops, "_molecular_splits", lambda name: ("train",))
        return _TinyQM9, ds_mod

    def test_streaming_writes_marker_and_shards(self, tmp_path, monkeypatch) -> None:
        """End-to-end clean run: marker + every shard land on disk."""
        smiles = ["CCO", "CC(=O)O", "CCC", "CCCC", "CCCCO", "CCN"]  # 6 → 2 shards of 3
        cls, _ = self._patch_dataset_cls(monkeypatch, smiles)
        report = prepare_molecular_dataset("qm9", cache_root=tmp_path)
        train = report["splits"]["train"]
        assert train["status"] == "ok"
        # 6 SMILES, shard_size=3 → 2 shards, 6 graphs total.
        assert train["shards"] == 2
        ds = cls(split="train", cache_root=tmp_path)
        shard_dir = ds._shard_dir()
        assert (shard_dir / _COMPLETE_MARKER).exists()
        assert sorted(p.name for p in shard_dir.glob("*.pt")) == ["0000.pt", "0001.pt"]
        assert not list(shard_dir.glob("*.pt.tmp"))

    def test_partial_state_without_marker_triggers_resume(
        self, tmp_path, monkeypatch
    ) -> None:
        """Partial shard layout + no marker MUST re-encode the missing tail."""
        smiles = ["CCO", "CC(=O)O", "CCC", "CCCC", "CCCCO", "CCN"]  # 2 shards of 3
        cls, _ = self._patch_dataset_cls(monkeypatch, smiles)

        # Simulate a preempted prepare: write only shard 0 manually.
        ds = cls(split="train", cache_root=tmp_path)
        shard_dir = ds._shard_dir()
        shard_dir.mkdir(parents=True, exist_ok=True)
        # Build the shard 0 content exactly as the streaming impl would.
        codec = ds.make_codec()
        partial_chunk = smiles[:3]
        partial_graphs, _ = codec.encode_dataset_with_stats(partial_chunk)
        torch.save(partial_graphs, shard_dir / "0000.pt")
        # Important: no _complete marker, no shard 1.

        report = prepare_molecular_dataset("qm9", cache_root=tmp_path)
        train = report["splits"]["train"]
        assert train["status"] == "ok"
        # Both shards must end up on disk after the resume.
        assert sorted(p.name for p in shard_dir.glob("*.pt")) == ["0000.pt", "0001.pt"]
        # And the marker should now be present.
        assert (shard_dir / _COMPLETE_MARKER).exists()

    def test_full_cache_hit_skips_preprocess(self, tmp_path, monkeypatch) -> None:
        """Marker present → cache hit → no re-encoding."""
        smiles = ["CCO", "CC(=O)O", "CCC"]
        cls, _ = self._patch_dataset_cls(monkeypatch, smiles)
        # First run: warm cache.
        prepare_molecular_dataset("qm9", cache_root=tmp_path)
        # Second run: would re-download SMILES from
        # ``download_smiles_split``; mutate sentinel to detect re-call.
        cls.SAMPLE_SMILES = ["NEVER_CALLED"]  # type: ignore[attr-defined]
        report = prepare_molecular_dataset("qm9", cache_root=tmp_path)
        train = report["splits"]["train"]
        assert train["status"] == "ok"
        assert train["shards"] == 1
        # Counters key only set on the streaming path; cache-hit branch
        # omits it. Use that to confirm we took the short-circuit.
        assert "counters" not in train


class TestSpectreValidate:
    def test_missing_fixture_marks_missing(self, tmp_path: Path) -> None:
        # Don't write anything — validator must report 'missing' rather
        # than raise. tmp_path subdir 'spectre/' won't exist.
        report = validate_dataset("planar", cache_root=tmp_path)
        assert report["status"] == "missing"
        assert report["exists"] is False
        assert report["fixture_path"].endswith("planar_64_200.pt")

    def test_corrupt_fixture_marks_load_error(self, tmp_path: Path) -> None:
        # Write a non-torch file at the expected path.
        spectre_dir = tmp_path / "spectre"
        spectre_dir.mkdir()
        (spectre_dir / "sbm_200.pt").write_bytes(b"\x00\x01\x02 not a torch zip")
        report = validate_dataset("sbm", cache_root=tmp_path)
        assert report["status"] == "load_error"
        assert "detail" in report

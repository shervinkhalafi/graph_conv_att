"""On-disk train↔test MMD baselines for DiGress-style ratio reporting.

DiGress (Vignac et al. 2023) Appendix F.1 reports MMD as
``r = MMD²(generated, test) / MMD²(training, test)`` rather than the raw
MMD². The baseline denominator depends only on the dataset and the kernel
configuration, so it is a one-shot computation that can be cached and
shared across all training runs that emit the same raw MMD².

The cache stores **both** forms per metric:

- ``mmd_squared`` — what ``compute_mmd`` returns directly
  (``k11 + k22 - 2 k12``). This is the form used in DiGress's ratio.
- ``mmd`` — the square root, the "raw MMD" in the GraphRNN/GRAN
  reporting convention. Some papers (incl. HiGen Table 1) report the
  unsquared form, so we keep both side by side to avoid ambiguity.

This module defines:

- The on-disk schema (``MMDBaseline``) — JSON file per dataset, keyed by
  every parameter that affects the MMD² value.
- ``baseline_path(dataset, root)`` — canonical filesystem location.
- ``load_baseline(dataset, root)`` — loader with parameter-fingerprint
  validation.
- ``compute_ratios(raw, baseline)`` — divide a fresh ``MMD²(generated,
  test)`` dict by the cached ``MMD²(train, test)`` to recover DiGress's
  reported ratio units.

The baseline files themselves are produced by
``scripts/compute_mmd_baselines.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Default location for cached baselines. ``data/`` is gitignored, so
# baselines are local artefacts; regenerate via
# ``scripts/compute_mmd_baselines.py``.
DEFAULT_BASELINE_ROOT = Path("data/eval/mmd_baselines")


@dataclass(frozen=True)
class MMDBaselineParams:
    """Parameters that affect the MMD² value.

    Anything that influences ``compute_mmd`` output belongs here so two
    baselines computed with different settings cannot silently overwrite
    or be mistaken for each other.

    The ``subsample_*`` fields are present because molecular datasets
    (QM9, MOSES, GuacaMol) have 1e5–1e6 graphs per split — full
    pairwise MMD is N², so we cap each split to a deterministic random
    subsample. ``None`` (the default) means "use the full split". When
    set, the resulting MMD² value depends on ``subsample_seed``, so all
    three fields participate in the fingerprint.
    """

    kernel: str
    degree_sigma: float
    clustering_sigma: float
    spectral_sigma: float
    orbit_sigma: float
    clustering_bins: int
    spectral_bins: int
    subsample_n_train: int | None = None
    subsample_n_test: int | None = None
    subsample_seed: int = 42

    def fingerprint(self) -> str:
        """16-hex-char content hash over the parameter tuple."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class MMDBaseline:
    """A cached MMD²(train, test) baseline for one dataset.

    Attributes
    ----------
    dataset
        Dataset name (matches the script's ``--dataset`` flag).
    n_train, n_test
        Materialised split sizes used for the baseline.
    params
        Parameter set the baseline was computed under.
    mmd_squared
        Per-metric ``MMD²(train, test)`` values. Keys: ``degree_mmd``,
        ``clustering_mmd``, ``spectral_mmd``, ``orbit_mmd``. The
        ``compute_mmd`` function in this codebase already returns the
        squared statistic (``k11 + k22 - 2 k12``), so these can be used
        directly as the ratio denominator without re-squaring.
    mmd
        Per-metric raw MMD values (square root of ``mmd_squared``).
        Same keys. Provided for convenience when comparing against
        papers that report unsquared MMD (e.g. HiGen Table 1) without
        forcing every caller to take a square root.
    computed_at
        ISO-8601 UTC timestamp of when the baseline was computed.
    git_sha
        Repo HEAD at compute time (best-effort; ``"unknown"`` outside a
        git checkout).
    extra
        Free-form metadata (e.g. fixture path, datamodule class name).
    """

    dataset: str
    n_train: int
    n_test: int
    params: MMDBaselineParams
    mmd_squared: dict[str, float]
    mmd: dict[str, float]
    computed_at: str
    git_sha: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "dataset": self.dataset,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "params": asdict(self.params),
            "fingerprint": self.params.fingerprint(),
            "mmd_squared": self.mmd_squared,
            "mmd": self.mmd,
            "computed_at": self.computed_at,
            "git_sha": self.git_sha,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MMDBaseline:
        """Round-trip from ``to_dict`` output.

        Older files predating the ``mmd`` field are accepted: when the
        key is absent we derive it from ``mmd_squared`` rather than
        forcing a regenerate. New files always carry both.
        """
        params = MMDBaselineParams(**payload["params"])
        stored_fp = payload.get("fingerprint")
        if stored_fp is not None and stored_fp != params.fingerprint():
            raise ValueError(
                f"MMD baseline for dataset={payload['dataset']!r} has "
                f"fingerprint mismatch: file says {stored_fp!r}, "
                f"recomputed from params is {params.fingerprint()!r}. "
                "The file was edited or the schema drifted; regenerate "
                "via scripts/compute_mmd_baselines.py."
            )
        mmd_squared: dict[str, float] = payload["mmd_squared"]
        mmd: dict[str, float] = payload.get(
            "mmd",
            {k: float(v) ** 0.5 for k, v in mmd_squared.items()},
        )
        return cls(
            dataset=payload["dataset"],
            n_train=payload["n_train"],
            n_test=payload["n_test"],
            params=params,
            mmd_squared=mmd_squared,
            mmd=mmd,
            computed_at=payload["computed_at"],
            git_sha=payload["git_sha"],
            extra=payload.get("extra", {}),
        )


def baseline_path(dataset: str, root: Path | None = None) -> Path:
    """Canonical filesystem path for a dataset's cached baseline.

    Parameters
    ----------
    dataset
        Dataset name. One file per dataset; if a single dataset needs
        multiple baselines (e.g. comparing kernel choices) callers can
        supply different ``root`` directories.
    root
        Override the default ``data/eval/mmd_baselines`` location.

    Returns
    -------
    pathlib.Path
        ``{root}/{dataset}.json``. The parent directory is *not* created.
    """
    base = root if root is not None else DEFAULT_BASELINE_ROOT
    return base / f"{dataset}.json"


def load_baseline(dataset: str, root: Path | None = None) -> MMDBaseline:
    """Load a cached baseline.

    Parameters
    ----------
    dataset
        Dataset name.
    root
        Override directory. Defaults to ``data/eval/mmd_baselines``.

    Returns
    -------
    MMDBaseline
        The deserialised, fingerprint-validated baseline.

    Raises
    ------
    FileNotFoundError
        When no baseline file exists for ``dataset``. Callers should
        regenerate via ``scripts/compute_mmd_baselines.py``.
    ValueError
        When the on-disk fingerprint disagrees with the params it was
        stored alongside, indicating manual edits or schema drift.
    """
    path = baseline_path(dataset, root=root)
    if not path.exists():
        raise FileNotFoundError(
            f"No MMD baseline cached for dataset={dataset!r} at {path}. "
            "Regenerate via: "
            "uv run scripts/compute_mmd_baselines.py --dataset "
            f"{dataset}"
        )
    with path.open() as f:
        payload = json.load(f)
    return MMDBaseline.from_dict(payload)


def save_baseline(baseline: MMDBaseline, root: Path | None = None) -> Path:
    """Atomically write a baseline to disk.

    Writes to ``{path}.tmp`` then ``rename``s to ``{path}`` so a partial
    write cannot leave a corrupt cache file behind.

    Parameters
    ----------
    baseline
        The baseline to persist.
    root
        Override directory.

    Returns
    -------
    pathlib.Path
        Final on-disk path (``baseline_path(dataset, root)``).
    """
    path = baseline_path(baseline.dataset, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(baseline.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)
    return path


def compute_ratios(
    raw_mmd_squared: dict[str, float],
    baseline: MMDBaseline,
) -> dict[str, float]:
    """Divide a ``MMD²(generated, test)`` dict by the cached baseline.

    Parameters
    ----------
    raw_mmd_squared
        Per-metric MMD² values, e.g. the ``gen-val/*_mmd`` keys logged
        during training. Only metrics present in *both* dicts produce
        ratio entries; missing metrics are skipped silently because not
        every training config logs every metric (e.g. orbit MMD requires
        the ORCA binary).
    baseline
        Cached train↔test baseline, typically loaded via
        ``load_baseline``.

    Returns
    -------
    dict[str, float]
        Map ``{metric}: r`` where
        ``r = MMD²(generated, test) / MMD²(train, test)``. Matches
        DiGress Appendix F.1's reported ratio convention.

    Raises
    ------
    ZeroDivisionError
        When a baseline metric is exactly zero. This indicates either a
        degenerate dataset split or a bug in baseline computation; we
        fail loud rather than silently yield ``inf``.
    """
    ratios: dict[str, float] = {}
    for metric, value in raw_mmd_squared.items():
        if metric not in baseline.mmd_squared:
            continue
        denom = baseline.mmd_squared[metric]
        if denom == 0.0:
            raise ZeroDivisionError(
                f"MMD baseline for dataset={baseline.dataset!r} has "
                f"{metric}=0.0 — cannot form the DiGress ratio. "
                "Inspect the baseline and regenerate."
            )
        ratios[metric] = value / denom
    return ratios

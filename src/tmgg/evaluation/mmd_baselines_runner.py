"""Reusable core for computing MMD² baselines.

Lifted out of ``scripts/compute_mmd_baselines.py`` so the same logic
can run from a Modal function without going through the CLI. The
script keeps the argparse wrapper and re-exports
``DATASET_BUILDERS`` / ``DEFAULT_SUBSAMPLE`` / ``compute_baseline``
from here.

Naming: this module is the *runner* — it knows how to build a
datamodule, materialise splits, subsample, and call ``compute_mmd``.
The serialisation schema lives in :mod:`tmgg.evaluation.mmd_baselines`.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import random
import subprocess
import time
from collections.abc import Callable
from typing import Any

import networkx as nx  # pyright: ignore[reportMissingModuleSource]
from torch_geometric.data import Data  # pyright: ignore[reportMissingImports]
from torch_geometric.utils import to_networkx  # pyright: ignore[reportMissingImports]

from tmgg.evaluation.graph_evaluator import compute_orbit_mmd
from tmgg.evaluation.mmd_baselines import MMDBaseline, MMDBaselineParams
from tmgg.evaluation.mmd_metrics import compute_mmd_metrics

logger = logging.getLogger(__name__)


# Production sigma defaults: these match
# ``configs/models/discrete/discrete_sbm_official.yaml`` and the
# ``compute_orbit_mmd`` default in ``graph_evaluator.py``. Changing any
# of these changes the meaning of ``gen-val/*_mmd`` numbers and MUST be
# reflected in the produced baseline's fingerprint.
DEFAULT_KERNEL = "gaussian_tv"
DEFAULT_DEGREE_SIGMA = 1.0
DEFAULT_CLUSTERING_SIGMA = 0.1  # DiGress uses 0.1 for clustering
DEFAULT_SPECTRAL_SIGMA = 1.0
DEFAULT_ORBIT_SIGMA = 30.0
DEFAULT_CLUSTERING_BINS = 100
DEFAULT_SPECTRAL_BINS = 200


# ---------------------------------------------------------------------------
# Datamodule construction
# ---------------------------------------------------------------------------


def _build_spectre_sbm() -> Any:
    from tmgg.data.data_modules.spectre_sbm import SpectreSBMDataModule

    return SpectreSBMDataModule()


def _build_spectre_planar() -> Any:
    from tmgg.data.data_modules.spectre_planar import SpectrePlanarDataModule

    return SpectrePlanarDataModule()


def _build_pyg_enzymes() -> Any:
    from tmgg.data import GraphDataModule

    # Mirrors ``configs/data/pyg_enzymes.yaml``. Splits are 70 / 10 / 20.
    return GraphDataModule(
        graph_type="enzymes",
        graph_config={
            "num_nodes": 126,  # published ENZYMES max; PyG datasets ignore at runtime
            "root": None,
            "max_graphs": None,
            "seed": 42,
        },
        samples_per_graph=1,  # baseline uses the raw set, not repeated samples
        train_ratio=0.7,
        val_ratio=0.1,
        seed=42,
    )


def _build_pyg_proteins() -> Any:
    from tmgg.data import GraphDataModule

    # Mirrors ``configs/data/pyg_proteins.yaml``. Splits are 70 / 10 / 20.
    return GraphDataModule(
        graph_type="proteins",
        graph_config={
            "num_nodes": 620,  # published PROTEINS max
            "root": None,
            "max_graphs": None,
            "seed": 42,
        },
        samples_per_graph=1,
        train_ratio=0.7,
        val_ratio=0.1,
        seed=42,
    )


def _build_qm9() -> Any:
    from tmgg.data.data_modules.molecular.qm9 import QM9DataModule

    # Mirrors ``configs/data/qm9_digress.yaml``. The molecular module
    # does its own internal train/val/test split per upstream cvignac
    # /DiGress. ``cache_root=None`` lets it use the default.
    return QM9DataModule(
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        seed=42,
        cache_root=None,
    )


def _build_moses() -> Any:
    from tmgg.data.data_modules.molecular.moses import MOSESDataModule

    return MOSESDataModule(
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        seed=42,
        cache_root=None,
    )


def _build_guacamol() -> Any:
    from tmgg.data.data_modules.molecular.guacamol import GuacaMolDataModule

    return GuacaMolDataModule(
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        seed=42,
        cache_root=None,
    )


# Registry: dataset name → zero-arg builder. Add new entries here when a
# new dataset config lands.
DATASET_BUILDERS: dict[str, Callable[[], Any]] = {
    "spectre_sbm": _build_spectre_sbm,
    "spectre_planar": _build_spectre_planar,
    "pyg_enzymes": _build_pyg_enzymes,
    "pyg_proteins": _build_pyg_proteins,
    "qm9": _build_qm9,
    "moses": _build_moses,
    "guacamol": _build_guacamol,
}

# Per-dataset suggested subsample caps. Molecular datasets have
# 1e5–1e6 graphs per split; full pairwise MMD is N², so we cap each
# side to keep one Modal CPU container under an hour. ``None`` =
# use the full split (default for the small graph-statistics datasets).
DEFAULT_SUBSAMPLE: dict[str, tuple[int | None, int | None]] = {
    "spectre_sbm": (None, None),
    "spectre_planar": (None, None),
    "pyg_enzymes": (None, None),
    "pyg_proteins": (None, None),
    "qm9": (1024, 1024),
    "moses": (1024, 1024),
    "guacamol": (1024, 1024),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Return the repo HEAD's short SHA, or ``"unknown"`` if not in a checkout."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _data_to_nx(d: Data) -> nx.Graph[Any]:
    """Convert a single PyG ``Data`` to an undirected NetworkX graph.

    Drops node / edge attributes — the MMD metrics in this codebase
    operate on plain undirected simple graphs.
    """
    return to_networkx(d, to_undirected=True)


def _split_graphs(dm: Any, split: str) -> list[nx.Graph[Any]]:
    """Pull the materialised graphs for a split and convert to NetworkX.

    Supports both datamodule shapes used in this repo:

    - PyG / SPECTRE convention: ``dm._{split}_data`` is a ``list[Data]``.
      Used by ``GraphDataModule`` (enzymes, proteins) and by
      ``SpectreSBMDataModule`` / ``SpectrePlanarDataModule``.
    - Molecular convention: ``dm._{split}_dataset._graphs`` is a
      ``list[DenseGraphState]`` (the in-house dense state container
      with a ``to_networkx_list()`` method). Used by ``QM9DataModule``,
      ``MOSESDataModule``, ``GuacaMolDataModule``.

    Crashes loud if neither convention applies.
    """
    pyg_attr = f"_{split}_data"
    if hasattr(dm, pyg_attr) and getattr(dm, pyg_attr) is not None:
        payload = getattr(dm, pyg_attr)
        return [_data_to_nx(d) for d in payload]

    mol_attr = f"_{split}_dataset"
    if hasattr(dm, mol_attr) and getattr(dm, mol_attr) is not None:
        ds = getattr(dm, mol_attr)
        graphs = ds._graphs
        if graphs is None:
            raise RuntimeError(
                f"{type(dm).__name__}.{mol_attr}._graphs is None; "
                "did setup() populate the split?"
            )
        # Each element of ``_graphs`` is a ``DenseGraphState`` that may
        # carry a batch dimension (even of size 1). ``to_networkx_list()``
        # expands batched data and returns a single-element list for
        # unbatched — safe in both cases. Flatten the result.
        return [h for g in graphs for h in g.to_networkx_list()]

    raise RuntimeError(
        f"{type(dm).__name__} does not expose split {split!r} via "
        f"either ``_{split}_data`` (PyG) or ``_{split}_dataset._graphs`` "
        "(molecular). Add support for the new datamodule shape."
    )


def _subsample(
    graphs: list[nx.Graph[Any]],
    n: int | None,
    seed: int,
) -> list[nx.Graph[Any]]:
    """Deterministically downsample to ``n`` graphs.

    Returns the input unchanged when ``n`` is ``None`` or larger than
    the population. Used so QM9 / MOSES / GuacaMol-scale splits don't
    blow up the N² kernel evaluation.
    """
    if n is None or n >= len(graphs):
        return graphs
    rng = random.Random(seed)
    return rng.sample(graphs, n)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_baseline(
    dataset: str,
    params: MMDBaselineParams,
    *,
    max_workers: int | None = None,
) -> MMDBaseline:
    """Materialise splits and compute the four MMD² values.

    Parameters
    ----------
    dataset
        Dataset name. Must be a key of :data:`DATASET_BUILDERS`.
    params
        Kernel / bandwidth / bin / subsample configuration. Embedded in
        the returned baseline's fingerprint.
    max_workers
        Forwarded to ``compute_graph_statistics`` for parallel histogram
        computation.

    Returns
    -------
    MMDBaseline
        Ready-to-persist baseline record.
    """
    if dataset not in DATASET_BUILDERS:
        known = ", ".join(sorted(DATASET_BUILDERS))
        raise ValueError(f"Unknown dataset {dataset!r}. Known: {known}.")

    logger.info("[%s] instantiating datamodule…", dataset)
    dm = DATASET_BUILDERS[dataset]()

    logger.info("[%s] running setup()…", dataset)
    t0 = time.time()
    dm.setup()
    logger.info("[%s] setup() finished in %.1fs", dataset, time.time() - t0)

    train_g_full = _split_graphs(dm, "train")
    test_g_full = _split_graphs(dm, "test")
    logger.info(
        "[%s] full split sizes: train=%d, test=%d",
        dataset,
        len(train_g_full),
        len(test_g_full),
    )

    train_g = _subsample(train_g_full, params.subsample_n_train, params.subsample_seed)
    test_g = _subsample(test_g_full, params.subsample_n_test, params.subsample_seed + 1)
    if len(train_g) != len(train_g_full) or len(test_g) != len(test_g_full):
        logger.info(
            "[%s] subsampled to train=%d, test=%d (seed=%d)",
            dataset,
            len(train_g),
            len(test_g),
            params.subsample_seed,
        )

    if len(train_g) < 2 or len(test_g) < 2:
        raise RuntimeError(
            f"Dataset {dataset!r} has insufficient graphs for MMD: "
            f"train={len(train_g)}, test={len(test_g)} (need ≥ 2 each)."
        )

    logger.info("[%s] computing degree/clustering/spectral MMD²…", dataset)
    t1 = time.time()
    base_three = compute_mmd_metrics(
        ref_graphs=train_g,
        gen_graphs=test_g,
        kernel=params.kernel,  # type: ignore[arg-type]
        sigma=1.0,
        max_workers=max_workers,
        degree_sigma=params.degree_sigma,
        clustering_sigma=params.clustering_sigma,
        spectral_sigma=params.spectral_sigma,
    )
    logger.info("[%s] three-stat MMD² done in %.1fs", dataset, time.time() - t1)

    logger.info("[%s] computing orbit MMD² (ORCA)…", dataset)
    t2 = time.time()
    orbit_value = compute_orbit_mmd(
        ref_graphs=train_g,
        gen_graphs=test_g,
        kernel=params.kernel,  # type: ignore[arg-type]
        sigma=params.orbit_sigma,
    )
    logger.info("[%s] orbit MMD² done in %.1fs", dataset, time.time() - t2)

    mmd_squared = {
        "degree_mmd": base_three.degree_mmd,
        "clustering_mmd": base_three.clustering_mmd,
        "spectral_mmd": base_three.spectral_mmd,
        "orbit_mmd": orbit_value,
    }
    # Raw (un-squared) MMD — the GraphRNN/GRAN reporting convention used
    # by some reproductions (e.g. HiGen Table 1). ``compute_mmd`` clamps
    # the V-statistic to ``[0, ∞)``, so the sqrt is always real.
    mmd = {k: math.sqrt(v) for k, v in mmd_squared.items()}

    return MMDBaseline(
        dataset=dataset,
        n_train=len(train_g),
        n_test=len(test_g),
        params=params,
        mmd_squared=mmd_squared,
        mmd=mmd,
        computed_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        extra={
            "datamodule_class": type(dm).__name__,
            "n_train_full": len(train_g_full),
            "n_test_full": len(test_g_full),
        },
    )

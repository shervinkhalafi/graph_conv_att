"""Modal app: train↔test MMD² baseline computation for the cached anchor.

One CPU-only ``@app.function`` parametrised on dataset name. Computes
the four-metric ``MMD²(train, test)`` via the same code path as
``scripts/compute_mmd_baselines.py``, writing the resulting JSON to
the ``graph-denoising-outputs`` Modal volume at::

    /data/outputs/mmd_baselines/<dataset>.json

then pulled back to the host via ``modal volume get``.

CPU-only because every metric is histogram + kernel arithmetic;
ORCA (the orbit-counting binary) is bundled into the experiment image
by ``src/tmgg/modal/_lib/image.py``, so this app does not need GPU.

Deploy::

    uv run modal deploy -m tmgg.modal._mmd_baseline_functions

Run one dataset (after deploy)::

    uv run modal run \\
        tmgg.modal._mmd_baseline_functions::compute_baseline \\
        --dataset spectre_sbm

See :mod:`tmgg.evaluation.mmd_baselines_runner` for the registry of
supported datasets and per-dataset subsample defaults.
"""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from typing import Any

import modal

from tmgg.modal._lib.image import create_tmgg_image
from tmgg.modal._lib.paths import discover_source_checkout_path
from tmgg.modal._lib.volumes import get_volume_mounts
from tmgg.modal.app import (
    CPU_PROFILES,
    DEFAULT_SCALEDOWN_WINDOW,
)

MMD_BASELINE_APP_NAME = "graph-denoising-mmd-baselines"
# 1h is plenty for the small graph-statistics datasets and is the upper
# bound for a 1024-cap molecular run with 200-bin spectral kernels and
# 30-sigma ORCA orbit MMD.
MMD_BASELINE_TIMEOUT = 3600
# Output directory on the ``graph-denoising-outputs`` volume. Does not collide
# with training run outputs (which live at ``/data/outputs/<exp>/<run>/``).
OUTPUT_DIR = "/data/outputs/mmd_baselines"

app = modal.App(MMD_BASELINE_APP_NAME, include_source=False)

experiment_image = create_tmgg_image(discover_source_checkout_path())


@app.function(
    name="compute_baseline",
    image=experiment_image,
    cpu=CPU_PROFILES["fast"],  # 8 vCPU; histogram/kernel work parallelises well
    timeout=MMD_BASELINE_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    # get_volume_mounts() returns dict[str, Any] (shared infra) which
    # basedpyright rejects against modal's invariant dict[str | PurePosixPath,
    # Volume | CloudBucketMount]. Silenced per shared-helper pattern.
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def compute_baseline(
    dataset: str,
    *,
    kernel: str = "gaussian_tv",
    degree_sigma: float = 1.0,
    clustering_sigma: float = 0.1,
    spectral_sigma: float = 1.0,
    orbit_sigma: float = 30.0,
    clustering_bins: int = 100,
    spectral_bins: int = 200,
    subsample_n_train: int | None = None,
    subsample_n_test: int | None = None,
    subsample_seed: int = 42,
    use_dataset_default_subsample: bool = True,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Compute and persist the MMD²(train, test) baseline for ``dataset``.

    Mirrors ``scripts/compute_mmd_baselines.py`` but as a Modal function.
    The serialisation format matches that script byte-for-byte (same
    ``MMDBaseline.to_dict()``), so a baseline produced here is loadable
    via ``tmgg.evaluation.load_baseline`` on the host side.

    Parameters
    ----------
    dataset
        One of the keys of :data:`DATASET_BUILDERS`.
    kernel, degree_sigma, clustering_sigma, spectral_sigma, orbit_sigma,
    clustering_bins, spectral_bins
        Kernel and bandwidth knobs. Defaults match the production eval
        path (``configs/models/discrete/discrete_sbm_official.yaml``).
    subsample_n_train, subsample_n_test
        Override the dataset-default cap. ``None`` plus
        ``use_dataset_default_subsample=True`` means "use the cap from
        ``DEFAULT_SUBSAMPLE``" (i.e. 1024 for QM9/MOSES/GuacaMol, no cap
        elsewhere). Pass an explicit number — including zero or negative
        — to override.
    subsample_seed
        Deterministic seed for the random sample. Embedded in the
        baseline fingerprint.
    use_dataset_default_subsample
        When ``True`` (the default), missing ``subsample_n_*`` fall back
        to ``DEFAULT_SUBSAMPLE[dataset]``. When ``False``, missing
        values mean "no cap regardless of dataset".
    max_workers
        Forwarded to ``compute_graph_statistics``. ``None`` = library
        default (cpu_count).

    Returns
    -------
    dict
        ``MMDBaseline.to_dict()`` payload — JSON-safe dict with
        ``dataset``, ``n_train``, ``n_test``, ``params``, ``fingerprint``,
        ``mmd_squared``, ``mmd``, ``computed_at``, ``git_sha``, ``extra``.
        The same JSON is also written to
        ``/data/outputs/mmd_baselines/{dataset}.json``.
    """
    import logging
    from pathlib import Path

    from tmgg.evaluation.mmd_baselines import MMDBaselineParams, save_baseline
    from tmgg.evaluation.mmd_baselines_runner import (
        DEFAULT_SUBSAMPLE,
    )
    from tmgg.evaluation.mmd_baselines_runner import (
        compute_baseline as _compute_baseline,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("tmgg.modal.mmd_baselines")

    # Resolve subsample defaults. Mirrors scripts/compute_mmd_baselines.py
    # but inline so the Modal function is self-contained.
    if use_dataset_default_subsample:
        default_train, default_test = DEFAULT_SUBSAMPLE.get(dataset, (None, None))
        if subsample_n_train is None:
            subsample_n_train = default_train
        if subsample_n_test is None:
            subsample_n_test = default_test

    params = MMDBaselineParams(
        kernel=kernel,
        degree_sigma=degree_sigma,
        clustering_sigma=clustering_sigma,
        spectral_sigma=spectral_sigma,
        orbit_sigma=orbit_sigma,
        clustering_bins=clustering_bins,
        spectral_bins=spectral_bins,
        subsample_n_train=subsample_n_train,
        subsample_n_test=subsample_n_test,
        subsample_seed=subsample_seed,
    )
    logger.info(
        "[%s] params fingerprint=%s subsample=(train=%s, test=%s)",
        dataset,
        params.fingerprint(),
        subsample_n_train,
        subsample_n_test,
    )

    baseline = _compute_baseline(dataset, params, max_workers=max_workers)
    out_root = Path(OUTPUT_DIR)
    out_path = save_baseline(baseline, root=out_root)
    logger.info("[%s] wrote %s", dataset, out_path)
    for k in sorted(baseline.mmd_squared):
        logger.info(
            "[%s]   %-16s mmd²=%.6e  mmd=%.6e",
            dataset,
            k,
            baseline.mmd_squared[k],
            baseline.mmd[k],
        )

    # Modal commits volume writes on function return automatically; no
    # explicit commit() needed here. ``modal volume get graph-denoising-outputs
    # /data/outputs/mmd_baselines/{dataset}.json …`` will see this file
    # once the function returns.

    return baseline.to_dict()

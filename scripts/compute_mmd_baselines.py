#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tmgg",
# ]
# [tool.uv.sources]
# tmgg = { path = "..", editable = true }
# ///
"""Compute and cache the per-dataset ``MMD²(train, test)`` baseline.

DiGress (Vignac et al. 2023) Appendix F.1 reports MMD as a ratio
``r = MMD²(generated, test) / MMD²(training, test)``. The denominator is
dataset-determined and kernel-determined — once per dataset, then reused
by every training run. This script materialises the train / test splits
through the project's datamodules, computes ``MMD²(train, test)`` for the
four metrics our pipeline emits (degree, clustering, spectral, orbit),
and writes one JSON per dataset to ``data/eval/mmd_baselines/``.

Schema and loader live in ``src/tmgg/evaluation/mmd_baselines.py``.
The compute logic lives in ``src/tmgg/evaluation/mmd_baselines_runner.py``
so the same path is reachable from a Modal function (see
``src/tmgg/modal/_mmd_baseline_functions.py``).

Usage
-----
    # All built-in datasets:
    uv run scripts/compute_mmd_baselines.py --all

    # One dataset:
    uv run scripts/compute_mmd_baselines.py --dataset spectre_sbm

    # Custom output root:
    uv run scripts/compute_mmd_baselines.py --all --out path/to/baselines

The script is **not** idempotent on existing files by default — pass
``--skip-existing`` to keep a previously computed baseline rather than
recomputing it. Parameter overrides (kernel/sigma/bins) get embedded in
the file's fingerprint, so they can never silently overwrite a baseline
computed under a different configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tmgg.evaluation.mmd_baselines import (
    MMDBaselineParams,
    baseline_path,
    save_baseline,
)
from tmgg.evaluation.mmd_baselines_runner import (
    DATASET_BUILDERS,
    DEFAULT_CLUSTERING_BINS,
    DEFAULT_CLUSTERING_SIGMA,
    DEFAULT_DEGREE_SIGMA,
    DEFAULT_KERNEL,
    DEFAULT_ORBIT_SIGMA,
    DEFAULT_SPECTRAL_BINS,
    DEFAULT_SPECTRAL_SIGMA,
    DEFAULT_SUBSAMPLE,
    compute_baseline,
)

logger = logging.getLogger("compute_mmd_baselines")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    doc = __doc__ or ""
    description, _, epilog = doc.partition("\n")
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog or None,
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_BUILDERS),
        help="Dataset to compute. Repeat for multiple.",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Compute baselines for every registered dataset.",
    )

    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: data/eval/mmd_baselines).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recompute when an output JSON already exists.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel workers for histogram computation.",
    )

    g = p.add_argument_group("MMD parameters (defaults match production eval)")
    g.add_argument(
        "--kernel", default=DEFAULT_KERNEL, choices=["gaussian", "gaussian_tv"]
    )
    g.add_argument("--degree-sigma", type=float, default=DEFAULT_DEGREE_SIGMA)
    g.add_argument("--clustering-sigma", type=float, default=DEFAULT_CLUSTERING_SIGMA)
    g.add_argument("--spectral-sigma", type=float, default=DEFAULT_SPECTRAL_SIGMA)
    g.add_argument("--orbit-sigma", type=float, default=DEFAULT_ORBIT_SIGMA)
    g.add_argument("--clustering-bins", type=int, default=DEFAULT_CLUSTERING_BINS)
    g.add_argument("--spectral-bins", type=int, default=DEFAULT_SPECTRAL_BINS)

    s = p.add_argument_group(
        "Subsampling (molecular sets need this; small datasets do not)"
    )
    s.add_argument(
        "--subsample-train",
        type=int,
        default=None,
        help=(
            "Cap train-side graph count. Default: dataset-specific "
            "(see DEFAULT_SUBSAMPLE in this script). 0 or negative → use full split."
        ),
    )
    s.add_argument(
        "--subsample-test",
        type=int,
        default=None,
        help="Cap test-side graph count. Default: dataset-specific.",
    )
    s.add_argument(
        "--subsample-seed",
        type=int,
        default=42,
        help="RNG seed used by the deterministic subsample. Embedded in fingerprint.",
    )
    s.add_argument(
        "--no-subsample",
        action="store_true",
        help=(
            "Disable subsampling for ALL datasets, regardless of "
            "DEFAULT_SUBSAMPLE. Use this for full-set baselines on "
            "molecular datasets if you have hours of CPU to burn."
        ),
    )

    return p


def _resolve_subsample(
    dataset: str,
    args: argparse.Namespace,
) -> tuple[int | None, int | None]:
    """Pick the subsample sizes for ``dataset`` based on CLI flags + defaults."""
    if args.no_subsample:
        return None, None
    default_train, default_test = DEFAULT_SUBSAMPLE.get(dataset, (None, None))
    sub_train = (
        args.subsample_train if args.subsample_train is not None else default_train
    )
    sub_test = args.subsample_test if args.subsample_test is not None else default_test
    if sub_train is not None and sub_train <= 0:
        sub_train = None
    if sub_test is not None and sub_test <= 0:
        sub_test = None
    return sub_train, sub_test


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    datasets: list[str]
    if args.all:
        datasets = sorted(DATASET_BUILDERS)
    else:
        # ``--dataset`` is ``action="append"``; one or more entries.
        datasets = list(dict.fromkeys(args.dataset))  # de-dup, keep order

    failures: list[str] = []
    for ds in datasets:
        sub_train, sub_test = _resolve_subsample(ds, args)
        params = MMDBaselineParams(
            kernel=args.kernel,
            degree_sigma=args.degree_sigma,
            clustering_sigma=args.clustering_sigma,
            spectral_sigma=args.spectral_sigma,
            orbit_sigma=args.orbit_sigma,
            clustering_bins=args.clustering_bins,
            spectral_bins=args.spectral_bins,
            subsample_n_train=sub_train,
            subsample_n_test=sub_test,
            subsample_seed=args.subsample_seed,
        )
        logger.info(
            "[%s] params fingerprint=%s subsample=(train=%s, test=%s)",
            ds,
            params.fingerprint(),
            sub_train,
            sub_test,
        )

        path = baseline_path(ds, root=args.out)
        if args.skip_existing and path.exists():
            logger.info("[%s] %s exists; skipping (--skip-existing)", ds, path)
            continue
        try:
            baseline = compute_baseline(ds, params, max_workers=args.max_workers)
        except Exception as exc:
            # Fail loud per CLAUDE.md (no graceful fallback) but still
            # try the remaining datasets so one bad fixture does not
            # block the whole batch. Per-dataset error printed; non-zero
            # exit at the end.
            logger.exception("[%s] baseline computation failed: %s", ds, exc)
            failures.append(ds)
            continue
        out_path = save_baseline(baseline, root=args.out)
        logger.info("[%s] wrote %s", ds, out_path)
        for k in sorted(baseline.mmd_squared):
            logger.info(
                "[%s]   %-16s mmd²=%.6e  mmd=%.6e",
                ds,
                k,
                baseline.mmd_squared[k],
                baseline.mmd[k],
            )

    if failures:
        logger.error(
            "Completed with failures: %s",
            ", ".join(failures),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

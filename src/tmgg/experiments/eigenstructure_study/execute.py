"""Hydra-compatible execution entry point for eigenstructure study.

Dispatches to the appropriate pipeline phase (collect, analyze, noised,
compare, covariance) based on ``config.phase``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def execute_eigenstructure(config: DictConfig) -> dict[str, Any]:
    """Execute an eigenstructure study phase from Hydra config.

    Parameters
    ----------
    config
        Resolved OmegaConf configuration. Required keys:

        - ``phase``: one of ``"collect"``, ``"analyze"``, ``"noised"``,
          ``"compare"``, ``"covariance"``
        - ``paths.output_dir``: base output directory
        - ``seed``: random seed

        Phase-specific keys are documented in the corresponding
        ``_run_*`` functions.

    Returns
    -------
    dict
        Result dict with at least ``{"status": "completed", "phase": ...}``.
    """
    phase = str(config.phase)
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "collect": _run_collect,
        "analyze": _run_analyze,
        "noised": _run_noised,
        "compare": _run_compare,
        "covariance": _run_covariance,
    }

    if phase not in dispatch:
        raise ValueError(
            f"Unknown eigenstructure phase: {phase!r}. "
            f"Expected one of {list(dispatch)}."
        )

    return dispatch[phase](config, output_dir)


def _run_collect(config: DictConfig, output_dir: Path) -> dict[str, Any]:
    from tmgg.experiments.eigenstructure_study.collector import EigenstructureCollector

    dataset_config_raw = OmegaConf.to_container(config.dataset.config, resolve=True)
    if not isinstance(dataset_config_raw, dict):
        raise TypeError(
            f"Expected dict for dataset.config, got {type(dataset_config_raw)}"
        )
    dataset_config: dict[str, Any] = {str(k): v for k, v in dataset_config_raw.items()}

    collector = EigenstructureCollector(
        dataset_name=config.dataset.name,
        dataset_config=dataset_config,
        output_dir=output_dir / "original",
        batch_size=config.get("batch_size", 64),
        seed=config.seed,
    )
    collector.collect()

    return {"status": "completed", "phase": "collect"}


def _run_analyze(config: DictConfig, output_dir: Path) -> dict[str, Any]:
    from tmgg.experiments.eigenstructure_study.analyzer import SpectralAnalyzer

    input_dir = output_dir / "original"
    analysis_cfg = config.get("analysis", {})
    subspace_k: int = analysis_cfg.get("subspace_k", 10)
    compute_covariance: bool = analysis_cfg.get("compute_covariance", True)
    matrix_type: str = analysis_cfg.get("matrix_type", "adjacency")

    analyzer = SpectralAnalyzer(input_dir)
    result = analyzer.analyze()

    analysis_dir = output_dir / "analysis"
    analyzer.save_results(result, analysis_dir)

    if subspace_k > 0:
        subspace_results = analyzer.compute_subspace_distances(k=subspace_k)
        with open(analysis_dir / "subspace_analysis.json", "w") as f:
            json.dump(subspace_results, f, indent=2)

    if compute_covariance:
        cov_result = analyzer.compute_eigenvalue_covariance(matrix_type)
        with open(analysis_dir / "covariance.json", "w") as f:
            json.dump(asdict(cov_result), f, indent=2)

    return {
        "status": "completed",
        "phase": "analyze",
        "spectral_gap_mean": result.spectral_gap_mean,
        "algebraic_connectivity_mean": result.algebraic_connectivity_mean,
    }


def _run_noised(config: DictConfig, output_dir: Path) -> dict[str, Any]:
    from tmgg.experiments.eigenstructure_study.noised_collector import (
        NoisedEigenstructureCollector,
    )

    noise_cfg = config.noise
    noise_levels = list(noise_cfg.levels)

    collector = NoisedEigenstructureCollector(
        input_dir=output_dir / "original",
        output_dir=output_dir / "noised",
        noise_type=noise_cfg.type,
        noise_levels=noise_levels,
        rotation_k=noise_cfg.get("rotation_k"),
        seed=config.seed,
    )
    collector.collect()

    return {
        "status": "completed",
        "phase": "noised",
        "noise_levels": noise_levels,
    }


def _run_compare(config: DictConfig, output_dir: Path) -> dict[str, Any]:
    from tmgg.experiments.eigenstructure_study.noised_collector import (
        NoisedAnalysisComparator,
    )

    comp_cfg = config.get("comparison", {})
    subspace_k: int = comp_cfg.get("subspace_k", 10)
    procrustes_k_values: list[int] = list(
        comp_cfg.get("procrustes_k_values", [1, 2, 4, 8, 16])
    )
    compute_cov_evolution: bool = comp_cfg.get("compute_covariance_evolution", True)
    matrix_type: str = config.get("analysis", {}).get("matrix_type", "adjacency")

    original_dir = output_dir / "original"
    noised_dir = output_dir / "noised"

    comparator = NoisedAnalysisComparator(original_dir, noised_dir)
    results = comparator.compute_full_comparison(
        k=subspace_k, procrustes_k_values=procrustes_k_values
    )

    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    with open(comparison_dir / "comparison.json", "w") as f:
        json.dump([result.to_json_dict() for result in results], f, indent=2)

    if compute_cov_evolution:
        evolution = comparator.compute_covariance_evolution(matrix_type)
        with open(comparison_dir / "covariance_evolution.json", "w") as f:
            json.dump(evolution.to_json_dict(), f, indent=2)

    return {
        "status": "completed",
        "phase": "compare",
        "num_comparisons": len(results),
    }


def _run_covariance(config: DictConfig, output_dir: Path) -> dict[str, Any]:
    from tmgg.experiments.eigenstructure_study.analyzer import SpectralAnalyzer

    analysis_cfg = config.get("analysis", {})
    matrix_type: str = analysis_cfg.get("matrix_type", "adjacency")

    analyzer = SpectralAnalyzer(output_dir / "original")
    cov_result = analyzer.compute_eigenvalue_covariance(matrix_type)

    cov_dir = output_dir / "covariance"
    cov_dir.mkdir(parents=True, exist_ok=True)
    with open(cov_dir / "covariance.json", "w") as f:
        json.dump(asdict(cov_result), f, indent=2)

    return {
        "status": "completed",
        "phase": "covariance",
        "frobenius_norm": cov_result.frobenius_norm,
    }

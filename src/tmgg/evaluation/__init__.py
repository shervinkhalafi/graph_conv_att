"""Evaluation metrics for graph denoising and generation quality.

Combines per-graph spectral metrics (eigenvalue error, subspace distance,
accuracy) with distributional metrics (MMD on degree/clustering/spectral
distributions, ORCA orbit counting) and structural checks (SBM accuracy,
planarity, uniqueness, novelty).
"""

from tmgg.evaluation.graph_evaluator import (
    EvaluationResults,
    GraphEvaluator,
)
from tmgg.evaluation.mmd_baselines import (
    DEFAULT_BASELINE_ROOT,
    MMDBaseline,
    MMDBaselineParams,
    baseline_path,
    compute_ratios,
    load_baseline,
    save_baseline,
)
from tmgg.evaluation.reference_graphs import (
    generate_reference_graphs,
)

__all__ = [
    "DEFAULT_BASELINE_ROOT",
    "EvaluationResults",
    "GraphEvaluator",
    "MMDBaseline",
    "MMDBaselineParams",
    "baseline_path",
    "compute_ratios",
    "generate_reference_graphs",
    "load_baseline",
    "save_baseline",
]

"""Graph eigenstructure study module.

This module provides tools for collecting and analyzing the eigenstructure
of graph adjacency and Laplacian matrices across various dataset types.

Three-phase pipeline:
1. collect: Compute and store eigendecompositions
2. analyze: Band gap, spectral gap, eigenvector coherence analysis
3. noised: Same collection with noise applied at configurable levels

CLI Commands
------------
tmgg-eigenstructure collect   # Phase 1
tmgg-eigenstructure analyze   # Phase 2
tmgg-eigenstructure noised    # Phase 3
tmgg-eigenstructure compare   # Compare original vs noised
"""

from tmgg.utils.spectral.laplacian import compute_laplacian
from tmgg.utils.spectral.spectral_deltas import (
    compute_alg_connectivity_delta,
    compute_eigenvalue_drift,
    compute_subspace_distance_from_eigenvectors,
)

from .analyzer import (
    ImprovementGapResult,
    SpectralAnalysisResult,
    SpectralAnalyzer,
    compute_algebraic_connectivity,
    compute_algebraic_connectivity_delta,
    compute_B_invariants,
    compute_effective_rank,
    compute_eigengap_delta,
    compute_eigenvalue_entropy,
    compute_eigenvector_coherence,
    compute_principal_angles,
    compute_spectral_gap,
    estimate_improvement_gap,
)
from .collector import EigenstructureCollector
from .noised_collector import (
    CovarianceEvolutionItem,
    CovarianceEvolutionResult,
    DriftComparisonResult,
    GapDeltaComparisonResult,
    MeanStdMetric,
    NoisedAnalysisComparator,
    NoisedEigenstructureCollector,
    NoiseLevelComparisonResult,
    PrincipalAngleResult,
    ProcrustesKStats,
    ProcrustesRotationResult,
    SubspaceDistanceResult,
)
from .storage import (
    iter_batches,
    load_decomposition_batch,
    load_manifest,
    save_dataset_manifest,
    save_decomposition_batch,
)

__all__ = [
    # Laplacian computation
    "compute_laplacian",
    # Collectors
    "EigenstructureCollector",
    "NoisedEigenstructureCollector",
    # Analysis
    "SpectralAnalyzer",
    "SpectralAnalysisResult",
    "NoisedAnalysisComparator",
    # Analysis functions
    "compute_spectral_gap",
    "compute_algebraic_connectivity",
    "compute_eigenvector_coherence",
    "compute_eigenvalue_entropy",
    "compute_effective_rank",
    "compute_principal_angles",
    # Improvement-gap surrogate (eq. 18 in NeurIPS 2026 paper draft)
    "ImprovementGapResult",
    "compute_B_invariants",
    "estimate_improvement_gap",
    # Typed delta comparison result objects
    "MeanStdMetric",
    "DriftComparisonResult",
    "GapDeltaComparisonResult",
    "SubspaceDistanceResult",
    "PrincipalAngleResult",
    "ProcrustesKStats",
    "ProcrustesRotationResult",
    "NoiseLevelComparisonResult",
    "CovarianceEvolutionItem",
    "CovarianceEvolutionResult",
    # Delta comparison functions
    "compute_eigengap_delta",
    "compute_algebraic_connectivity_delta",
    # Spectral delta primitives (canonical implementations from tmgg.utils.spectral)
    "compute_eigenvalue_drift",
    "compute_subspace_distance_from_eigenvectors",
    "compute_alg_connectivity_delta",
    # Storage utilities
    "save_decomposition_batch",
    "load_decomposition_batch",
    "save_dataset_manifest",
    "load_manifest",
    "iter_batches",
]

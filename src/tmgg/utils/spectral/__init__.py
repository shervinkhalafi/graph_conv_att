"""Spectral analysis primitives for eigenstructure comparison.

Pure linear-algebra utilities for computing Laplacians, eigenvector subspace
distances, Procrustes rotations, and spectral deltas between graph pairs.
"""

from tmgg.utils.spectral.laplacian import compute_laplacian
from tmgg.utils.spectral.spectral_deltas import (
    compute_alg_connectivity_delta,
    compute_eigengap_delta,
    compute_eigenvalue_drift,
    compute_spectral_deltas,
    compute_subspace_distance_from_eigenvectors,
)
from tmgg.utils.spectral.subspace import (
    compute_procrustes_rotation,
    compute_procrustes_rotation_batch,
)

__all__ = [
    "compute_alg_connectivity_delta",
    "compute_eigengap_delta",
    "compute_eigenvalue_drift",
    "compute_laplacian",
    "compute_procrustes_rotation",
    "compute_procrustes_rotation_batch",
    "compute_spectral_deltas",
    "compute_subspace_distance_from_eigenvectors",
]

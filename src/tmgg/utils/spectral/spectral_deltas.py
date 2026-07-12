"""Spectral delta primitives and orchestrator for comparing graph spectra.

Four primitive metrics operate on pre-decomposed eigendata (eigenvalues
and/or eigenvectors). The ``compute_spectral_deltas`` orchestrator takes
raw adjacency matrices, runs eigendecomposition, and calls all four.

Primitives
----------
- ``compute_eigenvalue_drift``: relative L2 distance between eigenvalue spectra
- ``compute_subspace_distance_from_eigenvectors``: Frobenius norm of projection matrix difference
- ``compute_eigengap_delta``: relative change in spectral gap (lambda_max - lambda_{max-1})
- ``compute_alg_connectivity_delta``: relative change in Fiedler value (lambda_2)
"""

from __future__ import annotations

from typing import Literal, overload

import torch

from tmgg.utils.spectral.laplacian import compute_laplacian

_EPS = 1e-10  # numerical stability for relative metrics


# ---------------------------------------------------------------------------
# Primitive functions — operate on eigenvalues / eigenvectors directly
# ---------------------------------------------------------------------------


def compute_eigenvalue_drift(
    eigenvalues_clean: torch.Tensor, eigenvalues_other: torch.Tensor
) -> torch.Tensor:
    """Relative L2 distance between eigenvalue spectra.

    Parameters
    ----------
    eigenvalues_clean
        Reference eigenvalues, shape ``(batch, n)`` or ``(n,)``.
    eigenvalues_other
        Comparison eigenvalues, same shape.

    Returns
    -------
    torch.Tensor
        ``‖lambda_other - lambda_clean‖_2 / ‖lambda_clean‖_2``,
        shape ``(batch,)`` or scalar.
    """
    diff = torch.norm(eigenvalues_other - eigenvalues_clean, dim=-1)
    norm = torch.norm(eigenvalues_clean, dim=-1) + _EPS
    return diff / norm


def compute_subspace_distance_from_eigenvectors(
    V_clean: torch.Tensor, V_other: torch.Tensor, k: int
) -> torch.Tensor:
    """Frobenius norm of projection matrix difference for top-k eigenvectors.

    Parameters
    ----------
    V_clean
        Reference eigenvectors, shape ``(batch, n, n)`` or ``(n, n)``.
        Columns sorted by ascending eigenvalue (``eigh`` convention).
    V_other
        Comparison eigenvectors, same shape.
    k
        Number of top eigenvectors (last *k* columns).

    Returns
    -------
    torch.Tensor
        ``‖P_clean - P_other‖_F``, shape ``(batch,)`` or scalar.
    """
    V_clean_k = V_clean[..., -k:]
    V_other_k = V_other[..., -k:]
    P_clean = V_clean_k @ V_clean_k.transpose(-2, -1)
    P_other = V_other_k @ V_other_k.transpose(-2, -1)
    return torch.norm(P_clean - P_other, p="fro", dim=(-2, -1))


def compute_eigengap_delta(
    eigenvalues_clean: torch.Tensor, eigenvalues_other: torch.Tensor
) -> torch.Tensor:
    """Relative change in spectral gap (lambda_max - lambda_{max-1}).

    Parameters
    ----------
    eigenvalues_clean
        Reference eigenvalues, ascending order, shape ``(batch, n)``.
    eigenvalues_other
        Comparison eigenvalues, same shape.

    Returns
    -------
    torch.Tensor
        Relative delta, shape ``(batch,)``. Positive means gap increased.

    Notes
    -----
    This computes the gap between the two largest adjacency eigenvalues, not
    the Laplacian spectral gap (λ₂) common in spectral graph theory. The
    adjacency eigengap measures separation between the leading eigenspace and
    the rest.
    """
    gap_clean = eigenvalues_clean[..., -1] - eigenvalues_clean[..., -2]
    gap_other = eigenvalues_other[..., -1] - eigenvalues_other[..., -2]
    return (gap_other - gap_clean) / (gap_clean.abs() + _EPS)


def compute_alg_connectivity_delta(
    lap_eigenvalues_clean: torch.Tensor, lap_eigenvalues_other: torch.Tensor
) -> torch.Tensor:
    """Relative change in algebraic connectivity (Fiedler value lambda_2).

    Parameters
    ----------
    lap_eigenvalues_clean
        Reference Laplacian eigenvalues, ascending order, shape ``(batch, n)``.
    lap_eigenvalues_other
        Comparison Laplacian eigenvalues, same shape.

    Returns
    -------
    torch.Tensor
        Relative delta, shape ``(batch,)``. Positive means connectivity increased.
    """
    alg_conn_clean = lap_eigenvalues_clean[..., 1]
    alg_conn_other = lap_eigenvalues_other[..., 1]
    return (alg_conn_other - alg_conn_clean) / (alg_conn_clean.abs() + _EPS)


# ---------------------------------------------------------------------------
# Orchestrator — takes raw adjacency, decomposes, calls all primitives
# ---------------------------------------------------------------------------


@overload
def compute_spectral_deltas(
    A_clean: torch.Tensor,
    A_other: torch.Tensor,
    k: int = 4,
    compute_rotation: bool = False,
    *,
    reduce: None = None,
) -> dict[str, torch.Tensor]: ...


@overload
def compute_spectral_deltas(
    A_clean: torch.Tensor,
    A_other: torch.Tensor,
    k: int = 4,
    compute_rotation: bool = False,
    *,
    reduce: Literal["mean"],
) -> dict[str, float]: ...


def compute_spectral_deltas(
    A_clean: torch.Tensor,
    A_other: torch.Tensor,
    k: int = 4,
    compute_rotation: bool = False,
    *,
    reduce: str | None = None,
) -> dict[str, torch.Tensor] | dict[str, float]:
    """Compute spectral delta metrics between clean and other (noisy/denoised) graphs.

    Parameters
    ----------
    A_clean : torch.Tensor
        Clean adjacency matrices, shape ``(n, n)`` or ``(batch, n, n)``.
    A_other : torch.Tensor
        Noisy or denoised adjacency matrices, same shape as *A_clean*.
    k : int
        Number of top eigenvectors for subspace comparison.
    compute_rotation : bool
        If True, also compute Procrustes rotation angle and residual metrics.
    reduce : str or None
        If ``"mean"``, return batch-averaged ``dict[str, float]`` suitable
        for direct logging.  If None (default), return per-graph tensors
        of shape ``(batch,)``.

    Returns
    -------
    dict[str, torch.Tensor] or dict[str, float]
        Delta metrics.  Keys always include ``eigengap_delta``,
        ``alg_conn_delta``, ``eigenvalue_drift``, ``subspace_distance``.
        When *compute_rotation* is True, also ``rotation_angle`` and
        ``rotation_residual``.  Positive eigengap/alg_conn deltas mean
        the quantity increased in *A_other*.
    """
    # Handle 2D input (single graph) by adding batch dimension
    if A_clean.dim() == 2:
        A_clean = A_clean.unsqueeze(0)
        A_other = A_other.unsqueeze(0)

    # Compute eigendecompositions for adjacency matrices
    eig_clean, vec_clean = torch.linalg.eigh(A_clean)
    eig_other, vec_other = torch.linalg.eigh(A_other)

    # Compute Laplacian eigendecompositions
    L_clean = compute_laplacian(A_clean)
    L_other = compute_laplacian(A_other)
    lap_eig_clean, _ = torch.linalg.eigh(L_clean)
    lap_eig_other, _ = torch.linalg.eigh(L_other)

    result: dict[str, torch.Tensor] = {
        "eigengap_delta": compute_eigengap_delta(eig_clean, eig_other),
        "alg_conn_delta": compute_alg_connectivity_delta(lap_eig_clean, lap_eig_other),
        "eigenvalue_drift": compute_eigenvalue_drift(eig_clean, eig_other),
        "subspace_distance": compute_subspace_distance_from_eigenvectors(
            vec_clean, vec_other, k
        ),
    }

    # Optional: Procrustes rotation metrics
    if compute_rotation:
        from tmgg.utils.spectral.subspace import (
            compute_procrustes_rotation_batch,
        )

        proc = compute_procrustes_rotation_batch(vec_clean, vec_other, k)
        result["rotation_angle"] = proc["angles"]
        result["rotation_residual"] = proc["residuals"]

    if reduce == "mean":
        return {key: val.mean().item() for key, val in result.items()}
    return result

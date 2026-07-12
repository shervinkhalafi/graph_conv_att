"""Subspace comparison via orthogonal Procrustes rotation.

Quantifies how much the eigenvector basis rotates when a graph is
perturbed (by noise, denoising, etc.).  The batch implementation is the
canonical one; the single-graph variant delegates to it.
"""

from __future__ import annotations

import torch


def compute_procrustes_rotation_batch(
    V1_batch: torch.Tensor,
    V2_batch: torch.Tensor,
    k: int,
    *,
    return_rotation: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute optimal orthogonal Procrustes rotation for a batch of eigenvector pairs.

    Finds the rotation R minimizing ``||U1 - U2 @ R||_F`` per sample, where
    U1/U2 are the top-*k* eigenvector subspaces (last *k* columns in
    ascending-eigenvalue order).

    Parameters
    ----------
    V1_batch, V2_batch : torch.Tensor
        Batched eigenvector matrices of shape ``(batch, n, n)``.
    k : int
        Number of top eigenvectors to consider (last *k* columns).
    return_rotation : bool
        If True, also return the ``(batch, k, k)`` rotation matrices.

    Returns
    -------
    dict[str, torch.Tensor]
        Always contains ``angles`` and ``residuals``, each ``(batch,)``.
        When *return_rotation* is True, also contains ``rotations`` of
        shape ``(batch, k, k)``.
    """
    U1 = V1_batch[:, :, -k:]  # (batch, n, k)
    U2 = V2_batch[:, :, -k:]
    M = U1.transpose(-2, -1) @ U2  # (batch, k, k)
    U, _, Vt = torch.linalg.svd(M)
    R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)  # (batch, k, k)

    trace_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace_R - 1) / max(k - 1, 1)).clamp(-1.0, 1.0)
    angles = torch.acos(cos_theta)
    residuals = torch.norm(U1 - U2 @ R, p="fro", dim=(-2, -1))

    result: dict[str, torch.Tensor] = {"angles": angles, "residuals": residuals}
    if return_rotation:
        result["rotations"] = R
    return result


def compute_procrustes_rotation(
    V1: torch.Tensor, V2: torch.Tensor, k: int
) -> dict[str, torch.Tensor]:
    """Compute Procrustes rotation for a single pair of eigenvector matrices.

    Thin wrapper around :func:`compute_procrustes_rotation_batch` that
    accepts unbatched ``(n, n)`` inputs and returns the rotation matrix,
    angle, and residual as scalar tensors.

    Parameters
    ----------
    V1, V2 : torch.Tensor
        Eigenvector matrices of shape ``(n, n)``, columns sorted by
        ascending eigenvalue.
    k : int
        Number of top eigenvectors to consider.

    Returns
    -------
    dict[str, torch.Tensor]
        ``rotation`` (k, k), ``angle`` (scalar), ``residual`` (scalar).
    """
    batch = compute_procrustes_rotation_batch(
        V1.unsqueeze(0), V2.unsqueeze(0), k, return_rotation=True
    )
    return {
        "rotation": batch["rotations"][0],
        "angle": batch["angles"][0],
        "residual": batch["residuals"][0],
    }


def align_top_k_to_reference_batch(
    V_ref: torch.Tensor,
    V_to_align_full: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Procrustes-align each batched top-*k* block to a single reference.

    Solves, per batch index ``i``,
    ``R_i = argmin_R ||V_ref - U_i R||_F`` over orthogonal ``R`` with
    ``U_i = V_to_align_full[i, :, -k:]``, and returns the aligned
    ``U_i R_i``. This is the multi-graph Procrustes step used when every
    graph should land in the same ``V_ref`` frame — for example, when
    aligning noisy eigenvector blocks to a dataset-wide Fréchet mean
    subspace so that cross-graph averages of ``B = V̂^T A V̂`` are
    frame-coherent.

    Parameters
    ----------
    V_ref : torch.Tensor
        Reference subspace of shape ``(n, k)`` with orthonormal columns.
    V_to_align_full : torch.Tensor
        Full eigenvector matrices of shape ``(batch, n, n)`` in
        ascending-eigenvalue order. The last *k* columns are aligned.
    k : int
        Subspace dimension; ``k <= V_ref.shape[1]``.

    Returns
    -------
    torch.Tensor
        Aligned top-*k* blocks of shape ``(batch, n, k)``.

    Notes
    -----
    Uses the same SVD-based closed form as
    :func:`compute_procrustes_rotation_batch` but broadcasts a shared
    reference instead of comparing paired batches. SVD includes
    reflections (``det R = ±1``), which subsumes the sign canonicalisation
    step that a first-cut largest-magnitude-positive rule would provide.
    """
    if V_ref.shape[1] < k:
        raise ValueError(
            f"V_ref has {V_ref.shape[1]} columns, less than requested k={k}"
        )
    V_ref_k = V_ref[:, :k]  # (n, k)
    V_noisy_k = V_to_align_full[:, :, -k:]  # (batch, n, k)
    # M[i] = V_ref_k^T @ V_noisy_k[i], shape (k, k) per batch element.
    M = V_ref_k.T.unsqueeze(0) @ V_noisy_k  # (batch, k, k)
    U, _, Vt = torch.linalg.svd(M)
    R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)  # (batch, k, k)
    return V_noisy_k @ R


def compute_frechet_mean_subspace(
    V_blocks: torch.Tensor,
) -> torch.Tensor:
    """Extrinsic (SVD-based) Fréchet mean on the Grassmannian.

    Given ``N`` orthonormal ``(n, k)`` blocks, computes a single
    representative ``(n, k)`` orthonormal basis that minimises, in the
    extrinsic chordal sense, the total squared Frobenius distance
    ``Σ_i ||V_i V_i^T - V* V*^T||_F²``. Concretely, concatenates the
    blocks along their second axis and returns the top-*k* left singular
    vectors of the resulting ``(n, N*k)`` matrix.

    This is the standard closed-form approximation to the geodesic
    Fréchet mean and coincides with it when subspaces are close. For
    diverse datasets it yields the best single-subspace summary under the
    chordal metric, which is the natural choice for aligning every
    graph's top-*k* noisy eigenvector block into one common frame prior
    to cross-graph averaging of ``B = V̂^T A V̂``.

    Parameters
    ----------
    V_blocks : torch.Tensor
        Stacked orthonormal blocks of shape ``(N, n, k)``.

    Returns
    -------
    torch.Tensor
        Fréchet mean subspace with orthonormal columns, shape ``(n, k)``.

    Notes
    -----
    Assumes the input columns are already orthonormal (they will be if
    they come from ``torch.linalg.eigh``). No re-orthonormalisation is
    performed before the concatenation; the SVD handles any conditioning.
    """
    if V_blocks.ndim != 3:
        raise ValueError(
            f"V_blocks must have shape (N, n, k); got {tuple(V_blocks.shape)}"
        )
    N, n, k = V_blocks.shape
    if N == 0:
        raise ValueError("Need at least one block to compute a Fréchet mean")
    # Concatenate into (n, N*k) and take top-k left singular vectors.
    M = V_blocks.permute(1, 0, 2).reshape(n, N * k)
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    return U[:, :k].contiguous()

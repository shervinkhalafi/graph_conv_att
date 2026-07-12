"""Laplacian matrix computation.

Computes the combinatorial Laplacian L = D - A or the symmetric normalized
Laplacian L_sym = I - D^{-1/2} A D^{-1/2}, controlled by a single flag.
"""

import torch


def compute_laplacian(
    A: torch.Tensor,
    *,
    normalized: bool = False,
    symmetrize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the graph Laplacian from an adjacency matrix.

    Parameters
    ----------
    A : torch.Tensor
        Adjacency matrix of shape ``(n, n)`` or ``(batch, n, n)``.
    normalized : bool
        If False (default), returns the combinatorial Laplacian ``L = D - A``
        with eigenvalues in ``[0, 2*d_max]``.  If True, returns the symmetric
        normalized Laplacian ``L_sym = I - D^{-1/2} A D^{-1/2}`` with
        eigenvalues in ``[0, 2]``.
    symmetrize : bool
        If True, forces the result to be exactly symmetric via
        ``(L + L^T) / 2``. Useful as a numerical guard when floating-point
        accumulation in ``D - A`` or the normalized product introduces tiny
        asymmetries. Default is False.
    eps : float
        Small value added to degrees before inversion (only used when
        ``normalized=True``) to handle isolated nodes.

    Returns
    -------
    torch.Tensor
        Laplacian matrix with the same shape as *A*.
    """
    squeeze = A.dim() == 2
    if squeeze:
        A = A.unsqueeze(0)
    elif A.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {A.shape}")

    degrees = A.sum(dim=-1)

    if normalized:
        deg_inv_sqrt = (degrees + eps).pow(-0.5)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
        n = A.shape[-1]
        identity = (
            torch.eye(n, device=A.device, dtype=A.dtype)
            .unsqueeze(0)
            .expand(A.shape[0], -1, -1)
        )
        L = identity - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        D = torch.diag_embed(degrees)
        L = D - A

    if symmetrize:
        L = (L + L.transpose(-2, -1)) / 2

    return L.squeeze(0) if squeeze else L

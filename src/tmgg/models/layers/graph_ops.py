"""Shared graph convolution primitives.

Symmetric normalization and polynomial graph convolution are used across
multiple GCN variants. Centralizing them here avoids duplication and
ensures consistent numerical behavior.
"""

from collections.abc import Sequence

import torch


def sym_normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization D^{-1/2} A D^{-1/2}.

    Bounds the spectral radius to [-1, 1], preventing overflow in
    matrix-power polynomial filters applied to dense graphs.

    Parameters
    ----------
    A
        Adjacency matrix, shape ``(batch, n, n)``.

    Returns
    -------
    torch.Tensor
        Normalized adjacency, same shape as *A*.
    """
    D = A.sum(dim=-1)  # (batch, n)
    D_inv_sqrt = torch.where(D > 0, D.pow(-0.5), torch.zeros_like(D))
    D_inv_sqrt_mat = torch.diag_embed(D_inv_sqrt)  # (batch, n, n)
    return torch.bmm(torch.bmm(D_inv_sqrt_mat, A), D_inv_sqrt_mat)


def poly_graph_conv(
    A_norm: torch.Tensor, X: torch.Tensor, H: torch.Tensor
) -> torch.Tensor:
    """Polynomial graph convolution Y = sum_i A_norm^i @ X @ H[i].

    Parameters
    ----------
    A_norm
        Normalized adjacency, shape ``(batch, n, n)``.
    X
        Node features, shape ``(batch, n, channels_in)``.
    H
        Filter coefficients, shape ``(num_terms+1, channels_in, channels_out)``.

    Returns
    -------
    torch.Tensor
        Convolved features, shape ``(batch, n, channels_out)``.
    """
    num_terms = H.shape[0] - 1  # H has num_terms+1 entries (including identity)
    Y = X @ H[0]  # Identity term (A^0 = I)
    A_power = A_norm
    for i in range(1, num_terms + 1):
        Y = Y + torch.bmm(A_power, X) @ H[i]
        if i < num_terms:
            A_power = torch.bmm(A_power, A_norm)
    return Y


def spectral_polynomial(
    Lambda: torch.Tensor,
    coefficients: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Compute spectral polynomial W = sum_l Lambda^l * H^{(l)}.

    Parameters
    ----------
    Lambda
        Normalized eigenvalues of shape (batch, k).
    coefficients
        Coefficient matrices H^{(l)}, each of shape (k, k).

    Returns
    -------
    torch.Tensor
        Polynomial result W of shape (batch, k, k).
    """
    batch_size, k = Lambda.shape
    W = torch.zeros(batch_size, k, k, device=Lambda.device, dtype=Lambda.dtype)
    Lambda_power = torch.ones_like(Lambda)
    for H_l in coefficients:
        Lambda_matrix = Lambda_power.unsqueeze(-1).expand(-1, -1, k)
        W = W + Lambda_matrix * H_l.unsqueeze(0)
        Lambda_power = Lambda_power * Lambda
    return W

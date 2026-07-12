"""Eigenvector embedding layers for GNN spatial models.

Provides ``_EigenEmbedding`` (full eigenvector matrix, private) and
``TruncatedEigenEmbedding`` (fixed-width truncation wrapper) built on
top of ``TopKEigenLayer`` from ``.topk_eigen``.
"""

import torch
import torch.nn as nn

from .topk_eigen import TopKEigenLayer


class _EigenEmbedding(nn.Module):
    """Full eigenvector embedding via eigendecomposition.

    Thin wrapper around ``TopKEigenLayer`` that returns all n eigenvectors
    (k=n) without sign normalization, preserving the original interface
    for GNN spatial models.

    Parameters
    ----------
    eigenvalue_reg
        Diagonal regularization before eigendecomposition. Spreads
        eigenvalues apart, improving gradient stability through eigh.
        Default 0.0. Values around 1e-4 to 1e-2 help when training
        produces NaN gradients or unstable loss.
    """

    def __init__(self, eigenvalue_reg: float = 0.0):
        super().__init__()
        self.eigenvalue_reg = eigenvalue_reg
        self._inner = TopKEigenLayer(
            k=10_000,  # effectively "all" — clamped to n inside forward()
            eigenvalue_reg=eigenvalue_reg,
            sign_normalize=False,
        )

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """Return full eigenvector matrix of shape (batch, n, n).

        Parameters
        ----------
        A
            Adjacency matrices of shape (batch, n, n).

        Returns
        -------
        torch.Tensor
            Eigenvector embeddings of shape (batch, n, n).

        Raises
        ------
        EigenDecompositionError
            If eigendecomposition fails, with diagnostic context.
        """
        V, _ = self._inner(A)
        return V


class TruncatedEigenEmbedding(nn.Module):
    """Eigenvector embedding truncated to a fixed feature dimension.

    Wraps ``_EigenEmbedding`` and truncates (or zero-pads) the output to
    ``target_dim`` columns. Rank-deficient matrices that yield fewer than
    ``target_dim`` eigenvectors are padded with zeros.

    Parameters
    ----------
    target_dim
        Number of eigenvector columns to return.
    eigenvalue_reg
        Diagonal regularization forwarded to ``_EigenEmbedding``.
    """

    def __init__(self, target_dim: int, eigenvalue_reg: float = 0.0):
        super().__init__()
        self.target_dim = target_dim
        self._inner = _EigenEmbedding(eigenvalue_reg=eigenvalue_reg)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """Return truncated eigenvector matrix of shape (batch, n, target_dim)."""
        z = self._inner(A)
        actual = min(z.shape[2], self.target_dim)
        z = z[:, :, :actual]
        if actual < self.target_dim:
            padding = torch.zeros(
                z.shape[0],
                z.shape[1],
                self.target_dim - actual,
                device=z.device,
                dtype=z.dtype,
            )
            z = torch.cat([z, padding], dim=2)
        return z

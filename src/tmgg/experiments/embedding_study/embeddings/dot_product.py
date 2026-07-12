"""Dot product embeddings for low-rank graph approximation.

Implements simple low-rank factorization where A ≈ X·Xᵀ for symmetric
embeddings or A ≈ X·Yᵀ for asymmetric. The output is continuous and
unbounded, making this suitable for MSE loss optimization.
"""

from __future__ import annotations

import torch

from .base import AsymmetricEmbedding, SymmetricEmbedding


class DotProductSymmetric(SymmetricEmbedding):
    """Symmetric dot product embedding: A ≈ X·Xᵀ.

    A rank-d approximation of the adjacency matrix via a single set of
    node embeddings. The reconstruction is the Gram matrix of embeddings.
    """

    def __init__(
        self,
        dimension: int,
        num_nodes: int,
        init_scale: float = 0.1,
    ) -> None:
        """Initialize symmetric dot product embedding.

        Parameters
        ----------
        dimension
            Embedding dimension d (rank of approximation).
        num_nodes
            Number of nodes n.
        init_scale
            Scale for random initialization.
        """
        super().__init__(dimension, num_nodes, init_scale)

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct adjacency via X·Xᵀ.

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency of shape (n, n). Values are unbounded
            and may need clamping for binary interpretation.
        """
        return self.X @ self.X.T


class DotProductAsymmetric(AsymmetricEmbedding):
    """Asymmetric dot product embedding: A ≈ X·Yᵀ.

    A rank-d approximation using separate source and target embeddings,
    allowing representation of asymmetric relationships or more flexible
    factorizations of symmetric matrices.
    """

    def __init__(
        self,
        dimension: int,
        num_nodes: int,
        init_scale: float = 0.1,
    ) -> None:
        """Initialize asymmetric dot product embedding.

        Parameters
        ----------
        dimension
            Embedding dimension d.
        num_nodes
            Number of nodes n.
        init_scale
            Scale for random initialization.
        """
        super().__init__(dimension, num_nodes, init_scale)

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct adjacency via X·Yᵀ.

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency of shape (n, n).
        """
        return self.X @ self.Y.T

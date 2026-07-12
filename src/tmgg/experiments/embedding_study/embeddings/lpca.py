"""Logistic PCA embeddings for graph reconstruction.

Implements the logistic PCA model from:
    "Node Embeddings and Exact Low-Rank Representations of Complex Networks"
    (NeurIPS 2020)

The model approximates adjacency as A ≈ σ(X·Xᵀ) for symmetric embeddings
or A ≈ σ(X·Yᵀ) for asymmetric embeddings, where σ is the sigmoid function.
"""

from __future__ import annotations

import torch

from .base import AsymmetricEmbedding, SymmetricEmbedding


class LPCASymmetric(SymmetricEmbedding):
    """Symmetric logistic PCA: A ≈ σ(X·Xᵀ).

    Each node i has embedding xᵢ ∈ ℝᵈ, and the probability of edge (i,j)
    is modeled as σ(xᵢ·xⱼ) where σ is the sigmoid function.
    """

    def __init__(
        self,
        dimension: int,
        num_nodes: int,
        init_scale: float = 0.1,
    ) -> None:
        """Initialize symmetric LPCA embedding.

        Parameters
        ----------
        dimension
            Embedding dimension d.
        num_nodes
            Number of nodes n.
        init_scale
            Scale for random initialization of embeddings.
        """
        super().__init__(dimension, num_nodes, init_scale)

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct adjacency via σ(X·Xᵀ).

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency probabilities of shape (n, n).
        """
        logits = self.X @ self.X.T
        return torch.sigmoid(logits)


class LPCAAsymmetric(AsymmetricEmbedding):
    """Asymmetric logistic PCA: A ≈ σ(X·Yᵀ).

    Each node i has source embedding xᵢ and target embedding yᵢ in ℝᵈ.
    The probability of edge (i,j) is modeled as σ(xᵢ·yⱼ).
    """

    def __init__(
        self,
        dimension: int,
        num_nodes: int,
        init_scale: float = 0.1,
    ) -> None:
        """Initialize asymmetric LPCA embedding.

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
        """Reconstruct adjacency via σ(X·Yᵀ).

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency probabilities of shape (n, n).
        """
        logits = self.X @ self.Y.T
        return torch.sigmoid(logits)

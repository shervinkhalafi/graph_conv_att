"""Threshold-based dot product embeddings.

Implements embeddings where edges exist iff inner product exceeds a threshold:
    A_ij = 1 iff xᵢ·xⱼ > τ

The hard threshold is relaxed via sigmoid for differentiability:
    Â_ij = σ(β(xᵢ·xⱼ - τ))

where β is a temperature parameter controlling sharpness.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import AsymmetricEmbedding, SymmetricEmbedding


class DotThresholdSymmetric(SymmetricEmbedding):
    """Symmetric threshold embedding: A_ij = 1 iff xᵢ·xⱼ > τ.

    Uses sigmoid relaxation σ(β(X·Xᵀ - τ)) for differentiable optimization.
    The threshold τ is learnable; temperature β controls sigmoid sharpness.
    """

    def __init__(
        self,
        dimension: int,
        num_nodes: int,
        init_scale: float = 0.1,
        init_threshold: float = 0.0,
        temperature: float = 10.0,
        learn_threshold: bool = True,
    ) -> None:
        """Initialize symmetric threshold embedding.

        Parameters
        ----------
        dimension
            Embedding dimension d.
        num_nodes
            Number of nodes n.
        init_scale
            Scale for random initialization of embeddings.
        init_threshold
            Initial threshold value τ.
        temperature
            Sigmoid temperature β. Higher values → sharper threshold.
        learn_threshold
            Whether to learn the threshold via gradient descent.
        """
        super().__init__(dimension, num_nodes, init_scale)
        self.temperature = temperature

        if learn_threshold:
            self._threshold = nn.Parameter(torch.tensor(init_threshold))
        else:
            self.register_buffer("_threshold", torch.tensor(init_threshold))

    @property
    def threshold(self) -> float:
        """Get current threshold value."""
        return self._threshold.item()

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct adjacency via σ(β(X·Xᵀ - τ)).

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency probabilities of shape (n, n).
        """
        dots = self.X @ self.X.T
        logits = self.temperature * (dots - self._threshold)
        return torch.sigmoid(logits)

    def reconstruct_hard(self) -> torch.Tensor:
        """Reconstruct with hard thresholding (non-differentiable).

        Returns
        -------
        torch.Tensor
            Binary adjacency of shape (n, n).
        """
        dots = self.X @ self.X.T
        return (dots > self._threshold).float()


class DotThresholdAsymmetric(AsymmetricEmbedding):
    """Asymmetric threshold embedding: A_ij = 1 iff xᵢ·yⱼ > τ.

    Uses sigmoid relaxation σ(β(X·Yᵀ - τ)) for differentiable optimization.
    """

    def __init__(
        self,
        dimension: int,
        num_nodes: int,
        init_scale: float = 0.1,
        init_threshold: float = 0.0,
        temperature: float = 10.0,
        learn_threshold: bool = True,
    ) -> None:
        """Initialize asymmetric threshold embedding.

        Parameters
        ----------
        dimension
            Embedding dimension d.
        num_nodes
            Number of nodes n.
        init_scale
            Scale for random initialization.
        init_threshold
            Initial threshold value τ.
        temperature
            Sigmoid temperature β.
        learn_threshold
            Whether to learn the threshold via gradient descent.
        """
        super().__init__(dimension, num_nodes, init_scale)
        self.temperature = temperature

        if learn_threshold:
            self._threshold = nn.Parameter(torch.tensor(init_threshold))
        else:
            self.register_buffer("_threshold", torch.tensor(init_threshold))

    @property
    def threshold(self) -> float:
        """Get current threshold value."""
        return self._threshold.item()

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct adjacency via σ(β(X·Yᵀ - τ)).

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency probabilities of shape (n, n).
        """
        dots = self.X @ self.Y.T
        logits = self.temperature * (dots - self._threshold)
        return torch.sigmoid(logits)

    def reconstruct_hard(self) -> torch.Tensor:
        """Reconstruct with hard thresholding.

        Returns
        -------
        torch.Tensor
            Binary adjacency of shape (n, n).
        """
        dots = self.X @ self.Y.T
        return (dots > self._threshold).float()

"""Distance-based threshold embeddings.

Implements embeddings where edges exist iff Euclidean distance is below
a threshold:
    A_ij = 1 iff ‖xᵢ - xⱼ‖ < τ

The hard threshold is relaxed via sigmoid for differentiability:
    Â_ij = σ(β(τ - ‖xᵢ - xⱼ‖))

Note: Distance-based thresholds are inherently symmetric, so no asymmetric
variant is provided.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import SymmetricEmbedding


class DistanceThresholdSymmetric(SymmetricEmbedding):
    """Distance threshold embedding: A_ij = 1 iff ‖xᵢ - xⱼ‖ < τ.

    Uses sigmoid relaxation σ(β(τ - ‖xᵢ-xⱼ‖)) for differentiable optimization.
    The threshold τ is learnable; temperature β controls sigmoid sharpness.
    """

    def __init__(
        self,
        dimension: int,
        num_nodes: int,
        init_scale: float = 0.1,
        init_threshold: float = 1.0,
        temperature: float = 10.0,
        learn_threshold: bool = True,
    ) -> None:
        """Initialize distance threshold embedding.

        Parameters
        ----------
        dimension
            Embedding dimension d.
        num_nodes
            Number of nodes n.
        init_scale
            Scale for random initialization of embeddings.
        init_threshold
            Initial threshold value τ. Should be positive.
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

    def _compute_distances(self) -> torch.Tensor:
        """Compute pairwise Euclidean distances between embeddings.

        Returns
        -------
        torch.Tensor
            Distance matrix of shape (n, n).
        """
        # ‖xᵢ - xⱼ‖² = ‖xᵢ‖² + ‖xⱼ‖² - 2xᵢ·xⱼ
        norms_sq = (self.X**2).sum(dim=1, keepdim=True)
        dots = self.X @ self.X.T
        dist_sq = norms_sq + norms_sq.T - 2 * dots
        # Clamp to avoid negative values from numerical errors
        dist_sq = dist_sq.clamp(min=0.0)
        return torch.sqrt(dist_sq)

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct adjacency via σ(β(τ - ‖xᵢ-xⱼ‖)).

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency probabilities of shape (n, n).
        """
        distances = self._compute_distances()
        logits = self.temperature * (self._threshold - distances)
        return torch.sigmoid(logits)

    def reconstruct_hard(self) -> torch.Tensor:
        """Reconstruct with hard thresholding (non-differentiable).

        Returns
        -------
        torch.Tensor
            Binary adjacency of shape (n, n).
        """
        distances = self._compute_distances()
        return (distances < self._threshold).float()

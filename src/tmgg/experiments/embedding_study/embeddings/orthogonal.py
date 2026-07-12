"""Orthogonal representation embeddings.

Implements Lovász's orthogonal representation where:
    - Non-edges: vectors are orthogonal (xᵢ·xⱼ = 0)
    - Edges: vectors are NOT orthogonal (xᵢ·xⱼ ≠ 0)

Thus: A_ij = 1 ⟺ xᵢ·xⱼ ≠ 0

The hard condition is relaxed via sigmoid for differentiability:
    Â_ij = σ(β(|xᵢ·xⱼ| - ε))

where ε is a small threshold and β is the temperature.

The minimum dimension d for which such a representation exists equals
the graph's orthogonality dimension, which is bounded by the chromatic
number and clique cover number.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import SymmetricEmbedding


class OrthogonalRepSymmetric(SymmetricEmbedding):
    """Orthogonal representation: A_ij = 1 iff xᵢ·xⱼ ≠ 0.

    Uses sigmoid relaxation σ(β(|xᵢ·xⱼ| - ε)) for differentiable optimization.
    The threshold ε and temperature β control the sharpness of the boundary
    between "orthogonal" (zero inner product) and "not orthogonal."
    """

    def __init__(
        self,
        dimension: int,
        num_nodes: int,
        init_scale: float = 0.1,
        epsilon: float = 0.01,
        temperature: float = 10.0,
        learn_epsilon: bool = False,
    ) -> None:
        """Initialize orthogonal representation embedding.

        Parameters
        ----------
        dimension
            Embedding dimension d.
        num_nodes
            Number of nodes n.
        init_scale
            Scale for random initialization of embeddings.
        epsilon
            Threshold ε for considering inner product as "zero."
        temperature
            Sigmoid temperature β. Higher values → sharper threshold.
        learn_epsilon
            Whether to learn epsilon via gradient descent.
        """
        super().__init__(dimension, num_nodes, init_scale)
        self.temperature = temperature

        if learn_epsilon:
            self._epsilon = nn.Parameter(torch.tensor(epsilon))
        else:
            self.register_buffer("_epsilon", torch.tensor(epsilon))

    @property
    def threshold(self) -> float:
        """Get current epsilon threshold value."""
        return self._epsilon.item()

    def reconstruct(self) -> torch.Tensor:
        """Reconstruct adjacency via σ(β(|xᵢ·xⱼ| - ε)).

        Edge exists when inner product magnitude exceeds epsilon.

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency probabilities of shape (n, n).
        """
        dots = self.X @ self.X.T
        abs_dots = torch.abs(dots)
        logits = self.temperature * (abs_dots - self._epsilon)
        return torch.sigmoid(logits)

    def reconstruct_hard(self) -> torch.Tensor:
        """Reconstruct with hard thresholding (non-differentiable).

        Returns
        -------
        torch.Tensor
            Binary adjacency of shape (n, n).
        """
        dots = self.X @ self.X.T
        abs_dots = torch.abs(dots)
        return (abs_dots > self._epsilon).float()

    def compute_orthogonality_loss(self, target: torch.Tensor) -> torch.Tensor:
        """Compute loss specifically for orthogonal representation.

        This loss directly penalizes:
        - Non-zero inner products where target has no edge
        - Near-zero inner products where target has an edge

        Parameters
        ----------
        target
            Target adjacency matrix of shape (n, n).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        dots = self.X @ self.X.T
        dots_sq = dots**2

        # For non-edges: penalize non-zero inner products (want orthogonal)
        non_edge_loss = (dots_sq * (1 - target)).mean()

        # For edges: penalize near-zero inner products (want non-orthogonal)
        # Use -log(|dot|² + eps) to encourage large absolute values
        edge_loss = (-torch.log(dots_sq + 1e-8) * target).mean()

        return non_edge_loss + 0.1 * edge_loss

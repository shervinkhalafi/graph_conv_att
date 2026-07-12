"""Base classes and protocols for graph embeddings.

Defines the interface for graph embedding models that map vertices to vectors
such that the graph structure can be reconstructed from inner products or
distances between vectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class EmbeddingResult:
    """Result of fitting an embedding to a graph.

    Attributes
    ----------
    X
        Node embeddings of shape (n, d) for symmetric, or source embeddings
        for asymmetric.
    Y
        Target embeddings of shape (n, d) for asymmetric, None for symmetric.
    dimension
        Embedding dimension d.
    reconstruction
        Reconstructed adjacency matrix of shape (n, n).
    fnorm_error
        Frobenius norm of (A - reconstruction).
    edge_accuracy
        Fraction of edges correctly predicted after rounding.
    converged
        Whether fitting converged to the tolerance criteria.
    threshold
        Learned threshold for threshold-based embeddings, None otherwise.
    """

    X: torch.Tensor
    Y: torch.Tensor | None
    dimension: int
    reconstruction: torch.Tensor
    fnorm_error: float
    edge_accuracy: float
    converged: bool
    threshold: float | None = None


class GraphEmbedding(nn.Module, ABC):
    """Abstract base class for graph embedding models.

    Graph embeddings map each vertex v to a vector x_v in R^d such that
    the adjacency matrix can be (approximately) reconstructed from the
    embeddings via some function f(x_i, x_j).
    """

    def __init__(self, dimension: int, num_nodes: int) -> None:
        """Initialize embedding.

        Parameters
        ----------
        dimension
            Embedding dimension d.
        num_nodes
            Number of nodes n in the graph.
        """
        super().__init__()
        self.dimension = dimension
        self.num_nodes = num_nodes

    @abstractmethod
    def reconstruct(self) -> torch.Tensor:
        """Reconstruct adjacency matrix from current embeddings.

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency (or logits) of shape (n, n).
        """
        ...

    @abstractmethod
    def get_embeddings(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get current node embeddings.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            (X, Y) where X is shape (n, d). Y is None for symmetric embeddings,
            or shape (n, d) for asymmetric.
        """
        ...

    def compute_loss(
        self,
        target: torch.Tensor,
        loss_type: Literal["bce", "mse"] = "bce",
    ) -> torch.Tensor:
        """Compute reconstruction loss.

        Parameters
        ----------
        target
            Target adjacency matrix of shape (n, n).
        loss_type
            Loss function: "bce" for binary cross-entropy, "mse" for mean
            squared error.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        reconstruction = self.reconstruct()

        if loss_type == "bce":
            # BCE expects logits if using BCEWithLogitsLoss
            # Our reconstruct() returns probabilities for most models
            return nn.functional.binary_cross_entropy(
                reconstruction.clamp(1e-7, 1 - 1e-7),
                target,
                reduction="mean",
            )
        elif loss_type == "mse":
            return nn.functional.mse_loss(reconstruction, target)
        else:  # pyright: ignore[reportUnreachable]  # defensive guard
            raise ValueError(f"Unknown loss type: {loss_type}")

    def evaluate(self, target: torch.Tensor) -> tuple[float, float]:
        """Evaluate reconstruction quality.

        Parameters
        ----------
        target
            Target adjacency matrix of shape (n, n).

        Returns
        -------
        tuple[float, float]
            (fnorm_error, edge_accuracy) where fnorm_error is the Frobenius
            norm of (A - Â) and edge_accuracy is the fraction of correctly
            predicted edges after rounding.
        """
        with torch.no_grad():
            reconstruction = self.reconstruct()
            fnorm = torch.norm(target - reconstruction, p="fro").item()

            # Round to binary and compute accuracy
            pred_binary = (reconstruction > 0.5).float()
            accuracy = (pred_binary == target).float().mean().item()

        return fnorm, accuracy

    def to_result(self, target: torch.Tensor) -> EmbeddingResult:
        """Create EmbeddingResult from current state.

        Parameters
        ----------
        target
            Target adjacency matrix for evaluation.

        Returns
        -------
        EmbeddingResult
            Result containing embeddings and evaluation metrics.
        """
        X, Y = self.get_embeddings()
        reconstruction = self.reconstruct().detach()
        fnorm, accuracy = self.evaluate(target)

        return EmbeddingResult(
            X=X.detach().clone(),
            Y=Y.detach().clone() if Y is not None else None,
            dimension=self.dimension,
            reconstruction=reconstruction,
            fnorm_error=fnorm,
            edge_accuracy=accuracy,
            converged=False,  # Set by fitter
            threshold=getattr(self, "threshold", None),
        )


class SymmetricEmbedding(GraphEmbedding, ABC):
    """Base class for symmetric embeddings where A ≈ f(X, X).

    In symmetric embeddings, each node has a single embedding vector,
    and the adjacency is reconstructed via a symmetric function of
    the embeddings (e.g., inner product, distance).
    """

    def __init__(
        self,
        dimension: int,
        num_nodes: int,
        init_scale: float = 0.1,
    ) -> None:
        """Initialize symmetric embedding.

        Parameters
        ----------
        dimension
            Embedding dimension d.
        num_nodes
            Number of nodes n.
        init_scale
            Scale for random initialization.
        """
        super().__init__(dimension, num_nodes)
        self.X = nn.Parameter(torch.randn(num_nodes, dimension) * init_scale)

    def get_embeddings(self) -> tuple[torch.Tensor, None]:
        """Get node embeddings."""
        return self.X, None


class AsymmetricEmbedding(GraphEmbedding, ABC):
    """Base class for asymmetric embeddings where A ≈ f(X, Y).

    In asymmetric embeddings, each node has separate source (X) and
    target (Y) embeddings, allowing representation of directed graphs
    or more flexible factorizations.
    """

    def __init__(
        self,
        dimension: int,
        num_nodes: int,
        init_scale: float = 0.1,
    ) -> None:
        """Initialize asymmetric embedding.

        Parameters
        ----------
        dimension
            Embedding dimension d.
        num_nodes
            Number of nodes n.
        init_scale
            Scale for random initialization.
        """
        super().__init__(dimension, num_nodes)
        self.X = nn.Parameter(torch.randn(num_nodes, dimension) * init_scale)
        self.Y = nn.Parameter(torch.randn(num_nodes, dimension) * init_scale)

    def get_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get source and target embeddings."""
        return self.X, self.Y

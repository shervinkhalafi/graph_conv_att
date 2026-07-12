"""Binary search for minimal embedding dimension.

Finds the smallest dimension d such that a graph can be reconstructed
with near-perfect accuracy using a given embedding method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch

from .base import EmbeddingResult, GraphEmbedding
from .distance_threshold import DistanceThresholdSymmetric
from .dot_product import (
    DotProductAsymmetric,
    DotProductSymmetric,
)
from .dot_threshold import (
    DotThresholdAsymmetric,
    DotThresholdSymmetric,
)
from .fitters.gauge_stabilized import (
    GaugeStabilizedConfig,
    GaugeStabilizedFitter,
)
from .fitters.gradient import FitConfig, GradientFitter
from .fitters.spectral import SpectralFitter
from .lpca import LPCAAsymmetric, LPCASymmetric
from .orthogonal import OrthogonalRepSymmetric


class EmbeddingType(Enum):
    """Available embedding types for dimension search."""

    LPCA_SYMMETRIC = "lpca_symmetric"
    LPCA_ASYMMETRIC = "lpca_asymmetric"
    DOT_PRODUCT_SYMMETRIC = "dot_product_symmetric"
    DOT_PRODUCT_ASYMMETRIC = "dot_product_asymmetric"
    DOT_THRESHOLD_SYMMETRIC = "dot_threshold_symmetric"
    DOT_THRESHOLD_ASYMMETRIC = "dot_threshold_asymmetric"
    DISTANCE_THRESHOLD = "distance_threshold"
    ORTHOGONAL = "orthogonal"


@dataclass
class DimensionResult:
    """Result of dimension search for a single graph.

    Attributes
    ----------
    min_dimension
        Minimum dimension achieving target reconstruction quality.
    embedding_type
        Type of embedding used.
    fitter_type
        Fitter used ("gradient", "spectral", or "both").
    final_fnorm
        Frobenius norm error at min_dimension.
    final_accuracy
        Edge accuracy at min_dimension.
    converged
        Whether fitting converged at min_dimension.
    embeddings
        Final embeddings at min_dimension.
    search_history
        List of (dimension, converged) pairs tried during search.
    """

    min_dimension: int
    embedding_type: EmbeddingType
    fitter_type: str
    final_fnorm: float
    final_accuracy: float
    converged: bool
    embeddings: EmbeddingResult | None
    search_history: list[tuple[int, bool]]


def _create_embedding(
    embedding_type: EmbeddingType,
    dimension: int,
    num_nodes: int,
) -> GraphEmbedding:
    """Create embedding model of specified type."""
    match embedding_type:
        case EmbeddingType.LPCA_SYMMETRIC:
            return LPCASymmetric(dimension, num_nodes)
        case EmbeddingType.LPCA_ASYMMETRIC:
            return LPCAAsymmetric(dimension, num_nodes)
        case EmbeddingType.DOT_PRODUCT_SYMMETRIC:
            return DotProductSymmetric(dimension, num_nodes)
        case EmbeddingType.DOT_PRODUCT_ASYMMETRIC:
            return DotProductAsymmetric(dimension, num_nodes)
        case EmbeddingType.DOT_THRESHOLD_SYMMETRIC:
            return DotThresholdSymmetric(dimension, num_nodes)
        case EmbeddingType.DOT_THRESHOLD_ASYMMETRIC:
            return DotThresholdAsymmetric(dimension, num_nodes)
        case EmbeddingType.DISTANCE_THRESHOLD:
            return DistanceThresholdSymmetric(dimension, num_nodes)
        case EmbeddingType.ORTHOGONAL:
            return OrthogonalRepSymmetric(dimension, num_nodes)


class DimensionSearcher:
    """Binary search for minimum embedding dimension.

    Searches from ceil(sqrt(n)) up to n to find the smallest dimension
    that achieves the target reconstruction quality.
    """

    def __init__(
        self,
        tol_fnorm: float = 0.01,
        tol_accuracy: float = 0.99,
        fitter: Literal[
            "gradient", "spectral", "both", "gauge-stabilized", "gauge-stabilized-svd"
        ] = "both",
        gradient_config: FitConfig | None = None,
        gauge_config: GaugeStabilizedConfig | None = None,
    ) -> None:
        """Initialize dimension searcher.

        Parameters
        ----------
        tol_fnorm
            Maximum allowed Frobenius norm error.
        tol_accuracy
            Minimum required edge accuracy.
        fitter
            Which fitter to use:
            - "gradient": Adam optimizer with BCE/MSE loss
            - "spectral": SVD/eigendecomposition for closed-form fitting
            - "both": spectral initialization followed by gradient descent
            - "gauge-stabilized": Hadamard-based Θ-space interpolation + anchor regularization
            - "gauge-stabilized-svd": Hadamard-based SVD-space interpolation + anchor regularization
            The gauge-stabilized variants only work with LPCA embeddings.
        gradient_config
            Configuration for gradient fitter. Uses defaults if None.
        gauge_config
            Configuration for gauge-stabilized fitter. Uses defaults if None.
        """
        self.tol_fnorm = tol_fnorm
        self.tol_accuracy = tol_accuracy
        self.fitter = fitter

        grad_cfg = gradient_config or FitConfig()
        grad_cfg.tol_fnorm = tol_fnorm
        grad_cfg.tol_accuracy = tol_accuracy
        self.gradient_fitter = GradientFitter(grad_cfg)
        self.spectral_fitter = SpectralFitter(tol_fnorm, tol_accuracy)

        # Set up gauge-stabilized fitter with appropriate init mode
        gauge_cfg = gauge_config or GaugeStabilizedConfig()
        gauge_cfg.tol_fnorm = tol_fnorm
        gauge_cfg.tol_accuracy = tol_accuracy
        if fitter == "gauge-stabilized-svd":
            gauge_cfg.init_mode = "svd"
        self.gauge_stabilized_fitter = GaugeStabilizedFitter(gauge_cfg)

    def _try_dimension(
        self,
        adjacency: torch.Tensor,
        embedding_type: EmbeddingType,
        dimension: int,
    ) -> tuple[bool, EmbeddingResult | None]:
        """Try fitting at a specific dimension.

        Returns (success, result) where success indicates whether
        reconstruction quality targets were met.
        """
        num_nodes = adjacency.shape[0]
        best_result: EmbeddingResult | None = None

        # Try gauge-stabilized fitter for LPCA embeddings
        if self.fitter in ("gauge-stabilized", "gauge-stabilized-svd"):
            if embedding_type not in (
                EmbeddingType.LPCA_SYMMETRIC,
                EmbeddingType.LPCA_ASYMMETRIC,
            ):
                raise ValueError(
                    f"{self.fitter} fitter only works with LPCA embeddings, got {embedding_type}"
                )

            embedding = _create_embedding(embedding_type, dimension, num_nodes)
            embedding = embedding.to(adjacency.device)
            result = self.gauge_stabilized_fitter.fit(embedding, adjacency)

            if result.converged:
                return True, result
            return False, result

        # Try spectral first if applicable
        if self.fitter in ("spectral", "both"):
            embedding = _create_embedding(embedding_type, dimension, num_nodes)
            embedding = embedding.to(adjacency.device)
            result = self.spectral_fitter.fit(embedding, adjacency)

            if result.converged:
                return True, result

            if best_result is None or result.fnorm_error < best_result.fnorm_error:
                best_result = result

        # Try gradient descent
        if self.fitter in ("gradient", "both"):
            embedding = _create_embedding(embedding_type, dimension, num_nodes)
            embedding = embedding.to(adjacency.device)

            # Optionally initialize with spectral
            if self.fitter == "both":
                self.spectral_fitter.initialize(embedding, adjacency)

            # Use temperature annealing for threshold models
            if embedding_type in (
                EmbeddingType.DOT_THRESHOLD_SYMMETRIC,
                EmbeddingType.DOT_THRESHOLD_ASYMMETRIC,
                EmbeddingType.DISTANCE_THRESHOLD,
                EmbeddingType.ORTHOGONAL,
            ):
                result = self.gradient_fitter.fit_with_temperature_annealing(
                    embedding, adjacency
                )
            else:
                result = self.gradient_fitter.fit(embedding, adjacency)

            if result.converged:
                return True, result

            if best_result is None or result.fnorm_error < best_result.fnorm_error:
                best_result = result

        return False, best_result

    def find_min_dimension(
        self,
        adjacency: torch.Tensor,
        embedding_type: EmbeddingType,
        min_dim: int | None = None,
        max_dim: int | None = None,
    ) -> DimensionResult:
        """Find minimum dimension via binary search.

        Parameters
        ----------
        adjacency
            Target adjacency matrix of shape (n, n).
        embedding_type
            Type of embedding to use.
        min_dim
            Starting dimension for search. Defaults to ceil(sqrt(n)).
        max_dim
            Maximum dimension to try. Defaults to n.

        Returns
        -------
        DimensionResult
            Result containing minimum dimension and final embeddings.
        """
        n = adjacency.shape[0]

        if min_dim is None:
            min_dim = math.ceil(math.sqrt(n))
        if max_dim is None:
            max_dim = n

        # Ensure min_dim >= 1
        min_dim = max(1, min_dim)

        # Help type checker understand these are now int
        assert isinstance(min_dim, int)
        assert isinstance(max_dim, int)

        search_history: list[tuple[int, bool]] = []
        best_success_dim: int | None = None
        best_success_result: EmbeddingResult | None = None

        # First try at min_dim (sqrt(n))
        success, result = self._try_dimension(adjacency, embedding_type, min_dim)
        search_history.append((min_dim, success))

        if success:
            # sqrt(n) works - search DOWN to find even smaller dimension
            best_success_dim = min_dim
            best_success_result = result

            # Binary search in [1, min_dim-1]
            low, high = 1, min_dim - 1
            while low <= high:
                mid = (low + high) // 2
                success, result = self._try_dimension(adjacency, embedding_type, mid)
                search_history.append((mid, success))

                if success:
                    best_success_dim = mid
                    best_success_result = result
                    high = mid - 1  # Try even smaller
                else:
                    low = mid + 1  # Need larger
        else:
            # sqrt(n) doesn't work - search UP to find dimension that works
            # Binary search in [min_dim+1, max_dim]
            low, high = min_dim + 1, max_dim
            while low <= high:
                mid = (low + high) // 2
                success, result = self._try_dimension(adjacency, embedding_type, mid)
                search_history.append((mid, success))

                if success:
                    best_success_dim = mid
                    best_success_result = result
                    high = mid - 1  # Try smaller
                else:
                    low = mid + 1  # Need larger

        # If no success found, return result for max_dim
        if best_success_dim is None:
            _, result = self._try_dimension(adjacency, embedding_type, max_dim)
            return DimensionResult(
                min_dimension=max_dim,
                embedding_type=embedding_type,
                fitter_type=self.fitter,
                final_fnorm=result.fnorm_error if result is not None else float("inf"),
                final_accuracy=result.edge_accuracy if result is not None else 0.0,
                converged=False,
                embeddings=result,
                search_history=search_history,
            )

        # best_success_result is guaranteed non-None when best_success_dim is set
        assert best_success_result is not None
        return DimensionResult(
            min_dimension=best_success_dim,
            embedding_type=embedding_type,
            fitter_type=self.fitter,
            final_fnorm=best_success_result.fnorm_error,
            final_accuracy=best_success_result.edge_accuracy,
            converged=True,
            embeddings=best_success_result,
            search_history=search_history,
        )

    def find_all_methods(
        self,
        adjacency: torch.Tensor,
        symmetric_only: bool = False,
    ) -> dict[EmbeddingType, DimensionResult]:
        """Find minimum dimensions for all embedding types.

        Parameters
        ----------
        adjacency
            Target adjacency matrix.
        symmetric_only
            If True, only use symmetric embedding types.

        Returns
        -------
        dict
            Mapping from embedding type to dimension result.
        """
        types = [
            EmbeddingType.LPCA_SYMMETRIC,
            EmbeddingType.DOT_PRODUCT_SYMMETRIC,
            EmbeddingType.DOT_THRESHOLD_SYMMETRIC,
            EmbeddingType.DISTANCE_THRESHOLD,
            EmbeddingType.ORTHOGONAL,
        ]

        if not symmetric_only:
            types.extend(
                [
                    EmbeddingType.LPCA_ASYMMETRIC,
                    EmbeddingType.DOT_PRODUCT_ASYMMETRIC,
                    EmbeddingType.DOT_THRESHOLD_ASYMMETRIC,
                ]
            )

        results = {}
        for etype in types:
            results[etype] = self.find_min_dimension(adjacency, etype)

        return results

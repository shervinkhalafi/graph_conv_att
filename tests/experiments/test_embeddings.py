"""Unit tests for graph embedding models.

Test rationale:
    These tests verify the correctness of graph embedding implementations:
    1. Embeddings produce valid reconstructions with correct shape
    2. Loss functions compute finite values
    3. Gradient-based fitting reduces loss
    4. Spectral fitting provides closed-form solutions for dot product
    5. Dimension search finds correct minimum dimensions

Invariants:
    - Reconstruction shape matches input adjacency shape
    - All losses are finite and non-negative
    - Symmetric embeddings produce symmetric reconstructions
    - Edge accuracy is in [0, 1]
"""

import math

import pytest
import torch

from tmgg.experiments.embedding_study.embeddings import (
    DistanceThresholdSymmetric,
    DotProductAsymmetric,
    DotProductSymmetric,
    DotThresholdAsymmetric,
    DotThresholdSymmetric,
    EmbeddingResult,
    LPCAAsymmetric,
    LPCASymmetric,
    OrthogonalRepSymmetric,
)
from tmgg.experiments.embedding_study.embeddings.fitters.gauge_stabilized import (
    GaugeStabilizedConfig,
    GaugeStabilizedFitter,
    canonicalize_eigenvectors,
    compute_hadamard_anchors,
    compute_hadamard_theta,
    interpolated_theta_init,
)
from tmgg.experiments.embedding_study.embeddings.fitters.gradient import (
    FitConfig,
    GradientFitter,
)
from tmgg.experiments.embedding_study.embeddings.fitters.spectral import SpectralFitter


@pytest.fixture
def simple_adjacency() -> torch.Tensor:
    """Create a simple block-diagonal adjacency matrix.

    Two blocks of size 5 each, high connectivity within blocks,
    low connectivity between blocks.
    """
    A = torch.zeros(10, 10)
    # Block 1: fully connected
    A[:5, :5] = 1.0
    # Block 2: fully connected
    A[5:, 5:] = 1.0
    # Remove self-loops
    A.fill_diagonal_(0)
    return A


@pytest.fixture
def random_adjacency() -> torch.Tensor:
    """Create a random symmetric adjacency matrix."""
    n = 15
    A = torch.bernoulli(torch.full((n, n), 0.3))
    A = ((A + A.T) > 0).float()
    A.fill_diagonal_(0)
    return A


class TestEmbeddingShapes:
    """Tests for embedding shape correctness."""

    @pytest.mark.parametrize(
        "embedding_cls",
        [
            LPCASymmetric,
            DotProductSymmetric,
            DotThresholdSymmetric,
            DistanceThresholdSymmetric,
            OrthogonalRepSymmetric,
        ],
    )
    def test_symmetric_embedding_shapes(
        self, embedding_cls: type, simple_adjacency: torch.Tensor
    ) -> None:
        """Symmetric embeddings produce correct shapes."""
        n = simple_adjacency.shape[0]
        d = 5

        embedding = embedding_cls(dimension=d, num_nodes=n)
        reconstruction = embedding.reconstruct()

        assert reconstruction.shape == (n, n)

        X, Y = embedding.get_embeddings()
        assert X.shape == (n, d)
        assert Y is None

    @pytest.mark.parametrize(
        "embedding_cls",
        [LPCAAsymmetric, DotProductAsymmetric, DotThresholdAsymmetric],
    )
    def test_asymmetric_embedding_shapes(
        self, embedding_cls: type, simple_adjacency: torch.Tensor
    ) -> None:
        """Asymmetric embeddings produce correct shapes."""
        n = simple_adjacency.shape[0]
        d = 5

        embedding = embedding_cls(dimension=d, num_nodes=n)
        reconstruction = embedding.reconstruct()

        assert reconstruction.shape == (n, n)

        X, Y = embedding.get_embeddings()
        assert X.shape == (n, d)
        assert Y is not None
        assert Y.shape == (n, d)


class TestEmbeddingLoss:
    """Tests for loss computation."""

    @pytest.mark.parametrize(
        "embedding_cls",
        [LPCASymmetric, DotProductSymmetric, DotThresholdSymmetric],
    )
    def test_bce_loss_is_finite(
        self, embedding_cls: type, simple_adjacency: torch.Tensor
    ) -> None:
        """BCE loss computes finite values."""
        n = simple_adjacency.shape[0]
        embedding = embedding_cls(dimension=5, num_nodes=n)

        loss = embedding.compute_loss(simple_adjacency, loss_type="bce")

        assert torch.isfinite(loss)
        assert loss >= 0

    @pytest.mark.parametrize(
        "embedding_cls",
        [DotProductSymmetric, LPCASymmetric],
    )
    def test_mse_loss_is_finite(
        self, embedding_cls: type, simple_adjacency: torch.Tensor
    ) -> None:
        """MSE loss computes finite values."""
        n = simple_adjacency.shape[0]
        embedding = embedding_cls(dimension=5, num_nodes=n)

        loss = embedding.compute_loss(simple_adjacency, loss_type="mse")

        assert torch.isfinite(loss)
        assert loss >= 0


class TestEmbeddingEvaluation:
    """Tests for evaluation metrics."""

    def test_evaluate_returns_valid_metrics(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """Evaluate returns valid fnorm and accuracy."""
        n = simple_adjacency.shape[0]
        embedding = LPCASymmetric(dimension=5, num_nodes=n)

        fnorm, accuracy = embedding.evaluate(simple_adjacency)

        assert fnorm >= 0
        assert 0 <= accuracy <= 1

    def test_to_result_creates_valid_result(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """to_result creates a valid EmbeddingResult."""
        n = simple_adjacency.shape[0]
        embedding = LPCASymmetric(dimension=5, num_nodes=n)

        result = embedding.to_result(simple_adjacency)

        assert isinstance(result, EmbeddingResult)
        assert result.X.shape == (n, 5)
        assert result.Y is None
        assert result.dimension == 5
        assert result.reconstruction.shape == (n, n)
        assert result.fnorm_error >= 0
        assert 0 <= result.edge_accuracy <= 1


class TestGradientFitter:
    """Tests for gradient-based fitting."""

    def test_fitting_reduces_loss(self, simple_adjacency: torch.Tensor) -> None:
        """Gradient fitting reduces loss over iterations."""
        n = simple_adjacency.shape[0]
        embedding = LPCASymmetric(dimension=n, num_nodes=n)

        initial_loss = embedding.compute_loss(simple_adjacency).item()

        config = FitConfig(lr=0.1, max_steps=100, patience=1000)
        fitter = GradientFitter(config)
        fitter.fit(embedding, simple_adjacency)

        final_loss = embedding.compute_loss(simple_adjacency).item()

        assert final_loss < initial_loss

    def test_high_dimension_achieves_good_accuracy(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """Full dimension fitting achieves high accuracy on simple graph."""
        n = simple_adjacency.shape[0]
        # Use full rank
        embedding = LPCASymmetric(dimension=n, num_nodes=n)

        config = FitConfig(
            lr=0.1,
            max_steps=500,
            tol_fnorm=0.1,
            tol_accuracy=0.95,
        )
        fitter = GradientFitter(config)
        result = fitter.fit(embedding, simple_adjacency)

        assert result.edge_accuracy > 0.8  # Should be quite good


class TestSpectralFitter:
    """Tests for spectral/SVD fitting."""

    def test_dot_product_svd_provides_good_accuracy(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """SVD provides good accuracy for dot product embedding.

        Note: X·Xᵀ can't perfectly match a zero-diagonal matrix since
        the diagonal elements are always positive (squared norms).
        """
        n = simple_adjacency.shape[0]

        # Full rank should provide good edge prediction
        embedding = DotProductSymmetric(dimension=n, num_nodes=n)

        fitter = SpectralFitter()
        result = fitter.fit(embedding, simple_adjacency)

        # Should achieve good edge accuracy (not fnorm, due to diagonal mismatch)
        assert result.edge_accuracy > 0.8

    def test_spectral_initialization_for_lpca(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """Spectral initialization provides reasonable starting point for LPCA."""
        n = simple_adjacency.shape[0]
        embedding = LPCASymmetric(dimension=n, num_nodes=n)

        fitter = SpectralFitter()
        fitter.initialize(embedding, simple_adjacency)

        # Should provide a reasonable starting point
        _, accuracy = embedding.evaluate(simple_adjacency)
        assert accuracy > 0.5  # Better than random


class TestThresholdEmbeddings:
    """Tests for threshold-based embeddings."""

    def test_dot_threshold_has_learnable_threshold(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """DotThreshold has a learnable threshold parameter."""
        n = simple_adjacency.shape[0]
        embedding = DotThresholdSymmetric(
            dimension=5, num_nodes=n, learn_threshold=True
        )

        assert hasattr(embedding, "_threshold")
        assert isinstance(embedding._threshold, torch.nn.Parameter)

    def test_distance_threshold_computes_distances(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """DistanceThreshold correctly computes pairwise distances."""
        n = simple_adjacency.shape[0]
        embedding = DistanceThresholdSymmetric(dimension=3, num_nodes=n)

        # Force specific embeddings
        with torch.no_grad():
            embedding.X[0] = torch.tensor([0.0, 0.0, 0.0])
            embedding.X[1] = torch.tensor([1.0, 0.0, 0.0])
            embedding.X[2] = torch.tensor([0.0, 1.0, 0.0])

        distances = embedding._compute_distances()

        # Distance from node 0 to node 1 should be 1.0
        assert abs(distances[0, 1].item() - 1.0) < 1e-5
        # Distance from node 0 to node 2 should be 1.0
        assert abs(distances[0, 2].item() - 1.0) < 1e-5
        # Distance from node 1 to node 2 should be sqrt(2)
        assert abs(distances[1, 2].item() - math.sqrt(2)) < 1e-5

    def test_orthogonal_uses_absolute_inner_product(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """OrthogonalRep uses absolute value of inner product."""
        n = simple_adjacency.shape[0]
        embedding = OrthogonalRepSymmetric(dimension=3, num_nodes=n)

        # Reconstruction should be symmetric
        reconstruction = embedding.reconstruct()

        assert torch.allclose(reconstruction, reconstruction.T)


class TestDimensionSearch:
    """Tests for dimension search functionality."""

    def test_search_finds_valid_dimension(self, simple_adjacency: torch.Tensor) -> None:
        """Dimension search finds a dimension that works."""
        from tmgg.experiments.embedding_study.embeddings.dimension_search import (
            DimensionSearcher,
            EmbeddingType,
        )

        searcher = DimensionSearcher(
            tol_fnorm=0.5,  # Relaxed for speed
            tol_accuracy=0.9,
            fitter="both",
        )

        result = searcher.find_min_dimension(
            simple_adjacency, EmbeddingType.DOT_PRODUCT_SYMMETRIC
        )

        assert result.min_dimension >= 1
        assert result.min_dimension <= simple_adjacency.shape[0]

    def test_binary_search_produces_history(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """Binary search records its attempts in history."""
        from tmgg.experiments.embedding_study.embeddings.dimension_search import (
            DimensionSearcher,
            EmbeddingType,
        )

        searcher = DimensionSearcher(tol_fnorm=0.5, tol_accuracy=0.9)

        result = searcher.find_min_dimension(
            simple_adjacency, EmbeddingType.LPCA_SYMMETRIC
        )

        assert len(result.search_history) > 0
        # Each entry is (dimension, converged)
        assert all(isinstance(h[0], int) for h in result.search_history)
        assert all(isinstance(h[1], bool) for h in result.search_history)


class TestGaugeStabilization:
    """Tests for gauge-stabilized LPCA fitting.

    Test rationale:
        The gauge stabilization techniques address GL(r) gauge freedom in LPCA
        factorizations. These tests verify:
        1. Canonicalization produces deterministic eigenvector ordering
        2. Hadamard anchors have expected flat-spectrum properties
        3. Θ-space interpolation produces valid initializations
        4. Gauge-stabilized fitter works with LPCA embeddings
    """

    def test_canonicalize_is_deterministic(self) -> None:
        """Canonicalization produces same result on repeated calls.

        The canonicalization rules (sign normalization + degenerate subspace
        sorting) should yield identical results for the same input.
        """
        n = 20
        # Create a symmetric matrix with some eigenvalue degeneracy
        A = torch.randn(n, n)
        A = (A + A.T) / 2

        eigenvalues, eigenvectors = torch.linalg.eigh(A)

        # Run canonicalization twice
        V1 = canonicalize_eigenvectors(eigenvectors, eigenvalues)
        V2 = canonicalize_eigenvectors(eigenvectors, eigenvalues)

        assert torch.allclose(V1, V2)

    def test_canonicalize_sign_rule(self) -> None:
        """Canonicalization applies sign rule: first nonzero entry positive.

        After canonicalization, the first nonzero entry of each eigenvector
        should be positive.
        """
        n = 10
        A = torch.randn(n, n)
        A = (A + A.T) / 2

        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        V_canon = canonicalize_eigenvectors(eigenvectors, eigenvalues)

        for j in range(n):
            v = V_canon[:, j]
            nonzero_mask = v.abs() > 1e-10
            if nonzero_mask.any():
                first_nonzero_idx = int(nonzero_mask.nonzero()[0, 0].item())
                assert (
                    v[first_nonzero_idx] >= 0
                ), f"Eigenvector {j} has negative first entry"

    def test_hadamard_theta_is_symmetric(self) -> None:
        """Hadamard Θ matrix is symmetric."""
        n = 16
        rank = 8
        theta = compute_hadamard_theta(n, rank)

        assert theta.shape == (n, n)
        assert torch.allclose(theta, theta.T)

    def test_hadamard_anchors_shape(self) -> None:
        """Hadamard anchors have correct shape."""
        n = 20
        rank = 5

        X_had, Y_had = compute_hadamard_anchors(n, rank)

        assert X_had.shape == (n, rank)
        assert Y_had.shape == (n, rank)
        # For symmetric case, X_had should equal Y_had
        assert torch.allclose(X_had, Y_had)

    def test_interpolated_init_produces_valid_embeddings(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """Θ-space interpolation produces valid embedding matrices."""
        n = simple_adjacency.shape[0]
        rank = 5

        X, Y = interpolated_theta_init(simple_adjacency, rank)

        assert X.shape == (n, rank)
        assert Y.shape == (n, rank)
        assert torch.isfinite(X).all()
        assert torch.isfinite(Y).all()

    def test_interpolated_init_alpha_blends(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """Different alpha values produce different initializations.

        alpha=0 should give fully graph-specific init (SVD of logit(A)),
        alpha=1 should give fully Hadamard-based init.
        """
        rank = 5

        X_0, _ = interpolated_theta_init(simple_adjacency, rank, alpha=0.0)
        X_1, _ = interpolated_theta_init(simple_adjacency, rank, alpha=1.0)

        # These should be different
        assert not torch.allclose(X_0, X_1)

    def test_gauge_stabilized_fitter_requires_lpca(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """GaugeStabilizedFitter rejects non-LPCA embeddings."""
        n = simple_adjacency.shape[0]
        embedding = DotProductSymmetric(dimension=5, num_nodes=n)

        fitter = GaugeStabilizedFitter()

        with pytest.raises(TypeError, match="LPCA embedding"):
            fitter.fit(embedding, simple_adjacency)

    def test_gauge_stabilized_fitter_works_with_lpca(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """GaugeStabilizedFitter successfully fits LPCA embeddings."""
        n = simple_adjacency.shape[0]
        embedding = LPCASymmetric(dimension=n, num_nodes=n)

        config = GaugeStabilizedConfig(
            max_steps=100,
            alpha=0.1,
            lambda_had=0.01,
            use_anchor=True,
        )
        fitter = GaugeStabilizedFitter(config)

        result = fitter.fit(embedding, simple_adjacency)

        assert isinstance(result, EmbeddingResult)
        assert result.fnorm_error >= 0
        assert 0 <= result.edge_accuracy <= 1

    def test_gauge_stabilized_init_only_mode(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """fit_init_only initializes without optimization."""
        n = simple_adjacency.shape[0]
        embedding = LPCASymmetric(dimension=n, num_nodes=n)

        initial_X = embedding.X.clone()

        fitter = GaugeStabilizedFitter()
        result = fitter.fit_init_only(embedding, simple_adjacency)

        # Embeddings should have changed (from random init to spectral init)
        assert not torch.allclose(embedding.X, initial_X)
        assert isinstance(result, EmbeddingResult)

    def test_gauge_stabilized_with_asymmetric_lpca(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """GaugeStabilizedFitter works with asymmetric LPCA."""
        n = simple_adjacency.shape[0]
        embedding = LPCAAsymmetric(dimension=5, num_nodes=n)

        config = GaugeStabilizedConfig(max_steps=50)
        fitter = GaugeStabilizedFitter(config)

        result = fitter.fit(embedding, simple_adjacency)

        assert isinstance(result, EmbeddingResult)
        assert result.Y is not None

    def test_dimension_search_with_gauge_stabilized(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """Dimension search works with gauge-stabilized fitter."""
        from tmgg.experiments.embedding_study.embeddings.dimension_search import (
            DimensionSearcher,
            EmbeddingType,
        )

        searcher = DimensionSearcher(
            tol_fnorm=0.5,
            tol_accuracy=0.9,
            fitter="gauge-stabilized",
        )

        result = searcher.find_min_dimension(
            simple_adjacency, EmbeddingType.LPCA_SYMMETRIC
        )

        assert result.min_dimension >= 1
        assert result.min_dimension <= simple_adjacency.shape[0]

    def test_interpolated_svd_init_produces_valid_embeddings(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """SVD-space interpolation produces valid embeddings.

        Unlike Θ-space interpolation which blends logit matrices before
        eigendecomposition, SVD-space interpolation decomposes first, aligns
        eigenvectors via Procrustes, then interpolates spectral components.
        """
        from tmgg.experiments.embedding_study.embeddings.fitters.gauge_stabilized import (
            interpolated_svd_init,
        )

        n = simple_adjacency.shape[0]
        rank = 5

        X, Y = interpolated_svd_init(simple_adjacency, rank)

        # Correct shapes
        assert X.shape == (n, rank)
        assert Y.shape == (n, rank)

        # Finite values
        assert torch.isfinite(X).all()
        assert torch.isfinite(Y).all()

    def test_interpolated_svd_init_differs_from_theta(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """SVD-space and Θ-space interpolation produce different results.

        Both methods blend graph-specific and Hadamard information, but the
        order of operations differs: SVD-space decomposes then blends, while
        Θ-space blends then decomposes.
        """
        from tmgg.experiments.embedding_study.embeddings.fitters.gauge_stabilized import (
            interpolated_svd_init,
        )

        rank = 5
        alpha = 0.3  # Use moderate alpha to see the difference

        X_theta, _ = interpolated_theta_init(simple_adjacency, rank, alpha=alpha)
        X_svd, _ = interpolated_svd_init(simple_adjacency, rank, alpha=alpha)

        # The two methods should produce different initializations
        # (unless by coincidence the two operations commute, which is unlikely)
        assert not torch.allclose(X_theta, X_svd, atol=1e-3)

    def test_gauge_stabilized_svd_init_mode(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """GaugeStabilizedFitter uses SVD-space init when configured."""
        n = simple_adjacency.shape[0]
        embedding_theta = LPCASymmetric(dimension=5, num_nodes=n)
        embedding_svd = LPCASymmetric(dimension=5, num_nodes=n)

        config_theta = GaugeStabilizedConfig(init_mode="theta", max_steps=0)
        config_svd = GaugeStabilizedConfig(init_mode="svd", max_steps=0)

        fitter_theta = GaugeStabilizedFitter(config_theta)
        fitter_svd = GaugeStabilizedFitter(config_svd)

        # Initialize both (no optimization due to max_steps=0)
        fitter_theta.fit_init_only(embedding_theta, simple_adjacency)
        fitter_svd.fit_init_only(embedding_svd, simple_adjacency)

        # Different init modes should produce different embeddings
        assert not torch.allclose(embedding_theta.X, embedding_svd.X, atol=1e-3)

    def test_dimension_search_with_gauge_stabilized_svd(
        self, simple_adjacency: torch.Tensor
    ) -> None:
        """Dimension search works with gauge-stabilized-svd fitter."""
        from tmgg.experiments.embedding_study.embeddings.dimension_search import (
            DimensionSearcher,
            EmbeddingType,
        )

        searcher = DimensionSearcher(
            tol_fnorm=0.5,
            tol_accuracy=0.9,
            fitter="gauge-stabilized-svd",
        )

        result = searcher.find_min_dimension(
            simple_adjacency, EmbeddingType.LPCA_SYMMETRIC
        )

        assert result.min_dimension >= 1
        assert result.min_dimension <= simple_adjacency.shape[0]

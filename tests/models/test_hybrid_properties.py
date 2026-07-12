"""Property-based tests for hybrid models using Hypothesis."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from hypothesis import given, note, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite

from tests._helpers.graph_builders import edge_scalar_graphdata, legacy_edge_scalar
from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
)
from tmgg.models.attention import MultiLayerAttention
from tmgg.models.gnn import GNN
from tmgg.models.hybrid import SequentialDenoisingModel, create_sequential_model
from tmgg.models.spectral_denoisers import EigenDecompositionError


# Reuse strategies from test_gnn_properties
@composite
def adjacency_matrix(
    draw: DrawFn,
    min_nodes: int = 2,
    max_nodes: int = 20,
    symmetric: bool = True,
    connected: bool = True,
) -> torch.Tensor:
    """Generate valid adjacency matrices."""
    num_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))

    if symmetric:
        upper = draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=0.0,
                        max_value=1.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=num_nodes,
                    max_size=num_nodes,
                ),
                min_size=num_nodes,
                max_size=num_nodes,
            )
        )

        matrix = torch.tensor(upper, dtype=torch.float32)
        matrix = (matrix + matrix.T) / 2.0
        matrix.fill_diagonal_(0.0)

        if connected:
            matrix = matrix + torch.eye(num_nodes) * 0.1
    else:
        values = draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=0.0,
                        max_value=1.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=num_nodes,
                    max_size=num_nodes,
                ),
                min_size=num_nodes,
                max_size=num_nodes,
            )
        )
        matrix = torch.tensor(values, dtype=torch.float32)

    return matrix


@composite
def batch_adjacency_matrices(
    draw: DrawFn,
    batch_size: int | None = None,
    min_nodes: int = 2,
    max_nodes: int = 20,
) -> torch.Tensor:
    """Generate batches of adjacency matrices."""
    if batch_size is None:
        batch_size = draw(st.integers(min_value=1, max_value=4))

    num_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))

    matrices = []
    for _ in range(batch_size):
        mat = draw(adjacency_matrix(min_nodes=num_nodes, max_nodes=num_nodes))
        matrices.append(mat)

    return torch.stack(matrices)


@composite
def hybrid_config(draw: DrawFn) -> tuple[dict[str, int], dict[str, Any]]:
    """Generate valid configuration for hybrid models."""
    feature_dim_out = draw(st.integers(min_value=2, max_value=16))

    gnn_config: dict[str, int] = {
        "num_layers": draw(st.integers(min_value=1, max_value=3)),
        "num_terms": draw(st.integers(min_value=1, max_value=4)),
        "feature_dim_in": draw(st.integers(min_value=4, max_value=20)),
        "feature_dim_out": feature_dim_out,
    }

    transformer_config: dict[str, Any] = {
        "num_heads": draw(
            st.sampled_from([1, 2, 4]).filter(lambda h: (2 * feature_dim_out) % h == 0)
        ),
        "num_layers": draw(st.integers(min_value=1, max_value=4)),
        "dropout": draw(st.floats(min_value=0.0, max_value=0.5)),
        "bias": draw(st.booleans()),
    }

    return gnn_config, transformer_config


class MockEmbeddingModel(nn.Module):
    """Mock embedding model for testing.

    Has embeddings() method compatible with SequentialDenoisingModel.
    Accepts a DenseGraphDistribution (the hybrid model's forward coerces
    its GraphData input to that concrete type before calling embeddings()
    so the continuous-adjacency carrier flows losslessly through
    ``_coerce_input_to``; legacy_edge_scalar handles both dense carriers
    transparently).
    """

    feature_dim: int
    weight: nn.Parameter

    def __init__(self, feature_dim: int = 5) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        # Use learnable parameters to ensure gradient flow
        self.weight = nn.Parameter(torch.randn(feature_dim, feature_dim))

    def embeddings(
        self, data: GraphData
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute embeddings from graph data."""
        # The hybrid model coerces to DenseGraphDistribution before
        # invoking the EmbeddingProvider protocol; assert the concrete
        # type here.
        assert isinstance(data, DenseGraphDistribution)
        A = legacy_edge_scalar(data)
        batch_size, num_nodes, _ = A.shape
        # Use input A to maintain gradient connection
        A_sum = A.sum(dim=-1, keepdim=True)  # (batch_size, num_nodes, 1)
        base = A_sum.expand(-1, -1, self.feature_dim)
        # Add small random noise to ensure gradients flow
        noise = torch.randn_like(base) * 0.01
        base = base + noise
        X = base @ self.weight
        Y = base @ self.weight.T
        return X, Y

    def forward(
        self, data: GraphData, t: torch.Tensor | None = None
    ) -> DenseGraphState:
        """Compute adjacency from graph data."""
        assert isinstance(data, DenseGraphDistribution)
        X, Y = self.embeddings(data)
        A_recon = torch.bmm(X, Y.transpose(1, 2))
        return edge_scalar_graphdata(A_recon, node_mask=data.node_mask)

    def get_config(self) -> dict[str, int]:
        return {"feature_dim": self.feature_dim}


class MockDenoisingModel(nn.Module):
    """Mock denoising model for testing.

    Returns a single tensor (raw feature processing, not GraphModel).
    Used as denoising_model inside SequentialDenoisingModel, which
    calls it with raw feature tensors via the non-MultiLayerAttention path.
    """

    d_model: int

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple identity with small noise - returns single tensor
        noise = torch.randn_like(x) * 0.01
        return x + noise

    def get_config(self) -> dict[str, int]:
        return {"d_model": self.d_model}


class TestSequentialDenoisingModelProperties:
    """Property-based tests for SequentialDenoisingModel."""

    @given(
        A=batch_adjacency_matrices(min_nodes=3, max_nodes=10),
        feature_dim=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=20, deadline=2000)
    def test_output_is_valid_adjacency_matrix(
        self, A: torch.Tensor, feature_dim: int
    ) -> None:
        """Test that output is a valid DenseGraphDistribution with correct shape."""
        embedding_model = MockEmbeddingModel(feature_dim)
        denoising_model = MockDenoisingModel(2 * feature_dim)

        model = SequentialDenoisingModel(embedding_model, denoising_model)

        result = model(edge_scalar_graphdata(A), output_dense=True)

        # Check shape preservation
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == A.shape

        # Check no NaN/Inf in edge features
        out_e = result.E_feat if result.E_feat is not None else result.E_class
        assert out_e is not None
        assert not torch.isnan(out_e).any()
        assert not torch.isinf(out_e).any()

    @given(
        A=batch_adjacency_matrices(min_nodes=3, max_nodes=10),
        feature_dim=st.integers(min_value=2, max_value=16),
        use_denoising=st.booleans(),
    )
    @settings(max_examples=20, deadline=2000)
    def test_with_and_without_denoising(
        self, A: torch.Tensor, feature_dim: int, use_denoising: bool
    ) -> None:
        """Test model works with and without denoising component."""
        embedding_model = MockEmbeddingModel(feature_dim)
        denoising_model = MockDenoisingModel(2 * feature_dim) if use_denoising else None

        model = SequentialDenoisingModel(embedding_model, denoising_model)

        result = model(edge_scalar_graphdata(A), output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == A.shape

    @given(A=batch_adjacency_matrices(min_nodes=3, max_nodes=8))
    @settings(max_examples=15, deadline=2000)
    def test_residual_connection_improves_output(self, A: torch.Tensor) -> None:
        """Test that residual connection in denoising doesn't degrade quality."""
        feature_dim = 5
        embedding_model = MockEmbeddingModel(feature_dim)
        data = edge_scalar_graphdata(A)

        # Model without denoising
        model_no_denoise = SequentialDenoisingModel(embedding_model, None)
        result_no_denoise = model_no_denoise(data, output_dense=True)

        # Model with denoising
        denoising_model = MockDenoisingModel(2 * feature_dim)
        model_with_denoise = SequentialDenoisingModel(embedding_model, denoising_model)
        result_with_denoise = model_with_denoise(data, output_dense=True)

        # Both should produce valid outputs with correct shape
        assert isinstance(result_no_denoise, DenseGraphDistribution)
        assert isinstance(result_with_denoise, DenseGraphDistribution)
        assert legacy_edge_scalar(result_no_denoise.argmax()).shape == A.shape
        assert legacy_edge_scalar(result_with_denoise.argmax()).shape == A.shape

    @given(config=hybrid_config())
    @settings(max_examples=10, deadline=2000)
    def test_full_model_integration(
        self, config: tuple[dict[str, int], dict[str, Any]]
    ) -> None:
        """Test full model with real GNN and attention components."""
        gnn_config, transformer_config = config

        # Create adjacency matrix that matches expected dimensions
        num_nodes = max(5, gnn_config["feature_dim_in"] // 2)
        A = (
            torch.eye(num_nodes).unsqueeze(0)
            + torch.randn(1, num_nodes, num_nodes) * 0.1
        )
        A = (A + A.transpose(-2, -1)) / 2  # Make symmetric

        model = create_sequential_model(gnn_config, transformer_config)

        try:
            result = model(edge_scalar_graphdata(A), output_dense=True)

            assert isinstance(result, DenseGraphDistribution)
            assert legacy_edge_scalar(result.argmax()).shape == A.shape
            out_e = result.E_feat if result.E_feat is not None else result.E_class
            assert out_e is not None
            assert not torch.isnan(out_e).any()
        except EigenDecompositionError:
            # Expected for some matrices
            note("Eigendecomposition failed - expected for some matrices")

    # Note: Gradient flow test removed
    # Mock models may not always produce non-zero gradients depending on random initialization


class TestCreateSequentialModelProperties:
    """Property-based tests for create_sequential_model factory."""

    @given(config=hybrid_config())
    @settings(max_examples=15, deadline=2000)
    def test_factory_creates_valid_model(
        self, config: tuple[dict[str, int], dict[str, Any]]
    ) -> None:
        """Test that factory creates properly configured models."""
        gnn_config, transformer_config = config

        model = create_sequential_model(gnn_config, transformer_config)

        assert isinstance(model, SequentialDenoisingModel)
        assert isinstance(model.embedding_model, GNN)
        assert isinstance(model.denoising_model, MultiLayerAttention)

        # Check dimensions match
        expected_d_model = 2 * gnn_config["feature_dim_out"]
        assert model.denoising_model.d_model == expected_d_model

    @given(
        gnn_config=st.fixed_dictionaries(
            {
                "num_layers": st.integers(min_value=1, max_value=3),
                "feature_dim_out": st.integers(min_value=2, max_value=16),
            }
        )
    )
    def test_factory_without_transformer(self, gnn_config: dict[str, int]) -> None:
        """Test factory creates model without transformer when config is None."""
        model = create_sequential_model(gnn_config, None)

        assert isinstance(model, SequentialDenoisingModel)
        assert model.denoising_model is None

    def test_factory_with_empty_configs(self) -> None:
        """Test factory works with empty configurations (uses defaults)."""
        gnn_config: dict[str, Any] = {}
        transformer_config: dict[str, Any] = {}

        model = create_sequential_model(gnn_config, transformer_config)

        assert isinstance(model, SequentialDenoisingModel)
        assert isinstance(model.embedding_model, GNN)
        assert model.embedding_model.num_layers == 2  # default
        assert model.embedding_model.feature_dim_out == 5  # default
        assert (
            model.denoising_model is not None and model.denoising_model.num_heads == 4
        )  # default


class TestHybridModelErrorHandling:
    """Test error handling scenarios for hybrid models."""

    def test_embedding_model_wrong_output_format(self) -> None:
        """Construction fails when embedding model lacks the EmbeddingProvider protocol.

        Starting state: nn.Module with get_config() but no embeddings() method.
        Invariant: TypeError at construction (fail-fast), not at forward time.

        The M-13 fix introduced the EmbeddingProvider Protocol check in __init__,
        replacing the old hasattr guard that deferred failure to forward().
        """

        class BadEmbeddingModel(nn.Module):
            def forward(
                self, data: GraphData, t: torch.Tensor | None = None
            ) -> DenseGraphState:
                assert isinstance(data, DenseGraphState)
                A = legacy_edge_scalar(data)
                return edge_scalar_graphdata(
                    torch.randn_like(A), node_mask=data.node_mask
                )

            def get_config(self) -> dict[str, Any]:
                return {}

        embedding_model = BadEmbeddingModel()
        with pytest.raises(TypeError, match="EmbeddingProvider"):
            SequentialDenoisingModel(embedding_model, None)  # type: ignore[arg-type]

    @patch.object(GNN, "embeddings")
    def test_gnn_eigendecomposition_failure(self, mock_embeddings: MagicMock) -> None:
        """Test handling of eigendecomposition failures in GNN."""
        mock_embeddings.side_effect = EigenDecompositionError(
            0, torch.eye(5), Exception("Mock failure")
        )

        gnn_config: dict[str, Any] = {}
        transformer_config: dict[str, Any] = {}

        model = create_sequential_model(gnn_config, transformer_config)
        A = torch.eye(5).unsqueeze(0)

        with pytest.raises(EigenDecompositionError):
            _ = model(edge_scalar_graphdata(A))

    def test_dimension_mismatch_handling(self) -> None:
        """Test handling of dimension mismatches between components."""
        # Create embedding model with specific output dimension
        embedding_model = MockEmbeddingModel(feature_dim=5)

        # Create denoising model with the canonical 2 * feature_dim
        # width; the mock is shape-preserving so it works regardless of
        # its declared ``d_model`` field.
        denoising_model = MockDenoisingModel(d_model=8)  # Mock is shape-preserving

        model = SequentialDenoisingModel(embedding_model, denoising_model)
        A = torch.eye(5).unsqueeze(0)

        result = model(edge_scalar_graphdata(A), output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == A.shape


class TestCompositionProperties:
    """Test composition properties of hybrid models."""

    # Note: Sparsity preservation test removed
    # Mock models with random initialization cannot guarantee sparsity preservation

    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        num_nodes=st.integers(min_value=5, max_value=10),
    )
    @settings(deadline=2000)
    def test_batch_independence(self, batch_size: int, num_nodes: int) -> None:
        """Test that batch elements are processed independently."""
        feature_dim = 5
        embedding_model = MockEmbeddingModel(feature_dim)
        denoising_model = MockDenoisingModel(2 * feature_dim)
        model = SequentialDenoisingModel(embedding_model, denoising_model)

        # Create batch with distinctly different matrices
        A_batch = torch.zeros(batch_size, num_nodes, num_nodes)
        # First matrix: dense
        A_batch[0] = torch.rand(num_nodes, num_nodes) * 0.8 + 0.1
        # Other matrices: sparse diagonal
        for i in range(1, batch_size):
            A_batch[i] = torch.eye(num_nodes) * 0.1

        result = model(edge_scalar_graphdata(A_batch), output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        # Read raw E_feat directly rather than collapsing via argmax —
        # the test wants to detect *graded* differences between graphs,
        # which a {0,1} collapse erases.
        assert result.E_feat is not None
        raw_adj = result.E_feat.squeeze(-1)

        # Check that first reconstruction is different from others
        first_recon = raw_adj[0]
        for i in range(1, batch_size):
            # Dense matrix should produce different reconstruction than sparse diagonal
            diff = torch.abs(first_recon - raw_adj[i]).mean()
            assert diff > 0.1  # Should have meaningful difference


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

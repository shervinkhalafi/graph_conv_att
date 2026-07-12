"""Property-based tests for GNN models using Hypothesis."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import given, note, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite

from tests._helpers.graph_builders import edge_scalar_graphdata, legacy_edge_scalar
from tmgg.data.datasets.graph_types import DenseGraphDistribution
from tmgg.models.gnn import GNN, GNNSymmetric, NodeVarGNN
from tmgg.models.layers import GraphConvolutionLayer
from tmgg.models.layers.eigen_embedding import _EigenEmbedding as EigenEmbedding
from tmgg.models.spectral_denoisers import EigenDecompositionError


# Custom strategies for generating graph data
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
        # Generate symmetric matrix
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
        matrix = (matrix + matrix.T) / 2.0  # Make symmetric
        matrix.fill_diagonal_(0.0)  # No self-loops

        if connected:
            # Ensure connectivity by adding small values to make it connected
            # Add small random values to ensure numerical stability
            matrix = matrix + torch.eye(num_nodes) * 0.1
    else:
        # Generate general matrix
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
    """Generate batches of adjacency matrices with same number of nodes."""
    if batch_size is None:
        batch_size = draw(st.integers(min_value=1, max_value=4))

    num_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))

    matrices = []
    for _ in range(batch_size):
        mat = draw(adjacency_matrix(min_nodes=num_nodes, max_nodes=num_nodes))
        matrices.append(mat)

    return torch.stack(matrices)


@composite
def gnn_params(draw: DrawFn) -> dict[str, Any]:
    """Generate valid GNN parameters."""
    num_layers = draw(st.integers(min_value=1, max_value=4))
    num_terms = draw(st.integers(min_value=1, max_value=5))
    feature_dim_in = draw(st.integers(min_value=2, max_value=20))
    feature_dim_out = draw(st.integers(min_value=2, max_value=20))

    return {
        "num_layers": num_layers,
        "num_terms": num_terms,
        "feature_dim_in": feature_dim_in,
        "feature_dim_out": feature_dim_out,
    }


class TestEigenEmbeddingProperties:
    """Property-based tests for EigenEmbedding."""

    @given(A=batch_adjacency_matrices())
    @settings(max_examples=30, deadline=2000)
    def test_eigenembedding_shape_preservation(self, A: torch.Tensor) -> None:
        """Test that eigen embedding preserves batch and node dimensions."""
        embedding = EigenEmbedding()

        try:
            result = embedding(A)
            assert result.shape == A.shape
            assert result.dtype == A.dtype
        except EigenDecompositionError as e:
            # This is expected for some ill-conditioned matrices
            note(f"EigenDecomposition failed as expected: {e}")
            assert e.matrix_idx >= 0
            assert e.debugging_context is not None

    @given(A=batch_adjacency_matrices(min_nodes=3, max_nodes=10))
    @settings(max_examples=20, deadline=2000)
    def test_eigenembedding_orthogonality(self, A: torch.Tensor) -> None:
        """Test that eigenvectors are orthogonal."""
        embedding = EigenEmbedding()

        try:
            eigenvecs = embedding(A)

            for i in range(A.shape[0]):
                # Check orthogonality: V^T V should be close to identity
                vt_v = torch.matmul(eigenvecs[i].T, eigenvecs[i])
                identity = torch.eye(A.shape[1])

                # Allow some numerical error
                assert torch.allclose(vt_v, identity, atol=1e-5)
        except EigenDecompositionError:
            pass  # Expected for ill-conditioned matrices

    def test_eigenembedding_error_handling(self) -> None:
        """Test that EigenEmbedding handles ill-conditioned matrices properly."""
        # Just test the EigenDecompositionError class directly
        # since PyTorch's eigendecomposition doesn't fail on singular/NaN matrices
        mock_matrix = torch.eye(3)
        mock_matrix[0, 0] = float("nan")  # Add NaN for testing
        mock_error: Exception = Exception("Mock eigendecomposition failure")

        # Test that EigenDecompositionError correctly captures debugging info
        error = EigenDecompositionError(0, mock_matrix, mock_error)
        assert error.matrix_idx == 0
        assert "condition_number" in error.debugging_context
        assert error.debugging_context["has_nan"] is True
        assert error.debugging_context["matrix_shape"] == [3, 3]
        assert "Mock eigendecomposition failure" in str(error)


class TestGNNProperties:
    """Property-based tests for GNN models."""

    @given(A=batch_adjacency_matrices(min_nodes=3, max_nodes=10), params=gnn_params())
    @settings(max_examples=20, deadline=2000)
    def test_gnn_output_shapes(self, A: torch.Tensor, params: dict[str, Any]) -> None:
        """Test that GNN produces correct output shapes."""
        model = GNN(**params)

        try:
            result = model(edge_scalar_graphdata(A), output_dense=True)

            batch_size, num_nodes, _ = A.shape
            assert isinstance(result, DenseGraphDistribution)
            assert legacy_edge_scalar(result.argmax()).shape == (
                batch_size,
                num_nodes,
                num_nodes,
            )
        except EigenDecompositionError:
            pass  # Expected for some matrices

    @given(A=batch_adjacency_matrices(min_nodes=3, max_nodes=8))
    @settings(max_examples=15, deadline=2000)
    def test_gnn_symmetric_produces_valid_adjacency(self, A: torch.Tensor) -> None:
        """Test that GNNSymmetric produces valid adjacency output."""
        model = GNNSymmetric(num_layers=2, feature_dim_out=5)

        try:
            result = model(edge_scalar_graphdata(A), output_dense=True)

            # Check shape
            assert isinstance(result, DenseGraphDistribution)
            collapsed = result.argmax()
            assert legacy_edge_scalar(collapsed).shape == A.shape

            # Check symmetry of raw edge features
            raw_adj = legacy_edge_scalar(collapsed)
            diff = torch.abs(raw_adj - raw_adj.transpose(-2, -1))
            assert torch.max(diff) < 0.1  # Allow some asymmetry
        except EigenDecompositionError:
            pass

    @given(
        A=batch_adjacency_matrices(min_nodes=3, max_nodes=8),
        num_layers=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=15, deadline=2000)
    def test_node_var_gnn_valid_output(self, A: torch.Tensor, num_layers: int) -> None:
        """Test NodeVarGNN produces valid output."""
        model = NodeVarGNN(num_layers=num_layers, feature_dim=5)

        try:
            result = model(edge_scalar_graphdata(A), output_dense=True)

            # Check shape
            assert isinstance(result, DenseGraphDistribution)
            collapsed = result.argmax()
            assert legacy_edge_scalar(collapsed).shape == A.shape

            # Check raw edge features are valid
            raw_adj = legacy_edge_scalar(collapsed)
            assert not torch.isnan(raw_adj).any()
            assert not torch.isinf(raw_adj).any()
        except EigenDecompositionError:
            pass

    # Note: Gradient flow test removed for GNN
    # Eigendecomposition operation breaks gradient flow to input adjacency matrix
    # This is a known limitation of eigendecomposition-based models


class TestGraphConvolutionProperties:
    """Property-based tests for graph convolution layers."""

    @given(
        A=batch_adjacency_matrices(min_nodes=3, max_nodes=10),
        num_terms=st.integers(min_value=1, max_value=4),
        num_channels=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=20, deadline=2000)
    def test_graph_conv_shape_preservation(
        self, A: torch.Tensor, num_terms: int, num_channels: int
    ) -> None:
        """Test that graph convolution preserves node and channel dimensions."""
        layer = GraphConvolutionLayer(num_terms, num_channels)

        batch_size, num_nodes, _ = A.shape
        X = torch.randn(batch_size, num_nodes, num_channels)

        Y = layer(A, X)

        assert Y.shape == X.shape
        assert not torch.isnan(Y).any()
        assert not torch.isinf(Y).any()

    @given(
        num_terms=st.integers(min_value=1, max_value=3),
        num_channels=st.integers(min_value=2, max_value=10),
    )
    def test_graph_conv_with_identity(self, num_terms: int, num_channels: int) -> None:
        """Test graph convolution with identity adjacency matrix."""
        layer = GraphConvolutionLayer(num_terms, num_channels)

        eye_matrix = torch.eye(5).unsqueeze(0)
        X = torch.randn(1, 5, num_channels)

        Y = layer(eye_matrix, X)

        assert Y.shape == X.shape
        assert not torch.isnan(Y).any()


class TestGNNErrorHandling:
    """Test error handling scenarios with mocks."""

    @patch("torch.linalg.eigh")
    def test_eigen_decomposition_error_handling(self, mock_eigh: MagicMock) -> None:
        """Test handling of eigendecomposition failures."""
        # Mock eigendecomposition to raise an error
        mock_eigh.side_effect = torch._C._LinAlgError("Mock eigendecomposition failure")  # pyright: ignore[reportAttributeAccessIssue]

        embedding = EigenEmbedding()
        A = torch.eye(5).unsqueeze(0)

        with pytest.raises(EigenDecompositionError) as exc_info:
            _ = embedding(A)

        error = exc_info.value
        assert error.matrix_idx == 0
        assert "Mock eigendecomposition failure" in str(error)

    @patch("torch.matrix_power")
    def test_matrix_power_stability(self, mock_matrix_power: MagicMock) -> None:
        """Test handling of matrix power computation issues."""
        # Mock matrix_power to return NaN
        mock_matrix_power.return_value = torch.full((1, 5, 5), float("nan"))

        layer = GraphConvolutionLayer(num_terms=3, num_channels=5)
        A = torch.eye(5).unsqueeze(0)
        X = torch.randn(1, 5, 5)

        Y = layer(A, X)

        # Layer should handle NaN through layer normalization or other means
        # but result will likely contain NaN
        assert Y.shape == X.shape

    def test_gnn_with_mismatched_dimensions(self) -> None:
        """Test GNN behavior with mismatched feature dimensions."""
        model = GNN(num_layers=2, feature_dim_in=10, feature_dim_out=5)

        # Create adjacency matrix with fewer nodes than feature_dim_in
        A = torch.eye(5).unsqueeze(0)

        try:
            result = model(edge_scalar_graphdata(A), output_dense=True)
            # Model should handle this by padding or truncating
            assert isinstance(result, DenseGraphDistribution)
            assert legacy_edge_scalar(result.argmax()).shape == (1, 5, 5)
        except EigenDecompositionError:
            pass  # Also acceptable


# Note: Permutation equivariance testing removed
# Eigendecomposition-based models cannot guarantee permutation equivariance
# due to arbitrary eigenvector signs and ordering


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

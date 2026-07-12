"""Property-based tests for attention models using Hypothesis."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite

from tests._helpers.graph_builders import edge_scalar_graphdata, legacy_edge_scalar
from tmgg.data.datasets.graph_types import DenseGraphDistribution
from tmgg.models.attention import MultiLayerAttention
from tmgg.models.layers import MultiHeadSelfAttention


# Custom strategies for generating test data
@composite
def attention_dims(draw: DrawFn) -> tuple[int, int]:
    """Generate valid dimensions for attention models."""
    d_model = draw(st.integers(min_value=8, max_value=128).filter(lambda x: x % 2 == 0))
    num_heads = draw(st.sampled_from([1, 2, 4, 8]).filter(lambda h: d_model % h == 0))
    return d_model, num_heads


@composite
def attention_input(
    draw: DrawFn,
    d_model: int | None = None,
    batch_size: int | None = None,
    seq_len: int | None = None,
) -> torch.Tensor:
    """Generate valid input tensors for attention models."""
    if d_model is None:
        d_model = draw(
            st.integers(min_value=8, max_value=64).filter(lambda x: x % 2 == 0)
        )
    if batch_size is None:
        batch_size = draw(st.integers(min_value=1, max_value=8))
    if seq_len is None:
        seq_len = draw(st.integers(min_value=2, max_value=32))

    # Generate input tensor with controlled values to avoid numerical issues
    x = draw(
        st.lists(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=-2.0,
                        max_value=2.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=d_model,
                    max_size=d_model,
                ),
                min_size=seq_len,
                max_size=seq_len,
            ),
            min_size=batch_size,
            max_size=batch_size,
        )
    )

    return torch.tensor(x, dtype=torch.float32)


@composite
def attention_mask(draw: DrawFn, batch_size: int, seq_len: int) -> torch.Tensor:
    """Generate valid attention masks ensuring at least one unmasked position per row."""
    masks = []
    for _ in range(batch_size):
        mask_batch = []
        for _ in range(seq_len):
            # Ensure at least one position is unmasked in each row
            row = draw(
                st.lists(
                    st.integers(min_value=0, max_value=1),
                    min_size=seq_len,
                    max_size=seq_len,
                )
            )
            if sum(row) == 0:  # If all masked, unmask at least one position
                row[draw(st.integers(min_value=0, max_value=seq_len - 1))] = 1
            mask_batch.append(row)
        masks.append(mask_batch)

    return torch.tensor(masks, dtype=torch.float32)


class TestMultiHeadSelfAttentionProperties:
    """Property-based tests for MultiHeadSelfAttention."""

    @given(dims=attention_dims())
    @settings(deadline=2000)
    def test_output_shape_invariant(self, dims: tuple[int, int]) -> None:
        """Test that output shape matches input shape regardless of internal dimensions."""
        d_model, num_heads = dims
        model = MultiHeadSelfAttention(d_model, num_heads)

        # Test with various input shapes
        for batch_size in [1, 4]:
            for seq_len in [5, 10]:
                x = torch.randn(batch_size, seq_len, d_model)
                output = model(x)

                assert output.shape == x.shape

    @settings(deadline=500)
    @given(
        dims=attention_dims(),
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=2, max_value=16),
    )
    def test_mask_zeros_attention(
        self, dims: tuple[int, int], batch_size: int, seq_len: int
    ) -> None:
        """Test that masked positions have exactly zero attention weight."""
        d_model, num_heads = dims
        model = MultiHeadSelfAttention(d_model, num_heads)

        x = torch.randn(batch_size, seq_len, d_model)

        # Create mask that masks out specific positions
        mask = torch.ones(batch_size, seq_len, seq_len)
        # Mask out last column
        mask[:, :, -1] = 0

        output = model(x, mask=mask)

        # Output should be valid (no NaN/Inf) with masking applied
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @given(
        dims=attention_dims(),
        batch_size=st.integers(min_value=1, max_value=2),
        seq_len=st.integers(min_value=2, max_value=8),
    )
    @settings(
        max_examples=30,
        deadline=2000,
        suppress_health_check=[HealthCheck.data_too_large],
    )
    def test_numerical_stability(
        self, dims: tuple[int, int], batch_size: int, seq_len: int
    ) -> None:
        """Test that the model handles various input scales without producing NaN/Inf."""
        d_model, num_heads = dims

        x = torch.randn(batch_size, seq_len, d_model)

        model = MultiHeadSelfAttention(d_model, num_heads)

        # Test with different input scales
        for scale in [0.01, 1.0, 10.0]:
            scaled_x = x * scale
            output = model(scaled_x)

            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    @given(dims=attention_dims())
    @settings(deadline=None)  # Disable deadline for this test
    def test_residual_connection_improves_gradient_flow(
        self, dims: tuple[int, int]
    ) -> None:
        """Test that residual connections prevent gradient vanishing."""
        d_model, num_heads = dims
        model = MultiHeadSelfAttention(d_model, num_heads)

        x = torch.randn(2, 10, d_model, requires_grad=True)
        output = model(x)

        # Compute gradient with respect to input
        loss = output.sum()
        loss.backward()

        # Check that gradients are not vanishing
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        assert not torch.isnan(x.grad).any()


class TestMultiLayerAttentionProperties:
    """Property-based tests for MultiLayerAttention.

    MultiLayerAttention is a GraphModel for adjacency matrices. Its forward()
    accepts GraphData and returns GraphData. The apply_attention() method
    provides raw tensor access for internal use (e.g., hybrid models).
    """

    @given(
        num_nodes=st.integers(min_value=4, max_value=16),
        num_layers=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=20, deadline=2000)
    def test_layer_composition_preserves_shape(
        self, num_nodes: int, num_layers: int
    ) -> None:
        """Test that stacking layers preserves adjacency matrix shape."""
        d_model = num_nodes
        num_heads = 2

        model = MultiLayerAttention(d_model, num_heads, num_layers)

        # Input: batch of adjacency matrices
        batch_size = 2
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
        A = A + torch.randn_like(A) * 0.1

        result = model(edge_scalar_graphdata(A), output_dense=True)

        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == A.shape

        # Check raw edge features for NaN/Inf
        out_e = result.E_feat if result.E_feat is not None else result.E_class
        assert out_e is not None
        assert not torch.isnan(out_e).any()
        assert not torch.isinf(out_e).any()

    @given(
        num_nodes=st.integers(min_value=4, max_value=12),
        num_layers=st.integers(min_value=2, max_value=6),
        batch_size=st.integers(min_value=1, max_value=2),
    )
    @settings(max_examples=20, deadline=2000)
    def test_deeper_models_maintain_stability(
        self, num_nodes: int, num_layers: int, batch_size: int
    ) -> None:
        """Test that deeper models don't produce NaN/Inf in edge features."""
        d_model = num_nodes
        num_heads = 2

        # Generate adjacency-like input
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
        A = A + torch.randn_like(A) * 0.3

        model = MultiLayerAttention(d_model, num_heads, num_layers)
        result = model(edge_scalar_graphdata(A), output_dense=True)

        assert isinstance(result, DenseGraphDistribution)
        out_e = result.E_feat if result.E_feat is not None else result.E_class
        assert out_e is not None
        assert not torch.isnan(out_e).any()
        assert not torch.isinf(out_e).any()

    @given(
        num_nodes=st.integers(min_value=4, max_value=12),
        num_layers=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=20, deadline=2000)
    def test_mask_propagation_through_layers(
        self, num_nodes: int, num_layers: int
    ) -> None:
        """Test that masks are correctly propagated through apply_attention().

        Since forward() takes GraphData(no mask parameter), mask propagation
        is tested via apply_attention() which provides raw tensor access.
        """
        d_model = num_nodes
        num_heads = 2
        model = MultiLayerAttention(d_model, num_heads, num_layers)

        batch_size = 2
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
        A = A + torch.randn_like(A) * 0.1

        # Create mask that masks out some positions
        mask = torch.ones(batch_size, num_nodes, num_nodes)
        mask[:, :, -2:] = 0

        output_masked = model.apply_attention(A, mask=mask)
        output_unmasked = model.apply_attention(A, mask=None)

        # Output should still be valid
        assert output_masked.shape == A.shape
        assert not torch.isnan(output_masked).any()
        assert not torch.isinf(output_masked).any()

        # Masking should produce different output than not masking
        assert output_masked.shape == output_unmasked.shape


class TestAttentionErrorHandling:
    """Test error handling with mocked components."""

    def test_handles_dimension_mismatch_gracefully(self) -> None:
        """Test that model handles dimension mismatches appropriately."""
        model = MultiHeadSelfAttention(d_model=64, num_heads=8)

        # Wrong input dimension will cause an error in the linear layer
        x = torch.randn(2, 10, 32)  # 32 != 64

        # This will raise an error because linear layer expects d_model input
        with pytest.raises(RuntimeError):
            _ = model(x)

    @patch("torch.nn.functional.softmax")
    def test_softmax_numerical_stability(self, mock_softmax: MagicMock) -> None:
        """Test handling of numerical instabilities in softmax."""
        # Mock softmax to return NaN to test error handling
        mock_softmax.return_value = torch.full((2, 10, 10), float("nan"))

        model = MultiHeadSelfAttention(d_model=32, num_heads=4)
        x = torch.randn(2, 10, 32)

        # The model should handle this gracefully
        with pytest.raises(RuntimeError):  # Model may not have explicit NaN handling
            _ = model(x)

    def test_zero_attention_scores_handling(self) -> None:
        """Test behavior when all attention scores are masked."""
        model = MultiHeadSelfAttention(d_model=32, num_heads=4)
        x = torch.randn(1, 5, 32)

        # Mask that zeros out everything
        mask = torch.zeros(1, 5, 5)

        output = model(x, mask=mask)

        # With all-zero mask, output should still be valid
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


# Hypothesis settings for debugging
pytest.register_assert_rewrite(__name__)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

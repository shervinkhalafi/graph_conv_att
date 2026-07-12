"""Tests for attention models."""

import pytest
import torch

from tests._helpers.graph_builders import edge_scalar_graphdata, legacy_edge_scalar
from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
)
from tmgg.models.attention import MultiLayerAttention
from tmgg.models.layers import MultiHeadSelfAttention


class TestMultiHeadSelfAttention:
    """Test MultiHeadSelfAttention module."""

    def test_init(self):
        """Test initialization."""
        d_model = 64
        num_heads = 8

        attention = MultiHeadSelfAttention(d_model, num_heads)

        assert attention.d_model == d_model
        assert attention.num_heads == num_heads
        assert attention.d_k == d_model // num_heads
        assert attention.d_v == d_model // num_heads

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        batch_size = 2
        seq_len = 10
        d_model = 64
        num_heads = 8

        attention = MultiHeadSelfAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output = attention(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_mask_functionality(self):
        """Test that masking works correctly."""
        batch_size = 1
        seq_len = 5
        d_model = 32
        num_heads = 4

        attention = MultiHeadSelfAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # Create mask that masks out last position
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, -1] = 0

        output = attention(x, mask=mask)

        assert output.shape == (batch_size, seq_len, d_model)


class TestMultiLayerAttention:
    """Test MultiLayerAttention module.

    MultiLayerAttention is a denoising model that processes adjacency matrices.
    It returns a single tensor (reconstructed adjacency matrix), not a tuple.
    """

    def test_init(self):
        """Test initialization."""
        d_model = 64
        num_heads = 8
        num_layers = 4

        model = MultiLayerAttention(d_model, num_heads, num_layers)

        assert len(model.layers) == num_layers
        assert model.d_model == d_model
        assert model.num_heads == num_heads
        assert model.num_layers == num_layers

    def test_forward_shape(self):
        """Test forward pass output shapes.

        MultiLayerAttention takes GraphData and returns GraphData.
        The dense edge state is extracted/reconstructed via
        to_edge_state()/from_edge_state().
        """
        batch_size = 2
        num_nodes = 10
        d_model = num_nodes  # d_model should match adjacency matrix dimension

        model = MultiLayerAttention(d_model, num_heads=2, num_layers=2)

        # Input: batch of dense edge states wrapped as GraphData
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
        A = A + torch.randn_like(A) * 0.1  # Add some noise
        data = edge_scalar_graphdata(A)

        result = model(data, output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == (
            batch_size,
            num_nodes,
            num_nodes,
        )

    def test_forward_does_not_touch_binary_projection(self, monkeypatch):
        """Continuous attention should not use binary-topology extraction."""

        def _raise(*_args, **_kwargs):
            raise AssertionError("binary topology should not be used here")

        # Patch on every concrete class so any internal coercion path is
        # also covered. ``GraphData`` is the abstract base; the runtime
        # object is one of the four concrete subclasses.
        monkeypatch.setattr(DenseGraphState, "dense_adjacency", _raise)
        monkeypatch.setattr(DenseGraphDistribution, "dense_adjacency", _raise)

        model = MultiLayerAttention(d_model=6, num_heads=2, num_layers=1)
        data = edge_scalar_graphdata(torch.randn(1, 6, 6))

        result = model(data, output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == (1, 6, 6)

    def test_get_config(self):
        """Test configuration retrieval."""
        d_model = 64
        num_heads = 8
        num_layers = 4
        dropout = 0.1
        bias = True

        model = MultiLayerAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            bias=bias,
        )

        config = model.get_config()

        assert config["d_model"] == d_model
        assert config["num_heads"] == num_heads
        assert config["num_layers"] == num_layers
        assert config["dropout"] == dropout
        assert config["bias"] == bias


if __name__ == "__main__":
    pytest.main([__file__])

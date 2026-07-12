"""Tests for hybrid models combining GNN and attention mechanisms."""

import pytest
import torch

from tests._helpers.graph_builders import edge_scalar_graphdata, legacy_edge_scalar
from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
)
from tmgg.models.attention import MultiLayerAttention
from tmgg.models.gnn import GNN
from tmgg.models.hybrid import SequentialDenoisingModel, create_sequential_model


class TestSequentialDenoisingModel:
    """Test SequentialDenoisingModel class."""

    def test_init_with_denoising(self):
        """Test initialization with both embedding and denoising models."""
        embedding_model = GNN(
            num_layers=2, num_terms=3, feature_dim_in=20, feature_dim_out=5
        )

        denoising_model = MultiLayerAttention(
            d_model=10,  # 2 * feature_dim_out
            num_heads=2,
            num_layers=2,
        )

        model = SequentialDenoisingModel(embedding_model, denoising_model)

        assert model.embedding_model is embedding_model
        assert model.denoising_model is denoising_model

    def test_init_without_denoising(self):
        """Test initialization with only embedding model."""
        embedding_model = GNN(
            num_layers=2, num_terms=3, feature_dim_in=20, feature_dim_out=5
        )

        model = SequentialDenoisingModel(embedding_model, None)

        assert model.embedding_model is embedding_model
        assert model.denoising_model is None

    def test_forward_with_denoising(self):
        """Test forward pass with denoising model returns GraphData."""
        batch_size = 2
        num_nodes = 10
        feature_dim_out = 5

        embedding_model = GNN(
            num_layers=1,
            num_terms=2,
            feature_dim_in=num_nodes,
            feature_dim_out=feature_dim_out,
        )

        denoising_model = MultiLayerAttention(
            d_model=2 * feature_dim_out, num_heads=2, num_layers=1
        )

        model = SequentialDenoisingModel(embedding_model, denoising_model)

        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
        data = edge_scalar_graphdata(A)

        result = model(data, output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == (
            batch_size,
            num_nodes,
            num_nodes,
        )

    def test_forward_does_not_touch_binary_projection(self, monkeypatch):
        """Sequential denoising should stay in edge-state space."""

        def _raise(*_args, **_kwargs):
            raise AssertionError("binary topology should not be used here")

        monkeypatch.setattr(DenseGraphState, "dense_adjacency", _raise)
        monkeypatch.setattr(DenseGraphDistribution, "dense_adjacency", _raise)

        embedding_model = GNN(
            num_layers=1, num_terms=2, feature_dim_in=6, feature_dim_out=3
        )
        denoising_model = MultiLayerAttention(d_model=6, num_heads=2, num_layers=1)
        model = SequentialDenoisingModel(embedding_model, denoising_model)
        data = edge_scalar_graphdata(torch.randn(1, 6, 6))

        result = model(data, output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == (1, 6, 6)

    def test_forward_without_denoising(self):
        """Test forward pass without denoising model returns DenseGraphDistribution."""
        batch_size = 2
        num_nodes = 10
        feature_dim_out = 5

        embedding_model = GNN(
            num_layers=1,
            num_terms=2,
            feature_dim_in=num_nodes,
            feature_dim_out=feature_dim_out,
        )

        model = SequentialDenoisingModel(embedding_model, None)

        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
        data = edge_scalar_graphdata(A)

        result = model(data, output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == (
            batch_size,
            num_nodes,
            num_nodes,
        )

    def test_get_config(self):
        """Test configuration retrieval."""
        embedding_model = GNN(
            num_layers=2, num_terms=3, feature_dim_in=20, feature_dim_out=5
        )

        denoising_model = MultiLayerAttention(d_model=10, num_heads=2, num_layers=2)

        model = SequentialDenoisingModel(embedding_model, denoising_model)

        config = model.get_config()

        assert "embedding_model" in config
        assert "has_denoising" in config
        assert config["has_denoising"] is True
        assert "denoising_model" in config
        assert config["model_type"] == "SequentialDenoisingModel"


class TestCreateSequentialModel:
    """Test factory function for creating sequential models."""

    def test_create_with_transformer(self):
        """Test creating model with transformer denoising."""
        gnn_config = {
            "num_layers": 2,
            "num_terms": 3,
            "feature_dim_in": 20,
            "feature_dim_out": 5,
        }

        transformer_config = {
            "num_heads": 4,
            "num_layers": 2,
            "d_k": 10,
            "d_v": 10,
            "dropout": 0.1,
            "bias": True,
        }

        model = create_sequential_model(gnn_config, transformer_config)

        assert isinstance(model, SequentialDenoisingModel)
        assert isinstance(model.embedding_model, GNN)
        assert isinstance(model.denoising_model, MultiLayerAttention)

        # Check GNN config
        assert model.embedding_model.num_layers == 2
        assert model.embedding_model.feature_dim_out == 5

        # Check transformer config
        assert model.denoising_model.d_model == 10  # 2 * feature_dim_out
        assert model.denoising_model.num_heads == 4
        assert model.denoising_model.num_layers == 2

    def test_create_without_transformer(self):
        """Test creating model without transformer denoising."""
        gnn_config = {
            "num_layers": 2,
            "num_terms": 3,
            "feature_dim_in": 20,
            "feature_dim_out": 5,
        }

        model = create_sequential_model(gnn_config, None)

        assert isinstance(model, SequentialDenoisingModel)
        assert isinstance(model.embedding_model, GNN)
        assert model.denoising_model is None

    def test_create_with_defaults(self):
        """Test creating model with default configurations."""
        model = create_sequential_model({}, {})

        assert isinstance(model, SequentialDenoisingModel)
        assert isinstance(model.embedding_model, GNN)
        assert isinstance(model.denoising_model, MultiLayerAttention)

        # Check defaults
        assert model.embedding_model.num_layers == 2
        assert model.embedding_model.feature_dim_out == 5
        assert model.denoising_model.num_heads == 4
        assert model.denoising_model.num_layers == 4

    def test_forward_pass_integration(self):
        """Test full forward pass through created model returns GraphData."""
        batch_size = 2
        num_nodes = 10

        gnn_config = {
            "num_layers": 1,
            "feature_dim_in": num_nodes,
            "feature_dim_out": 5,
        }

        transformer_config = {"num_heads": 2, "num_layers": 1}

        model = create_sequential_model(gnn_config, transformer_config)

        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
        data = edge_scalar_graphdata(A)

        result = model(data, output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == (
            batch_size,
            num_nodes,
            num_nodes,
        )


if __name__ == "__main__":
    pytest.main([__file__])

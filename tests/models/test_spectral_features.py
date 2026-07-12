"""Tests for new spectral model features.

Tests cover:
- SpectralProjectionLayer for eigenvalue-polynomial filtering
- PEARLEmbedding for GNN-based positional encodings
- Asymmetric reconstruction in LinearPE and GraphFilterBank
- PEARL embedding integration with spectral denoisers
"""

import pytest
import torch

from tests._helpers.graph_builders import edge_scalar_graphdata, legacy_edge_scalar
from tmgg.data.datasets.graph_types import DenseGraphDistribution
from tmgg.models.layers import PEARLEmbedding, SpectralProjectionLayer
from tmgg.models.spectral_denoisers import (
    GraphFilterBank,
    LinearPE,
    SelfAttentionDenoiser,
)


class TestSpectralProjectionLayer:
    """Tests for SpectralProjectionLayer."""

    def test_init(self):
        """Test initialization with default parameters."""
        layer = SpectralProjectionLayer(k=16, out_dim=64, num_terms=3)

        assert layer.k == 16
        assert layer.out_dim == 64
        assert layer.num_terms == 3
        assert len(layer.H) == 3

    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        k, out_dim = 16, 64
        batch_size, n = 4, 50

        layer = SpectralProjectionLayer(k=k, out_dim=out_dim, num_terms=3)
        V = torch.randn(batch_size, n, k)
        Lambda = torch.randn(batch_size, k)

        output = layer(V, Lambda)

        assert output.shape == (batch_size, n, out_dim)

    def test_different_num_terms(self):
        """Test layer works with different polynomial degrees."""
        for num_terms in [1, 2, 5, 10]:
            layer = SpectralProjectionLayer(k=8, out_dim=32, num_terms=num_terms)
            V = torch.randn(2, 20, 8)
            Lambda = torch.randn(2, 8)

            output = layer(V, Lambda)

            assert output.shape == (2, 20, 32)
            assert len(layer.H) == num_terms

    def test_numerical_stability(self):
        """Test layer handles extreme eigenvalues without NaN/Inf."""
        layer = SpectralProjectionLayer(k=8, out_dim=32, num_terms=3)
        V = torch.randn(2, 20, 8)

        # Test with very large eigenvalues
        Lambda_large = torch.randn(2, 8) * 1000
        output_large = layer(V, Lambda_large)
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()

        # Test with very small eigenvalues
        Lambda_small = torch.randn(2, 8) * 1e-6
        output_small = layer(V, Lambda_small)
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()


class TestPEARLEmbedding:
    """Tests for PEARLEmbedding layer."""

    def test_init_random_mode(self):
        """Test R-PEARL initialization."""
        layer = PEARLEmbedding(output_dim=16, num_layers=3, mode="random")

        assert layer.output_dim == 16
        assert layer.num_layers == 3
        assert layer.mode == "random"

    def test_init_basis_mode(self):
        """Test B-PEARL initialization."""
        layer = PEARLEmbedding(output_dim=16, num_layers=3, mode="basis")

        assert layer.mode == "basis"

    def test_init_invalid_mode(self):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError, match="mode must be 'random' or 'basis'"):
            PEARLEmbedding(output_dim=16, mode="invalid")

    def test_forward_shape_batched(self):
        """Test forward pass with batched input."""
        layer = PEARLEmbedding(output_dim=16, num_layers=3, mode="random")
        A = torch.randn(4, 50, 50)
        A = (A + A.transpose(-1, -2)) / 2  # Symmetrize

        output = layer(A)

        assert output.shape == (4, 50, 16)

    def test_forward_shape_unbatched(self):
        """Test forward pass with single graph."""
        layer = PEARLEmbedding(output_dim=16, num_layers=3, mode="random")
        A = torch.randn(50, 50)
        A = (A + A.T) / 2

        output = layer(A)

        assert output.shape == (50, 16)

    def test_random_mode_different_in_training(self):
        """Test R-PEARL uses different random features in training."""
        layer = PEARLEmbedding(output_dim=16, num_layers=3, mode="random")
        layer.train()

        A = torch.randn(2, 20, 20)
        A = (A + A.transpose(-1, -2)) / 2

        # Multiple forward passes should give different results in training
        out1 = layer(A)
        out2 = layer(A)

        # With random initialization, outputs should differ
        assert not torch.allclose(out1, out2)

    def test_eval_mode_deterministic(self):
        """Test layer is deterministic in eval mode."""
        layer = PEARLEmbedding(output_dim=16, num_layers=3, mode="random")
        layer.eval()

        A = torch.randn(2, 20, 20)
        A = (A + A.transpose(-1, -2)) / 2

        out1 = layer(A)
        out2 = layer(A)

        assert torch.allclose(out1, out2)

    def test_basis_mode_shape(self):
        """Test B-PEARL output shape."""
        layer = PEARLEmbedding(output_dim=16, num_layers=3, mode="basis", max_nodes=100)
        A = torch.randn(2, 50, 50)
        A = (A + A.transpose(-1, -2)) / 2

        output = layer(A)

        assert output.shape == (2, 50, 16)


class TestAsymmetricLinearPE:
    """Tests for asymmetric reconstruction in LinearPE."""

    def test_symmetric_mode(self):
        """Test default symmetric mode still works."""
        model = LinearPE(k=8, asymmetric=False)

        # Should have single W matrix
        assert hasattr(model, "W")
        assert model.W is not None
        assert not hasattr(model, "W_X") or model.W_X is None
        assert not hasattr(model, "W_Y") or model.W_Y is None

    def test_asymmetric_mode(self):
        """Test asymmetric mode has separate matrices."""
        model = LinearPE(k=8, asymmetric=True)

        # Should have separate W_X and W_Y matrices
        assert model.W_X is not None
        assert model.W_Y is not None
        assert model.W_X.shape == (8, 8)
        assert model.W_Y.shape == (8, 8)

    def test_asymmetric_forward_shape(self):
        """Test asymmetric mode produces correct output shape."""
        model = LinearPE(k=8, asymmetric=True)
        A = torch.randn(2, 20, 20)
        A = (A + A.transpose(-1, -2)) / 2

        result = model(edge_scalar_graphdata(A), output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == (2, 20, 20)

    def test_asymmetric_output_differs_from_symmetric(self):
        """Test asymmetric produces different output than symmetric."""
        torch.manual_seed(42)
        model_sym = LinearPE(k=8, asymmetric=False)

        torch.manual_seed(42)
        model_asym = LinearPE(k=8, asymmetric=True)

        A = torch.randn(2, 20, 20)
        A = (A + A.transpose(-1, -2)) / 2
        data = edge_scalar_graphdata(A)

        out_sym = model_sym(data, output_dense=True)
        out_asym = model_asym(data, output_dense=True)
        assert isinstance(out_sym, DenseGraphDistribution)
        assert isinstance(out_asym, DenseGraphDistribution)

        # Different parameterizations should give different raw edge features
        out_sym_e = out_sym.E_feat if out_sym.E_feat is not None else out_sym.E_class
        out_asym_e = (
            out_asym.E_feat if out_asym.E_feat is not None else out_asym.E_class
        )
        assert out_sym_e is not None and out_asym_e is not None
        assert not torch.allclose(out_sym_e, out_asym_e)


class TestAsymmetricGraphFilterBank:
    """Tests for asymmetric reconstruction in GraphFilterBank."""

    def test_symmetric_mode(self):
        """Test default symmetric mode still works."""
        model = GraphFilterBank(k=8, polynomial_degree=5, asymmetric=False)

        assert hasattr(model, "H")
        assert model.H is not None
        assert len(model.H) == 5

    def test_asymmetric_mode(self):
        """Test asymmetric mode has separate coefficient lists."""
        model = GraphFilterBank(k=8, polynomial_degree=5, asymmetric=True)

        assert model.H_X is not None
        assert model.H_Y is not None
        assert len(model.H_X) == 5
        assert len(model.H_Y) == 5

    def test_asymmetric_forward_shape(self):
        """Test asymmetric mode produces correct output shape."""
        model = GraphFilterBank(k=8, polynomial_degree=5, asymmetric=True)
        A = torch.randn(2, 20, 20)
        A = (A + A.transpose(-1, -2)) / 2

        result = model(edge_scalar_graphdata(A), output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == (2, 20, 20)


class TestPEARLIntegration:
    """Tests for PEARL embedding integration with spectral denoisers."""

    def test_linear_pe_with_pearl(self):
        """Test LinearPE with PEARL embeddings."""
        model = LinearPE(k=8, embedding_source="pearl_random", pearl_num_layers=2)

        A = torch.randn(2, 20, 20)
        A = (A + A.transpose(-1, -2)) / 2

        result = model(edge_scalar_graphdata(A), output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == (2, 20, 20)
        assert model.embedding_source == "pearl_random"

    def test_filter_bank_with_pearl(self):
        """Test GraphFilterBank with PEARL embeddings."""
        model = GraphFilterBank(
            k=8,
            polynomial_degree=3,
            embedding_source="pearl_random",
            pearl_num_layers=2,
        )

        A = torch.randn(2, 20, 20)
        A = (A + A.transpose(-1, -2)) / 2

        result = model(edge_scalar_graphdata(A), output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == (2, 20, 20)

    def test_self_attention_with_pearl(self):
        """Test SelfAttentionDenoiser with PEARL embeddings."""
        model = SelfAttentionDenoiser(
            k=8, d_k=16, embedding_source="pearl_random", pearl_num_layers=2
        )

        A = torch.randn(2, 20, 20)
        A = (A + A.transpose(-1, -2)) / 2

        result = model(edge_scalar_graphdata(A), output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == (2, 20, 20)

    def test_pearl_basis_mode_integration(self):
        """Test B-PEARL integration."""
        model = LinearPE(
            k=8, embedding_source="pearl_basis", pearl_num_layers=2, pearl_max_nodes=50
        )

        A = torch.randn(2, 30, 30)
        A = (A + A.transpose(-1, -2)) / 2

        result = model(edge_scalar_graphdata(A), output_dense=True)
        assert isinstance(result, DenseGraphDistribution)
        assert legacy_edge_scalar(result.argmax()).shape == (2, 30, 30)

    def test_invalid_embedding_source(self):
        """Test invalid embedding source raises error."""
        with pytest.raises(ValueError, match="embedding_source must be"):
            LinearPE(k=8, embedding_source="invalid")  # type: ignore[arg-type]


if __name__ == "__main__":
    pytest.main([__file__])

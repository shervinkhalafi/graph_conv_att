"""Forward pass sanity tests for spectral denoising models.

These tests verify that models produce non-trivial outputs and that gradients
flow to parameters. A model that outputs uniform edge probabilities indicates
training failure.

Test Rationale
--------------
- test_model_produces_nonzero_logits: Verifies untrained model doesn't start
  at a degenerate point where edge logits are exactly zero.
- test_gradients_flow_to_parameters: Confirms loss gradients actually reach
  model parameters through the forward pass.
- test_baseline_produces_nonzero_logits: Same check for baseline models.
"""

import pytest
import torch

from tests._helpers.graph_builders import edge_scalar_graphdata, legacy_edge_scalar
from tmgg.models.baselines import LinearBaseline, MLPBaseline
from tmgg.models.spectral_denoisers import (
    GraphFilterBank,
    LinearPE,
    SelfAttentionDenoiser,
)


class TestSpectralModelsForwardSanity:
    """Verify spectral denoising models produce valid outputs."""

    @pytest.fixture
    def identity_data(self):
        """Create identity adjacency matrix batch as GraphData."""
        return edge_scalar_graphdata(torch.eye(32).unsqueeze(0))

    @pytest.fixture
    def random_data(self):
        """Create random symmetric adjacency matrix batch as GraphData."""
        A = torch.rand(4, 32, 32)
        A = (A + A.transpose(-2, -1)) / 2
        return edge_scalar_graphdata(A)

    def test_linear_pe_produces_nonzero_logits(self, identity_data):
        """Verify LinearPE doesn't output uniform edge probabilities."""
        model = LinearPE(k=8, max_nodes=32)
        result = model(identity_data, output_dense=True)

        # Edge-channel logits should not be exactly zero
        edge_logits = legacy_edge_scalar(result)
        assert edge_logits.abs().mean() > 1e-6, (
            f"LinearPE outputs near-zero edge logits: mean={edge_logits.mean():.2e}, "
            f"std={edge_logits.std():.2e}"
        )

    def test_filter_bank_produces_nonzero_logits(self, identity_data):
        """Verify GraphFilterBank doesn't output uniform edge probabilities."""
        model = GraphFilterBank(k=8, polynomial_degree=5)
        result = model(identity_data, output_dense=True)

        edge_logits = legacy_edge_scalar(result)
        assert (
            edge_logits.abs().mean() > 1e-6
        ), f"GraphFilterBank outputs near-zero edge logits: mean={edge_logits.mean():.2e}"

    def test_self_attention_produces_nonzero_logits(self, identity_data):
        """Verify SelfAttentionDenoiser doesn't output uniform edge probabilities."""
        model = SelfAttentionDenoiser(k=8, d_k=64)
        result = model(identity_data, output_dense=True)

        edge_logits = legacy_edge_scalar(result)
        assert (
            edge_logits.abs().mean() > 1e-6
        ), f"SelfAttentionDenoiser outputs near-zero edge logits: mean={edge_logits.mean():.2e}"

    @pytest.mark.parametrize(
        "model_class,kwargs",
        [
            (LinearPE, {"k": 8, "max_nodes": 32}),
            (GraphFilterBank, {"k": 8, "polynomial_degree": 5}),
            (SelfAttentionDenoiser, {"k": 8, "d_k": 64}),
        ],
    )
    def test_gradients_flow_to_parameters(self, model_class, kwargs, identity_data):
        """Verify loss gradients reach model parameters."""
        model = model_class(**kwargs)

        result = model(identity_data, output_dense=True)
        # Use only edge-probability channel: both channels of 2-class encoding
        # sum to 1.0 per position, so E.sum() is constant with zero gradient.
        loss = legacy_edge_scalar(result).sum()
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = False
        max_grad = 0.0
        for _, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.abs().max().item()
                max_grad = max(max_grad, grad_norm)
                if grad_norm > 1e-10:
                    has_grad = True

        assert has_grad, (
            f"{model_class.__name__} has no meaningful gradients. "
            f"Max gradient: {max_grad:.2e}"
        )


class TestBaselineModelsForwardSanity:
    """Verify baseline models produce valid outputs."""

    @pytest.fixture
    def random_data(self):
        """Create random symmetric adjacency matrix batch as GraphData."""
        A = torch.rand(4, 32, 32)
        A = (A + A.transpose(-2, -1)) / 2
        return edge_scalar_graphdata(A)

    def test_linear_baseline_produces_nonzero_logits(self, random_data):
        """Verify LinearBaseline doesn't output uniform edge probabilities.

        LinearBaseline starts at identity, so initial output ~ input.
        With random input, output should be non-trivial.
        """
        model = LinearBaseline(max_nodes=32)
        result = model(random_data, output_dense=True)

        edge_logits = legacy_edge_scalar(result)
        assert (
            edge_logits.abs().mean() > 0.1
        ), f"LinearBaseline outputs near-zero edge logits: mean={edge_logits.mean():.2e}"

    def test_mlp_baseline_produces_nonzero_logits(self, random_data):
        """Verify MLPBaseline doesn't output uniform edge probabilities."""
        model = MLPBaseline(max_nodes=32, hidden_dim=256)
        result = model(random_data, output_dense=True)

        edge_logits = legacy_edge_scalar(result)
        assert (
            edge_logits.abs().mean() > 1e-6
        ), f"MLPBaseline outputs near-zero edge logits: mean={edge_logits.mean():.2e}"

    @pytest.mark.parametrize(
        "model_class,kwargs",
        [
            (LinearBaseline, {"max_nodes": 32}),
            (MLPBaseline, {"max_nodes": 32, "hidden_dim": 256}),
        ],
    )
    def test_baseline_gradients_flow(self, model_class, kwargs, random_data):
        """Verify baseline model gradients flow properly."""
        model = model_class(**kwargs)

        result = model(random_data, output_dense=True)
        # Use only edge-probability channel to avoid constant-sum gradient trap.
        loss = legacy_edge_scalar(result).sum()
        loss.backward()

        has_grad = False
        for _, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().max() > 1e-10:
                has_grad = True
                break

        assert has_grad, f"{model_class.__name__} has no meaningful gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

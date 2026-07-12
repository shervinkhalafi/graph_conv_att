"""Tests for DiGress extra features (P1.4).

Testing strategy
----------------
Verifies that each feature mode produces tensors with correct shapes, finite
values, and proper masking. Uses small random one-hot graphs (bs=4, n=12,
dx=de=2) so eigendecompositions stay fast and numerically stable.

Key invariants:
- DummyExtraFeatures returns zero-width tensors
- ExtraFeatures("cycles", ...) adds (3, 0, 5) dimensions
- ExtraFeatures("eigenvalues", ...) adds (3, 0, 11) dimensions
- ExtraFeatures("all", ...) adds (6, 0, 11) dimensions
- Cycle counts are non-negative after scaling
- Eigenfeatures are finite (no NaN/Inf)
- Masked nodes have zero extra_X values
- Invalid mode raises ValueError
- extra_features_dims returns correct widths for all modes
"""

from __future__ import annotations

import pytest
import torch

from tmgg.models.digress.extra_features import (
    DummyExtraFeatures,
    ExtraFeatures,
    extra_features_dims,
)


@pytest.fixture()
def graph_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random one-hot graph batch: bs=4, n=12, dx=2, de=2.

    Constructs symmetric one-hot edge features with some masked nodes
    (last 2 nodes in each graph are masked).
    """
    bs, n, dx, de = 4, 12, 2, 2
    torch.manual_seed(42)

    # Random one-hot node features
    X = torch.nn.functional.one_hot(torch.randint(0, dx, (bs, n)), dx).float()

    # Random symmetric one-hot edge features
    edge_classes = torch.randint(0, de, (bs, n, n))
    # Symmetrise
    upper = torch.triu(edge_classes, diagonal=1)
    edge_classes = upper + upper.transpose(1, 2)
    # Zero diagonal
    diag_mask = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)
    edge_classes[diag_mask] = 0
    E = torch.nn.functional.one_hot(edge_classes, de).float()

    y = torch.randn(bs, 1)

    # Mask: last 2 nodes invalid
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    node_mask[:, -2:] = False

    return X, E, y, node_mask


class TestDummyExtraFeatures:
    def test_returns_zero_width(
        self,
        graph_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """DummyExtraFeatures should return tensors with zero-width last dim."""
        X, E, y, node_mask = graph_batch
        ef = DummyExtraFeatures()
        extra_X, extra_E, extra_y = ef(X, E, y, node_mask)

        assert extra_X.shape == (4, 12, 0)
        assert extra_E.shape == (4, 12, 12, 0)
        assert extra_y.shape == (4, 0)


class TestExtraFeaturesDims:
    def test_cycles(self) -> None:
        assert extra_features_dims("cycles") == (3, 0, 5)

    def test_eigenvalues(self) -> None:
        assert extra_features_dims("eigenvalues") == (3, 0, 11)

    def test_all(self) -> None:
        assert extra_features_dims("all") == (6, 0, 11)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown features_type"):
            extra_features_dims("invalid_mode")


class TestCyclesMode:
    def test_shapes(
        self,
        graph_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Cycles mode should produce (bs, n, 3), (bs, n, n, 0), (bs, 5)."""
        X, E, y, node_mask = graph_batch
        ef = ExtraFeatures("cycles", max_n_nodes=12)
        extra_X, extra_E, extra_y = ef(X, E, y, node_mask)

        assert extra_X.shape == (4, 12, 3), f"extra_X shape: {extra_X.shape}"
        assert extra_E.shape == (4, 12, 12, 0), f"extra_E shape: {extra_E.shape}"
        assert extra_y.shape == (4, 5), f"extra_y shape: {extra_y.shape}"

    def test_cycle_counts_nonnegative(
        self,
        graph_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Scaled cycle counts should be non-negative (>= 0)."""
        X, E, y, node_mask = graph_batch
        ef = ExtraFeatures("cycles", max_n_nodes=12)
        extra_X, _extra_E, extra_y = ef(X, E, y, node_mask)

        # x_cycles are scaled by /10 and clipped to [0, 1]
        assert (extra_X >= 0).all(), f"Negative x_cycles: min={extra_X.min()}"
        assert (extra_y >= 0).all(), f"Negative y_cycles: min={extra_y.min()}"

    def test_masked_nodes_zeroed(
        self,
        graph_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Cycle counts at masked positions should be zero."""
        X, E, y, node_mask = graph_batch
        ef = ExtraFeatures("cycles", max_n_nodes=12)
        extra_X, _extra_E, _extra_y = ef(X, E, y, node_mask)

        # Last 2 nodes are masked
        assert (extra_X[:, -2:, :] == 0).all(), "Masked positions should be zero"


class TestEigenvaluesMode:
    def test_shapes(
        self,
        graph_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Eigenvalues mode: (bs, n, 3), (bs, n, n, 0), (bs, 11)."""
        X, E, y, node_mask = graph_batch
        ef = ExtraFeatures("eigenvalues", max_n_nodes=12)
        extra_X, extra_E, extra_y = ef(X, E, y, node_mask)

        assert extra_X.shape == (4, 12, 3), f"extra_X shape: {extra_X.shape}"
        assert extra_E.shape == (4, 12, 12, 0), f"extra_E shape: {extra_E.shape}"
        assert extra_y.shape == (4, 11), f"extra_y shape: {extra_y.shape}"

    def test_eigenfeatures_finite(
        self,
        graph_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Eigenvalue features should have no NaN or Inf values."""
        X, E, y, node_mask = graph_batch
        ef = ExtraFeatures("eigenvalues", max_n_nodes=12)
        extra_X, _extra_E, extra_y = ef(X, E, y, node_mask)

        assert torch.isfinite(extra_X).all(), "extra_X contains NaN/Inf"
        assert torch.isfinite(extra_y).all(), "extra_y contains NaN/Inf"


class TestAllMode:
    def test_shapes(
        self,
        graph_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """All mode: (bs, n, 6), (bs, n, n, 0), (bs, 11)."""
        X, E, y, node_mask = graph_batch
        ef = ExtraFeatures("all", max_n_nodes=12)
        extra_X, extra_E, extra_y = ef(X, E, y, node_mask)

        assert extra_X.shape == (4, 12, 6), f"extra_X shape: {extra_X.shape}"
        assert extra_E.shape == (4, 12, 12, 0), f"extra_E shape: {extra_E.shape}"
        assert extra_y.shape == (4, 11), f"extra_y shape: {extra_y.shape}"

    def test_eigenfeatures_finite(
        self,
        graph_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """All eigenfeatures should be finite."""
        X, E, y, node_mask = graph_batch
        ef = ExtraFeatures("all", max_n_nodes=12)
        extra_X, _extra_E, extra_y = ef(X, E, y, node_mask)

        assert torch.isfinite(extra_X).all(), "extra_X contains NaN/Inf"
        assert torch.isfinite(extra_y).all(), "extra_y contains NaN/Inf"

    def test_masked_nodes_zeroed(
        self,
        graph_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Extra_X at masked positions should be zero (cycle + eigenvector parts)."""
        X, E, y, node_mask = graph_batch
        ef = ExtraFeatures("all", max_n_nodes=12)
        extra_X, _extra_E, _extra_y = ef(X, E, y, node_mask)

        # Last 2 nodes are masked
        assert (extra_X[:, -2:, :] == 0).all(), "Masked positions should be zero"


class TestInvalidMode:
    def test_constructor_raises(self) -> None:
        """ExtraFeatures with invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown extra_features_type"):
            ExtraFeatures("nonexistent_mode", max_n_nodes=10)

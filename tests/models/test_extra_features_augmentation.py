"""Tests for feature augmentation classes and their adjust_dims interface.

Test rationale: the adjust_dims method encapsulates dimension arithmetic that
GraphTransformer delegates to each augmentation. If dims are wrong, the
GraphTransformer's mlp_in layers will silently accept mismatched widths and
produce garbage. These tests verify both the dim computation and the runtime
__call__ output shapes.
"""

import pytest
import torch
from torch.nn import functional as F

from tmgg.models.digress.extra_features import (
    DummyExtraFeatures,
    EigenvectorAugmentation,
    ExtraFeatures,
)

BS = 3
N = 8
DX = 2
DE = 2
DY = 0


@pytest.fixture()
def node_mask() -> torch.Tensor:
    return torch.ones(BS, N, dtype=torch.bool)


@pytest.fixture()
def categorical_input(node_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """One-hot categorical X and E tensors with symmetric edges.

    The cycle counter relies on symmetric adjacency, so edge classes are
    symmetrised (upper triangle mirrored) with zero diagonal.
    """
    torch.manual_seed(0)
    X = F.one_hot(torch.randint(0, DX, (BS, N)), DX).float()
    edge_classes = torch.randint(0, DE, (BS, N, N))
    upper = torch.triu(edge_classes, diagonal=1)
    edge_classes = upper + upper.transpose(1, 2)
    diag_mask = torch.eye(N, dtype=torch.bool).unsqueeze(0).expand(BS, -1, -1)
    edge_classes[diag_mask] = 0
    E = F.one_hot(edge_classes, DE).float()
    y = torch.zeros(BS, DY)
    return X, E, y, node_mask


class TestDummyExtraFeatures:
    def test_adjust_dims_identity(self) -> None:
        aug = DummyExtraFeatures()
        base = {"X": 5, "E": 3, "y": 1}
        assert aug.adjust_dims(base) == {"X": 5, "E": 3, "y": 1}

    def test_call_shapes(self, categorical_input: tuple[torch.Tensor, ...]) -> None:
        aug = DummyExtraFeatures()
        X, E, y, mask = categorical_input
        ex_X, ex_E, ex_y = aug(X, E, y, mask)
        assert ex_X.shape == (BS, N, 0)
        assert ex_E.shape == (BS, N, N, 0)
        assert ex_y.shape == (BS, 0)


class TestExtraFeatures:
    @pytest.mark.parametrize(
        "features_type,expected_extra",
        [("cycles", (3, 0, 5)), ("eigenvalues", (3, 0, 11)), ("all", (6, 0, 11))],
    )
    def test_adjust_dims(
        self, features_type: str, expected_extra: tuple[int, int, int]
    ) -> None:
        aug = ExtraFeatures(features_type, max_n_nodes=N)
        base = {"X": 2, "E": 2, "y": 0}
        result = aug.adjust_dims(base)
        dx, de, dy = expected_extra
        assert result == {"X": 2 + dx, "E": 2 + de, "y": 0 + dy}

    def test_call_shapes_cycles(
        self, categorical_input: tuple[torch.Tensor, ...]
    ) -> None:
        aug = ExtraFeatures("cycles", max_n_nodes=N)
        X, E, y, mask = categorical_input
        ex_X, ex_E, ex_y = aug(X, E, y, mask)
        assert ex_X.shape[0] == BS
        assert ex_X.shape[1] == N
        assert ex_X.shape[2] == 3
        assert ex_E.shape[-1] == 0
        assert ex_y.shape[0] == BS
        assert ex_y.shape[1] == 5


class TestEigenvectorAugmentation:
    K = 5

    def test_adjust_dims(self) -> None:
        aug = EigenvectorAugmentation(k=self.K)
        base = {"X": 2, "E": 2, "y": 0}
        result = aug.adjust_dims(base)
        assert result == {"X": 2 + self.K, "E": 2, "y": 0}

    def test_adjust_dims_preserves_other_keys(self) -> None:
        aug = EigenvectorAugmentation(k=self.K)
        base = {"X": 2, "E": 4, "y": 3}
        result = aug.adjust_dims(base)
        assert result["E"] == 4
        assert result["y"] == 3

    def test_call_shapes(self, categorical_input: tuple[torch.Tensor, ...]) -> None:
        aug = EigenvectorAugmentation(k=self.K)
        X, E, y, mask = categorical_input
        ex_X, ex_E, ex_y = aug(X, E, y, mask)
        assert ex_X.shape == (BS, N, self.K)
        assert ex_E.shape == (BS, N, N, 0)
        assert ex_y.shape == (BS, 0)

    def test_output_finite(self, categorical_input: tuple[torch.Tensor, ...]) -> None:
        aug = EigenvectorAugmentation(k=self.K)
        X, E, y, mask = categorical_input
        ex_X, _, _ = aug(X, E, y, mask)
        assert torch.isfinite(ex_X).all()

    def test_small_graph_pads(self) -> None:
        k = 20
        aug = EigenvectorAugmentation(k=k)
        X = torch.randn(1, 4, DX)
        E = F.one_hot(torch.randint(0, DE, (1, 4, 4)), DE).float()
        y = torch.zeros(1, 0)
        mask = torch.ones(1, 4, dtype=torch.bool)
        ex_X, _, _ = aug(X, E, y, mask)
        assert ex_X.shape == (1, 4, k)
        assert torch.isfinite(ex_X).all()

    def test_single_channel_edge_state_matches_categorical_topology(self) -> None:
        """Equivalent scalar and categorical edges should give the same features.

        Regression rationale
        --------------------
        Single-step denoising lifts graphs through edge_scalar_graphdata(),
        which stores edges as a single scalar channel. DiGress denoising still
        uses EigenvectorAugmentation, so the augmenter must derive the same
        adjacency from both ``E[..., 1]`` binary-topology tensors and
        ``E[..., 0]`` scalar edge-state tensors.
        """
        aug = EigenvectorAugmentation(k=1)
        X = torch.zeros(1, 5, DX)
        y = torch.zeros(1, 0)
        mask = torch.ones(1, 5, dtype=torch.bool)

        adjacency = torch.tensor(
            [
                [
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 0.0],
                ]
            ]
        )
        categorical_edges = torch.stack([1.0 - adjacency, adjacency], dim=-1)
        scalar_edges = adjacency.unsqueeze(-1)

        categorical_X, _, _ = aug(X, categorical_edges, y, mask)
        scalar_X, _, _ = aug(X, scalar_edges, y, mask)

        assert torch.allclose(categorical_X.abs(), scalar_X.abs(), atol=1e-5)

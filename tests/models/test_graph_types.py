"""Tests for the typed graph data containers in graph_types.py.

Testing Strategy
----------------
These tests verify the frozen dataclasses (GraphState, GraphDistribution,
DenseGraphState, DenseGraphDistribution) replacing the legacy
PlaceHolder, focusing on immutability, masking semantics, symmetry
assertions, and class-index collapse helpers.
"""

from __future__ import annotations

import pytest
import torch

from tests._helpers.graph_builders import binary_graphdata, legacy_edge_scalar
from tmgg.data.datasets.graph_types import DenseGraphState, collapse_to_indices


def _make_symmetric_E(bs: int, n: int, de: int) -> torch.Tensor:
    """Build a symmetric edge tensor for tests."""
    E_upper = torch.randn(bs, n, n, de)
    return (E_upper + E_upper.transpose(1, 2)) / 2


def _state(
    *,
    bs: int,
    n: int,
    X: torch.Tensor,
    E: torch.Tensor,
    num_nodes_per_graph: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
) -> DenseGraphState:
    """Helper: build DenseGraphState with default num_nodes_per_graph."""
    if num_nodes_per_graph is None:
        num_nodes_per_graph = torch.full((bs,), n, dtype=torch.long)
    if y is None:
        y = torch.zeros(bs, 0)
    return DenseGraphState(
        num_nodes_per_graph=num_nodes_per_graph,
        y=y,
        X_class=X,
        E_class=E,
    )


# -- DenseGraphState.mask() -------------------------------------------------


class TestDenseGraphStateMask:
    """Verify that mask() zeros masked positions and asserts E symmetry.

    Ported from the original PlaceHolder.mask() tests. Key difference:
    frozen dataclasses return new instances rather than mutating in place.
    """

    def test_symmetric_edges_pass(self) -> None:
        """Symmetric E after masking raises no error."""
        bs, n, dx, de = 2, 4, 3, 2
        X = torch.randn(bs, n, dx)
        E = _make_symmetric_E(bs, n, de)

        data = _state(bs=bs, n=n, X=X, E=E)
        result = data.mask()
        assert result is not data  # frozen — returns new instance

    def test_symmetric_edges_with_partial_mask(self) -> None:
        """Masking some nodes preserves symmetry when E starts symmetric."""
        bs, n, dx, de = 2, 5, 3, 2
        X = torch.randn(bs, n, dx)
        E = _make_symmetric_E(bs, n, de)
        # Drop the trailing position.
        num_nodes = torch.full((bs,), n - 1, dtype=torch.long)

        data = _state(bs=bs, n=n, X=X, E=E, num_nodes_per_graph=num_nodes)
        result = data.mask()
        assert result is not data

    @pytest.mark.skipif(
        not __debug__,
        reason=(
            "Symmetry assert is __debug__-guarded so production runs "
            "(PYTHONOPTIMIZE=1) skip the per-step bool(allclose) sync; "
            "see internal notes (not included)"
        ),
    )
    def test_asymmetric_edges_raise(self) -> None:
        """Asymmetric E triggers AssertionError after masking."""
        bs, n, dx, de = 1, 4, 3, 2
        X = torch.randn(bs, n, dx)
        E = torch.randn(bs, n, n, de)
        E[0, 0, 1, 0] = 10.0
        E[0, 1, 0, 0] = -10.0

        data = _state(bs=bs, n=n, X=X, E=E)
        with pytest.raises(AssertionError, match="symmetric"):
            data.mask()

    def test_masked_positions_are_zero(self) -> None:
        """Node features and edges involving masked nodes are zeroed."""
        bs, n, dx, de = 1, 3, 2, 2
        X = torch.ones(bs, n, dx)
        E = _make_symmetric_E(bs, n, de).abs() + 0.1
        # Make E symmetric and positive
        E = (E + E.transpose(1, 2)) / 2
        # Drop node 2 (n_real = 2).
        num_nodes = torch.tensor([2], dtype=torch.long)

        result = _state(
            bs=bs, n=n, X=X, E=E, num_nodes_per_graph=num_nodes
        ).mask()
        assert result.X_class is not None
        assert result.E_class is not None
        assert result.X_class[0, 2].abs().sum() == 0
        assert result.E_class[0, 2, :, :].abs().sum() == 0
        assert result.E_class[0, :, 2, :].abs().sum() == 0


# -- DenseGraphState.mask_zero_diag() ---------------------------------------


class TestDenseGraphStateMaskZeroDiag:
    """Verify mask_zero_diag zeros both masked positions and the diagonal."""

    def test_diagonal_zeroed(self) -> None:
        """Edge diagonal E[b, i, i, :] is zero after mask_zero_diag."""
        bs, n, dx, de = 2, 5, 3, 2
        X = torch.randn(bs, n, dx)
        E = torch.randn(bs, n, n, de)

        result = _state(bs=bs, n=n, X=X, E=E).mask_zero_diag()
        assert result.E_class is not None
        for b in range(bs):
            for i in range(n):
                assert result.E_class[b, i, i].abs().sum() == 0

    def test_masked_positions_zeroed(self) -> None:
        """Masked node features are zeroed."""
        bs, n, dx, de = 1, 4, 2, 2
        X = torch.ones(bs, n, dx)
        E = torch.ones(bs, n, n, de)
        # Drop node 3.
        num_nodes = torch.tensor([3], dtype=torch.long)

        result = _state(
            bs=bs, n=n, X=X, E=E, num_nodes_per_graph=num_nodes
        ).mask_zero_diag()
        assert result.X_class is not None
        assert result.X_class[0, 3].abs().sum() == 0


# -- DenseGraphState.type_as() ----------------------------------------------


class TestDenseGraphStateTypeAs:
    """Verify type_as returns a new instance with correct dtype."""

    def test_dtype_changes(self) -> None:
        X = torch.randn(1, 3, 2)
        E = torch.randn(1, 3, 3, 2)
        data = _state(bs=1, n=3, X=X, E=E)
        target = torch.zeros(1, dtype=torch.float64)
        result = data.type_as(target)
        assert result is not data
        assert result.X_class is not None
        assert result.E_class is not None
        assert result.X_class.dtype == torch.float64
        assert result.E_class.dtype == torch.float64
        assert result.y.dtype == torch.float64


# -- DenseGraphState immutability -------------------------------------------


class TestDenseGraphStateFrozen:
    """Frozen dataclass prevents field assignment."""

    def test_cannot_assign_X(self) -> None:
        X = torch.randn(1, 3, 2)
        E = torch.randn(1, 3, 3, 2)
        data = _state(bs=1, n=3, X=X, E=E)
        with pytest.raises(AttributeError):
            data.X_class = torch.zeros(1, 3, 2)  # type: ignore[misc]


# -- collapse_to_indices() --------------------------------------------------


class TestCollapseToIndices:
    """Verify argmax collapse with -1 sentinel for masked positions."""

    def test_basic_collapse(self) -> None:
        """One-hot X and E are collapsed to integer class indices."""
        bs, n, dx, de = 1, 3, 4, 2
        X = torch.zeros(bs, n, dx)
        X[0, 0, 2] = 1.0  # node 0 -> class 2
        X[0, 1, 0] = 1.0  # node 1 -> class 0
        X[0, 2, 3] = 1.0  # node 2 -> class 3

        E = torch.zeros(bs, n, n, de)
        E[0, 0, 1, 1] = 1.0  # edge (0,1) -> class 1
        E[0, 1, 0, 1] = 1.0  # symmetric

        E_idx, X_idx = collapse_to_indices(_state(bs=bs, n=n, X=X, E=E))
        assert X_idx is not None
        assert X_idx[0, 0] == 2
        assert X_idx[0, 1] == 0
        assert X_idx[0, 2] == 3
        assert E_idx[0, 0, 1] == 1

    def test_masked_positions_are_negative_one(self) -> None:
        """Masked nodes and edges get sentinel value -1."""
        bs, n, dx, de = 1, 3, 2, 2
        X = torch.zeros(bs, n, dx)
        X[..., 0] = 1.0
        E = torch.zeros(bs, n, n, de)
        E[..., 0] = 1.0
        # Drop node 2.
        num_nodes = torch.tensor([2], dtype=torch.long)

        E_idx, X_idx = collapse_to_indices(
            _state(bs=bs, n=n, X=X, E=E, num_nodes_per_graph=num_nodes)
        )
        assert X_idx is not None
        assert X_idx[0, 2] == -1
        assert E_idx[0, 2, 0] == -1
        assert E_idx[0, 0, 2] == -1


# -- Explicit graph boundaries ----------------------------------------------


class TestExplicitGraphBoundaries:
    """Explicit topology and edge-state accessors keep their semantics separate."""

    def test_binary_adjacency_round_trip(self) -> None:
        """Binary graph boundaries round-trip through the topology accessors."""
        adj = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
        data = binary_graphdata(adj)
        torch.testing.assert_close(data.dense_adjacency(), adj)

    def test_binary_topology_lifts_into_edge_state_space(self) -> None:
        """Binary-topology graphs expose their edge indicator channel as edge state."""
        adj = torch.tensor([[[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        data = binary_graphdata(adj)
        torch.testing.assert_close(legacy_edge_scalar(data), adj)

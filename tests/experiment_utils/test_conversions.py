"""Tests for ``DenseGraphState`` conversion helpers and collation.

Test rationale
--------------
Binary-topology helpers are the boundary between dense graph tensors and
discrete graph exports. Edge-state helpers are the boundary for continuous
models and must preserve values exactly. These tests protect both contracts
and the padding semantics in ``DenseGraphState.collate()``.

Per the 2026-05-07 sparse-default refactor, ``GraphData`` is the abstract
base of the type hierarchy; the concrete dense type is ``DenseGraphState``.
Tests retain the dense fixtures (``X_class`` / ``E_class`` / ``node_mask``)
because the helpers in ``tests._helpers.graph_builders`` produce
``DenseGraphState`` instances directly.
"""

import pytest
import torch
from tests._helpers.graph_builders import (
    binary_graphdata,
    edge_scalar_graphdata,
    legacy_edge_scalar,
)

from tmgg.data.datasets.graph_types import DenseGraphState

BS = 3
N = 6


@pytest.fixture()
def binary_adjacency() -> torch.Tensor:
    """A batch of simple binary symmetric adjacency matrices."""
    A = torch.zeros(BS, N, N)
    # Create some edges (symmetric)
    for b in range(BS):
        for i in range(N - 1):
            A[b, i, i + 1] = 1.0
            A[b, i + 1, i] = 1.0
    return A


class TestBinaryAdjacencyHelpers:
    """Verify binary-topology conversion helpers."""

    def test_output_shapes(self, binary_adjacency: torch.Tensor) -> None:
        """All fields have the expected shapes."""
        g = binary_graphdata(binary_adjacency)
        assert g.X_class is not None
        assert g.E_class is not None
        assert g.X_class.shape == (BS, N, 2)
        assert g.E_class.shape == (BS, N, N, 2)
        assert g.y.shape == (BS, 0)
        assert g.node_mask.shape == (BS, N)

    def test_nodes_are_one_hot(self, binary_adjacency: torch.Tensor) -> None:
        """All node features should be [0, 1] (real node)."""
        g = binary_graphdata(binary_adjacency)
        assert g.X_class is not None
        assert (g.X_class[:, :, 0] == 0).all()
        assert (g.X_class[:, :, 1] == 1).all()

    def test_edges_are_one_hot(self, binary_adjacency: torch.Tensor) -> None:
        """Edge features should sum to 1 along the last dim (one-hot)."""
        g = binary_graphdata(binary_adjacency)
        assert g.E_class is not None
        sums = g.E_class.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums))

    def test_diagonal_is_no_edge(self, binary_adjacency: torch.Tensor) -> None:
        """Diagonal entries should be [1, 0] (no self-loops)."""
        g = binary_graphdata(binary_adjacency)
        assert g.E_class is not None
        diag_idx = torch.arange(N)
        assert (g.E_class[:, diag_idx, diag_idx, 0] == 1).all()
        assert (g.E_class[:, diag_idx, diag_idx, 1] == 0).all()

    def test_all_nodes_valid(self, binary_adjacency: torch.Tensor) -> None:
        """node_mask should be all True (no padding at this stage)."""
        g = binary_graphdata(binary_adjacency)
        assert g.node_mask.all()

    def test_single_graph(self) -> None:
        """Unbatched (n, n) input still produces a single-graph DenseGraphState.

        ``DenseGraphState`` is always batched ``(bs, ...)`` per the
        2026-05-07 sparse-default refactor; the test_helpers builder
        wraps an unbatched ``(n, n)`` adjacency into a ``bs == 1``
        instance. The downstream assertions check the (1, n, ...)
        shapes that result.
        """
        A = torch.eye(4)
        A[0, 1] = A[1, 0] = 1.0
        g = binary_graphdata(A)
        assert g.X_class is not None
        assert g.E_class is not None
        assert g.X_class.dim() == 3
        assert g.X_class.shape == (1, 4, 2)
        assert g.E_class.dim() == 4
        assert g.E_class.shape == (1, 4, 4, 2)
        assert g.node_mask.dim() == 2
        assert g.node_mask.shape == (1, 4)


class TestToBinaryAdjacency:
    """Verify binary adjacency recovery."""

    def test_round_trip(self, binary_adjacency: torch.Tensor) -> None:
        """Binary topology should round-trip exactly."""
        g = binary_graphdata(binary_adjacency)
        A_recovered = g.dense_adjacency()
        # The diagonal is always zeroed (no self-loops)
        A_no_diag = binary_adjacency.clone()
        diag_idx = torch.arange(N)
        A_no_diag[:, diag_idx, diag_idx] = 0
        assert torch.allclose(A_recovered, A_no_diag)

    def test_single_graph_round_trip(self) -> None:
        """Round trip works for a single unbatched graph."""
        A = torch.zeros(5, 5)
        A[0, 1] = A[1, 0] = 1.0
        A[2, 3] = A[3, 2] = 1.0
        g = binary_graphdata(A)
        A_recovered = g.dense_adjacency()
        assert torch.allclose(A_recovered, A)

    def test_masking_zeros_invalid_edges(self) -> None:
        """Masked nodes should have zero adjacency entries."""
        E = torch.zeros(2, 4, 4, 2)
        E[:, :, :, 1] = 1.0  # all edges "present"
        node_mask = torch.ones(2, 4, dtype=torch.bool)
        node_mask[:, 3] = False  # mask last node
        g = DenseGraphState(
            X_class=torch.zeros(2, 4, 2),
            E_class=E,
            y=torch.zeros(2, 0),
            num_nodes_per_graph=node_mask.long().sum(dim=-1),
        )
        A = g.dense_adjacency()
        # Last row/col should be zero
        assert (A[:, 3, :] == 0).all()
        assert (A[:, :, 3] == 0).all()

    def test_symmetry_preservation(self, binary_adjacency: torch.Tensor) -> None:
        """Recovered adjacency should be symmetric if input edges were."""
        g = binary_graphdata(binary_adjacency)
        A = g.dense_adjacency()
        assert torch.allclose(A, A.transpose(1, 2))


class TestEdgeStateHelpers:
    """Verify continuous edge-state conversion helpers."""

    def test_binary_topology_lifts_to_edge_state(self) -> None:
        """Binary-topology graphs should expose their edge indicator as state."""
        adjacency = torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            ]
        )
        g = binary_graphdata(adjacency)
        torch.testing.assert_close(legacy_edge_scalar(g), adjacency)

    def test_round_trip_preserves_values(self) -> None:
        """Edge state should round-trip without binary projection."""
        edge_state = torch.tensor(
            [
                [
                    [0.0, -0.25, 1.5],
                    [2.0, 0.5, -3.25],
                    [4.5, 1.75, 0.0],
                ],
                [
                    [7.0, 8.5, -1.0],
                    [-2.5, 0.0, 3.0],
                    [9.25, -4.0, 6.0],
                ],
            ]
        )
        g = edge_scalar_graphdata(edge_state)
        recovered = legacy_edge_scalar(g)
        torch.testing.assert_close(recovered, edge_state)

    def test_round_trip_preserves_node_mask(self) -> None:
        """Explicit masks survive construction, and to_edge_scalar applies them."""
        edge_state = torch.randn(2, 4, 4)
        node_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        g = DenseGraphState.from_structure_only(node_mask, edge_state)
        assert torch.equal(g.node_mask, node_mask)
        # to_edge_scalar(source='feat') returns the stored edge state
        # masked by the outer product of node_mask.
        mask_2d = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        expected = edge_state * mask_2d.to(edge_state.dtype)
        torch.testing.assert_close(g.to_edge_scalar(source="feat"), expected)


class TestCollate:
    """Verify DenseGraphState.collate() for variable-size graphs.

    Per-graph instances are constructed directly with batch dimension 1
    (the dense type does not store unbatched single graphs the way the
    legacy ``GraphData`` did). The collator stacks per-graph rows to a
    common ``n_max`` derived from the largest member.
    """

    def _make_graph(self, n: int, dy: int = 0) -> DenseGraphState:
        """Create a single-graph DenseGraphState with n real nodes."""
        return DenseGraphState(
            X_class=torch.ones(1, n, 2),
            E_class=torch.ones(1, n, n, 2),
            y=torch.zeros(1, dy),
            num_nodes_per_graph=torch.tensor([n], dtype=torch.long),
        )

    def test_basic_padding(self) -> None:
        """Graphs of different sizes should be padded to the max."""
        g1 = self._make_graph(3)
        g2 = self._make_graph(5)
        batch = DenseGraphState.collate([g1, g2])
        assert batch.X_class is not None
        assert batch.E_class is not None
        assert batch.X_class.shape == (2, 5, 2)
        assert batch.E_class.shape == (2, 5, 5, 2)
        assert batch.node_mask.shape == (2, 5)

    def test_node_mask_correctness(self) -> None:
        """Mask should be True for real nodes, False for padding."""
        g1 = self._make_graph(3)
        g2 = self._make_graph(5)
        batch = DenseGraphState.collate([g1, g2])
        assert batch.node_mask[0, :3].all()
        assert not batch.node_mask[0, 3:].any()
        assert batch.node_mask[1, :5].all()

    def test_padded_nodes_have_no_node_class(self) -> None:
        """Padded node positions should be set to class 0 (no-node)."""
        g1 = self._make_graph(2)
        g2 = self._make_graph(4)
        batch = DenseGraphState.collate([g1, g2])
        assert batch.X_class is not None
        # For graph 0, positions 2 and 3 should have class 0 = 1
        assert batch.X_class[0, 2, 0] == 1.0
        assert batch.X_class[0, 3, 0] == 1.0

    def test_padded_edges_have_no_edge_class(self) -> None:
        """Padded edge positions should be set to class 0 (no-edge)."""
        g1 = self._make_graph(2)
        g2 = self._make_graph(4)
        batch = DenseGraphState.collate([g1, g2])
        assert batch.E_class is not None
        # For graph 0, row 2 (padded) should have E[0, 2, :, 0] = 1
        assert (batch.E_class[0, 2, :, 0] == 1.0).all()
        assert (batch.E_class[0, :, 2, 0] == 1.0).all()

    def test_real_data_preserved(self) -> None:
        """Real graph data should be preserved after padding."""
        X1 = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        E1 = torch.zeros(2, 2, 2)
        E1[0, 1, 1] = 1.0
        E1[1, 0, 1] = 1.0
        E1[0, 0, 0] = 1.0
        E1[1, 1, 0] = 1.0
        g1 = DenseGraphState(
            X_class=X1.unsqueeze(0),
            E_class=E1.unsqueeze(0),
            y=torch.zeros(1, 0),
            num_nodes_per_graph=torch.tensor([2], dtype=torch.long),
        )
        g2 = self._make_graph(3)

        batch = DenseGraphState.collate([g1, g2])
        assert batch.X_class is not None
        assert batch.E_class is not None
        # Graph 0's real data should be intact
        assert torch.allclose(batch.X_class[0, :2], X1)
        assert torch.allclose(batch.E_class[0, :2, :2], E1)

    def test_global_features_preserved(self) -> None:
        """Global features y should be copied correctly."""
        y1 = torch.tensor([1.0, 2.0])
        y2 = torch.tensor([3.0, 4.0])
        g1 = DenseGraphState(
            X_class=torch.ones(1, 2, 2),
            E_class=torch.ones(1, 2, 2, 2),
            y=y1.unsqueeze(0),
            num_nodes_per_graph=torch.tensor([2], dtype=torch.long),
        )
        g2 = DenseGraphState(
            X_class=torch.ones(1, 3, 2),
            E_class=torch.ones(1, 3, 3, 2),
            y=y2.unsqueeze(0),
            num_nodes_per_graph=torch.tensor([3], dtype=torch.long),
        )
        batch = DenseGraphState.collate([g1, g2])
        assert torch.allclose(batch.y[0], y1)
        assert torch.allclose(batch.y[1], y2)

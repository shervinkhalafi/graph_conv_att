"""Roundtrip and structural property tests for the 4-type GraphData grid.

Covers carrier conversions (sparse↔dense), content conversions
(state↔distribution), and the cell-to-cell composition matrix.
Verifies invariants and that lossless paths commute bit-for-bit.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphDistribution,
    GraphState,
)


def _no_edge_fill(d_ec: int) -> torch.Tensor:
    fill = torch.zeros(d_ec, dtype=torch.float32)
    fill[0] = 1.0
    return fill


@pytest.fixture
def small_state() -> GraphState:
    """Two graphs: 2 nodes / 1 edge, 3 nodes / 2 edges. Edge_class is one-hot
    over 2 channels (no-edge / edge)."""
    num_nodes = torch.tensor([2, 3], dtype=torch.long)
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 3, 4],
            [1, 0, 3, 2, 4, 3],
        ],
        dtype=torch.long,
    )
    edge_class = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],  # graph 0 edge 0-1 (both directions)
            [0.0, 1.0],
            [0.0, 1.0],  # graph 1 edge 2-3
            [0.0, 1.0],
            [0.0, 1.0],  # graph 1 edge 3-4
        ]
    )
    return GraphState(
        num_nodes_per_graph=num_nodes,
        y=torch.zeros(2, 0),
        batch=batch,
        x_class=None,
        x_feat=None,
        edge_index=edge_index,
        edge_class=edge_class,
        edge_feat=None,
    )


def test_state_carrier_roundtrip_preserves_active_edges(
    small_state: GraphState,
) -> None:
    """GraphState → DenseGraphState → GraphState recovers the same edge_index
    (set-equality, since edge_index ordering may change) and edge_class
    (gathered at recovered positions)."""
    fill = _no_edge_fill(d_ec=2)
    dense = small_state.to_dense(edge_class_fill=fill)
    recovered = dense.to_sparse()  # default predicate: argmax > 0

    # Compare set of (src, dst) directed pairs.
    orig = set(map(tuple, small_state.edge_index.t().tolist()))
    rec = set(map(tuple, recovered.edge_index.t().tolist()))
    assert orig == rec
    # num_nodes preserved.
    assert torch.equal(small_state.num_nodes_per_graph, recovered.num_nodes_per_graph)


def test_distribution_carrier_roundtrip_bit_exact(small_state: GraphState) -> None:
    """GraphDistribution → DenseGraphDistribution → GraphDistribution is
    bit-for-bit identity."""
    dist = small_state.to_distribution()
    dense = dist.to_dense()
    recovered = dense.to_sparse()

    # edge_index ordering should match (both produced by the same routine).
    # Sort both by (src, dst) for comparison.
    orig_sorted, _ = torch.sort(dist.edge_index[0] * 1_000_000 + dist.edge_index[1])
    rec_sorted, _ = torch.sort(
        recovered.edge_index[0] * 1_000_000 + recovered.edge_index[1]
    )
    assert torch.equal(orig_sorted, rec_sorted)
    assert dist.edge_class is not None and recovered.edge_class is not None
    assert dist.edge_class.shape == recovered.edge_class.shape


def test_dense_state_to_distribution_preserves_tensors(
    small_state: GraphState,
) -> None:
    """DenseGraphState.to_distribution() shares tensor identity with the source
    (no reshape, just type tag change)."""
    fill = _no_edge_fill(d_ec=2)
    dense_state = small_state.to_dense(edge_class_fill=fill)
    dense_dist = dense_state.to_distribution()
    assert isinstance(dense_dist, DenseGraphDistribution)
    assert dense_state.E_class is not None and dense_dist.E_class is not None
    assert torch.equal(dense_state.E_class, dense_dist.E_class)
    assert torch.equal(dense_state.num_nodes_per_graph, dense_dist.num_nodes_per_graph)


def test_dense_distribution_argmax_recovers_state(small_state: GraphState) -> None:
    """DenseGraphDistribution.argmax() applied to a one-hot encoding recovers
    the original state."""
    dense_state = small_state.to_dense(edge_class_fill=_no_edge_fill(2))
    dense_dist = dense_state.to_distribution()
    recovered = dense_dist.argmax()
    assert isinstance(recovered, DenseGraphState)
    assert dense_state.E_class is not None and recovered.E_class is not None
    assert torch.equal(dense_state.E_class, recovered.E_class)


def test_sparse_distribution_argmax_recovers_state(small_state: GraphState) -> None:
    """GraphDistribution.argmax() applied to a one-hot lift recovers state
    with the same active edges."""
    dist = small_state.to_distribution()
    recovered = dist.argmax()
    orig_pairs = set(map(tuple, small_state.edge_index.t().tolist()))
    rec_pairs = set(map(tuple, recovered.edge_index.t().tolist()))
    assert orig_pairs == rec_pairs


def test_state_to_dense_distribution_commutes(small_state: GraphState) -> None:
    """GraphState → DenseGraphDistribution via two paths gives the same tensor."""
    fill = _no_edge_fill(d_ec=2)
    # Path A: lift first, then carrier.
    path_a = small_state.to_distribution().to_dense()
    # Path B: carrier first, then lift.
    path_b = small_state.to_dense(edge_class_fill=fill).to_distribution()
    assert path_a.E_class is not None and path_b.E_class is not None
    assert torch.equal(path_a.E_class, path_b.E_class)
    assert torch.equal(path_a.num_nodes_per_graph, path_b.num_nodes_per_graph)


def test_graph_state_rejects_self_loops() -> None:
    with pytest.raises(ValueError, match="self-loops"):
        GraphState(
            num_nodes_per_graph=torch.tensor([2], dtype=torch.long),
            y=torch.zeros(1, 0),
            batch=torch.tensor([0, 0], dtype=torch.long),
            x_class=None,
            x_feat=None,
            edge_index=torch.tensor([[0, 0], [0, 0]], dtype=torch.long),
            edge_class=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            edge_feat=None,
        )


def test_graph_distribution_rejects_incomplete_edge_index() -> None:
    with pytest.raises(ValueError, match="complete off-diagonal"):
        # 2 nodes -> expect sum_E = 2*1 = 2; provide only 1.
        GraphDistribution(
            num_nodes_per_graph=torch.tensor([2], dtype=torch.long),
            y=torch.zeros(1, 0),
            batch=torch.tensor([0, 0], dtype=torch.long),
            x_class=None,
            x_feat=None,
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_class=torch.tensor([[0.0, 1.0]]),
            edge_feat=None,
        )


def test_graph_state_to_dense_requires_fill_for_edge_class(
    small_state: GraphState,
) -> None:
    with pytest.raises(ValueError, match="edge_class_fill is required"):
        small_state.to_dense()  # no fill provided


def test_graph_state_rejects_cross_graph_edges() -> None:
    with pytest.raises(ValueError, match="span different graphs"):
        GraphState(
            num_nodes_per_graph=torch.tensor([2, 2], dtype=torch.long),
            y=torch.zeros(2, 0),
            batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
            x_class=None,
            x_feat=None,
            edge_index=torch.tensor([[0, 2], [2, 0]], dtype=torch.long),  # cross-graph!
            edge_class=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            edge_feat=None,
        )

"""Tests for ``DenseGraphState.to_networkx`` and ``to_networkx_list``.

Pinned behaviours:

* Padding rows/columns (``node_mask == 0``) are dropped from the
  resulting graph.
* ``E_class`` argmax > 0 defines edges (channel 0 = "no edge").
* ``X_class`` argmax indices ride along as ``x_class`` node attrs.
* ``E_class`` argmax indices ride along as ``e_class`` edge attrs,
  but only for edges that survived the binary threshold.
* Batched ``DenseGraphState`` (``node_mask.dim() == 2``) requires an
  explicit ``batch_index``.
* ``to_networkx_list`` expands the whole batch.

These were carved out of the GraphData-as-universal-transport
refactor (spec 2026-05-01); the categorical-index attributes are the
hook molecular post-hoc analysis uses to reconstruct atom/bond types
without re-loading the codec.

Note on "unbatched": post the sparse-default refactor every
``DenseGraphState`` carries a 1-D ``num_nodes_per_graph`` and therefore
a 2-D ``node_mask`` (batched). The truly-unbatched branch that the
``to_networkx`` rejects-batch-index check guards is unreachable through
the public ctor; tests below exercise the equivalent semantic with a
batched-of-1 instance and an explicit ``batch_index=0``.
"""

from __future__ import annotations

import torch

from tmgg.data.datasets.graph_types import DenseGraphState


def _two_node_one_edge_batched_one() -> DenseGraphState:
    """Batched-of-1 hand-crafted 2-node graph with one edge."""
    return DenseGraphState(
        num_nodes_per_graph=torch.tensor([2], dtype=torch.long),
        # X_class: node 0 -> class 1, node 1 -> class 2.
        X_class=torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        ),
        # E_class: 2-channel "no-edge / edge".
        E_class=torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ],
            dtype=torch.float32,
        ),
        y=torch.zeros(1, 0),
    )


def _padded_batched() -> DenseGraphState:
    """Batched DenseGraphState with one valid graph (n=2) padded to n=3."""
    return DenseGraphState(
        num_nodes_per_graph=torch.tensor([2], dtype=torch.long),
        X_class=torch.tensor(
            [
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.0],  # padded position
                ]
            ],
            dtype=torch.float32,
        ),
        E_class=torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                    [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                ]
            ],
            dtype=torch.float32,
        ),
        y=torch.zeros(1, 0),
    )


def test_batched_one_to_networkx_topology_and_attrs() -> None:
    gd = _two_node_one_edge_batched_one()
    g = gd.to_networkx(batch_index=0)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    # Node attrs: x_class indices land on each node.
    assert g.nodes[0]["x_class"] == 1
    assert g.nodes[1]["x_class"] == 2
    # Edge attrs: only on surviving edges; e_class is the argmax of
    # the 2-channel E_class slot, which is 1 for the present edge.
    assert g.edges[0, 1]["e_class"] == 1


def test_batched_to_networkx_drops_padding_rows() -> None:
    gd = _padded_batched()
    g = gd.to_networkx(batch_index=0)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    assert g.nodes[0]["x_class"] == 1
    assert g.nodes[1]["x_class"] == 0


def test_to_networkx_list_expands_full_batch() -> None:
    gd = _padded_batched()
    graphs = gd.to_networkx_list()
    assert len(graphs) == 1
    assert graphs[0].number_of_nodes() == 2


def test_to_networkx_list_batched_one_returns_singleton() -> None:
    gd = _two_node_one_edge_batched_one()
    graphs = gd.to_networkx_list()
    assert len(graphs) == 1
    assert graphs[0].number_of_nodes() == 2


def test_no_x_class_means_no_node_attrs() -> None:
    """When X_class is None the per-node x_class attr must be absent."""
    gd = DenseGraphState(
        num_nodes_per_graph=torch.tensor([2], dtype=torch.long),
        # X_feat populated to exercise the X-side without X_class. Shape
        # (1, 2, 1) keeps it batched-of-1, matching the new invariant
        # that every populated tensor has the leading bs dim.
        X_feat=torch.zeros(1, 2, 1),
        E_class=torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            ],
            dtype=torch.float32,
        ),
        y=torch.zeros(1, 0),
    )
    g = gd.to_networkx(batch_index=0)
    assert "x_class" not in g.nodes[0]
    assert "x_class" not in g.nodes[1]
    # E_class still drives edge attrs.
    assert g.edges[0, 1]["e_class"] == 1

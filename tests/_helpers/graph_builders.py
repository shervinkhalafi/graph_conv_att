"""Test-local GraphData constructors replacing the removed legacy helpers.

The Wave 9.2 removal deletes ``GraphData.from_edge_state`` and
``GraphData.from_binary_adjacency`` from production code (see
``internal notes (not included)``).
Many test fixtures depended on those shapes — a dense adjacency producing a
two-channel ``E_class`` with a degenerate "node-or-not" ``X_class`` — and
replicating that inline at every call site would bloat diffs. This module
provides shared builders that tests import directly.

None of these helpers are production surface: they exist only so test
authors keep a concise way to construct the exact tensor layouts the legacy
helpers produced, without resurrecting the helpers themselves.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
)


def legacy_edge_scalar(data: GraphData) -> Tensor:
    """Return a dense scalar adjacency regardless of which edge field is populated.

    Replaces the removed ``GraphData.to_edge_state``. Accepts the universal
    :class:`GraphData` base for caller ergonomics — model forwards declared
    as returning ``GraphDistribution | DenseGraphDistribution`` widen
    cleanly to the base, and a runtime ``isinstance`` check keeps the body
    typed. ``DenseGraphState`` and ``DenseGraphDistribution`` carry the
    edge fields with identical shapes (the difference is purely the content
    interpretation). For ``DenseGraphState`` we delegate to
    :meth:`DenseGraphState.to_edge_scalar`; for ``DenseGraphDistribution``
    (which lacks that method per the type hierarchy) we replicate its body
    directly. Sparse carriers fail loudly: convert to dense at the call
    site.
    """
    if isinstance(data, DenseGraphState):
        if data.E_feat is not None:
            return data.to_edge_scalar(source="feat")
        return data.to_edge_scalar(source="class")

    if not isinstance(data, DenseGraphDistribution):
        raise TypeError(
            "legacy_edge_scalar() requires a dense carrier "
            "(DenseGraphState or DenseGraphDistribution); "
            f"got {type(data).__name__}. Convert at the call site via "
            "model(..., output_dense=True) or .to_dense()."
        )

    # DenseGraphDistribution path: inline the edge-scalar reduction.
    if data.E_feat is not None:
        e = data.E_feat
        edge_scalar = e.squeeze(-1) if e.shape[-1] == 1 else e[..., 0]
    else:
        if data.E_class is None:
            raise ValueError(
                "legacy_edge_scalar() requires E_feat or E_class to be populated."
            )
        if data.E_class.shape[-1] > 1:
            edge_scalar = 1.0 - data.E_class[..., 0]
        else:
            edge_scalar = data.E_class[..., 0]
    nm = data.node_mask
    mask2d = nm.unsqueeze(-1) * nm.unsqueeze(-2)
    return edge_scalar * mask2d.to(edge_scalar.dtype)


def binary_graphdata(adj: Tensor) -> DenseGraphState:
    """Build a DenseGraphState from a binary adjacency matching the legacy helper layout.

    Replaces the removed ``GraphData.from_binary_adjacency``. Produces the
    same two-channel ``E_class`` (``[no-edge, edge]``) with zero diagonal and
    a degenerate ``X_class`` marking every position as a "real" node. All
    node positions are valid (``num_nodes_per_graph`` covers every row).

    Parameters
    ----------
    adj
        Binary adjacency, shape ``(n, n)`` for a single graph or
        ``(bs, n, n)`` for a batch. Values should be 0 or 1.

    Returns
    -------
    DenseGraphState
        One-hot encoded graph with ``dx_class=2`` and ``de_class=2``.
    """
    single = adj.dim() == 2
    if single:
        adj = adj.unsqueeze(0)

    bs, n, _ = adj.shape
    adj = adj.float()

    x_class = torch.zeros(bs, n, 2, device=adj.device, dtype=adj.dtype)
    x_class[:, :, 1] = 1.0

    e_class = torch.zeros(bs, n, n, 2, device=adj.device, dtype=adj.dtype)
    e_class[:, :, :, 0] = 1.0 - adj
    e_class[:, :, :, 1] = adj

    diag_idx = torch.arange(n, device=adj.device)
    e_class[:, diag_idx, diag_idx, :] = 0
    e_class[:, diag_idx, diag_idx, 0] = 1.0

    y_out = torch.zeros(bs, 0, device=adj.device, dtype=adj.dtype)
    num_nodes_per_graph = torch.full((bs,), n, device=adj.device, dtype=torch.long)

    return DenseGraphState(
        num_nodes_per_graph=num_nodes_per_graph,
        y=y_out,
        X_class=x_class,
        E_class=e_class,
    )


def edge_scalar_graphdata(
    edge_state: Tensor, *, node_mask: Tensor | None = None
) -> DenseGraphState:
    """Build a structure-only DenseGraphState from a dense scalar edge tensor.

    Replaces the removed ``GraphData.from_edge_state``. Wraps a dense scalar
    adjacency as ``E_feat`` with shape ``(..., 1)`` and no categorical
    fields. When ``node_mask`` is not provided, every position is marked
    valid.

    Parameters
    ----------
    edge_state
        Dense scalar edges, ``(n, n)`` or ``(bs, n, n)``; a trailing
        single-channel axis is accepted and squeezed.
    node_mask
        Optional node-validity mask matching the batch shape of
        ``edge_state``. When ``None``, an all-True mask is synthesised.

    Returns
    -------
    DenseGraphState
        Instance with ``E_feat`` populated.
    """
    single = edge_state.dim() == 2
    if single:
        edge_state = edge_state.unsqueeze(0)

    if edge_state.dim() == 4 and edge_state.shape[-1] == 1:
        edge_state = edge_state[..., 0]

    if edge_state.dim() != 3:
        raise ValueError(
            "edge_scalar_graphdata() expects a 2D or 3D edge-state tensor, "
            f"got shape {tuple(edge_state.shape)}"
        )

    bs, n, _ = edge_state.shape

    if node_mask is None:
        node_mask = torch.ones(bs, n, device=edge_state.device, dtype=torch.bool)
    elif node_mask.dim() == 1:
        node_mask = node_mask.unsqueeze(0)

    if node_mask.shape != (bs, n):
        raise ValueError(
            "node_mask must have shape (bs, n) for edge_scalar_graphdata(), "
            f"got {tuple(node_mask.shape)}"
        )

    return DenseGraphState.from_structure_only(node_mask, edge_state)

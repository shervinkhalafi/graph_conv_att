"""Smoke tests for ``tmgg.training.losses``.

Rationale
---------
These tests verify the type-checked dispatch contract of
:func:`masked_ce_loss` and :func:`masked_mse_loss` introduced in Phase 5
of the sparse-default GraphData refactor. The contract:

1.  Carrier-matched calls (sparse pred + sparse target, or dense pred +
    dense target) must produce a finite zero-dim ``Tensor``.
2.  Carrier-mismatched calls (one sparse, one dense; or two states; or
    two distributions) must raise ``TypeError`` — silent densification
    or sparsification across the boundary is forbidden.

The fixtures construct a tiny two-graph batch (2 + 3 nodes) with a
2-channel one-hot ``edge_class`` and a 1-D continuous ``edge_feat``
sufficient to exercise both fields without depending on any production
dataset.
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
from tmgg.training.losses import masked_ce_loss, masked_mse_loss


def _build_sparse_state() -> GraphState:
    """Two graphs (sizes 2 and 3) with a single edge each, one-hot
    ``edge_class`` over 2 channels, scalar ``edge_feat`` set to 1.0 on
    every active directed edge."""
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
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    edge_feat = torch.ones(edge_index.shape[1], 1)
    return GraphState(
        num_nodes_per_graph=num_nodes,
        y=torch.zeros(2, 0),
        batch=batch,
        x_class=None,
        x_feat=None,
        edge_index=edge_index,
        edge_class=edge_class,
        edge_feat=edge_feat,
    )


def _sparse_distribution_from(state: GraphState) -> GraphDistribution:
    """Lift the sparse state onto the complete edge grid as a
    distribution we can use as the prediction carrier."""
    return state.to_distribution()


def _dense_state_and_dist() -> tuple[DenseGraphState, DenseGraphDistribution]:
    """Construct a dense state on a 2-graph batch (n_max=3) with a
    one-hot ``E_class`` and a scalar ``E_feat``, plus a paired
    distribution carrier with the same shape (logits = the one-hot
    target — sufficient for finite-loss smoke checks)."""
    num_nodes = torch.tensor([2, 3], dtype=torch.long)
    bs, n_max, d_ec = 2, 3, 2
    e_class = torch.zeros(bs, n_max, n_max, d_ec)
    # Channel 0 is the "no-edge" filler; mark active edges in channel 1.
    e_class[..., 0] = 1.0
    e_class[0, 0, 1] = torch.tensor([0.0, 1.0])
    e_class[0, 1, 0] = torch.tensor([0.0, 1.0])
    e_class[1, 0, 1] = torch.tensor([0.0, 1.0])
    e_class[1, 1, 0] = torch.tensor([0.0, 1.0])
    e_class[1, 1, 2] = torch.tensor([0.0, 1.0])
    e_class[1, 2, 1] = torch.tensor([0.0, 1.0])
    # Zero the diagonal (no self-loops).
    diag = torch.arange(n_max)
    e_class[:, diag, diag, :] = 0.0
    e_feat = torch.zeros(bs, n_max, n_max, 1)
    e_feat[0, 0, 1, 0] = 1.0
    e_feat[0, 1, 0, 0] = 1.0

    state = DenseGraphState(
        num_nodes_per_graph=num_nodes,
        y=torch.zeros(2, 0),
        X_class=None,
        X_feat=None,
        E_class=e_class,
        E_feat=e_feat,
    )
    dist = DenseGraphDistribution(
        num_nodes_per_graph=num_nodes,
        y=torch.zeros(2, 0),
        X_class=None,
        X_feat=None,
        E_class=e_class.clone(),
        E_feat=e_feat.clone(),
    )
    return state, dist


def test_masked_ce_sparse_pair_returns_finite_scalar() -> None:
    state = _build_sparse_state()
    dist = _sparse_distribution_from(state)
    loss = masked_ce_loss(dist, state, field="edge_class")
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_masked_ce_dense_pair_returns_finite_scalar() -> None:
    state, dist = _dense_state_and_dist()
    loss = masked_ce_loss(dist, state, field="edge_class")
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_masked_ce_mismatched_carriers_raise_type_error() -> None:
    sparse_state = _build_sparse_state()
    sparse_dist = _sparse_distribution_from(sparse_state)
    dense_state, dense_dist = _dense_state_and_dist()

    # sparse pred + dense target
    with pytest.raises(TypeError, match="masked_ce_loss requires"):
        masked_ce_loss(sparse_dist, dense_state, field="edge_class")

    # dense pred + sparse target
    with pytest.raises(TypeError, match="masked_ce_loss requires"):
        masked_ce_loss(dense_dist, sparse_state, field="edge_class")


def test_masked_mse_sparse_pair_returns_finite_scalar() -> None:
    state = _build_sparse_state()
    dist = _sparse_distribution_from(state)
    loss = masked_mse_loss(dist, state, field="edge_feat")
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_masked_mse_dense_pair_returns_finite_scalar() -> None:
    state, dist = _dense_state_and_dist()
    loss = masked_mse_loss(dist, state, field="edge_feat")
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_masked_mse_mismatched_carriers_raise_type_error() -> None:
    sparse_state = _build_sparse_state()
    sparse_dist = _sparse_distribution_from(sparse_state)
    dense_state, dense_dist = _dense_state_and_dist()

    with pytest.raises(TypeError, match="masked_mse_loss requires"):
        masked_mse_loss(sparse_dist, dense_state, field="edge_feat")

    with pytest.raises(TypeError, match="masked_mse_loss requires"):
        masked_mse_loss(dense_dist, sparse_state, field="edge_feat")

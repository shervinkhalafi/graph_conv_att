"""Sparse-pathway tests for the new 4-type GraphData siblings on
CategoricalNoiseProcess and GaussianNoiseProcess.

Test rationale
--------------
Phase 4 of the sparse-default GraphData refactor adds ``sample`` /
``sample_dense`` (and, for the categorical process, ``q_marginal`` /
``q_marginal_dense``) on the 2x2 carrier × content grid. The dense
siblings are exercised by the legacy dense-path tests in
``test_noise_process.py``; this module covers only the new
sparse-pathway public surface and its type contract.

Each test pins:

- the static return type (``GraphState`` for categorical-state output,
  ``GraphDistribution`` for the categorical marginal and every Gaussian
  output);
- the structural invariants of the sparse output: a
  :class:`GraphDistribution` covers the *complete* off-diagonal
  edge_index (sum_E == sum_i n_i (n_i - 1)), while a
  :class:`GraphState` covers a *subset* (sum_E ≤ sum_i n_i (n_i - 1));
- consistency between the sparse path and re-sparsified dense path: the
  sparse-output edge_index must match what
  ``DenseGraphState.to_sparse`` / ``DenseGraphDistribution.to_sparse``
  produce from the matching dense intermediate.
"""

from __future__ import annotations

import torch

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphDistribution,
    GraphState,
    state_to_dense_sample,
)
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    GaussianNoiseProcess,
)
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.utils.noising.noise import GaussianNoise


def _categorical_state(seed: int = 0) -> GraphState:
    """Two-graph batch of a triangle and a P_3 path with one-hot edges.

    Graph 0: triangle on 3 nodes (3 undirected edges → 6 directed).
    Graph 1: path 0-1-2 on 3 nodes (2 undirected edges → 4 directed).
    Encoded with ``e_classes=2`` (channel 0 = no-edge, channel 1 = edge)
    and ``x_classes=2`` (channel 0 = pad-style, channel 1 = real-node).
    """
    torch.manual_seed(seed)
    nn_per = torch.tensor([3, 3], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    # Triangle (graph 0) edges, then path (graph 1) edges; both directed.
    e0 = torch.tensor(
        [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]],
        dtype=torch.long,
    )
    # Graph-1 nodes are 3..5 globally; build path 3-4-5.
    e1 = torch.tensor(
        [[3, 4, 4, 5], [4, 3, 5, 4]],
        dtype=torch.long,
    )
    edge_index = torch.cat([e0, e1], dim=1)

    sum_E = int(edge_index.shape[1])
    edge_class = torch.zeros(sum_E, 2)
    edge_class[:, 1] = 1.0  # all retained edges are class "edge"

    x_class = torch.zeros(6, 2)
    x_class[:, 1] = 1.0
    return GraphState(
        num_nodes_per_graph=nn_per,
        y=torch.zeros(2, 0),
        batch=batch,
        x_class=x_class,
        x_feat=None,
        edge_index=edge_index,
        edge_class=edge_class,
        edge_feat=None,
    )


def _gaussian_state(seed: int = 0) -> GraphState:
    """Two-graph batch with a single-channel ``edge_feat`` adjacency.

    Same topology as ``_categorical_state`` but encoded as a continuous
    weight = 1.0 per active edge, with no categorical fields.
    """
    torch.manual_seed(seed)
    nn_per = torch.tensor([3, 3], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    e0 = torch.tensor(
        [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]],
        dtype=torch.long,
    )
    e1 = torch.tensor(
        [[3, 4, 4, 5], [4, 3, 5, 4]],
        dtype=torch.long,
    )
    edge_index = torch.cat([e0, e1], dim=1)
    sum_E = int(edge_index.shape[1])
    edge_feat = torch.ones(sum_E, 1)
    return GraphState(
        num_nodes_per_graph=nn_per,
        y=torch.zeros(2, 0),
        batch=batch,
        x_class=None,
        x_feat=None,
        edge_index=edge_index,
        edge_class=None,
        edge_feat=edge_feat,
    )


def _expected_complete_E(num_nodes_per_graph: torch.Tensor) -> int:
    """Total directed edges in a complete off-diagonal graph batch."""
    n = num_nodes_per_graph
    return int((n * (n - 1)).sum().item())


# ---------------------------------------------------------------------------
# CategoricalNoiseProcess
# ---------------------------------------------------------------------------


def test_categorical_sample_returns_graph_state() -> None:
    z0 = _categorical_state()
    schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=50)
    proc = CategoricalNoiseProcess(
        schedule=schedule,
        x_classes=int(z0.x_class.shape[-1]) if z0.x_class is not None else 0,
        e_classes=int(z0.edge_class.shape[-1]) if z0.edge_class is not None else 0,
        limit_distribution="uniform",
    )
    t = torch.tensor([10, 20])
    # Categorical sampling draws fresh randomness; pin the global RNG so
    # the sparse path and the matching dense → to_sparse path agree on the
    # same draw and therefore on the same active-edge count.
    torch.manual_seed(123)
    z_t = proc.sample(z0, t)

    assert isinstance(z_t, GraphState)
    # sum_E must not exceed the complete-pair upper bound.
    assert int(z_t.edge_index.shape[1]) <= _expected_complete_E(z0.num_nodes_per_graph)
    # Re-running with the same seed through the dense-then-sparse path
    # must produce the same sparse layout: the new ``sample`` is exactly
    # ``state_to_dense_sample → sample_dense → to_sparse``.
    torch.manual_seed(123)
    dense = state_to_dense_sample(z0)
    noised_dense = proc.sample_dense(dense, t)
    re_sparsified = noised_dense.to_sparse()
    assert int(z_t.edge_index.shape[1]) == int(re_sparsified.edge_index.shape[1])
    torch.testing.assert_close(z_t.edge_index, re_sparsified.edge_index)


def test_categorical_q_marginal_returns_graph_distribution() -> None:
    z0 = _categorical_state()
    schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=50)
    proc = CategoricalNoiseProcess(
        schedule=schedule,
        x_classes=int(z0.x_class.shape[-1]) if z0.x_class is not None else 0,
        e_classes=int(z0.edge_class.shape[-1]) if z0.edge_class is not None else 0,
        limit_distribution="uniform",
    )
    t = torch.tensor([10, 20])
    qt = proc.q_marginal(z0, t)

    assert isinstance(qt, GraphDistribution)
    # GraphDistribution requires the complete off-diagonal edge_index.
    assert int(qt.edge_index.shape[1]) == _expected_complete_E(z0.num_nodes_per_graph)
    # Per-position rows are PMFs over the e_classes channels.
    assert qt.edge_class is not None
    row_sums = qt.edge_class.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_categorical_sample_dense_returns_dense_graph_state() -> None:
    z0 = _categorical_state()
    schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=50)
    proc = CategoricalNoiseProcess(
        schedule=schedule,
        x_classes=2,
        e_classes=2,
        limit_distribution="uniform",
    )
    dense = state_to_dense_sample(z0)
    t = torch.tensor([5, 5])
    out = proc.sample_dense(dense, t)
    assert isinstance(out, DenseGraphState)
    # One-hot per active edge position (sum to 1 across the channel axis).
    assert out.E_class is not None
    nm = out.node_mask
    n_max = int(nm.shape[1])
    pair_mask = nm.unsqueeze(-1) & nm.unsqueeze(-2)
    eye = torch.eye(n_max, dtype=torch.bool).unsqueeze(0)
    valid = pair_mask & ~eye
    e_sums = out.E_class[valid].sum(dim=-1)
    assert torch.allclose(e_sums, torch.ones_like(e_sums), atol=1e-5)


def test_categorical_q_marginal_dense_returns_dense_graph_distribution() -> None:
    z0 = _categorical_state()
    schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=50)
    proc = CategoricalNoiseProcess(
        schedule=schedule,
        x_classes=2,
        e_classes=2,
        limit_distribution="uniform",
    )
    dense = state_to_dense_sample(z0)
    t = torch.tensor([10, 20])
    out = proc.q_marginal_dense(dense, t)
    assert isinstance(out, DenseGraphDistribution)
    assert out.E_class is not None
    # Inactive (padding / diagonal) rows are zero by construction; active
    # rows are PMFs that sum to 1.
    nm = out.node_mask
    n_max = int(nm.shape[1])
    pair_mask = nm.unsqueeze(-1) & nm.unsqueeze(-2)
    eye = torch.eye(n_max, dtype=torch.bool).unsqueeze(0)
    valid = pair_mask & ~eye
    row_sums = out.E_class[valid].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# GaussianNoiseProcess
# ---------------------------------------------------------------------------


def test_gaussian_sample_returns_graph_distribution() -> None:
    z0 = _gaussian_state()
    schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=50)
    proc = GaussianNoiseProcess(definition=GaussianNoise(), schedule=schedule)
    t = torch.tensor([10, 20])
    qt = proc.sample(z0, t)
    assert isinstance(qt, GraphDistribution)
    # Gaussian sample → distribution carrier covers the complete
    # off-diagonal edge_index (continuous mass everywhere).
    assert int(qt.edge_index.shape[1]) == _expected_complete_E(z0.num_nodes_per_graph)


def test_gaussian_sample_dense_returns_dense_graph_distribution() -> None:
    z0 = _gaussian_state()
    schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=50)
    proc = GaussianNoiseProcess(definition=GaussianNoise(), schedule=schedule)
    dense = state_to_dense_sample(z0)
    t = torch.tensor([10, 20])
    out = proc.sample_dense(dense, t)
    assert isinstance(out, DenseGraphDistribution)
    # Edge-feat output is symmetric in the edge axes.
    assert out.E_feat is not None
    torch.testing.assert_close(out.E_feat, out.E_feat.transpose(-3, -2))

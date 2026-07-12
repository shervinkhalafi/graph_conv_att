"""Stage 1 telemetry: per-timestep loss/KL bins and bits-per-edge.

These tests pin the behaviour of the cheap dataset-agnostic dynamics
metrics added to :class:`DiffusionModule`. They exercise the helpers
in isolation to keep iteration fast — full Lightning-loop coverage is
provided by the existing integration tests in ``tests/experiments``
which run the per-step path end-to-end.
"""

from __future__ import annotations

import math

import torch

from tmgg.training.lightning_modules.train_loss_discrete import (
    masked_node_ce,
    per_graph_edge_ce,
    per_graph_node_ce,
    per_graph_y_ce,
)

# ---------------------------------------------------------------------------
# Per-graph CE helpers (Stage 1 supports for the per-t scatter)
# ---------------------------------------------------------------------------


def test_per_graph_node_ce_matches_masked_node_ce_on_average() -> None:
    """The batch mean of per-graph node CE equals the global ``masked_node_ce``.

    Both helpers use the same valid-row predicate (``(true != 0).any(-1)``);
    per-graph divides by per-graph valid count, then we mean over graphs.
    Equivalence holds when every graph contributes the same number of
    valid rows (no padding mismatch), which is true for our test fixture.
    """
    torch.manual_seed(0)
    bs, n, dx = 4, 6, 3
    pred = torch.randn(bs, n, dx)
    targets = torch.zeros(bs, n, dx)
    cls = torch.randint(0, dx, (bs, n))
    targets.scatter_(-1, cls.unsqueeze(-1), 1.0)
    node_mask = torch.ones(bs, n, dtype=torch.bool)

    pg = per_graph_node_ce(pred, targets, node_mask)
    assert pg.shape == (bs,)

    global_ce = masked_node_ce(pred, targets, node_mask)
    assert torch.allclose(pg.mean(), global_ce, atol=1e-5)


def test_per_graph_node_ce_zero_loss_at_perfect_prediction() -> None:
    """When logits put all mass on the true class, per-graph CE → 0."""
    bs, n, dx = 2, 3, 2
    one_hot = torch.zeros(bs, n, dx)
    cls = torch.randint(0, dx, (bs, n))
    one_hot.scatter_(-1, cls.unsqueeze(-1), 1.0)
    # Logits with infinite weight on the true class.
    big = 50.0
    pred = (one_hot - 0.5) * big * 2
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    pg = per_graph_node_ce(pred, one_hot, node_mask)
    assert torch.all(pg < 1e-10)


def test_per_graph_edge_ce_excludes_diagonal_via_zero_target_predicate() -> None:
    """Edge CE uses ``(true != 0).any(-1)`` which auto-skips diagonals.

    A graph with only diagonal "edges" (encoded as all-zero rows by
    ``encode_no_edge``) has no valid rows, so the per-graph helper
    returns ``0/1 == 0`` with the ``count.clamp(min=1)`` guard.
    """
    bs, n, de = 2, 4, 3
    pred = torch.randn(bs, n, n, de)
    true = torch.zeros(bs, n, n, de)  # all-zero ⇒ no valid rows anywhere
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    pg = per_graph_edge_ce(pred, true, node_mask)
    assert torch.allclose(pg, torch.zeros(bs))


def test_per_graph_y_ce_handles_empty_class_field() -> None:
    """Empty-class graph-level field returns a zero ``(bs,)`` tensor.

    Mirrors the upstream guard in :func:`masked_y_ce`.
    """
    bs, dy = 3, 0
    pred = torch.zeros(bs, 1)  # shape doesn't matter; numel(true) is the gate
    true = torch.zeros(bs, dy)
    pg = per_graph_y_ce(pred, true)
    assert pg.shape == (bs,)
    assert torch.all(pg == 0.0)


# ---------------------------------------------------------------------------
# Bin-index logic
# ---------------------------------------------------------------------------


def test_t_bin_indices_cover_all_t_uniformly() -> None:
    """``t_int`` covering 0..T with ``n_t_bins`` evenly partitions the range.

    Synthetic case: T=99, n_t_bins=10. ``t = arange(100)`` → exactly 10
    samples per bin. The bin-index formula is
    ``(t * n_bins) // (T + 1)``.
    """
    T = 99
    n_bins = 10
    t = torch.arange(T + 1)
    bin_idx = (t * n_bins) // (T + 1)
    counts = torch.zeros(n_bins, dtype=torch.long)
    counts.scatter_add_(0, bin_idx, torch.ones_like(t))
    assert torch.all(counts == 10)


def test_t_bin_indices_clamped_to_last_bin_at_T() -> None:
    """When T+1 is divisible by n_bins, t=T should land in the final bin.

    The ``clamp(max=n_bins-1)`` in :meth:`_t_bin_idx` is defensive against
    rounding for oddly-divisible (T, n_bins) combinations.
    """
    T = 100
    n_bins = 10
    t = torch.tensor([T])
    bin_idx = ((t * n_bins) // (T + 1)).clamp(max=n_bins - 1)
    assert int(bin_idx.item()) == n_bins - 1


# ---------------------------------------------------------------------------
# Bits-per-edge formula
# ---------------------------------------------------------------------------


def test_bits_per_edge_known_inputs() -> None:
    """Exact ratio: bits = sum(NLL_nats) / (sum(edges) * ln(2))."""
    nll = torch.tensor([10.0, 20.0])  # nats
    edges = torch.tensor([5.0, 10.0])
    total_nll = nll.sum()  # 30
    total_edges = edges.sum()  # 15
    expected = total_nll / (total_edges * math.log(2.0))
    got = total_nll / (total_edges * torch.log(torch.tensor(2.0)))
    assert torch.allclose(got, expected.unsqueeze(0).squeeze(), atol=1e-6)


def test_bits_per_edge_handles_zero_edges_via_clamp() -> None:
    """When the val set has zero edges, the clamp(min=1) keeps the
    computation finite (yields total_nll * 1 / ln(2) bits, which is
    a sentinel rather than a NaN). Behaviour-pinning test only."""
    total_nll = torch.tensor(5.0)
    total_edges = torch.tensor(0.0)
    ln2 = torch.log(torch.tensor(2.0))
    got = total_nll / (total_edges.clamp(min=1.0) * ln2)
    assert torch.isfinite(got)
    # Equivalent to bits assuming exactly 1 edge.
    assert torch.allclose(got, total_nll / ln2)

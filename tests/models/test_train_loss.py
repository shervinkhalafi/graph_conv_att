"""Tests for TrainLossDiscrete and the per-field masked CE helpers.

Test rationale: The training loss directly determines gradient signal quality.
Incorrect masking or normalization leads to silent training failures where the
model appears to train but produces degenerate samples. These tests verify the
loss computation against known analytical results and confirm that masking
correctly excludes invalid positions.

Following the 2026-04-21 upstream-parity refactor, the categorical helpers
consume **raw logits** (not post-softmax probabilities) and dispatch to
``F.cross_entropy``. The tests below feed logits accordingly:

* "perfect" predictions use a large-valued one-hot-like logit so
  ``log_softmax`` collapses to ~0 loss;
* "uniform" predictions use all-zero logits so softmax is uniform over
  classes;
* padding rows in the **target** are set to all-zero so the
  ``(true != 0).any(-1)`` predicate inside the helpers drops them — this is
  exactly the convention upstream's ``encode_no_edge`` / node-pad logic
  follows.

The ``TestUpstreamParity`` class pins behavioural equivalence with an explicit
reference implementation using upstream's formula
(``F.cross_entropy(flat_logits, flat_target_indices, reduction='sum') /
num_valid_rows``) to ``atol=1e-6``. This is the regression test that keeps the
parity claim honest.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from tmgg.training.lightning_modules.train_loss_discrete import (
    TrainLossDiscrete,
    masked_edge_ce,
    masked_node_ce,
)

BS = 4
N = 8
DX = 3
DE = 2


def _make_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer indices to one-hot float tensors."""
    return torch.nn.functional.one_hot(indices, num_classes).float()


def _peaked_logits_from_one_hot(
    one_hot: torch.Tensor, peak: float = 20.0
) -> torch.Tensor:
    """Return logits whose softmax concentrates on the one-hot class.

    With ``peak=20``, softmax puts ~1.0 on the target class (exp(20)/(exp(20)+2)
    is within float32 rounding of 1.0), so cross-entropy on the correct target
    is ~exp(-20) ≈ 2e-9 per row.
    """
    return one_hot * peak


@pytest.fixture()
def loss_fn() -> TrainLossDiscrete:
    return TrainLossDiscrete(lambda_E=5.0)


@pytest.fixture()
def full_mask() -> torch.Tensor:
    """All nodes are valid."""
    return torch.ones(BS, N, dtype=torch.bool)


class TestPerfectPrediction:
    """Large-peak logits matching one-hot targets give near-zero CE."""

    def test_perfect_nodes_and_edges(
        self, loss_fn: TrainLossDiscrete, full_mask: torch.Tensor
    ) -> None:
        """Peaked logits on the true class yield near-zero loss."""
        true_X = _make_one_hot(torch.randint(0, DX, (BS, N)), DX)
        true_E = _make_one_hot(torch.randint(0, DE, (BS, N, N)), DE)

        pred_X_logits = _peaked_logits_from_one_hot(true_X)
        pred_E_logits = _peaked_logits_from_one_hot(true_E)

        loss = loss_fn(
            pred_X=pred_X_logits,
            pred_E=pred_E_logits,
            true_X=true_X.clone(),
            true_E=true_E.clone(),
            node_mask=full_mask,
        )
        assert loss.item() < 1e-6, f"Perfect prediction loss too high: {loss.item()}"


class TestUniformPrediction:
    """Zero logits give softmax = uniform PMF; expected CE is log(K)."""

    def test_uniform_node_prediction(self, full_mask: torch.Tensor) -> None:
        """With lambda_E=0, uniform node prediction gives loss = log(DX)."""
        loss_fn = TrainLossDiscrete(lambda_E=0.0)
        true_X = _make_one_hot(torch.zeros(BS, N, dtype=torch.long), DX)
        pred_X_logits = torch.zeros(BS, N, DX)
        # Edges don't matter since lambda_E=0; still need valid (non-zero) target
        true_E = _make_one_hot(torch.zeros(BS, N, N, dtype=torch.long), DE)
        pred_E_logits = _peaked_logits_from_one_hot(true_E)

        loss = loss_fn(
            pred_X=pred_X_logits,
            pred_E=pred_E_logits,
            true_X=true_X.clone(),
            true_E=true_E.clone(),
            node_mask=full_mask,
        )
        expected = math.log(DX)
        assert (
            abs(loss.item() - expected) < 1e-5
        ), f"Uniform node loss {loss.item():.6f} != expected log({DX})={expected:.6f}"

    def test_uniform_edge_prediction(self, full_mask: torch.Tensor) -> None:
        """With lambda_E=1 and perfect node pred, uniform edge prediction
        gives edge contribution = log(DE).
        """
        loss_fn = TrainLossDiscrete(lambda_E=1.0)
        # Perfect node prediction (peaked logits on true class)
        true_X = _make_one_hot(torch.zeros(BS, N, dtype=torch.long), DX)
        pred_X_logits = _peaked_logits_from_one_hot(true_X)
        # Uniform edge prediction via zero logits
        true_E = _make_one_hot(torch.zeros(BS, N, N, dtype=torch.long), DE)
        pred_E_logits = torch.zeros(BS, N, N, DE)

        loss = loss_fn(
            pred_X=pred_X_logits,
            pred_E=pred_E_logits,
            true_X=true_X.clone(),
            true_E=true_E.clone(),
            node_mask=full_mask,
        )
        expected_edge = math.log(DE)
        # Node loss is near zero, so total ~ 1.0 * log(DE)
        assert (
            abs(loss.item() - expected_edge) < 1e-5
        ), f"Uniform edge loss {loss.item():.6f} != expected log({DE})={expected_edge:.6f}"


class TestMasking:
    """Padding rows in the target (all-zero) should not contribute to the loss.

    The helpers drop rows via ``(true != 0).any(-1)`` — the same predicate
    upstream uses. Padding node/edge positions emit all-zero target rows via
    ``encode_no_edge`` / equivalent, so setting them to zero in the test
    mirrors production.
    """

    def test_masked_nodes_do_not_affect_loss(self, loss_fn: TrainLossDiscrete) -> None:
        """Loss should be the same whether extra nodes are masked or absent."""
        n_valid = 6
        node_mask_a = torch.ones(BS, n_valid, dtype=torch.bool)

        true_X_a = _make_one_hot(torch.randint(0, DX, (BS, n_valid)), DX)
        true_E_a = _make_one_hot(torch.randint(0, DE, (BS, n_valid, n_valid)), DE)
        pred_X_a_logits = torch.randn(BS, n_valid, DX)
        pred_E_a_logits = torch.randn(BS, n_valid, n_valid, DE)

        loss_a = loss_fn(
            pred_X=pred_X_a_logits.clone(),
            pred_E=pred_E_a_logits.clone(),
            true_X=true_X_a.clone(),
            true_E=true_E_a.clone(),
            node_mask=node_mask_a,
        )

        # Scenario B: pad with 2 extra nodes whose TARGET rows are all-zero
        # (matches upstream's ``encode_no_edge`` / node-pad convention).
        n_padded = n_valid + 2
        node_mask_b = torch.ones(BS, n_padded, dtype=torch.bool)
        node_mask_b[:, n_valid:] = False

        true_X_b = torch.zeros(BS, n_padded, DX)
        true_X_b[:, :n_valid] = true_X_a
        true_E_b = torch.zeros(BS, n_padded, n_padded, DE)
        true_E_b[:, :n_valid, :n_valid] = true_E_a

        # Pred logits at padding positions can be arbitrary — dropped by the
        # (true != 0) predicate.
        pred_X_b_logits = torch.randn(BS, n_padded, DX) * 100
        pred_X_b_logits[:, :n_valid] = pred_X_a_logits
        pred_E_b_logits = torch.randn(BS, n_padded, n_padded, DE) * 100
        pred_E_b_logits[:, :n_valid, :n_valid] = pred_E_a_logits

        loss_b = loss_fn(
            pred_X=pred_X_b_logits,
            pred_E=pred_E_b_logits,
            true_X=true_X_b,
            true_E=true_E_b,
            node_mask=node_mask_b,
        )

        assert torch.allclose(loss_a, loss_b, atol=1e-6), (
            f"Masked loss {loss_b.item():.6f} differs from unmasked "
            f"{loss_a.item():.6f}"
        )


class TestLambdaEWeighting:
    """Verify that lambda_E scales the edge loss contribution."""

    def test_higher_lambda_increases_loss(self, full_mask: torch.Tensor) -> None:
        """Doubling lambda_E should exactly double the total loss when node
        prediction is perfect (contributes ~0) and edge prediction is uniform
        (contributes log(DE)).
        """
        true_X = _make_one_hot(torch.zeros(BS, N, dtype=torch.long), DX)
        true_E = _make_one_hot(torch.zeros(BS, N, N, dtype=torch.long), DE)
        pred_X_logits = _peaked_logits_from_one_hot(true_X)  # near-zero node CE
        pred_E_logits = torch.zeros(BS, N, N, DE)  # uniform -> CE = log(DE)

        loss_low = TrainLossDiscrete(lambda_E=1.0)(
            pred_X=pred_X_logits.clone(),
            pred_E=pred_E_logits.clone(),
            true_X=true_X.clone(),
            true_E=true_E.clone(),
            node_mask=full_mask,
        )
        loss_high = TrainLossDiscrete(lambda_E=2.0)(
            pred_X=pred_X_logits.clone(),
            pred_E=pred_E_logits.clone(),
            true_X=true_X.clone(),
            true_E=true_E.clone(),
            node_mask=full_mask,
        )
        assert loss_high.item() > loss_low.item(), (
            f"Higher lambda_E did not increase loss: "
            f"lambda_E=2.0 -> {loss_high.item():.4f}, "
            f"lambda_E=1.0 -> {loss_low.item():.4f}"
        )
        ratio = loss_high.item() / max(loss_low.item(), 1e-10)
        assert 1.99 < ratio < 2.01, f"Expected loss ratio ~2.0 but got {ratio:.4f}"


def _reference_cross_entropy_over_valid_rows(
    logits: torch.Tensor,
    true_onehot: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Upstream DiGress's exact formula for the CE-over-valid-rows metric.

    Mirrors ``src/metrics/abstract_metrics.py:CrossEntropyMetric`` — sum of
    per-row CE with ``reduction='sum'`` divided by the number of valid rows.
    Rows with an all-zero target are dropped, matching the
    ``(true != 0).any(-1)`` predicate the live helpers use.

    Parameters
    ----------
    logits
        Flat ``(N_total, C)`` or higher-rank logits; the function reshapes
        to ``(-1, C)`` internally.
    true_onehot
        Same leading shape as ``logits``; last axis is the class dimension.
    label_smoothing
        Forwarded to :func:`F.cross_entropy`.

    Returns
    -------
    Tensor
        Scalar ``sum(per_row_CE_over_valid_rows) / num_valid_rows``.
    """
    c = logits.size(-1)
    flat_logits = logits.reshape(-1, c)
    flat_true = true_onehot.reshape(-1, c)
    valid = (flat_true != 0).any(dim=-1)
    flat_logits = flat_logits[valid]
    flat_targets = flat_true[valid].argmax(dim=-1)
    total_ce = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="sum",
        label_smoothing=label_smoothing,
    )
    return total_ce / flat_logits.size(0)


class TestUpstreamParity:
    """Pin behavioural parity with upstream's ``F.cross_entropy`` formula.

    Both helpers must match the reference implementation to float32 precision.
    These tests are the regression anchor: any future accidental re-introduction
    of epsilon smoothing or softmax-then-log gymnastics will break them.
    """

    def test_masked_node_ce_matches_upstream(self) -> None:
        """``masked_node_ce`` == ``CE_sum_over_valid / num_valid`` at atol=1e-6."""
        torch.manual_seed(42)
        # Mix of valid and padding rows to exercise the (true != 0) predicate.
        true_X = torch.zeros(BS, N, DX)
        # Put valid one-hot targets in half the positions
        true_X[:, : N // 2] = _make_one_hot(torch.randint(0, DX, (BS, N // 2)), DX)
        pred_X_logits = torch.randn(BS, N, DX)
        node_mask = torch.ones(BS, N, dtype=torch.bool)

        got = masked_node_ce(pred_X_logits, true_X, node_mask)
        expected = _reference_cross_entropy_over_valid_rows(pred_X_logits, true_X)
        assert torch.allclose(
            got, expected, atol=1e-6
        ), f"masked_node_ce {got.item()} != reference {expected.item()}"

    def test_masked_edge_ce_matches_upstream(self) -> None:
        """``masked_edge_ce`` == upstream CE at atol=1e-6."""
        torch.manual_seed(43)
        # Mix of valid and all-zero (diagonal / padding) rows.
        true_E = torch.zeros(BS, N, N, DE)
        # Mark upper-triangle (off-diagonal) as valid one-hot
        for b in range(BS):
            for i in range(N):
                for j in range(i + 1, N):
                    cls = int(torch.randint(0, DE, (1,)).item())
                    true_E[b, i, j, cls] = 1.0
                    true_E[b, j, i, cls] = 1.0  # symmetric
        pred_E_logits = torch.randn(BS, N, N, DE)
        node_mask = torch.ones(BS, N, dtype=torch.bool)

        got = masked_edge_ce(pred_E_logits, true_E, node_mask)
        expected = _reference_cross_entropy_over_valid_rows(pred_E_logits, true_E)
        assert torch.allclose(
            got, expected, atol=1e-6
        ), f"masked_edge_ce {got.item()} != reference {expected.item()}"

    def test_masked_edge_ce_matches_upstream_on_pipeline_shaped_targets(
        self,
    ) -> None:
        """``masked_edge_ce`` must match upstream on pipeline-shaped E targets.

        Pins the invariant from ``GraphData.from_pyg_batch`` that our
        training targets arrive at the CE with the E diagonal explicitly
        zeroed, mirroring upstream DiGress's ``utils.encode_no_edge``
        (``digress-upstream-readonly/src/utils.py:73-74``: ``E[diag] = 0``).

        The earlier ``test_masked_edge_ce_matches_upstream`` constructs
        ``true_E`` with a manual upper-triangular fill — it never exercises
        a full ``[1-adj, adj]``-stacked one-hot batch. This test builds
        exactly what the production data path now emits (diagonal = zero
        after the 2026-04-21 upstream-parity fix in ``from_pyg_batch``) and
        pins ``masked_edge_ce`` to upstream's ``F.cross_entropy`` formula
        at ``atol=1e-6``.

        If the data layer ever stops zeroing the diagonal, this test will
        fail loudly. See ``an internal validation harness (not included)`` for
        the full tradeoff discussion.
        """
        torch.manual_seed(99)
        # Reconstruct what ``GraphData.from_pyg_batch`` emits AFTER the
        # 2026-04-21 upstream-parity fix: symmetric adjacency with zeroed
        # diagonal, stacked into ``[1-adj, adj]`` one-hot, with the
        # resulting E_class diagonal **explicitly zeroed** to match
        # upstream's ``encode_no_edge``.
        adj = torch.bernoulli(torch.full((BS, N, N), 0.47))
        adj = (adj + adj.transpose(1, 2)).clamp(max=1.0)
        diag_idx = torch.arange(N)
        adj[:, diag_idx, diag_idx] = 0.0
        true_E_pipeline = torch.stack([1.0 - adj, adj], dim=-1)
        true_E_pipeline[:, diag_idx, diag_idx, :] = 0.0
        # Sanity: diagonal must be ``[0, 0]`` in this fixture — that's the
        # whole point.
        assert (
            true_E_pipeline[:, diag_idx, diag_idx] == torch.tensor([0.0, 0.0])
        ).all(), "Test fixture construction error: diagonal should be [0, 0]."

        pred_E_logits = torch.randn(BS, N, N, DE)
        node_mask = torch.ones(BS, N, dtype=torch.bool)

        got = masked_edge_ce(pred_E_logits, true_E_pipeline, node_mask)
        expected = _reference_cross_entropy_over_valid_rows(
            pred_E_logits, true_E_pipeline
        )
        assert torch.allclose(got, expected, atol=1e-6), (
            f"masked_edge_ce={got.item():.6f} diverges from upstream "
            f"reference={expected.item():.6f} on pipeline-shaped input. "
            f"If this fails, the data layer ``from_pyg_batch`` may have "
            f"stopped zeroing the E diagonal; see BUG_REPORT.md."
        )

    def test_label_smoothing_passed_through(self) -> None:
        """Non-zero ``label_smoothing`` reaches ``F.cross_entropy``."""
        torch.manual_seed(44)
        true_X = _make_one_hot(torch.randint(0, DX, (BS, N)), DX)
        pred_X_logits = torch.randn(BS, N, DX)
        node_mask = torch.ones(BS, N, dtype=torch.bool)

        got_hard = masked_node_ce(pred_X_logits, true_X, node_mask, label_smoothing=0.0)
        got_smooth = masked_node_ce(
            pred_X_logits, true_X, node_mask, label_smoothing=0.1
        )
        expected_smooth = _reference_cross_entropy_over_valid_rows(
            pred_X_logits, true_X, label_smoothing=0.1
        )

        # Hard vs smooth must actually differ, and smooth must match the
        # reference with the same smoothing factor.
        assert not torch.allclose(
            got_hard, got_smooth
        ), "label_smoothing=0.1 had no effect — kwarg ignored?"
        assert torch.allclose(got_smooth, expected_smooth, atol=1e-6), (
            f"masked_node_ce(smoothing=0.1) {got_smooth.item()} != "
            f"reference {expected_smooth.item()}"
        )

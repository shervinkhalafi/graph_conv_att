"""Discrete diffusion training loss — per-field masked CE helpers.

The Wave 5.1 refactor splits the original joint ``(X, E)`` loss into per-field
helpers so :class:`~tmgg.training.lightning_modules.diffusion_module.DiffusionModule`
can iterate over ``noise_process.fields``. :class:`TrainLossDiscrete` is now a
thin compatibility wrapper that composes the node and edge helpers with a
``lambda_E`` weighting, keeping upstream DiGress parity tests that call it
directly green. New callers should prefer the helpers below.

The categorical helpers now consume **raw logits** and dispatch to
``torch.nn.functional.cross_entropy`` exactly as upstream DiGress does
(`digress-upstream-readonly/src/metrics/train_metrics.py:95-102`). Invalid
rows are dropped with the same ``(true != 0).any(-1)`` predicate upstream
uses after flattening to ``(bs*n, dx)`` / ``(bs*n*n, de)``, so edge-diagonal
positions (encoded as all-zero rows via ``encode_no_edge``) are excluded
automatically. ``label_smoothing`` is exposed as an opt-in keyword-only
parameter; the default ``0.0`` preserves bit-for-bit upstream parity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_node_ce(
    pred_X_logits: Tensor,
    true_X: Tensor,
    node_mask: Tensor,
    *,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Masked cross-entropy over a node-shaped categorical field.

    Parameters
    ----------
    pred_X_logits
        Raw logits, shape ``(bs, n, dx)``. No softmax expected or applied
        externally: the fused ``log_softmax`` inside
        :func:`torch.nn.functional.cross_entropy` handles it.
    true_X
        Target distribution, shape ``(bs, n, dx)``. Typically one-hot; soft
        targets work too but are converted to hard class indices internally
        via ``argmax(-1)`` to match upstream DiGress semantics.
    node_mask
        Boolean mask of valid nodes, shape ``(bs, n)``. Currently unused for
        row selection (the ``(true != 0).any(-1)`` predicate already excludes
        padding rows because invalid ``true_X`` entries are all-zero), but
        kept in the signature for API stability and future use.
    label_smoothing
        Forwarded to :func:`torch.nn.functional.cross_entropy`. Default
        ``0.0`` matches upstream. Pass e.g. ``0.1`` for classical label
        smoothing at the call site.

    Returns
    -------
    Tensor
        Scalar mean CE across valid node positions (flattened across batch
        and node axes), matching upstream's per-row mean semantics.
    """
    del node_mask  # unused — row selection uses the (true != 0) predicate

    dx = pred_X_logits.size(-1)
    flat_logits = pred_X_logits.reshape(-1, dx)
    flat_true = true_X.reshape(-1, dx)

    valid = (flat_true != 0).any(dim=-1)
    flat_logits = flat_logits[valid]
    flat_true = flat_true[valid]
    flat_targets = flat_true.argmax(dim=-1)

    return F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="mean",
        label_smoothing=label_smoothing,
    )


def masked_edge_ce(
    pred_E_logits: Tensor,
    true_E: Tensor,
    node_mask: Tensor,
    *,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Masked cross-entropy over an edge-shaped categorical field.

    Parameters
    ----------
    pred_E_logits
        Raw logits, shape ``(bs, n, n, de)``. No softmax expected or applied.
    true_E
        Target distribution, shape ``(bs, n, n, de)``. Diagonal positions
        and padding-node pairs are encoded as all-zero rows by
        :func:`tmgg.data.datasets.graph_types.encode_no_edge`; the
        ``(true != 0).any(-1)`` predicate below excludes them automatically
        — no additional diagonal mask is needed.
    node_mask
        Boolean mask of valid nodes, shape ``(bs, n)``. Currently unused
        (see :func:`masked_node_ce` note); kept for API stability.
    label_smoothing
        Forwarded to :func:`torch.nn.functional.cross_entropy`. Default
        ``0.0`` matches upstream.

    Returns
    -------
    Tensor
        Scalar mean CE across valid edge positions.
    """
    del node_mask  # unused — row selection uses the (true != 0) predicate

    de = pred_E_logits.size(-1)
    flat_logits = pred_E_logits.reshape(-1, de)
    flat_true = true_E.reshape(-1, de)

    valid = (flat_true != 0).any(dim=-1)
    flat_logits = flat_logits[valid]
    flat_true = flat_true[valid]
    flat_targets = flat_true.argmax(dim=-1)

    return F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="mean",
        label_smoothing=label_smoothing,
    )


def masked_y_ce(
    pred_y_logits: Tensor,
    true_y: Tensor,
    *,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross-entropy over a graph-level (global) categorical field.

    Mirrors upstream DiGress's ``loss_y`` term in
    ``digress-upstream-readonly/src/metrics/train_metrics.py:62-123``.
    The graph-level ``y`` tensor is shape ``(bs, dy)``: there is no
    spatial mask and no diagonal to exclude — every batch element
    contributes one classification example.

    Parameters
    ----------
    pred_y_logits
        Raw logits, shape ``(bs, dy)``. No softmax applied externally.
    true_y
        Target distribution, shape ``(bs, dy)``. Typically one-hot;
        soft targets are converted to hard class indices via
        ``argmax(-1)`` to match upstream DiGress semantics.
    label_smoothing
        Forwarded to :func:`torch.nn.functional.cross_entropy`. Default
        ``0.0`` matches upstream.

    Returns
    -------
    Tensor
        Scalar mean CE across the batch. Empty-batch / empty-class
        inputs (``true_y.numel() == 0``) return a zero scalar with no
        gradient, matching upstream's ``if true_y.numel() > 0 else 0.0``
        guard at ``train_metrics.py:99-103``.
    """
    if true_y.numel() == 0:
        # Upstream guard: a graph-level field with zero classes (the
        # SBM convention before this wire-up) contributes no loss.
        return pred_y_logits.new_zeros(())
    flat_targets = true_y.argmax(dim=-1)
    return F.cross_entropy(
        pred_y_logits,
        flat_targets,
        reduction="mean",
        label_smoothing=label_smoothing,
    )


def masked_y_mse(
    pred_y: Tensor,
    true_y: Tensor,
) -> Tensor:
    """Mean-squared error over a graph-level (global) continuous field.

    Sister helper to :func:`masked_y_ce` for continuous global
    targets. There is no spatial mask: every batch element contributes
    one regression example, averaged over the feature axis.

    Parameters
    ----------
    pred_y
        Predicted features, shape ``(bs, dy)``.
    true_y
        Target features, shape ``(bs, dy)``.

    Returns
    -------
    Tensor
        Scalar mean MSE across the batch. Empty-feature inputs
        (``true_y.numel() == 0``) return a zero scalar with no gradient.
    """
    if true_y.numel() == 0:
        return pred_y.new_zeros(())
    diff_sq = (pred_y - true_y) ** 2  # (bs, dy)
    per_graph = diff_sq.mean(dim=-1)  # (bs,)
    return per_graph.mean()


def masked_node_mse(
    pred_X: Tensor,
    true_X: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Per-graph masked MSE over a node-shaped continuous field.

    Zeros out contributions at padding positions; remaining positions are
    averaged per graph and then over the batch.
    """
    diff_sq = (pred_X - true_X) ** 2  # (bs, n, d)
    diff_sum = diff_sq.sum(dim=-1)  # (bs, n)
    num_nodes = node_mask.sum(dim=-1).clamp(min=1).to(diff_sum.dtype)  # (bs,)
    per_graph = (diff_sum * node_mask).sum(dim=-1) / num_nodes  # (bs,)
    return per_graph.mean()


def masked_edge_mse(
    pred_E: Tensor,
    true_E: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Per-graph masked MSE over an edge-shaped continuous field.

    Zeros contributions at padding edges and along the diagonal; remaining
    off-diagonal entries are averaged per graph and then over the batch.
    """
    diff_sq = (pred_E - true_E) ** 2  # (bs, n, n, d)
    diff_sum = diff_sq.sum(dim=-1)  # (bs, n, n)
    diag_mask = ~torch.eye(
        node_mask.size(1), device=node_mask.device, dtype=torch.bool
    ).unsqueeze(0)
    edge_mask = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask).to(
        diff_sum.dtype
    )
    num_edges = edge_mask.sum(dim=(-1, -2)).clamp(min=1)  # (bs,)
    per_graph = (diff_sum * edge_mask).sum(dim=(-1, -2)) / num_edges  # (bs,)
    return per_graph.mean()


def per_graph_node_ce(
    pred_X_logits: Tensor,
    true_X: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Per-graph masked cross-entropy over a node-shaped categorical field.

    Returns ``(bs,)`` mean CE per graph (averaged over its valid node
    positions). The valid-row predicate matches :func:`masked_node_ce`
    (``(true != 0).any(-1)``) so padding rows that are encoded as
    all-zero contribute zero. Used by the per-timestep telemetry path
    in ``DiffusionModule``: per-graph loss + per-graph ``t`` lets the
    training-step scatter into ``t``-binned accumulators without
    losing batch-level resolution.
    """
    log_probs = F.log_softmax(pred_X_logits, dim=-1)  # (bs, n, dx)
    targets = true_X.argmax(dim=-1)  # (bs, n)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (bs, n)
    valid_row = (true_X != 0).any(dim=-1) & node_mask  # (bs, n)
    nll = nll * valid_row.to(nll.dtype)
    count = valid_row.sum(dim=-1).clamp(min=1).to(nll.dtype)  # (bs,)
    return nll.sum(dim=-1) / count


def per_graph_edge_ce(
    pred_E_logits: Tensor,
    true_E: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Per-graph masked cross-entropy over an edge-shaped categorical field.

    Returns ``(bs,)`` mean CE per graph. The valid-row predicate
    ``(true != 0).any(-1)`` already excludes the diagonal and padding
    pairs (encoded as all-zero rows by ``encode_no_edge``).
    """
    log_probs = F.log_softmax(pred_E_logits, dim=-1)  # (bs, n, n, de)
    targets = true_E.argmax(dim=-1)  # (bs, n, n)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (bs, n, n)
    valid_row = (true_E != 0).any(dim=-1)  # (bs, n, n)
    valid_row = valid_row & (node_mask.unsqueeze(1) & node_mask.unsqueeze(2))
    nll = nll * valid_row.to(nll.dtype)
    count = valid_row.sum(dim=(-1, -2)).clamp(min=1).to(nll.dtype)
    return nll.sum(dim=(-1, -2)) / count


def per_graph_y_ce(pred_y_logits: Tensor, true_y: Tensor) -> Tensor:
    """Per-graph cross-entropy over a graph-level categorical field.

    Returns ``(bs,)``. Empty-class inputs return a zero tensor of the
    right shape (matches the upstream guard in :func:`masked_y_ce`).
    """
    bs = pred_y_logits.shape[0]
    if true_y.numel() == 0:
        return pred_y_logits.new_zeros((bs,))
    log_probs = F.log_softmax(pred_y_logits, dim=-1)  # (bs, dy)
    targets = true_y.argmax(dim=-1)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def per_graph_node_mse(pred_X: Tensor, true_X: Tensor, node_mask: Tensor) -> Tensor:
    """Per-graph masked MSE over a node-shaped continuous field, ``(bs,)``."""
    diff_sq = (pred_X - true_X) ** 2  # (bs, n, d)
    diff_sum = diff_sq.sum(dim=-1)  # (bs, n)
    num_nodes = node_mask.sum(dim=-1).clamp(min=1).to(diff_sum.dtype)
    return (diff_sum * node_mask.to(diff_sum.dtype)).sum(dim=-1) / num_nodes


def per_graph_edge_mse(pred_E: Tensor, true_E: Tensor, node_mask: Tensor) -> Tensor:
    """Per-graph masked MSE over an edge-shaped continuous field, ``(bs,)``."""
    diff_sq = (pred_E - true_E) ** 2  # (bs, n, n, d)
    diff_sum = diff_sq.sum(dim=-1)
    diag_mask = ~torch.eye(
        node_mask.size(1), device=node_mask.device, dtype=torch.bool
    ).unsqueeze(0)
    edge_mask = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask).to(
        diff_sum.dtype
    )
    num_edges = edge_mask.sum(dim=(-1, -2)).clamp(min=1)
    return (diff_sum * edge_mask).sum(dim=(-1, -2)) / num_edges


def per_graph_y_mse(pred_y: Tensor, true_y: Tensor) -> Tensor:
    """Per-graph MSE over a graph-level continuous field, ``(bs,)``."""
    bs = pred_y.shape[0]
    if true_y.numel() == 0:
        return pred_y.new_zeros((bs,))
    diff_sq = (pred_y - true_y) ** 2  # (bs, dy)
    return diff_sq.mean(dim=-1)


class TrainLossDiscrete:
    """Masked cross-entropy loss for dual-field discrete diffusion training.

    Thin compatibility wrapper around :func:`masked_node_ce` and
    :func:`masked_edge_ce`. New code should call the helpers directly via the
    :class:`~tmgg.training.lightning_modules.diffusion_module.DiffusionModule`
    per-field loss loop; this class remains so the upstream DiGress parity
    regression tests stay green without changes.

    Parameters
    ----------
    lambda_E
        Weight for edge loss relative to node loss. Default is 5.0,
        following DiGress convention.
    label_smoothing
        Forwarded to the masked CE helpers. Default ``0.0`` preserves
        upstream parity.
    """

    def __init__(
        self,
        lambda_E: float = 5.0,
        *,
        label_smoothing: float = 0.0,
    ) -> None:
        self.lambda_E = lambda_E
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        pred_X: Tensor,
        pred_E: Tensor,
        true_X: Tensor,
        true_E: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        """Compute masked cross-entropy loss.

        Parameters
        ----------
        pred_X
            Raw node logits, shape ``(bs, n, dx)``. Not softmaxed.
        pred_E
            Raw edge logits, shape ``(bs, n, n, de)``. Not softmaxed.
        true_X
            True node class distribution (one-hot or soft targets),
            shape ``(bs, n, dx)``.
        true_E
            True edge class distribution, shape ``(bs, n, n, de)``.
        node_mask
            Boolean mask for valid nodes, shape ``(bs, n)``.

        Returns
        -------
        Tensor
            Scalar loss value, averaged over valid positions then weighted.
        """
        loss_x = masked_node_ce(
            pred_X, true_X, node_mask, label_smoothing=self.label_smoothing
        )
        loss_e = masked_edge_ce(
            pred_E, true_E, node_mask, label_smoothing=self.label_smoothing
        )
        return loss_x + self.lambda_E * loss_e

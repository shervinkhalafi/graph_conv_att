"""Masked mean-squared-error loss over the 4-type GraphData grid.

Mirrors :func:`tmgg.training.losses.masked_ce_loss` — distribution-content
``pred`` versus state-content ``target`` — but operates on continuous
fields (``edge_feat`` / ``x_feat``) by elementwise difference instead of
softmax cross-entropy. Carrier matching is enforced; cross-carrier pairs
raise ``TypeError``.
"""

# pyright: reportPrivateImportUsage=false
# torch's runtime API exposes ``zeros`` at module top level; the stub
# does not re-export it. Same precedent as ``tmgg.data.datasets.graph_types``.
from __future__ import annotations

import torch
from torch import Tensor

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphDistribution,
    GraphState,
    _DistributionGraph,
    _StateGraph,
)


def masked_mse_loss(
    pred: _DistributionGraph,
    target: _StateGraph,
    *,
    field: str = "edge_feat",
    weight: float = 1.0,
) -> Tensor:
    """Masked MSE between distribution-content ``pred`` and state-content
    ``target`` on the named field.

    Parameters
    ----------
    pred
        Prediction carrier (a :class:`_DistributionGraph` subtype).
    target
        Reference carrier (a :class:`_StateGraph` subtype).
    field
        One of ``"edge_feat"`` or ``"x_feat"``. Other fields raise
        ``KeyError`` via ``getattr``.
    weight
        Multiplicative scalar applied to the reduced loss. Default 1.0.

    Returns
    -------
    Tensor
        Zero-dim mean-squared error across valid positions, weighted by
        ``weight``. Returns zero when the field is ``None`` on either
        side.

    Raises
    ------
    TypeError
        If the (pred, target) carrier pair is not (dense, dense) or
        (sparse, sparse).
    """
    if isinstance(pred, DenseGraphDistribution) and isinstance(target, DenseGraphState):
        return _masked_mse_dense(pred, target, field=field, weight=weight)
    if isinstance(pred, GraphDistribution) and isinstance(target, GraphState):
        return _masked_mse_sparse(pred, target, field=field, weight=weight)
    raise TypeError(
        "masked_mse_loss requires (pred, target) carriers to match: got "
        f"pred={type(pred).__name__}, target={type(target).__name__}. "
        "Convert one side via to_dense() / to_sparse() before calling."
    )


def _dense_field_name(field: str) -> str:
    if field.startswith("edge_"):
        return "E_" + field[len("edge_") :]
    if field.startswith("x_"):
        return "X_" + field[len("x_") :]
    return field


def _masked_mse_dense(
    pred: DenseGraphDistribution,
    target: DenseGraphState,
    *,
    field: str,
    weight: float,
) -> Tensor:
    name = _dense_field_name(field)
    pred_t: Tensor | None = getattr(pred, name)
    tgt_t: Tensor | None = getattr(target, name)
    if pred_t is None or tgt_t is None:
        return torch.zeros((), device=pred.num_nodes_per_graph.device)

    nm = target.node_mask  # (B, n_max), bool
    if field.startswith("edge"):
        # Pair mask broadcast over the channel axis.
        pair_mask = (nm.unsqueeze(-1) & nm.unsqueeze(-2)).unsqueeze(-1)
        mask_f = pair_mask.to(pred_t.dtype)
        diff = (pred_t - tgt_t) * mask_f
        denom = mask_f.sum().clamp(min=1)
    else:
        node_mask = nm.unsqueeze(-1).to(pred_t.dtype)
        diff = (pred_t - tgt_t) * node_mask
        denom = node_mask.sum().clamp(min=1)
    return weight * (diff**2).sum() / denom


def _masked_mse_sparse(
    pred: GraphDistribution,
    target: GraphState,
    *,
    field: str,
    weight: float,
) -> Tensor:
    pred_t: Tensor | None = getattr(pred, field)
    if pred_t is None:
        return torch.zeros((), device=pred.num_nodes_per_graph.device)

    if field.startswith("edge"):
        # Lift target onto the complete edge_index so shapes line up.
        target_dist: GraphDistribution = target.to_distribution()
        tgt_t: Tensor | None = getattr(target_dist, field)
        if tgt_t is None:
            return torch.zeros((), device=pred.num_nodes_per_graph.device)
        return weight * ((pred_t - tgt_t) ** 2).mean()

    tgt_t = getattr(target, field)
    if tgt_t is None:
        return torch.zeros((), device=pred.num_nodes_per_graph.device)
    return weight * ((pred_t - tgt_t) ** 2).mean()

"""Masked cross-entropy loss over the 4-type GraphData grid.

``pred`` must be a :class:`_DistributionGraph` (logits / probabilities);
``target`` must be a :class:`_StateGraph` (one-hot reference). Both
carriers are accepted: sparse pred + sparse target uses the complete
edge_index path; dense pred + dense target uses the legacy padded-grid
implementation. Cross-carrier pairs raise ``TypeError`` — convert
explicitly via ``to_dense`` / ``to_sparse`` upstream.

The ``field`` argument selects which split tensor to attack:

- ``"edge_class"`` (default): edges over the categorical class channel
- ``"x_class"``: nodes over the categorical class channel

For the sparse path on edges, ``GraphState.to_distribution()`` is used to
lift the active edges onto the same complete edge_index that the
``GraphDistribution`` carries; this preserves the active-edge classes
unchanged and fills inactive positions with the canonical no-edge
one-hot, matching the upstream dense convention.
"""

# pyright: reportPrivateImportUsage=false
# torch's runtime API exposes ``zeros``, ``eye``, ``bool`` etc. at module
# top level; the stub does not re-export them, so basedpyright flags
# every use. Same precedent as ``tmgg.data.datasets.graph_types``.
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphDistribution,
    GraphState,
    _DistributionGraph,
    _StateGraph,
)


def masked_ce_loss(
    pred: _DistributionGraph,
    target: _StateGraph,
    *,
    field: str = "edge_class",
    weight: float = 1.0,
) -> Tensor:
    """Masked cross-entropy between distribution-content ``pred`` and
    state-content ``target`` on the named field.

    Parameters
    ----------
    pred
        Prediction carrier (a :class:`_DistributionGraph` subtype). For
        edges, the underlying tensor is over the *complete* edge grid
        (every off-diagonal pair) on the sparse path or the padded
        ``(B, n_max, n_max, K)`` block on the dense path.
    target
        Reference carrier (a :class:`_StateGraph` subtype) with one-hot
        class encoding on its active edges (sparse) or padded grid (dense).
    field
        One of ``"edge_class"`` or ``"x_class"``. Other fields raise
        ``KeyError`` via ``getattr``.
    weight
        Multiplicative scalar applied to the reduced loss. Default 1.0.

    Returns
    -------
    Tensor
        Zero-dim mean cross-entropy across valid positions, weighted by
        ``weight``. When the field is ``None`` on either side (the
        carrier was constructed without that channel), the loss is zero.

    Raises
    ------
    TypeError
        If the (pred, target) carrier pair is not (dense, dense) or
        (sparse, sparse). Cross-carrier mixing is forbidden — convert
        explicitly via ``to_dense`` / ``to_sparse`` upstream.
    """
    if isinstance(pred, DenseGraphDistribution) and isinstance(target, DenseGraphState):
        return _masked_ce_dense(pred, target, field=field, weight=weight)
    if isinstance(pred, GraphDistribution) and isinstance(target, GraphState):
        return _masked_ce_sparse(pred, target, field=field, weight=weight)
    raise TypeError(
        "masked_ce_loss requires (pred, target) carriers to match: got "
        f"pred={type(pred).__name__}, target={type(target).__name__}. "
        "Convert one side via to_dense() / to_sparse() before calling."
    )


def _dense_field_name(field: str) -> str:
    """Map sparse-style field names to the dense uppercase counterparts."""
    if field.startswith("edge_"):
        return "E_" + field[len("edge_") :]
    if field.startswith("x_"):
        return "X_" + field[len("x_") :]
    return field


def _masked_ce_dense(
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
        bs = int(nm.shape[0])
        n_max = int(nm.shape[1])
        pair_mask = nm.unsqueeze(-1) & nm.unsqueeze(-2)  # (B, n_max, n_max)
        eye_mask = (
            torch.eye(n_max, dtype=torch.bool, device=nm.device)
            .unsqueeze(0)
            .expand(bs, n_max, n_max)
        )
        valid = (pair_mask & ~eye_mask).reshape(-1)  # (B*n_max*n_max,)
        # CE inputs: (N, C) and (N,) integer targets.
        pred_flat = pred_t.reshape(-1, pred_t.shape[-1])
        tgt_idx = tgt_t.reshape(-1, tgt_t.shape[-1]).argmax(dim=-1)
    else:
        # Node CE: flatten over (B, n_max).
        valid = nm.reshape(-1)  # (B*n_max,)
        pred_flat = pred_t.reshape(-1, pred_t.shape[-1])
        tgt_idx = tgt_t.reshape(-1, tgt_t.shape[-1]).argmax(dim=-1)

    losses = F.cross_entropy(pred_flat, tgt_idx, reduction="none")
    return weight * (losses * valid.to(losses.dtype)).sum() / valid.sum().clamp(min=1)


def _masked_ce_sparse(
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
        # ``GraphDistribution`` carries the complete edge_index;
        # ``GraphState`` carries only active edges. Lift the target onto
        # the same complete grid so we can do an elementwise CE.
        target_dist: GraphDistribution = target.to_distribution()
        tgt_t: Tensor | None = (
            target_dist.edge_class
            if field == "edge_class"
            else (getattr(target_dist, field))
        )
        if tgt_t is None:
            return torch.zeros((), device=pred.num_nodes_per_graph.device)
        idx = tgt_t.argmax(dim=-1)
        losses = F.cross_entropy(pred_t, idx, reduction="none")
        # Match the dense convention: drop pairs whose target one-hot is
        # the all-zero "no-channel" filler (none in our scattering, but
        # guard anyway). Active positions are exactly those where the
        # one-hot has any non-zero entry — by construction in
        # ``to_distribution()`` that is every complete-grid position.
        valid = (tgt_t != 0).any(dim=-1)
        return (
            weight * (losses * valid.to(losses.dtype)).sum() / valid.sum().clamp(min=1)
        )

    # Node CE: pred and target are both indexed on per-graph nodes
    # (sum_n rows). Each target row is a one-hot.
    tgt_t = getattr(target, field)
    if tgt_t is None:
        return torch.zeros((), device=pred.num_nodes_per_graph.device)
    idx = tgt_t.argmax(dim=-1)
    losses = F.cross_entropy(pred_t, idx, reduction="none")
    return weight * losses.mean()

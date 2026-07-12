"""Field identifiers and loss-kind dispatch for unified graph carriers.

Declares the canonical field names (node/edge/graph class and feature
streams) and the loss kind each field dispatches to, shared across
datasets, models, and losses under the Wave 0 unified-carrier refactor
and Wave 1 sparse-default split (the four concrete leaves
``GraphState`` / ``GraphDistribution`` / ``DenseGraphState`` /
``DenseGraphDistribution`` carry these field streams under
representation-appropriate names).

The ``y_class`` and ``y_feat`` entries (parity #27 / #44 / D-13) cover
graph-level (global) targets, mirroring upstream DiGress's per-graph
``y`` tensor and its ``loss_y`` cross-entropy term in
``digress-upstream-readonly/src/metrics/train_metrics.py:62-123``. The
graph carrier carries the underlying tensor on ``y``; the per-field
loss loop reads it via the field-name dispatch in ``DiffusionModule``.
With the default ``lambda_y = 0.0`` the y-term contributes nothing to
training loss, preserving structure-only SBM behaviour bit-for-bit.
"""

from typing import Final, Literal

FieldName = Literal["X_class", "X_feat", "E_class", "E_feat", "y_class", "y_feat"]

FIELD_NAMES: Final[frozenset[FieldName]] = frozenset(
    {"X_class", "X_feat", "E_class", "E_feat", "y_class", "y_feat"}
)

GRAPHDATA_LOSS_KIND: Final[dict[FieldName, Literal["ce", "mse"]]] = {
    "X_class": "ce",
    "X_feat": "mse",
    "E_class": "ce",
    "E_feat": "mse",
    "y_class": "ce",
    "y_feat": "mse",
}

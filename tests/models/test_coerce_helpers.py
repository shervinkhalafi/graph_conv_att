"""Cell-by-cell tests for ``_coerce_input_to`` / ``_coerce_output_to``.

The 4 concrete :class:`~tmgg.data.datasets.graph_types.GraphData` types form
a 2x2 grid on (carrier, content). The coercion helpers in
:mod:`tmgg.models.base` are required to be:

- *Total* over the 16 ``(input_type, target_type)`` cells of the input grid,
  EXCEPT the four lossy ``distribution -> state`` collapses, which must be
  rejected with :class:`TypeError`.
- *Lossless and identity-preserving* on the four diagonal cells.

These tests exercise each cell with hand-built fixtures small enough to
keep the suite cheap (two graphs of 2 and 3 nodes, one feature channel).

Rationale
---------
The helpers are infrastructure called by every model's ``forward``; they
fan in from arbitrary call sites and must never silently inject a lossy
collapse. We therefore assert both shape (target type) on the legal cells
AND a hard rejection on the four illegal ones, rather than relying on
downstream unit tests to notice.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,  # used as upper bound in parametric type lists
    GraphDistribution,
    GraphState,
)
from tmgg.models.base import _coerce_input_to, _coerce_output_to


# --- Fixtures -----------------------------------------------------------


def _no_edge_fill(d_ec: int) -> torch.Tensor:
    fill = torch.zeros(d_ec, dtype=torch.float32)
    fill[0] = 1.0
    return fill


@pytest.fixture
def small_state() -> GraphState:
    """Two graphs (2 nodes / 1 edge, 3 nodes / 2 edges) with a one-hot
    edge_class over (no-edge, edge) channels. Identical layout to the
    fixture used in ``tests/data/test_graph_types_conversions.py`` to
    keep the property anchors aligned."""
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
    return GraphState(
        num_nodes_per_graph=num_nodes,
        y=torch.zeros(2, 0),
        batch=batch,
        x_class=None,
        x_feat=None,
        edge_index=edge_index,
        edge_class=edge_class,
        edge_feat=None,
    )


@pytest.fixture
def small_dense_state(small_state: GraphState) -> DenseGraphState:
    return small_state.to_dense(edge_class_fill=_no_edge_fill(2))


@pytest.fixture
def small_dense_distribution(
    small_dense_state: DenseGraphState,
) -> DenseGraphDistribution:
    return small_dense_state.to_distribution()


@pytest.fixture
def small_distribution(small_state: GraphState) -> GraphDistribution:
    return small_state.to_distribution()


# --- 4x4 grid for _coerce_input_to --------------------------------------

# All four concrete types, identified by name to drive parametric tests.
_TYPES: list[type[GraphData]] = [
    GraphState,
    GraphDistribution,
    DenseGraphState,
    DenseGraphDistribution,
]
_TYPE_NAMES = [t.__name__ for t in _TYPES]


def _instance_for(
    cls: type[GraphData],
    *,
    sparse_state: GraphState,
    sparse_distribution: GraphDistribution,
    dense_state: DenseGraphState,
    dense_distribution: DenseGraphDistribution,
) -> GraphData:
    """Return the supplied fixture matching ``cls``."""
    if cls is GraphState:
        return sparse_state
    if cls is GraphDistribution:
        return sparse_distribution
    if cls is DenseGraphState:
        return dense_state
    if cls is DenseGraphDistribution:
        return dense_distribution
    raise AssertionError(f"unhandled fixture class {cls.__name__}")


def _is_state(cls: type[GraphData]) -> bool:
    return cls in (GraphState, DenseGraphState)


def _is_distribution(cls: type[GraphData]) -> bool:
    return cls in (GraphDistribution, DenseGraphDistribution)


@pytest.mark.parametrize("input_cls", _TYPES, ids=_TYPE_NAMES)
@pytest.mark.parametrize("target_cls", _TYPES, ids=_TYPE_NAMES)
def test_coerce_input_grid(
    input_cls: type[GraphData],
    target_cls: type[GraphData],
    small_state: GraphState,
    small_distribution: GraphDistribution,
    small_dense_state: DenseGraphState,
    small_dense_distribution: DenseGraphDistribution,
) -> None:
    """For each of the 16 cells, ``_coerce_input_to`` either returns an
    instance of the target type (12 legal cells) or raises ``TypeError``
    (4 distribution -> state collapse cells)."""
    data = _instance_for(
        input_cls,
        sparse_state=small_state,
        sparse_distribution=small_distribution,
        dense_state=small_dense_state,
        dense_distribution=small_dense_distribution,
    )

    is_lossy_collapse = _is_distribution(input_cls) and _is_state(target_cls)
    if is_lossy_collapse:
        with pytest.raises(TypeError, match="lossy collapse"):
            _coerce_input_to(data, target=target_cls)
    else:
        coerced = _coerce_input_to(data, target=target_cls)
        assert isinstance(coerced, target_cls), (
            f"_coerce_input_to({input_cls.__name__} -> {target_cls.__name__}) "
            f"returned {type(coerced).__name__}"
        )


def test_coerce_input_identity_returns_same_instance(
    small_state: GraphState,
) -> None:
    """The diagonal of the grid (input_cls == target_cls) returns the
    object unchanged, not a copy. Important so callers don't pay a
    structural conversion cost when nothing needs to change."""
    assert _coerce_input_to(small_state, target=GraphState) is small_state


# --- _coerce_output_to: same-content carrier flips ----------------------


def test_coerce_output_dense_distribution_to_sparse(
    small_dense_distribution: DenseGraphDistribution,
) -> None:
    out = _coerce_output_to(small_dense_distribution, target=GraphDistribution)
    assert isinstance(out, GraphDistribution)


def test_coerce_output_sparse_distribution_to_dense(
    small_distribution: GraphDistribution,
) -> None:
    out = _coerce_output_to(small_distribution, target=DenseGraphDistribution)
    assert isinstance(out, DenseGraphDistribution)


def test_coerce_output_dense_state_to_sparse(
    small_dense_state: DenseGraphState,
) -> None:
    out = _coerce_output_to(small_dense_state, target=GraphState)
    assert isinstance(out, GraphState)


def test_coerce_output_sparse_state_to_dense(small_state: GraphState) -> None:
    out = _coerce_output_to(small_state, target=DenseGraphState)
    assert isinstance(out, DenseGraphState)


def test_coerce_output_identity_returns_same_instance(
    small_dense_distribution: DenseGraphDistribution,
) -> None:
    assert (
        _coerce_output_to(small_dense_distribution, target=DenseGraphDistribution)
        is small_dense_distribution
    )


def test_coerce_output_rejects_content_collapse(
    small_dense_distribution: DenseGraphDistribution,
) -> None:
    """The output coercer does not perform content collapse — even on the
    same carrier — because that collapse must be an explicit modelling
    decision, not a return-type cast."""
    with pytest.raises(TypeError, match="unsupported"):
        _coerce_output_to(small_dense_distribution, target=DenseGraphState)


def test_coerce_output_rejects_content_lift(small_dense_state: DenseGraphState) -> None:
    """Symmetric guard: lift is lossless but is also not a return-type
    cast; supporting it would let one model claim distribution-content
    output while the architecture only emits state-content. We require
    the architecture to lift before returning instead."""
    with pytest.raises(TypeError, match="unsupported"):
        _coerce_output_to(small_dense_state, target=DenseGraphDistribution)

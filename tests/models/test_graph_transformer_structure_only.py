"""Parametrized tests for ``GraphTransformer.forward`` on structure-only batches.

Test rationale
--------------
Targets the inline X synthesis at ``transformer_model.py:944`` flagged by
``internal notes (not included)`` (v5) §2 as one
of the four independent consumer sites that hardcode the historical
``C_x = 2`` (``[no-node, node]``) shape. Per spec §5.5 this site MUST
synthesise via ``GraphData.synth_structure_only_x_class(node_mask,
self.output_dims_x_class)`` so the model honors its own authoritative
``C_x`` rather than the legacy two-channel default.

The spec's distinction matters: ``self.output_dims_x_class`` *is* ``C_x``,
whereas ``self.input_dims["X"]`` is ``X_in = C_x + F_x`` (an aggregate
that includes any extra-feature continuous channels). The synthesised
tensor is the ``X_class`` slice specifically; extras get concatenated
downstream.

Assumed starting state
~~~~~~~~~~~~~~~~~~~~~~
- A small structure-only ``GraphData`` with ``X_class = None``,
  ``E_class`` populated K=2, and a fully-true ``node_mask`` of shape
  ``(B, N)`` with ``B = 2``, ``N = 4``.
- ``GraphTransformer`` is built minimally: ``n_layers=1``,
  ``extra_features=None``, ``use_timestep=False``. ``input_dims["X"]``
  is set to ``c_x`` so the inner MLP's linear input width matches the
  synthesised tensor (after the spec's fix lands the synth emits
  ``c_x`` channels). ``output_dims["X"] = c_x`` and
  ``output_dims_x_class = c_x`` keep the head consistent.

Expected current state (TDD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``c_x = 1`` parametrized case is expected to fail today: the inline
``torch.stack([1.0 - node_ind, node_ind], dim=-1)`` synth at line 944
emits a width-2 tensor; the inner ``mlp_in_X`` is ``Linear(1, ...)``
(because ``input_dims["X"] = 1`` per the test's setup); a ``RuntimeError``
about matrix shape mismatch fires. The ``c_x = 2`` case currently passes
(it accidentally agrees with the hardcoded width).

The ``c_x = 3`` negative test references the new helper
``GraphData.synth_structure_only_x_class``, which Phase 1 of the
implementation plan introduces. Currently the inline synth still emits
width 2 with no raise, so this test fails (no exception). After Phase 4
lands the model fix, the helper is called with ``c_x = 3`` and raises.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    GraphData,
    GraphState,
)
from tmgg.models.digress.transformer_model import GraphTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def structure_only_batch() -> GraphState:
    """A structure-only batch built via ``GraphState.from_pyg_batch`` (B=2, N up to 4).

    Two graphs of differing sizes (triangle, square). ``x_class`` is
    ``None`` and ``edge_class`` is populated K=2 (one-hot) along the
    actual edges, matching the data-layer convention codified by the
    2026-04-15 unified-graph-features spec. Sparse output is the
    canonical entry shape for production diffusion modules.
    """
    torch.manual_seed(0)
    triangle = Data(
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long
        ),
        num_nodes=3,
    )
    square = Data(
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
        ),
        num_nodes=4,
    )
    pyg_batch = Batch.from_data_list([triangle, square])
    data = GraphState.from_pyg_batch(pyg_batch)
    # Sanity: confirm assumed starting state.
    assert data.x_class is None
    assert data.edge_class is not None
    assert data.edge_class.shape[-1] == 2
    return data


def _build_transformer(c_x: int) -> GraphTransformer:
    """Construct a minimal ``GraphTransformer`` with the given ``c_x``.

    ``input_dims["X"]`` and ``output_dims["X"]`` track ``c_x`` so the
    inner Linear layers expect the synthesised width. ``y`` is set to
    width 1 so the inner ``mlp_in_y`` (``Linear(1, ...)``) has
    well-defined inputs; the ``y`` tensor on the batch will carry a
    single-column zero vector to match.
    """
    return GraphTransformer(
        n_layers=1,
        input_dims={"X": c_x, "E": 2, "y": 1},
        hidden_mlp_dims={"X": 8, "E": 8, "y": 8},
        hidden_dims={"dx": 8, "de": 4, "dy": 4, "n_head": 2},
        output_dims={"X": c_x, "E": 2, "y": 1},
        extra_features=None,
        use_timestep=False,
        output_dims_x_class=c_x,
        output_dims_x_feat=None,
        output_dims_e_class=2,
        output_dims_e_feat=None,
    )


def _attach_y_column(data: GraphData) -> GraphData:
    """Replace ``y`` on ``data`` with a width-1 zero vector.

    The minimal transformer expects ``input_dims["y"] = 1``; the
    structure-only batch produced by ``from_pyg_batch`` has ``y`` of
    shape ``(B, 0)``. Replace it so the inner ``mlp_in_y`` receives a
    well-shaped input.
    """
    bs = int(data.num_nodes_per_graph.shape[0])
    return data.replace(y=torch.zeros(bs, 1))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c_x", [1, 2])
def test_forward_handles_structure_only_input(
    c_x: int,
    structure_only_batch: GraphState,
) -> None:
    """``forward`` runs on a structure-only batch with ``output_dims_x_class=c_x``.

    Test rationale: the model's internal X synth at line 944 must emit
    a tensor of width ``c_x`` so the downstream ``Linear(c_x, ...)``
    accepts it. Per spec §5.5 the synth uses ``self.output_dims_x_class``
    as the authoritative ``C_x`` source, NOT ``self.input_dims["X"]``
    (which is the X_in aggregate). Output ``X_class`` width must equal
    the configured ``c_x``.
    """
    torch.manual_seed(42)
    model = _build_transformer(c_x=c_x)
    data = _attach_y_column(structure_only_batch)
    pred = model(data, output_dense=True)
    assert isinstance(pred, DenseGraphDistribution)
    assert pred.X_class is not None
    assert pred.X_class.shape[-1] == c_x
    assert pred.E_class is not None
    assert pred.E_class.shape[-1] == 2


def test_forward_raises_for_c_x_geq_3(
    structure_only_batch: GraphState,
) -> None:
    """``forward`` raises when ``output_dims_x_class >= 3`` and ``X_class is None``.

    Test rationale: spec §3 regime table classifies ``C_x >= 3`` as
    real categorical content (atom types, etc.). There is no canonical
    synthesis from ``node_mask`` alone for this regime; per spec §5.1
    + §5.5 the model's synth path MUST raise via the canonical helper.
    """
    torch.manual_seed(42)
    model = _build_transformer(c_x=3)
    data = _attach_y_column(structure_only_batch)
    with pytest.raises(ValueError):
        model(data)

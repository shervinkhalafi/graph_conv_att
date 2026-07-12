# pyright: reportAttributeAccessIssue=false
"""Parametrized tests for ``CategoricalNoiseProcess`` over ``C_x`` regimes.

Test rationale
--------------
Targets the contract introduced by
``internal notes (not included)`` (v5). The spec's
problem statement (§2) and root cause (§4) identify a defect: the
synthesis of structure-only ``X_class`` lives in four independent consumer
sites, all hardcoding the historical ``C_x = 2`` (``[no-node, node]``)
shape. When a config sets ``noise_process.x_classes = 1`` (the
upstream-DiGress / GDPO SBM convention), deeper paths -- ``forward_sample``,
``_posterior_probabilities``, ``_posterior_probabilities_marginalised``,
``forward_pmf``, ``prior_pmf``, ``posterior_sample`` (direct + marginalised) --
still emit ``C_x = 2`` tensors against a ``C_x = 1`` noise process. The
first crash on Modal (Vignac SBM repro) was a ``CUDA bmm`` shape mismatch
inside ``_posterior_probabilities``.

Each parametrized test below pins the post-fix contract from §5: every
consumer call site MUST honor the noise process's authoritative
``self.x_classes`` so the returned ``DenseGraphState`` tensors have ``X_class``
of width exactly ``C_x``. The negative-path tests pin §3's regime table:
``C_x >= 3`` is real categorical (synthesis is undefined and MUST raise),
and ``E`` is never synthesised regardless of ``C_e``.

Assumed starting state
~~~~~~~~~~~~~~~~~~~~~~
- ``CategoricalNoiseProcess`` constructed with ``limit_distribution="uniform"``
  to avoid needing an empirical-marginal initialisation (no datamodule
  fixtures are available at this layer).
- Structure-only batches built via ``DenseGraphState.from_pyg_batch`` so
  ``X_class is None`` and ``E_class`` is populated K=2 from adjacency
  (per the 2026-04-15 unified-graph-features spec).
- Every test uses a fixed seed for determinism.

Expected current state (TDD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These tests are written **before** the spec's §5 implementation lands.
Most cases for ``c_x = 1`` are expected to fail today (the synthesis
defaults to ``C_x = 2`` and downstream ``compute_posterior_distribution``
complains about shape mismatch). The negative-path test
``test_x_synthesis_raises_for_c_x_geq_3`` references the new classmethod
``GraphData.synth_structure_only_x_class`` that does not exist yet; it
will fail with ``AttributeError``. The ``test_e_classes_required_no_default``
test pins the post-fix signature ``_read_categorical_e(data, e_classes)``
that does not exist yet; it will fail with ``TypeError`` once the spec
adds the required argument. All these failures are the contract.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from tmgg.data.datasets.graph_types import DenseGraphState
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    _read_categorical_e,
)
from tmgg.diffusion.schedule import NoiseSchedule

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def schedule() -> NoiseSchedule:
    """Cosine schedule with T=50; mirrors ``conftest.cosine_schedule``."""
    return NoiseSchedule(schedule_type="cosine_iddpm", timesteps=50)


@pytest.fixture()
def structure_only_batch() -> DenseGraphState:
    """A small structure-only batch built via ``DenseGraphState.from_pyg_batch``.

    Contains two graphs of differing sizes (triangle, square). The
    resulting ``DenseGraphState`` has ``X_class = None`` and ``E_class``
    populated K=2 from the adjacency, matching the convention codified
    in the 2026-04-15 unified-graph-features spec. ``node_mask`` reflects
    actual node counts (3 and 4) padded to ``n_max = 4``.
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
    data = DenseGraphState.from_pyg_batch(pyg_batch)
    # Sanity: confirm assumed starting state.
    assert data.X_class is None, "from_pyg_batch must emit X_class=None"
    assert data.E_class is not None
    assert data.E_class.shape[-1] == 2, "from_pyg_batch yields K=2 E_class"
    return data


def _make_process(
    schedule: NoiseSchedule, c_x: int, c_e: int
) -> CategoricalNoiseProcess:
    """Build a ``CategoricalNoiseProcess`` with the parametrized class widths.

    ``limit_distribution="uniform"`` constructs stationary PMFs immediately
    so no ``initialize_from_data`` call is needed. The process is
    immediately ready for ``forward_sample`` / posterior queries.
    """
    return CategoricalNoiseProcess(
        schedule=schedule,
        x_classes=c_x,
        e_classes=c_e,
        limit_distribution="uniform",
    )


def _t_pair(batch: DenseGraphState) -> tuple[Tensor, Tensor]:
    """Return ``(t, s)`` integer timesteps with ``s = t - 1``.

    ``t = 25`` is mid-schedule for ``T = 50``, deep enough that posterior
    contributions from both ``q`` and the prior are non-trivial.
    """
    bs = batch.node_mask.shape[0]
    t = torch.full((bs,), 25, dtype=torch.long)
    s = t - 1
    return t, s


# ---------------------------------------------------------------------------
# Positive-path tests: parametrized over c_x
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c_x", [1, 2])
def test_forward_sample_preserves_c_x_shape(
    c_x: int,
    schedule: NoiseSchedule,
    structure_only_batch: DenseGraphState,
) -> None:
    """``forward_sample`` returns ``z_t`` with ``X_class.shape[-1] == c_x``.

    Test rationale: the forward path (``_apply_noise``) reads X via
    ``_read_categorical_x``. Per spec §5.2 the call site MUST pass the
    process's authoritative ``self.x_classes``; the result's ``X_class``
    width must therefore equal ``c_x``, never the historical default.
    """
    proc = _make_process(schedule, c_x=c_x, c_e=2)
    t = torch.full((structure_only_batch.node_mask.shape[0],), 25, dtype=torch.long)
    result = proc.forward_sample(structure_only_batch, t)
    assert result.z_t.X_class is not None
    assert result.z_t.X_class.shape[-1] == c_x
    assert result.z_t.E_class is not None
    assert result.z_t.E_class.shape[-1] == 2


@pytest.mark.parametrize("c_x", [1, 2])
def test_posterior_probabilities_preserves_c_x_shape(
    c_x: int,
    schedule: NoiseSchedule,
    structure_only_batch: DenseGraphState,
) -> None:
    """``_posterior_probabilities`` honors ``c_x`` against a structure-only batch.

    Test rationale: this is the originally-crashing site (Modal Vignac
    SBM repro). The bug-trigger condition is passing the structure-only
    batch as the ``x0_param`` argument so its ``X_class is None`` forces
    a synth call inside the posterior. Per spec §5.2 the synth must
    yield a ``c_x``-wide tensor matching the kernel widths.
    """
    proc = _make_process(schedule, c_x=c_x, c_e=2)
    t, s = _t_pair(structure_only_batch)
    z_t = proc.forward_sample(structure_only_batch, t).z_t
    result = proc._posterior_probabilities(z_t, structure_only_batch, t, s)
    assert result.X_class is not None
    assert result.X_class.shape[-1] == c_x
    assert result.E_class is not None
    assert result.E_class.shape[-1] == 2


@pytest.mark.parametrize("c_x", [1, 2])
def test_posterior_probabilities_marginalised_preserves_c_x_shape(
    c_x: int,
    schedule: NoiseSchedule,
    structure_only_batch: DenseGraphState,
) -> None:
    """Marginalised posterior preserves ``c_x`` shape on a structure-only batch.

    Test rationale: the marginalised variant has its own ``_read_*``
    call sites (lines 1334, 1336 in ``noise_process.py``). It's the
    upstream DiGress sampling path; a defect here corrupts every
    sampling step. Same contract as the direct posterior.
    """
    proc = _make_process(schedule, c_x=c_x, c_e=2)
    t, s = _t_pair(structure_only_batch)
    z_t = proc.forward_sample(structure_only_batch, t).z_t
    result = proc._posterior_probabilities_marginalised(z_t, structure_only_batch, t, s)
    assert result.X_class is not None
    assert result.X_class.shape[-1] == c_x
    assert result.E_class is not None
    assert result.E_class.shape[-1] == 2


@pytest.mark.parametrize("c_x", [1, 2])
def test_forward_pmf_preserves_c_x_shape(
    c_x: int,
    schedule: NoiseSchedule,
    structure_only_batch: DenseGraphState,
) -> None:
    """``forward_pmf`` honors ``c_x`` on a structure-only batch.

    Test rationale: ``forward_pmf`` is the next-crash site flagged in
    spec §2 ("deeper paths carry the same defect"). It reads X via
    ``_read_categorical_x`` and is used by analytic-KL VLB estimators.
    """
    proc = _make_process(schedule, c_x=c_x, c_e=2)
    t = torch.full((structure_only_batch.node_mask.shape[0],), 25, dtype=torch.long)
    result = proc.forward_pmf(structure_only_batch, t)
    assert result.X_class is not None
    assert result.X_class.shape[-1] == c_x
    assert result.E_class is not None
    assert result.E_class.shape[-1] == 2


@pytest.mark.parametrize("c_x", [1, 2])
def test_prior_pmf_preserves_c_x_shape(
    c_x: int,
    schedule: NoiseSchedule,
    structure_only_batch: DenseGraphState,
) -> None:
    """``prior_pmf`` honors ``c_x`` when tiled to a node mask.

    Test rationale: ``prior_pmf`` lifts the stationary PMF (which has
    width ``c_x`` by construction) onto a per-position tensor. The
    output width must equal ``c_x``; this verifies no implicit C_x=2
    assumption sneaks in via tiling code.
    """
    proc = _make_process(schedule, c_x=c_x, c_e=2)
    result = proc.prior_pmf(structure_only_batch.node_mask)
    assert result.X_class is not None
    assert result.X_class.shape[-1] == c_x
    assert result.E_class is not None
    assert result.E_class.shape[-1] == 2


@pytest.mark.parametrize("c_x", [1, 2])
def test_posterior_sample_direct_preserves_c_x_shape(
    c_x: int,
    schedule: NoiseSchedule,
    structure_only_batch: DenseGraphState,
) -> None:
    """``posterior_sample`` (direct path) yields ``c_x``-wide samples.

    Test rationale: ``posterior_sample`` runs ``_posterior_probabilities``
    then samples one-hot indices via ``F.one_hot(..., num_classes=...)``.
    The ``num_classes`` argument must come from ``self.x_classes`` (it
    does, by inspection of source line 1287); we verify the *output*
    width to catch a future regression where the hardcoded width slips
    back in.
    """
    torch.manual_seed(42)
    proc = _make_process(schedule, c_x=c_x, c_e=2)
    t, s = _t_pair(structure_only_batch)
    z_t = proc.forward_sample(structure_only_batch, t).z_t
    result = proc.posterior_sample(z_t, structure_only_batch, t, s)
    assert result.X_class is not None
    assert result.X_class.shape[-1] == c_x
    assert result.E_class is not None
    assert result.E_class.shape[-1] == 2


@pytest.mark.parametrize("c_x", [1, 2])
def test_posterior_sample_marginalised_preserves_c_x_shape(
    c_x: int,
    schedule: NoiseSchedule,
    structure_only_batch: DenseGraphState,
) -> None:
    """``posterior_sample_marginalised`` yields ``c_x``-wide samples.

    Test rationale: this is the upstream DiGress sampling path
    (``posterior_sample_from_model_output`` routes here per source line
    1430). A regression here would corrupt every reverse-step sample
    during evaluation.
    """
    torch.manual_seed(42)
    proc = _make_process(schedule, c_x=c_x, c_e=2)
    t, s = _t_pair(structure_only_batch)
    z_t = proc.forward_sample(structure_only_batch, t).z_t
    result = proc.posterior_sample_marginalised(z_t, structure_only_batch, t, s)
    assert result.X_class is not None
    assert result.X_class.shape[-1] == c_x
    assert result.E_class is not None
    assert result.E_class.shape[-1] == 2


# ---------------------------------------------------------------------------
# Negative-path tests: regime boundaries
# ---------------------------------------------------------------------------


def test_e_synthesis_raises_with_clear_message(
    structure_only_batch: DenseGraphState,
) -> None:
    """``_read_categorical_e`` raises with the asymmetry-explanation message.

    Test rationale: spec §3 codifies the X-vs-E asymmetry rule: X *can*
    be synthesised because every valid node trivially exists as its
    only class, but E *cannot* be synthesised because edges are an
    adjacency property orthogonal to ``node_mask``. Per spec §5.2 the
    helper MUST raise ``ValueError`` with a message naming the
    asymmetry ("E synthesis" / "adjacency") regardless of ``C_e``.

    Construction note: we manually build a ``DenseGraphState`` with
    ``E_class = None`` and a populated ``E_feat`` so the ``__post_init__``
    "at least one E_*" invariant is satisfied. This isolates the
    helper's contract from the dataclass invariant.
    """
    bs, n = structure_only_batch.node_mask.shape
    e_feat = torch.zeros(bs, n, n, 1)
    data_no_eclass = DenseGraphState(
        num_nodes_per_graph=structure_only_batch.num_nodes_per_graph,
        y=torch.zeros(bs, 0),
        E_feat=e_feat,
    )
    with pytest.raises(ValueError, match="(?i)E synthesis|adjacency"):
        _read_categorical_e(data_no_eclass, e_classes=2)  # pyright: ignore[reportCallIssue]


def test_x_synthesis_raises_for_c_x_geq_3(
    structure_only_batch: DenseGraphState,
) -> None:
    """``DenseGraphState.synth_structure_only_x_class`` raises for ``c_x >= 3``.

    Test rationale: spec §3 regime table: ``C_x >= 3`` is real categorical
    content (atom types, etc.). There is no canonical "structure-only
    C_x=3" interpretation; synthesis MUST raise. The classmethod is
    introduced in spec §5.1 (Phase 1 of the implementation plan).
    """
    with pytest.raises(ValueError):
        DenseGraphState.synth_structure_only_x_class(
            structure_only_batch.node_mask, c_x=3
        )


def test_e_classes_required_no_default(
    structure_only_batch: DenseGraphState,
) -> None:
    """``_read_categorical_e`` must require ``e_classes`` explicitly.

    Test rationale: spec §5.2 strips defaulted arguments from the helper
    signatures so type-checker / runtime catches missing arguments.
    Currently the helper has no ``e_classes`` argument at all (signature
    is ``_read_categorical_e(data)``); after Phase 2 lands, calling it
    without the arg MUST raise ``TypeError`` ("missing 1 required
    positional argument: 'e_classes'").

    Until Phase 2 lands this test will fail (the call succeeds because
    no positional arg is required). The failure encodes the contract:
    "no defaulted ``e_classes``" is part of what we are buying.
    """
    with pytest.raises(TypeError):
        _read_categorical_e(structure_only_batch)  # pyright: ignore[reportCallIssue]

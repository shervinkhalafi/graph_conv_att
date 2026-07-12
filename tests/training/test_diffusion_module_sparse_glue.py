# pyright: reportPrivateImportUsage=false
# torch's runtime API exposes ``zeros``, ``arange``, ``tensor`` etc. at
# module top level; the stub does not re-export them, so basedpyright
# flags every use. Same precedent as ``tmgg.training.losses`` tests.
"""Sparse-default refactor (Phase 5.2): DiffusionModule glue smoke tests.

Rationale
---------
Wave 3-A re-types ``DiffusionModule.training_step`` against ``GraphState``
(the carrier emitted by the Wave 2-A datamodules). Inside the step the
batch is densified to ``DenseGraphState`` once, then the sparse
``noise_process.sample`` produces ``z_t`` and the model is asked for a
``DenseGraphDistribution`` via ``output_dense=True`` so the legacy
diagnostic / VLB / loss helpers see uppercase fields.

These tests pin two contracts:

1. ``training_step`` consumes a sparse ``GraphState`` batch end-to-end and
   returns a finite zero-dim loss tensor. The actual model is stubbed to
   return a fixed ``DenseGraphDistribution`` whose ``E_class`` carries
   the same one-hot pattern as the densified target — under
   cross-entropy with no gradient flow the loss is well-defined and
   finite.
2. The ``GraphData`` -> ``GraphState`` retype on ``transfer_batch_to_device``
   in ``BaseGraphModule`` is a no-op at the call site (``GraphState.to``
   round-trips a sparse batch onto the requested device).

The goal is to exercise the type-checked contract at the
DiffusionModule call boundary without instantiating one of the real
production models, which are being refactored concurrently in Wave 3-B/C/D.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphDistribution,
    GraphState,
)
from tmgg.diffusion.noise_process import NoiseProcess, NoisedBatch
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.models.base import GraphModel
from tmgg.training.lightning_modules.diffusion_module import DiffusionModule
from tmgg.utils.noising.size_distribution import SizeDistribution


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _build_sparse_state(
    num_nodes: list[int] | None = None,
    *,
    num_edge_classes: int = 2,
) -> GraphState:
    """Build a small sparse ``GraphState`` with one-hot ``edge_class``.

    Two graphs of sizes 2 and 3 by default, with one undirected edge
    in each (active class = channel 1). The complete-pair structure is
    *not* materialised here — ``GraphState`` carries only active edges
    per its sparse invariant.
    """
    nn_list = num_nodes if num_nodes is not None else [2, 3]
    bs = len(nn_list)
    nm = torch.tensor(nn_list, dtype=torch.long)
    batch = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(nn_list)]
    )

    # Active edges: one undirected pair per graph, both directions.
    # Graph 0 (sizes [2,3]): nodes 0,1 → edge (0,1) and (1,0).
    # Graph 1: global node ids 2,3,4 → edge (2,3) and (3,2).
    cum = [0]
    for n in nn_list:
        cum.append(cum[-1] + n)
    src: list[int] = []
    dst: list[int] = []
    for i in range(bs):
        a = cum[i]
        b = cum[i] + 1
        src.extend([a, b])
        dst.extend([b, a])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    e_class = torch.zeros(edge_index.shape[1], num_edge_classes)
    # Class 1 = "edge present" by convention.
    e_class[:, 1] = 1.0

    return GraphState(
        num_nodes_per_graph=nm,
        y=torch.zeros(bs, 0),
        batch=batch,
        x_class=None,
        x_feat=None,
        edge_index=edge_index,
        edge_class=e_class,
        edge_feat=None,
    )


class _StubModel(GraphModel):
    """Stub ``GraphModel`` returning a fixed dense distribution.

    The forward signature accepts the new ``output_dense`` kwarg the
    Phase 5.2 ``training_step`` passes, plus the legacy ``data, t``
    positional/keyword pair so the stub is contract-compatible with
    callers that still use the legacy form.

    The fixed prediction is a ``DenseGraphDistribution`` whose
    ``E_class`` matches a hand-set one-hot pattern — under
    cross-entropy this gives a finite, easily-verifiable loss without
    requiring the stub to perform any real inference.
    """

    def __init__(
        self,
        num_nodes_per_graph: Tensor,
        num_edge_classes: int = 2,
        num_node_classes: int = 1,
    ) -> None:
        super().__init__()
        self._num_nodes_per_graph = num_nodes_per_graph
        self._num_edge_classes = num_edge_classes
        self._num_node_classes = num_node_classes
        # A trivial trainable parameter so the LightningModule's parameter
        # iteration sees something non-empty.
        self._weight = torch.nn.Parameter(torch.zeros(1))

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        data: GraphData,
        t: Tensor | None = None,
        *,
        output_dense: bool = False,
    ) -> GraphData:
        del data, t  # stub: input is ignored
        bs = int(self._num_nodes_per_graph.shape[0])
        n_max = int(self._num_nodes_per_graph.max().item())
        d_ec = self._num_edge_classes
        d_xc = self._num_node_classes
        # Uniform over classes; CE collapses to log(d_ec) per-position.
        e_class = torch.full(
            (bs, n_max, n_max, d_ec),
            1.0 / float(d_ec),
            dtype=torch.float32,
        )
        # Diagonal must be zeroed per the dense invariant.
        diag = torch.arange(n_max)
        e_class[:, diag, diag, :] = 0.0

        # Uniform per-node class distribution. Populated so the
        # diffusion module's ``_resolve_x_classes_for_loss`` defensive
        # fallback can read C_x from the prediction in test contexts
        # that do not use a real ``CategoricalNoiseProcess``.
        x_class = torch.full(
            (bs, n_max, d_xc),
            1.0 / float(d_xc),
            dtype=torch.float32,
        )

        dense_pred = DenseGraphDistribution(
            num_nodes_per_graph=self._num_nodes_per_graph,
            y=torch.zeros(bs, 0),
            X_class=x_class,
            X_feat=None,
            E_class=e_class,
            E_feat=None,
        )
        if output_dense:
            return dense_pred
        return dense_pred.to_sparse()

    def get_config(self) -> dict[str, Any]:
        return {"stub": True}


class _StubCategoricalNoiseProcess(NoiseProcess):
    """Minimal categorical-flavoured stub for the DiffusionModule wiring.

    Implements only the surface area exercised by ``training_step``: the
    new sparse ``sample(GraphState, t) -> GraphState`` entry point plus
    the legacy ``forward_sample`` / ``posterior_sample`` / ``sample_prior``
    abstracts (no-op implementations sufficient for the abstract-base
    contract).
    """

    fields = frozenset({"E_class"})

    def __init__(self, schedule: NoiseSchedule, num_edge_classes: int = 2) -> None:
        super().__init__()
        self._schedule = schedule
        self._num_edge_classes = num_edge_classes

    @property
    def timesteps(self) -> int:
        return int(self._schedule.timesteps)

    def process_state_condition_vector(self, t: Tensor) -> Tensor:
        # Normalised time as a scalar conditioning vector.
        return (t.float() / float(self.timesteps)).unsqueeze(-1)

    def sample(self, z_0: GraphState, t: Tensor) -> GraphState:  # pyright: ignore[reportIncompatibleMethodOverride]
        del t  # stub: identity (no actual noising)
        return z_0

    # The abstract methods below are never called by ``training_step`` in
    # this test — provided as no-op implementations to satisfy the ABC.

    def forward_sample(self, x_0: GraphData, t: Tensor) -> NoisedBatch:  # pragma: no cover - unused in this test
        raise NotImplementedError("stub forward_sample is not exercised in this test")

    def posterior_sample(  # pragma: no cover - unused in this test
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        raise NotImplementedError("stub posterior_sample is not exercised in this test")

    def sample_prior(self, node_mask: Tensor) -> GraphData:  # pragma: no cover - unused in this test
        raise NotImplementedError("stub sample_prior is not exercised in this test")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_training_step_consumes_graphstate_and_returns_finite_loss() -> None:
    """``training_step(GraphState, batch_idx)`` produces a finite scalar loss.

    The dataloader emits ``GraphState`` post-Wave 2-A; this test confirms
    the Wave 3-A re-typed entry point threads the sparse batch all the
    way through to a finite cross-entropy loss without needing a
    production model.
    """
    state = _build_sparse_state(num_nodes=[2, 3], num_edge_classes=2)
    schedule = NoiseSchedule("linear_ddpm", timesteps=10)
    noise_process = _StubCategoricalNoiseProcess(schedule, num_edge_classes=2)
    model = _StubModel(state.num_nodes_per_graph, num_edge_classes=2)

    module = DiffusionModule(
        model=model,
        noise_process=noise_process,
        sampler=None,
        noise_schedule=schedule,
        evaluator=None,
        loss_type="cross_entropy",
        num_nodes=3,
        eval_every_n_steps=1,
    )
    # Legacy ``setup`` populates ``_size_distribution`` from the
    # datamodule; tests bypass setup, so inject a stub directly.
    module._size_distribution = SizeDistribution(  # pyright: ignore[reportPrivateUsage]
        sizes=(2, 3), counts=(1, 1), max_size=3
    )
    module._cur_bs = int(state.num_nodes_per_graph.shape[0])  # pyright: ignore[reportPrivateUsage]

    loss = module.training_step(state, batch_idx=0)

    assert isinstance(loss, Tensor)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_transfer_batch_to_device_round_trips_graphstate_on_cpu() -> None:
    """``transfer_batch_to_device`` on CPU is an identity-on-tensors call.

    The Wave 3-A retype on ``BaseGraphModule.transfer_batch_to_device``
    forwards to ``GraphState.to(device)``. Calling it with a CPU device
    must return a ``GraphState`` whose tensors live on the same device
    and carry equal content.
    """
    state = _build_sparse_state(num_nodes=[2, 3], num_edge_classes=2)
    schedule = NoiseSchedule("linear_ddpm", timesteps=2)
    noise_process = _StubCategoricalNoiseProcess(schedule, num_edge_classes=2)
    model = _StubModel(state.num_nodes_per_graph, num_edge_classes=2)
    module = DiffusionModule(
        model=model,
        noise_process=noise_process,
        sampler=None,
        noise_schedule=schedule,
        evaluator=None,
        loss_type="cross_entropy",
        num_nodes=3,
        eval_every_n_steps=1,
    )

    moved = module.transfer_batch_to_device(state, torch.device("cpu"), 0)
    assert isinstance(moved, GraphState)
    assert moved.num_nodes_per_graph.device.type == "cpu"
    assert moved.edge_index.device.type == "cpu"
    assert torch.equal(moved.num_nodes_per_graph, state.num_nodes_per_graph)
    assert torch.equal(moved.edge_index, state.edge_index)


def test_no_edge_fill_helper_returns_channel_zero_one_hot() -> None:
    """``_no_edge_fill_for`` returns the canonical channel-0 one-hot fill.

    ``GraphState.to_dense(edge_class_fill=...)`` requires a ``(d_ec,)``
    fill vector when ``edge_class`` is populated. The helper returns the
    DiGress-canonical no-edge encoding (all mass on channel 0) so the
    densification path matches the upstream convention.
    """
    from tmgg.training.lightning_modules.diffusion_module import _no_edge_fill_for

    state = _build_sparse_state(num_nodes=[2, 3], num_edge_classes=4)
    fill = _no_edge_fill_for(state)
    assert fill is not None
    assert fill.shape == (4,)
    assert torch.allclose(fill, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_no_edge_fill_helper_returns_none_for_continuous_only_state() -> None:
    """No edge_class → no fill required; helper returns ``None``.

    Passing ``None`` to ``GraphState.to_dense(edge_class_fill=None)``
    short-circuits the categorical-fill path; the dense state then
    carries only continuous edge content.
    """
    from tmgg.training.lightning_modules.diffusion_module import _no_edge_fill_for

    bs = 2
    nm = torch.tensor([2, 3], dtype=torch.long)
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 3, 4],
            [1, 0, 3, 2, 4, 3],
        ],
        dtype=torch.long,
    )
    edge_feat = torch.ones(edge_index.shape[1], 1)

    state = GraphState(
        num_nodes_per_graph=nm,
        y=torch.zeros(bs, 0),
        batch=batch,
        x_class=None,
        x_feat=None,
        edge_index=edge_index,
        edge_class=None,
        edge_feat=edge_feat,
    )
    assert _no_edge_fill_for(state) is None

"""End-to-end smoke tests for the Wave 4-A sparse-default Sampler.

Test rationale
--------------
Wave 4-A retypes the Sampler's public surface to produce and consume
the new sparse-default carriers (:class:`GraphState` for warm-start
input and per-graph output). The legacy noise-process API still
consumes the dense ``GraphData``-shaped form internally, so the
sampler bridges between the two carriers across every reverse step.

This test file pins three things:

* ``DiffusionState`` accepts a sparse :class:`GraphState` warm start
  and rejects non-state types with a clear error.
* ``Sampler.sample`` calls the model with ``output_dense=True`` so the
  model emits a :class:`DenseGraphDistribution`, matching the Wave 3
  forward contract.
* The reverse loop returns a list of per-graph
  :class:`GraphState` objects (one per requested graph) with sane
  per-graph node counts.

The test uses fully stubbed model and noise-process objects so it
does not depend on the legacy ``noise_process.py`` runtime path being
fixed. That path is the responsibility of Wave 5 cleanup; pre-Wave-5
the live :class:`CategoricalNoiseProcess` cannot construct its
return values because ``_categorical_graphdata`` instantiates the
abstract :class:`GraphData` directly. The stubs side-step that
legacy crash so the public-surface contract can be pinned on its own.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphState,
)
from tmgg.diffusion.noise_process import NoiseProcess
from tmgg.diffusion.sampler import (
    DiffusionState,
    Sampler,
)
from tmgg.models.base import GraphModel  # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sparse_state(
    *,
    bs: int = 2,
    n: int = 3,
    dx: int = 1,
    de: int = 2,
    device: torch.device | None = None,
) -> GraphState:
    """Build a small batched :class:`GraphState` for sampler tests.

    The graph carries ``X_class`` of width ``dx`` (canonical
    structure-only encoding for ``dx == 1``) and ``edge_class`` of
    width ``de`` with channel 0 marking "no edge" (DiGress canonical
    encoding). Each graph has the same node count ``n`` and zero edges,
    so the sparse ``edge_index`` is empty -- enough for the sparse
    invariants to hold while staying tiny.
    """
    dev = device if device is not None else torch.device("cpu")
    num_nodes_per_graph = torch.full((bs,), n, dtype=torch.long, device=dev)
    batch = torch.arange(bs, device=dev).repeat_interleave(n)
    x_class = torch.zeros(bs * n, dx, dtype=torch.float32, device=dev)
    if dx >= 1:
        x_class[:, 0] = 1.0
    edge_index = torch.empty(2, 0, dtype=torch.long, device=dev)
    edge_class = torch.empty(0, de, dtype=torch.float32, device=dev)
    y = torch.zeros(bs, 0, dtype=torch.float32, device=dev)
    return GraphState(
        num_nodes_per_graph=num_nodes_per_graph,
        y=y,
        batch=batch,
        x_class=x_class,
        x_feat=None,
        edge_index=edge_index,
        edge_class=edge_class,
        edge_feat=None,
    )


def _make_dense_state(
    *, bs: int, n: int, dx: int, de: int, device: torch.device
) -> DenseGraphState:
    """Build a small batched :class:`DenseGraphState` (legacy form).

    ``X_class`` is filled at channel 0 (structure-only) and ``E_class``
    is one-hot at channel 0 ("no edge") at every off-diagonal pair.
    """
    num_nodes_per_graph = torch.full((bs,), n, dtype=torch.long, device=device)
    y = torch.zeros(bs, 0, dtype=torch.float32, device=device)
    x_class = torch.zeros(bs, n, dx, dtype=torch.float32, device=device)
    if dx >= 1:
        x_class[..., 0] = 1.0
    e_class = torch.zeros(bs, n, n, de, dtype=torch.float32, device=device)
    if de >= 1:
        e_class[..., 0] = 1.0
    diag = torch.arange(n, device=device)
    e_class[:, diag, diag, :] = 0.0
    return DenseGraphState(
        num_nodes_per_graph=num_nodes_per_graph,
        y=y,
        X_class=x_class,
        X_feat=None,
        E_class=e_class,
        E_feat=None,
    )


# ---------------------------------------------------------------------------
# Stub model and noise process
# ---------------------------------------------------------------------------


class _StubModel(GraphModel):
    """Stub model that records each forward call and returns a uniform PMF.

    The forward signature accepts the new ``output_dense`` keyword and
    returns a :class:`DenseGraphDistribution` so the sampler's call site
    is exercised without pulling in any real architecture.
    """

    def __init__(self, *, dx: int, de: int) -> None:
        super().__init__()
        self.dx = dx
        self.de = de
        self.calls: list[tuple[type, bool]] = []

    def get_config(self) -> dict[str, Any]:
        return {"dx": self.dx, "de": self.de}

    def forward(  # type: ignore[override]
        self,
        data: GraphData,
        t: Tensor | None = None,
        *,
        output_dense: bool = False,
    ) -> DenseGraphDistribution:
        _ = t
        self.calls.append((type(data), output_dense))
        if not output_dense:
            raise AssertionError(
                "Sampler must call the model with output_dense=True per the "
                "Wave 4-A contract."
            )
        # Coerce to a DenseGraphState and return a uniform distribution
        # over the declared class widths. The sampler does not require
        # any specific values; we just need a valid carrier.
        if isinstance(data, GraphState):
            num_nodes_per_graph = data.num_nodes_per_graph
            n = int(num_nodes_per_graph.max().item())
            bs = int(num_nodes_per_graph.shape[0])
            device = data.batch.device
            y = data.y
        elif isinstance(data, DenseGraphState):
            num_nodes_per_graph = data.num_nodes_per_graph
            bs = int(num_nodes_per_graph.shape[0])
            n = int(data.E_class.shape[1]) if data.E_class is not None else (
                int(data.X_class.shape[1])
                if data.X_class is not None
                else int(num_nodes_per_graph.max().item())
            )
            device = num_nodes_per_graph.device
            y = data.y
        else:
            raise TypeError(
                f"_StubModel.forward: unsupported carrier {type(data).__name__}"
            )
        x_dist = torch.full(
            (bs, n, self.dx),
            1.0 / self.dx,
            dtype=torch.float32,
            device=device,
        )
        e_dist = torch.full(
            (bs, n, n, self.de),
            1.0 / self.de,
            dtype=torch.float32,
            device=device,
        )
        return DenseGraphDistribution(
            num_nodes_per_graph=num_nodes_per_graph,
            y=y,
            X_class=x_dist,
            X_feat=None,
            E_class=e_dist,
            E_feat=None,
        )


class _StubNoiseProcess(NoiseProcess):
    """Stub noise process with a fixed timestep budget and identity hooks.

    Every per-step hook returns a fresh dense state with the same shape
    as ``z_t``; the concrete values do not matter for the surface tests.
    The sampler only checks that the per-step contract is honoured.
    """

    fields = frozenset({"X_class", "E_class"})

    def __init__(self, *, timesteps: int, dx: int, de: int) -> None:
        super().__init__()
        self._timesteps = timesteps
        self._dx = dx
        self._de = de

    @property
    def timesteps(self) -> int:
        return self._timesteps

    def process_state_condition_vector(self, t: Tensor) -> Tensor:
        return t.float() / self._timesteps

    def model_output_to_posterior_parameter(
        self, model_output: GraphData
    ) -> GraphData:
        return model_output

    def finalize_sample(self, z_0: GraphData) -> GraphData:
        return z_0

    def sample_prior(self, node_mask: Tensor) -> GraphData:
        bs, n = node_mask.shape
        device = node_mask.device
        num_nodes_per_graph = node_mask.sum(dim=-1).long()
        x_class = torch.zeros(bs, n, self._dx, dtype=torch.float32, device=device)
        if self._dx >= 1:
            x_class[..., 0] = 1.0
        e_class = torch.zeros(
            bs, n, n, self._de, dtype=torch.float32, device=device
        )
        if self._de >= 1:
            e_class[..., 0] = 1.0
        diag = torch.arange(n, device=device)
        e_class[:, diag, diag, :] = 0.0
        y = torch.zeros(bs, 0, dtype=torch.float32, device=device)
        return DenseGraphState(
            num_nodes_per_graph=num_nodes_per_graph,
            y=y,
            X_class=x_class,
            X_feat=None,
            E_class=e_class,
            E_feat=None,
        )

    def forward_sample(self, x_0: GraphData, t: Tensor) -> Any:
        raise NotImplementedError("_StubNoiseProcess does not support forward sampling")

    def posterior_sample(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        # Identity reverse step: pass z_t through unchanged. The
        # sampler's downstream symmetry guard reads ``E_class`` and
        # tolerates any symmetric tensor, which the prior fill
        # satisfies.
        _ = x0_param, t, s
        return z_t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiffusionStateAcceptsGraphState:
    """``DiffusionState.graph`` must be a sparse :class:`GraphState`."""

    def test_constructs_from_graph_state(self) -> None:
        graph = _make_sparse_state(bs=2, n=3)
        state = DiffusionState(graph=graph, t=5, max_t=5)
        assert state.t == 5
        assert state.max_t == 5
        assert state.graph is graph

    def test_rejects_non_graph_state(self) -> None:
        # A DenseGraphState is a state-content type but not the sparse
        # carrier the new contract demands; the type-tag check must
        # fire here.
        dense = _make_dense_state(
            bs=2, n=3, dx=1, de=2, device=torch.device("cpu")
        )
        with pytest.raises(TypeError, match="must be a GraphState"):
            DiffusionState(graph=dense, t=0, max_t=5)  # type: ignore[arg-type]

    def test_rejects_y_shape_mismatch(self) -> None:
        # Construct a sparse state then craft a mismatched y to force
        # the DiffusionState invariant. The sparse helper builds
        # ``y`` of shape ``(bs, 0)``; replace with ``(bs+1, 0)`` to
        # break the alignment.
        graph = _make_sparse_state(bs=2, n=3)
        bad_y = torch.zeros(3, 0, dtype=torch.float32)
        bad = GraphState(
            num_nodes_per_graph=graph.num_nodes_per_graph,
            y=bad_y,
            batch=graph.batch,
            x_class=graph.x_class,
            x_feat=None,
            edge_index=graph.edge_index,
            edge_class=graph.edge_class,
            edge_feat=None,
        )
        with pytest.raises(ValueError, match="must align with batch size"):
            DiffusionState(graph=bad, t=0, max_t=5)


class TestSamplerSurface:
    """End-to-end surface invariants of :meth:`Sampler.sample`."""

    def test_returns_list_of_graph_state(self) -> None:
        bs = 2
        n = 3
        dx = 1
        de = 2
        process = _StubNoiseProcess(timesteps=2, dx=dx, de=de)
        model = _StubModel(dx=dx, de=de)
        sampler = Sampler(assert_symmetric_e=False)

        result = sampler.sample(
            model=model,
            noise_process=process,
            num_graphs=bs,
            num_nodes=n,
            device=torch.device("cpu"),
        )

        assert isinstance(result, list)
        assert len(result) == bs
        for graph in result:
            assert isinstance(graph, GraphState)
            assert int(graph.num_nodes_per_graph.sum().item()) == n

    def test_model_called_with_output_dense_true(self) -> None:
        bs = 2
        n = 3
        timesteps = 3
        dx = 1
        de = 2
        process = _StubNoiseProcess(timesteps=timesteps, dx=dx, de=de)
        model = _StubModel(dx=dx, de=de)
        sampler = Sampler(assert_symmetric_e=False)

        sampler.sample(
            model=model,
            noise_process=process,
            num_graphs=bs,
            num_nodes=n,
            device=torch.device("cpu"),
        )

        # ``timesteps`` reverse steps means ``timesteps`` model calls.
        assert len(model.calls) == timesteps
        for _, output_dense in model.calls:
            assert output_dense is True

    def test_warm_start_consumes_graph_state(self) -> None:
        bs = 2
        n = 3
        timesteps = 2
        dx = 1
        de = 2
        process = _StubNoiseProcess(timesteps=timesteps, dx=dx, de=de)
        model = _StubModel(dx=dx, de=de)
        sampler = Sampler(assert_symmetric_e=False)

        sparse_warm = _make_sparse_state(bs=bs, n=n, dx=dx, de=de)
        warm_start = DiffusionState(graph=sparse_warm, t=1, max_t=timesteps)

        result = sampler.sample(
            model=model,
            noise_process=process,
            num_graphs=bs,
            num_nodes=n,
            device=torch.device("cpu"),
            start_from=warm_start,
        )

        # ``t=1`` so the reverse chain runs only one step.
        assert len(model.calls) == 1
        assert len(result) == bs
        for graph in result:
            assert isinstance(graph, GraphState)


class TestSamplerSingleStep:
    """Unit-level test: one reverse step end-to-end with stubs.

    The sampler's per-step contract is:

    * call ``noise_process.process_state_condition_vector(t)``;
    * call ``model(z_t, t=condition, output_dense=True)``;
    * pass the model output to ``model_output_to_posterior_parameter``;
    * call ``noise_process.posterior_sample_from_model_output``;
    * carry the result to the next iteration.

    By starting from a ``t=1`` warm start, we force exactly one
    iteration and pin every step in that contract.
    """

    def test_one_reverse_step(self) -> None:
        bs = 1
        n = 4
        dx = 1
        de = 2
        process = _StubNoiseProcess(timesteps=10, dx=dx, de=de)
        model = _StubModel(dx=dx, de=de)
        sampler = Sampler(assert_symmetric_e=True)

        sparse_warm = _make_sparse_state(bs=bs, n=n, dx=dx, de=de)
        warm_start = DiffusionState(graph=sparse_warm, t=1, max_t=10)

        result = sampler.sample(
            model=model,
            noise_process=process,
            num_graphs=bs,
            num_nodes=n,
            device=torch.device("cpu"),
            start_from=warm_start,
        )

        # Single reverse step on a one-graph batch.
        assert len(model.calls) == 1
        carrier_type, output_dense = model.calls[0]
        # The warm-start sparse GraphState gets densified inside the
        # sampler before the model call, so the model receives a
        # DenseGraphState.
        assert carrier_type is DenseGraphState
        assert output_dense is True

        # One graph out, sparse carrier, correct node count.
        assert len(result) == 1
        assert isinstance(result[0], GraphState)
        assert int(result[0].num_nodes_per_graph.sum().item()) == n

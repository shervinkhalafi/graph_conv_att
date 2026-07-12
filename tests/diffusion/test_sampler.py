"""Tests for the Sampler hierarchy (CategoricalSampler, ContinuousSampler).

Test rationale
--------------
The ``Sampler`` ABC defines a ``sample()`` method that performs reverse
diffusion to generate graphs. ``CategoricalSampler`` implements ancestral
sampling for categorical discrete diffusion. ``ContinuousSampler`` implements
Gaussian reverse diffusion for adjacency-based models.

Each test validates:
- ABC contract (subclass relationship)
- Shared sampler loop across the semantic wrapper classes
- Correct output types and shapes from ``sample()``
- Graph invariants: symmetry and valid one-hot encoding

Dummy models return trivial outputs (uniform or identity) so the tests
exercise only the sampling logic, not model quality.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from tests._helpers.graph_builders import binary_graphdata
from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphState,
)
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    ContinuousNoiseProcess,
)
from tmgg.diffusion.sampler import (
    CategoricalSampler,
    ContinuousSampler,
    DiffusionState,
    Sampler,
)
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.models.base import GraphModel  # pyright: ignore[reportAttributeAccessIssue]
from tmgg.utils.noising.noise import GaussianNoise

# ---------------------------------------------------------------------------
# Dummy models for testing
# ---------------------------------------------------------------------------


class UniformCategoricalModel(GraphModel):
    """Returns uniform distribution over classes for both X and E.

    Used to test that the categorical sampler produces valid outputs
    regardless of model quality; uniform predictions still lead to valid
    probability distributions through the posterior computation.

    Per the Wave 4-A model contract, ``forward`` accepts an
    ``output_dense`` keyword and returns a :class:`DenseGraphDistribution`
    when ``output_dense=True`` (the path the sampler always takes).
    """

    def __init__(self, dx: int, de: int) -> None:
        super().__init__()
        self.dx = dx
        self.de = de

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
        if not output_dense:
            raise AssertionError(
                "UniformCategoricalModel only supports output_dense=True; "
                "the sampler always sets it."
            )
        # The sampler densifies sparse inputs to a DenseGraphState before
        # the model call, so we can rely on the legacy dense-shape access.
        assert isinstance(data, DenseGraphState), (
            f"expected DenseGraphState, got {type(data).__name__}"
        )
        assert data.X_class is not None
        assert data.E_class is not None
        bs, n, _ = data.X_class.shape
        X = torch.ones(bs, n, self.dx, device=data.X_class.device) / self.dx
        E = torch.ones(bs, n, n, self.de, device=data.E_class.device) / self.de
        return DenseGraphDistribution(
            num_nodes_per_graph=data.num_nodes_per_graph,
            y=data.y,
            X_class=X,
            E_class=E,
        )


class IdentityContinuousModel(GraphModel):
    """Returns the input graph unchanged.

    Used to test that the continuous sampler pipeline works end-to-end.
    With an identity model, the sampler still exercises schedule lookups,
    posterior computation, and final thresholding.

    Per the Wave 4-A model contract, ``forward`` accepts an
    ``output_dense`` keyword and returns a :class:`DenseGraphDistribution`
    constructed from the input dense state (lossless lift; data is
    already in dense form).
    """

    def __init__(self) -> None:
        super().__init__()

    def get_config(self) -> dict[str, Any]:
        return {}

    def forward(  # type: ignore[override]
        self,
        data: GraphData,
        t: Tensor | None = None,
        *,
        output_dense: bool = False,
    ) -> DenseGraphDistribution:
        _ = t
        if not output_dense:
            raise AssertionError(
                "IdentityContinuousModel only supports output_dense=True; "
                "the sampler always sets it."
            )
        assert isinstance(data, DenseGraphState), (
            f"expected DenseGraphState, got {type(data).__name__}"
        )
        # Lift the dense state to a dense distribution — same data, only
        # the static type tag changes.
        return DenseGraphDistribution(
            num_nodes_per_graph=data.num_nodes_per_graph,
            y=data.y,
            X_class=data.X_class,
            X_feat=data.X_feat,
            E_class=data.E_class,
            E_feat=data.E_feat,
        )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

T_STEPS = 5
N_NODES = 4
BS = 2
DX = 3
DE = 2


# cosine_schedule_short fixture (T=5) provided by conftest.py


@pytest.fixture()
def unified_schedule() -> NoiseSchedule:
    """NoiseSchedule (our new unified wrapper) for both sampler types."""
    return NoiseSchedule("cosine_iddpm", timesteps=T_STEPS)


@pytest.fixture()
def categorical_noise_process(
    cosine_schedule_short: NoiseSchedule,
) -> CategoricalNoiseProcess:
    return CategoricalNoiseProcess(
        schedule=cosine_schedule_short,
        x_classes=DX,
        e_classes=DE,
        limit_distribution="uniform",
    )


@pytest.fixture()
def continuous_noise_process() -> ContinuousNoiseProcess:
    return ContinuousNoiseProcess(
        definition=GaussianNoise(),
        schedule=NoiseSchedule("cosine_iddpm", timesteps=T_STEPS),
    )


@pytest.fixture()
def uniform_model() -> UniformCategoricalModel:
    return UniformCategoricalModel(dx=DX, de=DE)


@pytest.fixture()
def identity_model() -> IdentityContinuousModel:
    return IdentityContinuousModel()


# ---------------------------------------------------------------------------
# DiffusionState validation tests
# ---------------------------------------------------------------------------


def _make_sparse_state_for_tests(bs: int = BS, n: int = N_NODES) -> GraphState:
    """Build a small batched :class:`GraphState` for DiffusionState tests.

    Per the Wave 4-A contract, :class:`DiffusionState` consumes the
    sparse-default carrier. The graphs here carry a structure-only
    ``x_class`` width 1 and an empty ``edge_index`` so the sparse
    invariants hold trivially.
    """
    num_nodes_per_graph = torch.full((bs,), n, dtype=torch.long)
    batch = torch.arange(bs).repeat_interleave(n)
    x_class = torch.ones(bs * n, 1, dtype=torch.float32)
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_class = torch.empty(0, 2, dtype=torch.float32)
    y = torch.zeros(bs, 0, dtype=torch.float32)
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


class TestDiffusionState:
    """DiffusionState should validate timestep metadata and batched graph shape."""

    def test_constructs_for_valid_batched_graph(self) -> None:
        graph = _make_sparse_state_for_tests()
        state = DiffusionState(graph=graph, t=3, max_t=T_STEPS)
        assert state.t == 3
        assert state.max_t == T_STEPS
        assert state.graph is graph

    def test_rejects_negative_timestep(self) -> None:
        graph = _make_sparse_state_for_tests()
        with pytest.raises(ValueError, match="0 <= t <= max_t"):
            DiffusionState(graph=graph, t=-1, max_t=T_STEPS)

    def test_rejects_timestep_above_max(self) -> None:
        graph = _make_sparse_state_for_tests()
        with pytest.raises(ValueError, match="0 <= t <= max_t"):
            DiffusionState(graph=graph, t=T_STEPS + 1, max_t=T_STEPS)

    def test_rejects_dense_graph(self) -> None:
        # Per the Wave 4-A contract, DiffusionState requires a sparse
        # GraphState. A DenseGraphState (state-content but wrong
        # carrier) must be rejected with a clear type error.
        dense = binary_graphdata(torch.zeros(BS, N_NODES, N_NODES))
        with pytest.raises(TypeError, match="must be a GraphState"):
            DiffusionState(graph=dense, t=1, max_t=T_STEPS)  # type: ignore[arg-type]

    def test_rejects_mismatched_dense_state_shapes(self) -> None:
        # Migrated from the legacy node_mask-mismatch check: under the
        # new dense type, ``DenseGraphState.__post_init__`` enforces
        # that every populated split tensor's leading dims agree with
        # ``num_nodes_per_graph`` and an internally consistent ``n_max``.
        # Mismatched X_class / E_class widths surface as a "leading
        # dims" error.
        with pytest.raises(ValueError, match="leading dims"):
            DenseGraphState(
                num_nodes_per_graph=torch.full((BS,), N_NODES, dtype=torch.long),
                y=torch.zeros(BS, 0),
                X_class=torch.zeros(BS, N_NODES + 1, DX),
                E_class=torch.zeros(BS, N_NODES, N_NODES, DE),
            )


# ---------------------------------------------------------------------------
# ABC contract tests
# ---------------------------------------------------------------------------


class TestSamplerABC:
    """Both sampler implementations must be instances of Sampler."""

    def test_categorical_sampler_is_sampler(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
    ) -> None:
        _ = categorical_noise_process
        _ = unified_schedule
        sampler = CategoricalSampler()
        assert isinstance(sampler, Sampler)

    def test_continuous_sampler_is_sampler(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
    ) -> None:
        _ = continuous_noise_process
        _ = unified_schedule
        sampler = ContinuousSampler()
        assert isinstance(sampler, Sampler)


# ---------------------------------------------------------------------------
# Unified-loop tests
# ---------------------------------------------------------------------------


class TestUnifiedSamplerAliases:
    """Categorical and continuous wrappers should share one sampler loop."""

    def test_categorical_alias_uses_base_sample_implementation(self) -> None:
        assert CategoricalSampler.sample is Sampler.sample

    def test_continuous_alias_uses_base_sample_implementation(self) -> None:
        assert ContinuousSampler.sample is Sampler.sample


# ---------------------------------------------------------------------------
# CategoricalSampler.sample() tests
# ---------------------------------------------------------------------------


class TestCategoricalSamplerSample:
    """Verify CategoricalSampler.sample() output type, shapes, and invariants."""

    def test_uses_noise_process_condition_vector_for_model_input(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Categorical sampling should delegate conditioning semantics.

        Test rationale: Step 1 removes sampler-owned timestep normalization.
        The sampler should ask the noise process for the model condition
        vector at each reverse step and forward that exact tensor.
        """
        sampler = CategoricalSampler()
        observed_condition: list[torch.Tensor | None] = []

        def condition_vector(t: Tensor) -> Tensor:
            return t.float() + 100.0

        original_forward = uniform_model.forward

        def spy_forward(
            data: GraphData,
            t: Tensor | None = None,
            *,
            output_dense: bool = False,
        ) -> DenseGraphDistribution:
            observed_condition.append(None if t is None else t.detach().clone())
            return original_forward(data, t=t, output_dense=output_dense)

        categorical_noise_process.process_state_condition_vector = condition_vector  # type: ignore[method-assign]
        uniform_model.forward = spy_forward  # type: ignore[assignment]

        sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )

        expected = [
            torch.full((BS,), float(step) + 100.0)
            for step in range(unified_schedule.timesteps, 0, -1)
        ]
        assert len(observed_condition) == len(expected)
        for actual, want in zip(observed_condition, expected, strict=True):
            assert actual is not None
            torch.testing.assert_close(actual, want)

    def test_returns_list_of_graph_state(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """sample() returns a list of sparse GraphState with the correct length.

        Wave 4-A: the sampler emits per-graph :class:`GraphState`
        instances (sparse-default carrier), one per requested graph.
        """
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        assert isinstance(results, list)
        assert len(results) == BS
        for g in results:
            assert isinstance(g, GraphState)

    def test_output_shape_correctness(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Generated graphs have the expected per-graph node counts.

        Each per-graph slice is a single-graph batch (B=1) with
        ``num_nodes_per_graph = [n]``. The sparse ``x_class`` carries
        ``(n, dx)`` content; the dense adjacency materialises as
        ``(1, n, n)`` via :meth:`GraphState.dense_adjacency`.
        """
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            assert int(g.num_nodes_per_graph.sum().item()) == N_NODES
            assert g.x_class is not None
            assert g.x_class.shape == (N_NODES, DX)
            adj = g.dense_adjacency()
            # Single-graph batch: shape (1, N, N).
            assert adj.shape == (1, N_NODES, N_NODES)

    def test_edge_symmetry(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Generated dense adjacency is symmetric.

        With sparse outputs the symmetry invariant lives on the
        ``edge_index`` directly (each undirected edge appears twice),
        which :meth:`GraphState.__post_init__` already enforces under
        ``__debug__``. The dense adjacency view is the user-facing
        check this test pins.
        """
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.dense_adjacency()
            assert torch.equal(
                adj, adj.transpose(-1, -2)
            ), "Dense adjacency view of GraphState must be symmetric"

    def test_valid_class_indices(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Per-position one-hot rows sum to 1 on every valid position.

        On the sparse carrier ``x_class`` rows are one per real node and
        ``edge_class`` rows are one per directed active edge.
        :meth:`GraphState.__post_init__` already drops self-loops and
        enforces the edge-index / class-row count agreement, so we just
        verify the one-hot row-sum invariant on the active sparse rows.
        """
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            assert g.x_class is not None
            x_sums = g.x_class.sum(dim=-1)
            assert torch.allclose(x_sums, torch.ones_like(x_sums))
            if g.edge_class is not None and g.edge_class.shape[0] > 0:
                e_sums = g.edge_class.sum(dim=-1)
                assert torch.allclose(e_sums, torch.ones_like(e_sums))

    def test_variable_node_counts(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Sampling with a Tensor of per-graph node counts works correctly."""
        sampler = CategoricalSampler()
        num_nodes = torch.tensor([3, 5])
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=2,
            num_nodes=num_nodes,
            device=torch.device("cpu"),
        )
        assert len(results) == 2
        assert int(results[0].num_nodes_per_graph.sum().item()) == 3
        assert int(results[1].num_nodes_per_graph.sum().item()) == 5

    def test_sample_no_gradients_tracked(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """sample() must not track gradients even when called outside no_grad().

        Test rationale: the reverse diffusion loop allocates many
        intermediates; tracking gradients wastes memory with no benefit
        since sampling is always inference-only.
        """
        sampler = CategoricalSampler()
        # Ensure we are NOT inside a no_grad context
        assert torch.is_grad_enabled()
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            assert g.x_class is not None
            assert not g.x_class.requires_grad
            if g.edge_class is not None:
                assert not g.edge_class.requires_grad

    def test_warm_start_still_works_with_explicit_noise_process(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Warm-start sampling should still work after ownership moves.

        Test rationale: Step 2 removes sampler-owned process state. Passing the
        process explicitly must not break partial reverse chains that resume
        from an existing latent graph.

        Per the Wave 4-A contract, ``DiffusionState`` consumes a sparse
        :class:`GraphState`; the noise-process prior is dense, so we
        sparsify it at the boundary.
        """
        sampler = CategoricalSampler()
        node_mask = torch.ones(BS, N_NODES, dtype=torch.bool)
        dense_prior = categorical_noise_process.sample_prior(node_mask)
        assert isinstance(dense_prior, DenseGraphState)
        start_graph = dense_prior.to_sparse()
        start_from = DiffusionState(
            graph=start_graph,
            t=unified_schedule.timesteps - 1,
            max_t=unified_schedule.timesteps,
        )

        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
            start_from=start_from,
        )

        assert len(results) == BS


# ---------------------------------------------------------------------------
# CategoricalSampler with marginal transitions (T-5)
# ---------------------------------------------------------------------------


class TestCategoricalSamplerMarginal:
    """CategoricalSampler with empirical stationary PMFs instead of uniform.

    Test rationale: all existing categorical sampler tests use the uniform
    stationary mode. These cases verify that loader-initialised empirical
    marginals also produce valid reverse sampling behavior.
    """

    @pytest.fixture()
    def marginal_noise_process(
        self,
        cosine_schedule_short: NoiseSchedule,
    ) -> CategoricalNoiseProcess:
        from torch_geometric.data import Batch, Data

        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule_short,
            x_classes=DX,
            e_classes=DE,
            limit_distribution="empirical_marginal",
        )
        # 4-node graph with edges (0,1), (0,2), (1,2). PyG enumerates both
        # directions per undirected edge → 6 directed edges total.
        edges = [(0, 1), (0, 2), (1, 2)]
        directed = [(i, j) for i, j in edges] + [(j, i) for i, j in edges]
        edge_index = torch.tensor(
            [[u for u, _ in directed], [v for _, v in directed]],
            dtype=torch.long,
        )
        pyg_batch = Batch.from_data_list(
            [Data(edge_index=edge_index, num_nodes=N_NODES)]
        )
        proc.initialize_from_data([pyg_batch])
        return proc

    def test_returns_valid_graph_state(
        self,
        marginal_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Marginal-transition sampling produces valid sparse GraphState output."""
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=marginal_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        assert len(results) == BS
        for g in results:
            assert isinstance(g, GraphState)
            assert g.x_class is not None
            assert g.x_class.shape[0] == N_NODES

    def test_valid_class_distributions(
        self,
        marginal_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """x_class rows are valid one-hot PMFs with empirical marginals."""
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=marginal_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            assert g.x_class is not None
            x_sums = g.x_class.sum(dim=-1)
            assert torch.allclose(x_sums, torch.ones_like(x_sums))

    def test_edge_symmetry(
        self,
        marginal_noise_process: CategoricalNoiseProcess,
        unified_schedule: NoiseSchedule,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Symmetry invariant holds for the dense adjacency view."""
        sampler = CategoricalSampler()
        results = sampler.sample(
            model=uniform_model,
            noise_process=marginal_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.dense_adjacency()
            assert torch.equal(adj, adj.transpose(-1, -2))


# ---------------------------------------------------------------------------
# ContinuousSampler.sample() tests
# ---------------------------------------------------------------------------


class TestContinuousSamplerSample:
    """Verify ContinuousSampler.sample() output type, shapes, and invariants."""

    def test_uses_noise_process_condition_vector_for_model_input(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """Continuous sampling should delegate conditioning semantics.

        Test rationale: the continuous sampler previously derived schedule
        noise levels directly, which diverged from training-time conditioning.
        This test locks the sampler to the noise-process-owned condition
        vector instead.
        """
        sampler = ContinuousSampler()
        observed_condition: list[torch.Tensor | None] = []

        def condition_vector(t: Tensor) -> Tensor:
            return t.float() + 100.0

        original_forward = identity_model.forward

        def spy_forward(
            data: GraphData,
            t: Tensor | None = None,
            *,
            output_dense: bool = False,
        ) -> DenseGraphDistribution:
            observed_condition.append(None if t is None else t.detach().clone())
            return original_forward(data, t=t, output_dense=output_dense)

        continuous_noise_process.process_state_condition_vector = condition_vector  # type: ignore[method-assign]
        identity_model.forward = spy_forward  # type: ignore[assignment]

        sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )

        expected = [
            torch.full((BS,), float(step) + 100.0)
            for step in range(unified_schedule.timesteps, 0, -1)
        ]
        assert len(observed_condition) == len(expected)
        for actual, want in zip(observed_condition, expected, strict=True):
            assert actual is not None
            torch.testing.assert_close(actual, want)

    def test_returns_list_of_graph_state(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """sample() returns a list of sparse GraphState with the correct length."""
        sampler = ContinuousSampler()
        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        assert isinstance(results, list)
        assert len(results) == BS
        for g in results:
            assert isinstance(g, GraphState)

    def test_output_shape_correctness(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """Generated graphs have expected per-graph node counts.

        The dense adjacency view is square ``(1, n, n)`` per graph;
        ``num_nodes_per_graph`` carries the per-graph node count.
        """
        sampler = ContinuousSampler()
        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            n = int(g.num_nodes_per_graph.sum().item())
            assert n == N_NODES
            adj = g.dense_adjacency()
            assert adj.shape[-2] == adj.shape[-1] == n

    def test_binary_adjacency(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """Final adjacency values should be binary (0 or 1)."""
        sampler = ContinuousSampler()
        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.dense_adjacency()
            unique_vals = torch.unique(adj)
            for v in unique_vals:
                assert v.item() in (
                    0.0,
                    1.0,
                ), f"Expected binary adjacency, found value {v.item()}"

    def test_edge_symmetry(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """Generated adjacency should be symmetric."""
        sampler = ContinuousSampler()
        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.dense_adjacency()
            if adj.dim() == 2:
                assert torch.equal(adj, adj.T), "Adjacency must be symmetric"
            elif adj.dim() == 3:
                assert torch.equal(
                    adj, adj.transpose(-1, -2)
                ), "Adjacency must be symmetric"

    def test_zero_diagonal(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """No self-loops: diagonal of adjacency should be zero."""
        sampler = ContinuousSampler()
        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.dense_adjacency()
            if adj.dim() == 2:
                assert torch.diag(adj).sum() == 0, "Self-loops must be zero"
            elif adj.dim() == 3:
                for b in range(adj.shape[0]):
                    assert torch.diag(adj[b]).sum() == 0, "Self-loops must be zero"

    def test_sample_no_gradients_tracked(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """sample() must not track gradients even when called outside no_grad().

        Test rationale: the reverse diffusion loop allocates many
        intermediates; tracking gradients wastes memory with no benefit
        since sampling is always inference-only.
        """
        sampler = ContinuousSampler()
        assert torch.is_grad_enabled()
        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        for g in results:
            adj = g.dense_adjacency()
            assert not adj.requires_grad

    def test_warm_start_still_works_with_explicit_noise_process(
        self,
        continuous_noise_process: ContinuousNoiseProcess,
        unified_schedule: NoiseSchedule,
        identity_model: IdentityContinuousModel,
    ) -> None:
        """Warm-start sampling should still work after ownership moves.

        Per the Wave 4-A contract, ``DiffusionState`` consumes a sparse
        :class:`GraphState`; the noise-process prior is dense, so we
        sparsify it at the boundary.
        """
        sampler = ContinuousSampler()
        node_mask = torch.ones(BS, N_NODES, dtype=torch.bool)
        dense_prior = continuous_noise_process.sample_prior(node_mask)
        assert isinstance(dense_prior, DenseGraphState)
        start_graph = dense_prior.to_sparse()
        start_from = DiffusionState(
            graph=start_graph,
            t=unified_schedule.timesteps - 1,
            max_t=unified_schedule.timesteps,
        )

        results = sampler.sample(
            model=identity_model,
            noise_process=continuous_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
            start_from=start_from,
        )

        assert len(results) == BS


class TestAssertSymmetricEToggle:
    """``Sampler.assert_symmetric_e`` controls the per-step E-symmetry guard.

    Test rationale (parity #28 / #29 / D-5): the toggle mirrors upstream
    DiGress's per-reverse-step ``assert (E_s == E_s.transpose(1, 2)).all()``
    at ``diffusion_model_discrete.py:649``. Default ``True`` matches
    upstream semantics; ``False`` lets production hot loops skip the
    check when fp32 batch sizes make it expensive. We pin both the
    default and the failure surface so a future bypass of the canonical
    symmetrisation primitive cannot silently land asymmetric edges
    downstream.
    """

    def test_default_passes_on_normal_categorical_sampling(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """Standard categorical sampling produces symmetric E_class with default toggle."""
        sampler = CategoricalSampler()  # default assert_symmetric_e=True
        assert sampler._assert_symmetric_e is True  # noqa: SLF001
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        # Reaches the end without firing the per-step assertion.
        assert len(results) == BS

    def test_disabling_toggle_lets_loop_complete_without_check(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """``assert_symmetric_e=False`` bypasses the per-step symmetry guard."""
        sampler = CategoricalSampler(assert_symmetric_e=False)
        assert sampler._assert_symmetric_e is False  # noqa: SLF001
        results = sampler.sample(
            model=uniform_model,
            noise_process=categorical_noise_process,
            num_graphs=BS,
            num_nodes=N_NODES,
            device=torch.device("cpu"),
        )
        assert len(results) == BS

    def test_assertion_fires_on_forced_asymmetric_e_class(
        self,
        categorical_noise_process: CategoricalNoiseProcess,
        uniform_model: UniformCategoricalModel,
    ) -> None:
        """A noise process that returns asymmetric E_class trips the guard."""
        sampler = CategoricalSampler()  # default assert_symmetric_e=True

        # Wrap posterior_sample_from_model_output to force asymmetry.
        original_hook = categorical_noise_process.posterior_sample_from_model_output

        def asymmetric_hook(
            z_t: GraphData,
            x0_param: GraphData,
            t: Tensor,
            s: Tensor,
        ) -> GraphData:
            out = original_hook(z_t, x0_param, t, s)
            # The categorical noise process returns a dense state from
            # ``posterior_sample_from_model_output``; narrow for typed
            # access to ``E_class``.
            assert isinstance(out, DenseGraphState)
            assert out.E_class is not None
            corrupted = out.E_class.clone()
            # Flip a single off-diagonal entry to break symmetry.
            corrupted[..., 0, 1, 0] = 1.0 - corrupted[..., 0, 1, 0]
            return out.replace(E_class=corrupted)

        categorical_noise_process.posterior_sample_from_model_output = (  # type: ignore[method-assign]
            asymmetric_hook
        )

        with pytest.raises(AssertionError, match="asymmetric E_class"):
            sampler.sample(
                model=uniform_model,
                noise_process=categorical_noise_process,
                num_graphs=BS,
                num_nodes=N_NODES,
                device=torch.device("cpu"),
            )

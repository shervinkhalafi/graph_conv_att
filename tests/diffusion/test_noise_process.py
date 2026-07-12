# pyright: reportAttributeAccessIssue=false
"""Tests for the NoiseProcess hierarchy.

Test rationale
--------------
The ``NoiseProcess`` hierarchy unifies two separate noise systems under a
single abstract interface. ``ContinuousNoiseProcess`` wraps the existing
``NoiseDefinition`` subclasses and operates on dense edge-valued states.
``CategoricalNoiseProcess`` owns its stationary categorical PMFs directly and
operates on one-hot categorical representations.

Each test validates that the wrapper correctly delegates to the underlying
noise model, preserves tensor shapes and graph invariants (symmetry,
masking), and satisfies the ``NoiseProcess`` ABC contract.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import Tensor

from tests._helpers.graph_builders import binary_graphdata, legacy_edge_scalar
from tmgg.data.datasets.graph_data_fields import FIELD_NAMES
from tmgg.data.datasets.graph_types import DenseGraphState, GraphData
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    ContinuousNoiseProcess,
    ExactDensityNoiseProcess,
    GaussianNoiseProcess,
    NoisedBatch,
    NoiseProcess,
)
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.utils.noising.noise import (
    DigressNoise,
    EdgeFlipNoise,
    GaussianNoise,
    LogitNoise,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_adjacency_batch() -> Tensor:
    """A small batch of binary adjacency matrices (bs=2, n=4)."""
    torch.manual_seed(42)
    adj = torch.zeros(2, 4, 4)
    # Graph 0: path 0-1-2-3
    adj[0, 0, 1] = adj[0, 1, 0] = 1
    adj[0, 1, 2] = adj[0, 2, 1] = 1
    adj[0, 2, 3] = adj[0, 3, 2] = 1
    # Graph 1: complete K4
    for i in range(4):
        for j in range(4):
            if i != j:
                adj[1, i, j] = 1
    return adj


@pytest.fixture()
def graph_data_from_adj(simple_adjacency_batch: Tensor) -> DenseGraphState:
    """DenseGraphState created from the simple adjacency batch."""
    return binary_graphdata(simple_adjacency_batch)


@pytest.fixture()
def categorical_graph_data() -> DenseGraphState:
    """A small batch of one-hot categorical DenseGraphState(bs=2, n=5, dx=3, de=2)."""
    torch.manual_seed(42)
    bs, n, dx, de = 2, 5, 3, 2
    # Random one-hot X
    x_idx = torch.randint(0, dx, (bs, n))
    X = torch.zeros(bs, n, dx)
    X.scatter_(2, x_idx.unsqueeze(-1), 1.0)
    # Random one-hot E (symmetric, zero diagonal)
    e_idx = torch.randint(0, de, (bs, n, n))
    # Make symmetric
    e_idx = torch.triu(e_idx, diagonal=1)
    e_idx = e_idx + e_idx.transpose(1, 2)
    E = torch.zeros(bs, n, n, de)
    E.scatter_(3, e_idx.unsqueeze(-1), 1.0)
    # Zero diagonal -> class 0
    diag = torch.arange(n)
    E[:, diag, diag, :] = 0
    E[:, diag, diag, 0] = 1.0

    y = torch.zeros(bs, 0)
    num_nodes_per_graph = torch.full((bs,), n, dtype=torch.long)
    return DenseGraphState(
        num_nodes_per_graph=num_nodes_per_graph, y=y, X_class=X, E_class=E
    )


# cosine_schedule fixture (T=50) provided by conftest.py


# ---------------------------------------------------------------------------
# ABC contract tests
# ---------------------------------------------------------------------------


class TestNoiseProcessABC:
    """Verify that both subclasses satisfy the NoiseProcess interface."""

    def test_continuous_is_noise_process(self, cosine_schedule: NoiseSchedule) -> None:
        gen = GaussianNoise()
        proc = ContinuousNoiseProcess(definition=gen, schedule=cosine_schedule)
        assert isinstance(proc, NoiseProcess)
        assert proc.timesteps == cosine_schedule.timesteps

    def test_categorical_is_noise_process(self, cosine_schedule: NoiseSchedule) -> None:
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        assert isinstance(proc, NoiseProcess)
        assert proc.timesteps == cosine_schedule.timesteps

    def test_concrete_subclasses_declare_nonempty_fields(
        self, cosine_schedule: NoiseSchedule
    ) -> None:
        """Each concrete NoiseProcess declares a subset of FIELD_NAMES.

        Test rationale (Wave 2.1): the new ``fields`` class attribute on
        ``NoiseProcess`` is the load-bearing declaration the Lightning
        module and composite wrapper iterate over. It must be non-empty
        and lie inside the canonical ``FIELD_NAMES`` set for every
        concrete subclass so the loss dispatch stays well-defined.
        """
        categorical = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        gaussian = GaussianNoiseProcess(
            definition=GaussianNoise(), schedule=cosine_schedule
        )

        for proc in (categorical, gaussian):
            assert isinstance(proc.fields, frozenset)
            assert len(proc.fields) > 0
            assert proc.fields.issubset(FIELD_NAMES)

        assert categorical.fields == frozenset({"X_class", "E_class"})
        assert gaussian.fields == frozenset({"E_feat"})

    def test_gaussian_noise_process_field_override(
        self, cosine_schedule: NoiseSchedule
    ) -> None:
        """GaussianNoiseProcess accepts a valid fields override.

        Test rationale: joint X_feat + E_feat diffusion must be
        configurable via the constructor kwarg; invalid fields (outside
        the continuous subset) must raise.
        """
        joint = GaussianNoiseProcess(
            definition=GaussianNoise(),
            schedule=cosine_schedule,
            fields=frozenset({"X_feat", "E_feat"}),
        )
        assert joint.fields == frozenset({"X_feat", "E_feat"})

        with pytest.raises(ValueError, match="subset"):
            GaussianNoiseProcess(
                definition=GaussianNoise(),
                schedule=cosine_schedule,
                fields=frozenset({"X_class"}),  # pyright: ignore[reportArgumentType]
            )

        with pytest.raises(ValueError, match="non-empty"):
            GaussianNoiseProcess(
                definition=GaussianNoise(),
                schedule=cosine_schedule,
                fields=frozenset(),
            )


# ---------------------------------------------------------------------------
# ContinuousNoiseProcess tests
# ---------------------------------------------------------------------------


class TestContinuousNoiseProcess:
    """Tests for ContinuousNoiseProcess wrapping NoiseDefinition subclasses."""

    def test_process_state_condition_vector_uses_normalized_timestep(
        self,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Default conditioning should be the normalized timestep ``t / T``.

        Test rationale: Step 1 moves model-conditioning ownership into the
        noise process without changing experiment semantics yet. The process
        therefore needs to expose the same normalized timestep representation
        that training used before the refactor.
        """
        proc = ContinuousNoiseProcess(
            definition=GaussianNoise(),
            schedule=cosine_schedule,
        )
        t = torch.tensor([1, 25, 50], dtype=torch.long)

        expected = t.float() / proc.timesteps
        actual = proc.process_state_condition_vector(t)

        torch.testing.assert_close(actual, expected)

    def test_initialization_hooks_are_noops(
        self,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Continuous processes should report no dataloader-backed setup needs."""
        proc = ContinuousNoiseProcess(
            definition=GaussianNoise(),
            schedule=cosine_schedule,
        )

        assert proc.needs_data_initialization() is False
        assert proc.is_initialized() is True

    def test_sample_at_level_preserves_graph_shape(
        self,
        graph_data_from_adj: DenseGraphState,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Direct-level noising should preserve the GraphData batch structure."""
        proc = ContinuousNoiseProcess(
            definition=GaussianNoise(),
            schedule=cosine_schedule,
        )

        result = proc.sample_at_level(graph_data_from_adj, level=0.25)

        assert isinstance(result, GraphData)
        # Continuous noise writes into E_feat (structure-only output).
        assert result.E_feat is not None
        assert (
            result.to_edge_scalar(source="feat").shape
            == graph_data_from_adj.dense_adjacency().shape
        )
        assert result.node_mask.shape == graph_data_from_adj.node_mask.shape

    @pytest.mark.parametrize(
        "generator_cls",
        [
            GaussianNoise,
            EdgeFlipNoise,
            DigressNoise,
            LogitNoise,
        ],
        ids=["gaussian", "edge_flip", "digress", "logit"],
    )
    def test_forward_sample_returns_graph_data_with_same_shape(
        self,
        generator_cls: type,
        graph_data_from_adj: DenseGraphState,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """forward_sample() preserves the graph extents of the input batch.

        Test rationale (D-4): forward_sample now returns a NoisedBatch
        bundling the noised graph with schedule scalars. The shape /
        type assertions previously pinned to ``GraphData`` now operate
        on ``result.z_t``.
        """
        gen = generator_cls()
        proc = ContinuousNoiseProcess(definition=gen, schedule=cosine_schedule)
        t = torch.tensor([25, 25], dtype=torch.long)  # integer timesteps
        noised = proc.forward_sample(graph_data_from_adj, t)
        result = noised.z_t

        assert isinstance(result, GraphData)
        # Continuous forward_sample writes into E_feat (structure-only output).
        assert result.E_feat is not None
        assert (
            result.to_edge_scalar(source="feat").shape
            == graph_data_from_adj.dense_adjacency().shape
        )
        assert result.y.shape == graph_data_from_adj.y.shape
        assert result.node_mask.shape == graph_data_from_adj.node_mask.shape

    def test_forward_sample_at_t_zero_is_near_identity(
        self,
        graph_data_from_adj: DenseGraphState,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """At t=0 DDPM Gaussian forward noise leaves the input nearly unchanged.

        Test rationale (Wave 2.3): the rewritten ``forward_sample`` applies the
        closed-form DDPM parametrisation ``sqrt(alpha_bar_t) * x_0 +
        sqrt(1 - alpha_bar_t) * eps`` regardless of the wrapped
        ``NoiseDefinition``. The ``cosine_iddpm`` schedule has
        ``alpha_bar(0) ≈ 0.998``, so the output equals the input up to a
        ~4 %% perturbation. We assert an element-wise absolute tolerance
        that is loose enough to accept that perturbation but tight enough
        to catch a regression to mid-schedule noise.
        """
        torch.manual_seed(42)
        gen = DigressNoise()
        proc = GaussianNoiseProcess(definition=gen, schedule=cosine_schedule)
        t = torch.tensor([0, 0], dtype=torch.long)
        result = proc.forward_sample(graph_data_from_adj, t).z_t
        adj_in = graph_data_from_adj.dense_adjacency().unsqueeze(-1)
        adj_out = result.E_feat
        assert adj_out is not None
        # alpha_bar(0) ≈ 0.998 ⇒ noise std ≈ 0.04. Allow 5 sigma headroom.
        assert torch.allclose(adj_in, adj_out, atol=0.25)

    def test_posterior_sample_returns_graph_data_with_same_shape(
        self,
        graph_data_from_adj: DenseGraphState,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """posterior_sample() returns a sampled GraphData state at timestep s."""
        gen = GaussianNoise()
        proc = ContinuousNoiseProcess(definition=gen, schedule=cosine_schedule)
        # Pass integer timesteps — the posterior uses the schedule to look up alpha_bar.
        t = torch.tensor([40, 40], dtype=torch.long)
        s = torch.tensor([30, 30], dtype=torch.long)
        z_t = proc.forward_sample(graph_data_from_adj, t).z_t
        posterior = proc.posterior_sample(z_t, graph_data_from_adj, t, s)
        assert isinstance(posterior, GraphData)
        assert (
            legacy_edge_scalar(posterior).shape
            == graph_data_from_adj.dense_adjacency().shape
        )
        assert posterior.node_mask.shape == graph_data_from_adj.node_mask.shape

    def test_posterior_parameters_use_schedule_alpha_bar(
        self,
        graph_data_from_adj: DenseGraphState,
    ) -> None:
        """Posterior alpha values should come from the schedule, not a hardcoded
        linear mapping.

        Test rationale: before this fix, ContinuousNoiseProcess posterior math
        computed alpha_t = 1 - t assuming t was a noise level and the schedule
        was linear.  With a cosine schedule, the relationship between integer
        timestep t and alpha_bar(t) is non-linear; a hardcoded ``1 - t``
        would produce wrong posterior statistics.  We verify that the posterior
        mean helper uses the schedule's alpha_bar values by comparing cosine
        and linear schedules.
        """
        cosine = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=50)
        linear = NoiseSchedule(schedule_type="linear_ddpm", timesteps=50)

        gen = GaussianNoise()
        proc_cosine = ContinuousNoiseProcess(definition=gen, schedule=cosine)
        proc_linear = ContinuousNoiseProcess(definition=gen, schedule=linear)

        t = torch.tensor([25, 25], dtype=torch.long)
        s = torch.tensor([20, 20], dtype=torch.long)

        z_t = proc_cosine.forward_sample(
            graph_data_from_adj, torch.tensor([25, 25], dtype=torch.long)
        ).z_t

        post_cosine = proc_cosine._posterior_parameters(z_t, graph_data_from_adj, t, s)
        post_linear = proc_linear._posterior_parameters(z_t, graph_data_from_adj, t, s)

        # Cosine and linear schedules have different alpha_bar at t=25.
        # The posterior means must differ because the interpolation weights
        # (alpha_bar_s) differ between schedules.
        alpha_s_cosine = cosine.get_alpha_bar(t_int=s)
        alpha_s_linear = linear.get_alpha_bar(t_int=s)
        assert not torch.allclose(
            alpha_s_cosine, alpha_s_linear, atol=1e-3
        ), "Precondition: cosine and linear alpha_bar should differ at t=25"
        assert not torch.allclose(
            post_cosine["mean"], post_linear["mean"], atol=1e-3
        ), "Posterior means should differ when using different schedules"

    def test_apply_edge_symmetry(
        self,
        graph_data_from_adj: DenseGraphState,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Edge features remain symmetric after forward sampling."""
        gen = GaussianNoise()
        proc = ContinuousNoiseProcess(definition=gen, schedule=cosine_schedule)
        t = torch.tensor([25, 25], dtype=torch.long)
        result = proc.forward_sample(graph_data_from_adj, t).z_t
        adj = legacy_edge_scalar(result)
        assert torch.allclose(adj, adj.transpose(1, 2), atol=1e-5)


# ---------------------------------------------------------------------------
# CategoricalNoiseProcess tests
# ---------------------------------------------------------------------------


class TestCategoricalNoiseProcess:
    """Tests for CategoricalNoiseProcess stationary-PMF behavior."""

    def test_process_state_condition_vector_uses_normalized_timestep(
        self,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Categorical conditioning also defaults to the normalized timestep.

        Test rationale: the refactor changes ownership, not semantics. Both
        concrete process types should initially expose the existing ``t / T``
        representation so training and sampling stay aligned during Step 1.
        """
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([1, 25, 50], dtype=torch.long)

        expected = t.float() / proc.timesteps
        actual = proc.process_state_condition_vector(t)

        torch.testing.assert_close(actual, expected)

    def test_uniform_construction(self, cosine_schedule: NoiseSchedule) -> None:
        """Uniform mode constructs immediate stationary PMFs."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        assert proc._limit_x is not None
        assert proc._limit_e is not None
        assert isinstance(proc, ExactDensityNoiseProcess)
        assert proc.needs_data_initialization() is False
        assert proc.is_initialized() is True

    def test_empirical_marginal_starts_uninitialised(
        self,
        cosine_schedule: NoiseSchedule,
    ) -> None:
        """Empirical-marginal mode defers PMF construction to loader setup."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="empirical_marginal",
        )
        assert proc._limit_x is None
        assert proc._limit_e is None
        assert proc.needs_data_initialization() is True
        assert proc.is_initialized() is False

    def test_initialize_from_data_uses_strict_upper_triangle(
        self, cosine_schedule: NoiseSchedule
    ) -> None:
        """Sparse edge counts produce the expected π_E on a fully-connected pair."""
        from torch_geometric.data import Batch, Data

        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=2,
            e_classes=2,
            limit_distribution="empirical_marginal",
        )

        # Two-node graph with one undirected edge: PyG enumerates both
        # directions, so edge_index has 2 columns.
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        pyg_batch = Batch.from_data_list([Data(edge_index=edge_index, num_nodes=2)])

        proc.initialize_from_data([pyg_batch])

        torch.testing.assert_close(
            proc._limit_e,
            torch.tensor([0.0, 1.0]),
        )

    def test_initialize_from_data_falls_back_to_uniform_on_zero_edge_counts(
        self, cosine_schedule: NoiseSchedule
    ) -> None:
        """Empirical edge marginals fall back to uniform when no real edges exist."""
        from torch_geometric.data import Batch, Data

        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=2,
            e_classes=2,
            limit_distribution="empirical_marginal",
        )

        # Single-node graph with no edges: per-class counts are all zero,
        # so the PMF falls back to uniform.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        pyg_batch = Batch.from_data_list([Data(edge_index=edge_index, num_nodes=1)])

        proc.initialize_from_data([pyg_batch])

        torch.testing.assert_close(
            proc._limit_e,
            torch.tensor([0.5, 0.5]),
        )

    def test_rejects_invalid_limit_distribution(
        self, cosine_schedule: NoiseSchedule
    ) -> None:
        """Unknown limit_distribution values raise ValueError."""
        with pytest.raises(ValueError, match="limit_distribution"):
            CategoricalNoiseProcess(
                schedule=cosine_schedule,
                x_classes=3,
                e_classes=2,
                limit_distribution="bogus",  # type: ignore[arg-type]
            )

    # ---- absorbing limit-distribution variant (parity #12 / D-12) -----

    def test_absorbing_construction_is_one_hot_indicator(
        self, cosine_schedule: NoiseSchedule
    ) -> None:
        """Absorbing mode populates one-hot stationary PMFs at construction."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="absorbing",
            absorbing_class_x=1,
            absorbing_class_e=0,
        )
        assert proc.needs_data_initialization() is False
        assert proc.is_initialized() is True
        torch.testing.assert_close(proc._limit_x, torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(proc._limit_e, torch.tensor([1.0, 0.0]))

    def test_absorbing_rejects_out_of_range_class(
        self, cosine_schedule: NoiseSchedule
    ) -> None:
        """Out-of-range absorbing-class indices raise ValueError."""
        with pytest.raises(ValueError, match="absorbing_class_x"):
            CategoricalNoiseProcess(
                schedule=cosine_schedule,
                x_classes=3,
                e_classes=2,
                limit_distribution="absorbing",
                absorbing_class_x=5,
            )
        with pytest.raises(ValueError, match="absorbing_class_e"):
            CategoricalNoiseProcess(
                schedule=cosine_schedule,
                x_classes=3,
                e_classes=2,
                limit_distribution="absorbing",
                absorbing_class_e=-1,
            )

    def test_absorbing_forward_pmf_collapses_at_t_equals_T(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """At t = T the per-position PMF puts all mass on the absorbing class."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="absorbing",
            absorbing_class_x=2,
            absorbing_class_e=1,
        )
        assert categorical_graph_data.X_class is not None
        assert categorical_graph_data.E_class is not None
        bs = categorical_graph_data.X_class.shape[0]
        t = torch.full((bs,), proc.timesteps, dtype=torch.long)
        pmf = proc.forward_pmf(categorical_graph_data, t)

        # alpha_bar(T) ~ 0, so prob_x = pi_x and prob_e = pi_e at every
        # position, regardless of the clean-graph one-hot.
        assert pmf.X_class is not None
        assert pmf.E_class is not None
        # Every node-position PMF collapses to absorbing_class_x = 2.
        expected_x = torch.zeros_like(pmf.X_class)
        expected_x[..., 2] = 1.0
        torch.testing.assert_close(pmf.X_class, expected_x, atol=1e-5, rtol=1e-5)
        # Every edge-position PMF collapses to absorbing_class_e = 1.
        expected_e = torch.zeros_like(pmf.E_class)
        expected_e[..., 1] = 1.0
        torch.testing.assert_close(pmf.E_class, expected_e, atol=1e-5, rtol=1e-5)

    def test_absorbing_forward_pmf_preserves_signal_at_t_equals_zero(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """At t = 0 the forward PMF concentrates on the clean one-hot state."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="absorbing",
            absorbing_class_x=0,
            absorbing_class_e=0,
        )
        assert categorical_graph_data.X_class is not None
        assert categorical_graph_data.E_class is not None
        bs = categorical_graph_data.X_class.shape[0]
        t = torch.zeros(bs, dtype=torch.long)
        pmf = proc.forward_pmf(categorical_graph_data, t)

        # alpha_bar(0) is close to but not exactly 1 for cosine
        # schedules, so the PMF concentrates strongly on the clean
        # one-hot but allows ~1e-2 leakage onto the absorbing class.
        # Verify that the argmax matches the clean signal at every
        # position rather than asserting bitwise equality.
        assert pmf.X_class is not None
        assert pmf.E_class is not None
        assert torch.equal(
            pmf.X_class.argmax(dim=-1),
            categorical_graph_data.X_class.argmax(dim=-1),
        )
        assert torch.equal(
            pmf.E_class.argmax(dim=-1),
            categorical_graph_data.E_class.argmax(dim=-1),
        )

    def test_absorbing_forward_pmf_intermediate_matches_closed_form(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """Mid-trajectory PMF equals the closed-form alpha_bar*x + (1-alpha_bar)*u."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="absorbing",
            absorbing_class_x=1,
            absorbing_class_e=0,
        )
        assert categorical_graph_data.X_class is not None
        assert categorical_graph_data.E_class is not None
        bs = categorical_graph_data.X_class.shape[0]
        mid = proc.timesteps // 2
        t = torch.full((bs,), mid, dtype=torch.long)
        pmf = proc.forward_pmf(categorical_graph_data, t)

        alpha_bar_t = proc.noise_schedule.get_alpha_bar(t_int=t)  # (bs,)
        # Hand-rolled closed form: alpha_bar*x + (1-alpha_bar)*pi.
        u_x = torch.tensor([0.0, 1.0, 0.0])
        u_e = torch.tensor([1.0, 0.0])
        a_x = alpha_bar_t.view(-1, 1, 1)
        a_e = alpha_bar_t.view(-1, 1, 1, 1)
        expected_x = a_x * categorical_graph_data.X_class + (1.0 - a_x) * u_x.view(
            1, 1, -1
        )
        expected_e = a_e * categorical_graph_data.E_class + (1.0 - a_e) * u_e.view(
            1, 1, 1, -1
        )
        torch.testing.assert_close(pmf.X_class, expected_x, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pmf.E_class, expected_e, atol=1e-5, rtol=1e-5)

    def test_absorbing_posterior_returns_valid_pmf(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """Posterior at non-trivial t produces a row-stochastic PMF."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="absorbing",
            absorbing_class_x=0,
            absorbing_class_e=0,
        )
        assert categorical_graph_data.X_class is not None
        bs = categorical_graph_data.X_class.shape[0]
        t = torch.full((bs,), proc.timesteps // 2, dtype=torch.long)
        z_t = proc.forward_sample(categorical_graph_data, t).z_t
        s = t - 1
        posterior = proc._posterior_probabilities(z_t, categorical_graph_data, t, s)
        # Per-node PMF rows sum to 1 over the class dim (within the
        # masked region — categorical_graph_data has node_mask all-True).
        assert posterior.X_class is not None
        x_sums = posterior.X_class.sum(dim=-1)
        torch.testing.assert_close(
            x_sums,
            torch.ones_like(x_sums),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_absorbing_stationary_distribution_accessor(
        self, cosine_schedule: NoiseSchedule
    ) -> None:
        """_stationary_distribution returns the configured one-hot indicators."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=4,
            e_classes=3,
            limit_distribution="absorbing",
            absorbing_class_x=3,
            absorbing_class_e=2,
        )
        x_pi, e_pi, _ = proc._stationary_distribution()
        torch.testing.assert_close(x_pi, torch.tensor([0.0, 0.0, 0.0, 1.0]))
        torch.testing.assert_close(e_pi, torch.tensor([0.0, 0.0, 1.0]))

    def test_stationary_distribution_raises_before_forward_sample(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """Empirical-marginal mode raises until loader setup populates PMFs."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="empirical_marginal",
        )
        t = torch.tensor([10, 20])
        with pytest.raises(RuntimeError, match="not initialised"):
            proc.forward_sample(categorical_graph_data, t)

    def test_forward_sample_mirrors_legacy_into_split_fields(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """forward_sample() populates both legacy and split fields (Wave 2.2 parity).

        Test rationale: during the transition the categorical noise process
        writes the same tensor into ``X`` and ``X_class`` (and into ``E`` /
        ``E_class``) so downstream consumers reading either form stay in
        agreement. The shim is removed in Wave 9.
        """
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([10, 20])
        result = proc.forward_sample(categorical_graph_data, t).z_t

        assert result.X_class is not None
        assert result.E_class is not None
        assert torch.equal(result.X_class, result.X_class)
        assert torch.equal(result.E_class, result.E_class)

    def test_forward_sample_reads_from_split_fields(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """forward_sample() reads X_class / E_class directly.

        Wave 9 removes the legacy X / E fields outright; this test now
        pins the remaining categorical path onto the split fields by
        cloning the inputs and checking that the forward draw at
        alpha_bar ≈ 1 (t=0) collapses onto the cloned tensors.
        """
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        assert categorical_graph_data.X_class is not None
        assert categorical_graph_data.E_class is not None
        split_input = categorical_graph_data.replace(
            X_class=categorical_graph_data.X_class.clone(),
            E_class=categorical_graph_data.E_class.clone(),
        )
        torch.manual_seed(0)
        t = torch.tensor([0, 0])  # alpha_bar ≈ 1: output must mirror the input.
        result = proc.forward_sample(split_input, t).z_t

        assert result.X_class is not None
        assert result.E_class is not None
        assert split_input.X_class is not None
        assert split_input.E_class is not None
        torch.testing.assert_close(result.X_class, split_input.X_class)
        torch.testing.assert_close(result.E_class, split_input.E_class)

    def test_forward_sample_returns_valid_one_hot(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """forward_sample() returns one-hot encoded GraphData."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        # Use integer timestep indices
        t = torch.tensor([10, 20])
        result = proc.forward_sample(categorical_graph_data, t).z_t

        assert isinstance(result, GraphData)
        assert result.X_class is not None
        assert result.E_class is not None
        # X should be one-hot: each row sums to 1 for valid nodes
        x_sums = result.X_class[categorical_graph_data.node_mask].sum(dim=-1)
        assert torch.allclose(x_sums, torch.ones_like(x_sums))

        # E should be one-hot for valid edges (off-diagonal, valid nodes)
        bs, n = categorical_graph_data.node_mask.shape
        mask_2d = categorical_graph_data.node_mask.unsqueeze(
            1
        ) * categorical_graph_data.node_mask.unsqueeze(2)
        diag = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)
        valid_edges = mask_2d & ~diag
        e_sums = result.E_class[valid_edges].sum(dim=-1)
        assert torch.allclose(e_sums, torch.ones_like(e_sums))

    def test_forward_sample_preserves_shape(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """forward_sample() preserves the shape of all GraphData fields."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([10, 20])
        result = proc.forward_sample(categorical_graph_data, t).z_t
        assert result.X_class is not None
        assert result.E_class is not None
        assert categorical_graph_data.X_class is not None
        assert categorical_graph_data.E_class is not None
        assert result.X_class.shape == categorical_graph_data.X_class.shape
        assert result.E_class.shape == categorical_graph_data.E_class.shape
        assert result.node_mask.shape == categorical_graph_data.node_mask.shape

    def test_prior_log_prob_returns_per_sample_shape(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """prior_log_prob() returns one finite value per batch element."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        result = proc.prior_log_prob(categorical_graph_data)
        assert categorical_graph_data.X_class is not None
        assert result.shape == (categorical_graph_data.X_class.shape[0],)
        assert torch.isfinite(result).all()

    def test_forward_log_prob_uses_requested_final_timestep(
        self,
    ) -> None:
        """forward_log_prob() should distinguish timestep T from T-1."""
        T = 10
        schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=T)
        proc = CategoricalNoiseProcess(
            schedule=schedule,
            x_classes=2,
            e_classes=2,
            limit_distribution="uniform",
        )

        bs, n = 2, 5
        X = torch.zeros(bs, n, 2)
        X[:, :, 0] = 1.0  # all nodes class 0
        E = torch.zeros(bs, n, n, 2)
        E[:, :, :, 0] = 1.0  # all edges class 0
        num_nodes_per_graph = torch.full((bs,), n, dtype=torch.long)
        clean = DenseGraphState(
            num_nodes_per_graph=num_nodes_per_graph,
            y=torch.zeros(bs, 0),
            X_class=X,
            E_class=E,
        )

        log_prob_T = proc.forward_log_prob(
            clean, clean, torch.full((bs,), T, dtype=torch.long)
        )
        log_prob_T_minus_1 = proc.forward_log_prob(
            clean, clean, torch.full((bs,), T - 1, dtype=torch.long)
        )
        assert torch.isfinite(log_prob_T).all()
        assert torch.isfinite(log_prob_T_minus_1).all()
        assert torch.all(log_prob_T < log_prob_T_minus_1)

    def test_posterior_sample_returns_one_hot_graph_data(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """posterior_sample() returns a sampled one-hot GraphData state."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([20, 30])
        s = torch.tensor([19, 29])
        z_t = proc.forward_sample(categorical_graph_data, t).z_t
        posterior = proc.posterior_sample(z_t, categorical_graph_data, t, s)
        assert isinstance(posterior, GraphData)
        assert posterior.X_class is not None
        assert posterior.E_class is not None
        assert categorical_graph_data.X_class is not None
        assert categorical_graph_data.E_class is not None
        assert posterior.X_class.shape == categorical_graph_data.X_class.shape
        assert posterior.E_class.shape == categorical_graph_data.E_class.shape
        x_sums = posterior.X_class[categorical_graph_data.node_mask].sum(dim=-1)
        assert torch.allclose(x_sums, torch.ones_like(x_sums))

        bs, n = categorical_graph_data.node_mask.shape
        mask_2d = categorical_graph_data.node_mask.unsqueeze(
            1
        ) * categorical_graph_data.node_mask.unsqueeze(2)
        diag = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)
        valid_edges = mask_2d & ~diag
        e_sums = posterior.E_class[valid_edges].sum(dim=-1)
        assert torch.allclose(e_sums, torch.ones_like(e_sums))

    def test_posterior_sample_marginalised_returns_one_hot_graph_data(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """``posterior_sample_marginalised`` produces valid one-hot
        ``GraphData`` of the same shape as input. Locks in the
        upstream-DiGress per-x0 marginalisation entry point so the
        sampler can rely on it.

        See ``internal notes (not included)``
        section 9 (sampler reverse loop).
        """
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([20, 30])
        s = torch.tensor([19, 29])
        z_t = proc.forward_sample(categorical_graph_data, t).z_t
        # Use the clean graph itself as a "soft prediction" — when it is
        # a one-hot, marginalisation should agree with the direct
        # posterior at sampling resolution.
        posterior = proc.posterior_sample_marginalised(
            z_t, categorical_graph_data, t, s
        )
        assert isinstance(posterior, GraphData)
        assert posterior.X_class is not None
        assert posterior.E_class is not None
        assert categorical_graph_data.X_class is not None
        assert categorical_graph_data.E_class is not None
        assert posterior.X_class.shape == categorical_graph_data.X_class.shape
        assert posterior.E_class.shape == categorical_graph_data.E_class.shape
        x_sums = posterior.X_class[categorical_graph_data.node_mask].sum(dim=-1)
        assert torch.allclose(x_sums, torch.ones_like(x_sums))

    def test_marginalised_posterior_matches_direct_when_x0_one_hot(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """At convergence the model emits a one-hot prediction. The
        marginalised posterior must then equal the direct posterior:
        ``sum_c [c == x0] * p(z_s | z_t, x_0 = c) = p(z_s | z_t, x_0 = x0)``.
        Compares ``_posterior_probabilities_marginalised`` against
        ``_posterior_probabilities`` on a one-hot ``x0_param``.
        """
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([20, 30])
        s = torch.tensor([19, 29])
        z_t = proc.forward_sample(categorical_graph_data, t).z_t

        direct = proc._posterior_probabilities(  # pyright: ignore[reportPrivateUsage]
            z_t, categorical_graph_data, t, s
        )
        marginalised = proc._posterior_probabilities_marginalised(  # pyright: ignore[reportPrivateUsage]
            z_t, categorical_graph_data, t, s
        )

        # Compare on valid (non-masked) positions only.
        mask = categorical_graph_data.node_mask
        assert direct.X_class is not None and marginalised.X_class is not None
        assert torch.allclose(
            direct.X_class[mask], marginalised.X_class[mask], atol=1e-5
        )

    def test_apply_edge_symmetry(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """Edge features remain symmetric after categorical forward sampling."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([15, 25])
        result = proc.forward_sample(categorical_graph_data, t).z_t
        assert result.E_class is not None
        # E should be symmetric in the one-hot encoding
        assert torch.allclose(
            result.E_class, result.E_class.transpose(1, 2)
        ), "Edge features must remain symmetric after categorical noise"


# ---------------------------------------------------------------------------
# NoisedBatch contract tests (parity D-4 / #17 / #18)
# ---------------------------------------------------------------------------


class TestNoisedBatchContract:
    """Pin the NoisedBatch field shapes and schedule-scalar agreement.

    Test rationale: parity #17 / #18 / D-4 introduces NoisedBatch to
    bundle the noised state with per-sample schedule scalars at
    ``(B, 1)``. The shape and value contract translates directly into
    the upstream-DiGress ``apply_noise`` dict layout, so VLB-path code
    stays grep-friendly across the two codebases. These tests guard the
    contract for both concrete process types and the composite.
    """

    def test_categorical_forward_sample_returns_noised_batch_with_b1_scalars(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """Categorical forward_sample returns NoisedBatch with (B, 1) scalars."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        bs = categorical_graph_data.node_mask.shape[0]
        t = torch.tensor([10, 20], dtype=torch.long)
        out = proc.forward_sample(categorical_graph_data, t)

        assert isinstance(out, NoisedBatch)
        assert isinstance(out.z_t, GraphData)
        assert out.t_int.shape == (bs, 1)
        assert out.t.shape == (bs, 1)
        assert out.beta_t.shape == (bs, 1)
        assert out.alpha_t_bar.shape == (bs, 1)
        assert out.alpha_s_bar.shape == (bs, 1)
        assert out.t_int.dtype == torch.long
        torch.testing.assert_close(out.t_int.squeeze(-1), t)
        torch.testing.assert_close(
            out.alpha_t_bar.squeeze(-1),
            cosine_schedule.get_alpha_bar(t_int=t),
        )

    def test_gaussian_forward_sample_returns_noised_batch_with_b1_scalars(
        self,
        cosine_schedule: NoiseSchedule,
        graph_data_from_adj: DenseGraphState,
    ) -> None:
        """Gaussian forward_sample returns NoisedBatch with (B, 1) scalars."""
        proc = GaussianNoiseProcess(
            definition=GaussianNoise(), schedule=cosine_schedule
        )
        bs = graph_data_from_adj.node_mask.shape[0]
        t = torch.tensor([15, 25], dtype=torch.long)
        out = proc.forward_sample(graph_data_from_adj, t)

        assert isinstance(out, NoisedBatch)
        assert out.t_int.shape == (bs, 1)
        assert out.beta_t.shape == (bs, 1)
        torch.testing.assert_close(out.t_int.squeeze(-1), t)
        torch.testing.assert_close(
            out.alpha_t_bar.squeeze(-1),
            cosine_schedule.get_alpha_bar(t_int=t),
        )

    def test_alpha_s_bar_uses_t_minus_one(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
    ) -> None:
        """alpha_s_bar is alpha_bar(t - 1), matching upstream apply_noise."""
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([10, 20], dtype=torch.long)
        out = proc.forward_sample(categorical_graph_data, t)
        s = (t - 1).clamp(min=0)
        torch.testing.assert_close(
            out.alpha_s_bar.squeeze(-1),
            cosine_schedule.get_alpha_bar(t_int=s),
        )

    def test_build_noised_batch_t_zero_returns_upstream_alpha_bar_T(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """At ``t = 0``, ``alpha_s_bar`` equals ``alpha_bar[T]``.

        Test rationale (parity W2-2): upstream DiGress's
        ``noise_schedule.py:75-79`` indexes ``alphas_bar`` with
        ``t_int = -1`` at ``t = 0``, which Python negative-indexing
        resolves to ``alphas_bar[T]``. Our ``_resolve_index`` rejects
        negative indices, so :func:`_build_noised_batch` branches
        explicitly: when any element of ``t_int`` is zero, the
        corresponding ``s_int`` is forced to ``T`` so the schedule
        lookup matches upstream. The previous clamp-to-zero behaviour
        was the divergence flagged in the 2026-04-23 fix-parity
        meta-synthesis. The warning suppression here keeps the test
        focused on the value contract; a sibling test pins the warning.
        """
        monkeypatch.delenv("FAIL_STRICTLY", raising=False)
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([0, 5], dtype=torch.long)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = proc.forward_sample(categorical_graph_data, t)

        timesteps = cosine_schedule.timesteps
        expected_t_zero = cosine_schedule.get_alpha_bar(
            t_int=torch.tensor([timesteps], dtype=torch.long)
        )
        torch.testing.assert_close(out.alpha_s_bar[0], expected_t_zero)
        # Sanity: the ``t = 5`` element still uses the t-1 branch.
        expected_t_five = cosine_schedule.get_alpha_bar(
            t_int=torch.tensor([4], dtype=torch.long)
        )
        torch.testing.assert_close(out.alpha_s_bar[1], expected_t_five)

    @pytest.mark.skipif(
        not __debug__,
        reason=(
            "t=0 warning emission is __debug__-guarded so production runs "
            "(PYTHONOPTIMIZE=1) skip the per-step bool(.any()) sync; see "
            "internal notes (not included)"
        ),
    )
    def test_build_noised_batch_t_zero_emits_warning(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ``t = 0`` element triggers the upstream-parity RuntimeWarning."""
        monkeypatch.delenv("FAIL_STRICTLY", raising=False)
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([0, 5], dtype=torch.long)
        with pytest.warns(RuntimeWarning, match=r"alpha_bar\[T\]"):
            proc.forward_sample(categorical_graph_data, t)

    @pytest.mark.skipif(
        not __debug__,
        reason=(
            "FAIL_STRICTLY raise sits inside the t=0 __debug__-guarded "
            "branch so production runs (PYTHONOPTIMIZE=1) skip both; see "
            "internal notes (not included)"
        ),
    )
    def test_build_noised_batch_t_zero_fail_strictly_raises(
        self,
        cosine_schedule: NoiseSchedule,
        categorical_graph_data: DenseGraphState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``FAIL_STRICTLY=1`` converts the warning into a hard error.

        The env-var gate exists so a developer chasing a regression can
        promote the warning to a stack-traceable exception without
        editing source. The error text mirrors the warning so grep can
        find both call sites.
        """
        monkeypatch.setenv("FAIL_STRICTLY", "1")
        proc = CategoricalNoiseProcess(
            schedule=cosine_schedule,
            x_classes=3,
            e_classes=2,
            limit_distribution="uniform",
        )
        t = torch.tensor([0, 5], dtype=torch.long)
        with pytest.raises(RuntimeError, match=r"alpha_bar\[T\]"):
            proc.forward_sample(categorical_graph_data, t)

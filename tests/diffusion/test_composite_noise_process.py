# pyright: reportAttributeAccessIssue=false
"""Tests for ``CompositeNoiseProcess`` (Wave 2.4).

Test rationale
--------------
The composite wrapper fans ``forward_sample`` / ``posterior_sample`` out to
a sequence of sub-processes with disjoint ``fields`` sets. The tests here
check:

- **Overlap detection.** Overlapping ``fields`` raise ``ValueError`` with a
  message that names every overlapping field so the operator can see which
  sub-processes collide.
- **Disjoint composition.** A DiGress-style categorical process and a
  Gaussian edge-weight process, composed, jointly noise their declared
  fields on a hybrid graph without dropping the other side's payload.
- **Conditioning vector shape.** ``process_state_condition_vector`` returns
  a 2-D tensor whose width is the sum of per-process projection widths, in
  list order.
- **Density delegation.** ``forward_log_prob`` / ``prior_log_prob``
  delegate to every ``ExactDensityNoiseProcess`` sub-process and sum.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.data.datasets.graph_types import DenseGraphState
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    CompositeNoiseProcess,
    GaussianNoiseProcess,
)
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.utils.noising.noise import GaussianNoise


@pytest.fixture()
def schedule() -> NoiseSchedule:
    """Cosine schedule with ``T = 50`` shared across all composite tests."""
    return NoiseSchedule(schedule_type="cosine_iddpm", timesteps=50)


@pytest.fixture()
def hybrid_graph_data() -> DenseGraphState:
    """Batch with one-hot ``X_class`` / ``E_class`` and continuous ``E_feat``.

    Shapes: ``bs=2``, ``n=4``, ``x_classes=3``, ``e_classes=2``,
    ``de_feat=1``. The edge-feature tensor is symmetric and zero-diagonal.
    """
    torch.manual_seed(0)
    bs, n, x_classes, e_classes = 2, 4, 3, 2
    x_class = torch.zeros(bs, n, x_classes)
    x_class[..., 0] = 1.0
    e_class = torch.zeros(bs, n, n, e_classes)
    e_class[..., 0] = 1.0
    e_feat = torch.randn(bs, n, n, 1)
    e_feat = 0.5 * (e_feat + e_feat.transpose(-2, -3))
    diag = torch.arange(n)
    e_feat[:, diag, diag, :] = 0.0
    num_nodes_per_graph = torch.full((bs,), n, dtype=torch.long)
    return DenseGraphState(
        num_nodes_per_graph=num_nodes_per_graph,
        y=torch.zeros(bs, 0),
        X_class=x_class,
        E_class=e_class,
        E_feat=e_feat,
    )


class TestCompositeConstruction:
    """Validation invariants at construction time."""

    def test_empty_processes_raise(self) -> None:
        """``CompositeNoiseProcess`` rejects empty sub-process sequences.

        Test rationale: the composite has no meaningful ``fields`` or
        timestep contract when no sub-processes are provided; catching
        this at construction prevents a confusing ``IndexError`` later.
        """
        with pytest.raises(ValueError, match="at least one sub-process"):
            CompositeNoiseProcess([])

    def test_overlapping_fields_raise(self, schedule: NoiseSchedule) -> None:
        """Overlapping field declarations raise with every conflict named.

        Test rationale: the disjointness invariant is the load-bearing
        guarantee that ``forward_sample`` / ``posterior_sample`` compose
        order-independently for each declared field. Surfacing the
        overlapping field names is the only way the operator can know
        which sub-processes to reconfigure.
        """
        cat1 = CategoricalNoiseProcess(
            schedule=schedule, x_classes=3, e_classes=2, limit_distribution="uniform"
        )
        cat2 = CategoricalNoiseProcess(
            schedule=schedule, x_classes=3, e_classes=2, limit_distribution="uniform"
        )
        with pytest.raises(ValueError, match="overlapping fields") as exc_info:
            CompositeNoiseProcess([cat1, cat2])
        # Both X_class and E_class are declared by both sub-processes.
        message = str(exc_info.value)
        assert "'X_class'" in message
        assert "'E_class'" in message

    def test_disjoint_construction_succeeds(self, schedule: NoiseSchedule) -> None:
        """Disjoint sub-processes construct successfully with the union ``fields``.

        Test rationale: categorical (X_class, E_class) + Gaussian
        (E_feat) is the motivating use case (hybrid continuous-edge
        DiGress); the composite must expose the union field set so the
        Lightning module's loss dispatch covers every owned field.
        """
        cat = CategoricalNoiseProcess(
            schedule=schedule, x_classes=3, e_classes=2, limit_distribution="uniform"
        )
        gauss = GaussianNoiseProcess(definition=GaussianNoise(), schedule=schedule)
        comp = CompositeNoiseProcess([cat, gauss])
        assert comp.fields == frozenset({"X_class", "E_class", "E_feat"})
        assert comp.timesteps == schedule.timesteps


class TestCompositeForward:
    """End-to-end composite forward + posterior sampling."""

    def test_forward_sample_noises_every_declared_field(
        self,
        hybrid_graph_data: DenseGraphState,
        schedule: NoiseSchedule,
    ) -> None:
        """Composite ``forward_sample`` noises all three declared fields jointly.

        Test rationale: disjoint fields imply no interference, so each
        field must end up noised by its owning sub-process and carry
        through to the composite output regardless of list order.
        """
        cat = CategoricalNoiseProcess(
            schedule=schedule, x_classes=3, e_classes=2, limit_distribution="uniform"
        )
        gauss = GaussianNoiseProcess(definition=GaussianNoise(), schedule=schedule)
        comp = CompositeNoiseProcess([cat, gauss])
        t = torch.tensor([40, 40], dtype=torch.long)
        out = comp.forward_sample(hybrid_graph_data, t).z_t

        # All declared fields must be present on the output.
        assert out.X_class is not None
        assert out.E_class is not None
        assert out.E_feat is not None

        # X_class and E_class are one-hot categorical outputs.
        torch.testing.assert_close(
            out.X_class.sum(dim=-1), torch.ones_like(out.X_class.sum(dim=-1))
        )
        torch.testing.assert_close(
            out.E_class.sum(dim=-1), torch.ones_like(out.E_class.sum(dim=-1))
        )

        # E_feat must differ from the input (Gaussian noise applied).
        assert hybrid_graph_data.E_feat is not None
        assert not torch.allclose(out.E_feat, hybrid_graph_data.E_feat)

    def test_condition_vector_concatenates_per_process_pieces(
        self,
        schedule: NoiseSchedule,
    ) -> None:
        """Composite conditioning has width equal to sum of sub-process widths.

        Test rationale: the architecture downstream consumes the
        composite conditioning directly; it must see a 2-D tensor of
        shape ``(bs, sum_d)`` where ``sum_d`` is fixed at construction
        time. Two scalar sub-processes produce width 2.
        """
        cat = CategoricalNoiseProcess(
            schedule=schedule, x_classes=3, e_classes=2, limit_distribution="uniform"
        )
        gauss = GaussianNoiseProcess(definition=GaussianNoise(), schedule=schedule)
        comp = CompositeNoiseProcess([cat, gauss])
        bs = 5
        t = torch.tensor([25] * bs, dtype=torch.long)
        vec = comp.process_state_condition_vector(t)
        assert vec.shape == (bs, 2)
        # Both sub-processes currently emit ``t / T``, so the two
        # columns are equal.
        torch.testing.assert_close(vec[:, 0], vec[:, 1])

    def test_loss_for_dispatches_per_field(self, schedule: NoiseSchedule) -> None:
        """``loss_for`` returns 'ce' for class fields and 'mse' for feat fields.

        Test rationale: the Lightning module's per-field loss sum
        consults ``loss_for(field)`` to pick between cross-entropy and
        mean-squared-error. Delegating to ``GRAPHDATA_LOSS_KIND``
        centralises the mapping so a future field change updates a
        single table.
        """
        cat = CategoricalNoiseProcess(
            schedule=schedule, x_classes=3, e_classes=2, limit_distribution="uniform"
        )
        gauss = GaussianNoiseProcess(definition=GaussianNoise(), schedule=schedule)
        comp = CompositeNoiseProcess([cat, gauss])
        assert comp.loss_for("X_class") == "ce"
        assert comp.loss_for("E_class") == "ce"
        assert comp.loss_for("E_feat") == "mse"
        with pytest.raises(KeyError, match="X_feat"):
            comp.loss_for("X_feat")

    def test_forward_log_prob_sums_per_sub_process(
        self,
        hybrid_graph_data: DenseGraphState,
        schedule: NoiseSchedule,
    ) -> None:
        """Composite ``forward_log_prob`` equals the sum of sub-process densities.

        Test rationale: with disjoint fields the joint forward density
        factorises across sub-processes, so the composite log-density
        is a straight sum. The test pins this invariant so a future
        cross-field coupling cannot silently break additivity.
        """
        cat = CategoricalNoiseProcess(
            schedule=schedule, x_classes=3, e_classes=2, limit_distribution="uniform"
        )
        gauss = GaussianNoiseProcess(definition=GaussianNoise(), schedule=schedule)
        comp = CompositeNoiseProcess([cat, gauss])
        t = torch.tensor([40, 40], dtype=torch.long)
        x_t = comp.forward_sample(hybrid_graph_data, t).z_t

        composite = comp.forward_log_prob(x_t, hybrid_graph_data, t)
        direct = cat.forward_log_prob(
            x_t, hybrid_graph_data, t
        ) + gauss.forward_log_prob(x_t, hybrid_graph_data, t)
        torch.testing.assert_close(composite, direct)


class TestReverseSamplerDispatchHook:
    """``posterior_sample_from_model_output`` routes every process uniformly.

    Test rationale (Wave 4.2): the reverse-sampler loop no longer
    branches on ``isinstance``; it calls
    ``posterior_sample_from_model_output`` on the active noise
    process. These tests pin the dispatch for each concrete subclass:

    - ``GaussianNoiseProcess`` falls back to the base-class default,
      which routes to :meth:`posterior_sample`.
    - ``CategoricalNoiseProcess`` overrides the hook to invoke
      :meth:`posterior_sample_marginalised` — the upstream-DiGress
      per-class marginalisation form.
    - ``CompositeNoiseProcess`` iterates sub-processes in list order,
      threading the running state through each hook. Because fields
      are disjoint, the composite result matches calling each
      sub-process's hook directly under the same random seed.
    """

    def test_gaussian_hook_matches_posterior_sample(
        self,
        hybrid_graph_data: DenseGraphState,
        schedule: NoiseSchedule,
    ) -> None:
        """Gaussian hook returns the same distribution as ``posterior_sample``."""
        gauss = GaussianNoiseProcess(definition=GaussianNoise(), schedule=schedule)
        t = torch.tensor([30, 30], dtype=torch.long)
        s = torch.tensor([29, 29], dtype=torch.long)
        # Sample ``z_t`` from the forward process so the posterior call
        # receives a well-shaped state for the declared fields.
        z_t = gauss.forward_sample(hybrid_graph_data, t).z_t

        torch.manual_seed(7)
        hook = gauss.posterior_sample_from_model_output(z_t, hybrid_graph_data, t, s)
        torch.manual_seed(7)
        direct = gauss.posterior_sample(z_t, hybrid_graph_data, t, s)
        assert hook.E_feat is not None
        assert direct.E_feat is not None
        torch.testing.assert_close(hook.E_feat, direct.E_feat)

    def test_categorical_hook_routes_to_marginalised_form(
        self,
        hybrid_graph_data: DenseGraphState,
        schedule: NoiseSchedule,
    ) -> None:
        """Categorical hook matches the marginalised posterior, not the direct.

        Seeds the RNG once before each call; the two sampling paths
        operate on the same distribution and therefore produce the
        same one-hot draw when fed with a converged one-hot
        prediction. For a softer prediction the two diverge, pinning
        the routing decision.
        """
        cat = CategoricalNoiseProcess(
            schedule=schedule, x_classes=3, e_classes=2, limit_distribution="uniform"
        )
        t = torch.tensor([30, 30], dtype=torch.long)
        s = torch.tensor([29, 29], dtype=torch.long)
        z_t = cat.forward_sample(hybrid_graph_data, t).z_t

        # Soft prediction: fully diffuse over classes. The two code
        # paths disagree numerically under this prediction, so
        # matching ``posterior_sample_marginalised`` (not
        # ``posterior_sample``) is a load-bearing assertion.
        bs, n = z_t.node_mask.shape
        soft_x = torch.full((bs, n, 3), 1.0 / 3.0)
        soft_e = torch.full((bs, n, n, 2), 0.5)
        soft = hybrid_graph_data.replace(X_class=soft_x, E_class=soft_e)

        torch.manual_seed(11)
        hook = cat.posterior_sample_from_model_output(z_t, soft, t, s)
        torch.manual_seed(11)
        marginalised = cat.posterior_sample_marginalised(z_t, soft, t, s)
        assert hook.X_class is not None
        assert marginalised.X_class is not None
        torch.testing.assert_close(hook.X_class, marginalised.X_class)
        torch.testing.assert_close(hook.E_class, marginalised.E_class)

    def test_composite_hook_threads_sub_processes_in_order(
        self,
        hybrid_graph_data: DenseGraphState,
        schedule: NoiseSchedule,
    ) -> None:
        """Composite hook equals iterating each sub-process's hook manually."""
        cat = CategoricalNoiseProcess(
            schedule=schedule, x_classes=3, e_classes=2, limit_distribution="uniform"
        )
        gauss = GaussianNoiseProcess(definition=GaussianNoise(), schedule=schedule)
        comp = CompositeNoiseProcess([cat, gauss])

        t = torch.tensor([30, 30], dtype=torch.long)
        s = torch.tensor([29, 29], dtype=torch.long)
        z_t = comp.forward_sample(hybrid_graph_data, t).z_t

        # Manual iteration under the same seed must agree with the
        # composite's hook call.
        torch.manual_seed(13)
        comp_out = comp.posterior_sample_from_model_output(z_t, hybrid_graph_data, t, s)

        torch.manual_seed(13)
        manual = cat.posterior_sample_from_model_output(z_t, hybrid_graph_data, t, s)
        manual = gauss.posterior_sample_from_model_output(
            manual, hybrid_graph_data, t, s
        )

        assert comp_out.X_class is not None
        assert manual.X_class is not None
        torch.testing.assert_close(comp_out.X_class, manual.X_class)
        torch.testing.assert_close(comp_out.E_class, manual.E_class)
        assert comp_out.E_feat is not None
        assert manual.E_feat is not None
        torch.testing.assert_close(comp_out.E_feat, manual.E_feat)

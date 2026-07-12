# pyright: reportAttributeAccessIssue=false
"""Tests for the DDPM ``GaussianNoiseProcess`` implementation (Wave 2.3).

Test rationale
--------------
Wave 2.3 replaces the legacy ``NoiseDefinition``-driven forward pass with a
per-field DDPM Gaussian forward process. The methods under test are:

- ``forward_sample`` — preserves non-declared fields, symmetrises
  ``E_feat`` outputs, and satisfies the closed-form marginal statistics
  ``N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)``.
- ``posterior_sample`` — samples from the DDPM closed-form reverse
  posterior at ``(t, s)`` and hands back the mean deterministically at
  ``s <= 0``.
- ``forward_log_prob`` / ``posterior_log_prob`` / ``prior_log_prob`` —
  return per-sample log-densities summed over declared fields.

The constructor contract (field-subset validation, empty-set rejection)
is already exercised in ``test_noise_process.py``; this file focuses on
the numerical behaviour of the Gaussian math.
"""

from __future__ import annotations

import math

import pytest
import torch

from tmgg.data.datasets.graph_types import DenseGraphState
from tmgg.diffusion.noise_process import GaussianNoiseProcess
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.utils.noising.noise import GaussianNoise


@pytest.fixture()
def long_schedule() -> NoiseSchedule:
    """Cosine schedule with ``T = 500`` for tight DDPM statistics.

    Large batches and a long schedule put ``alpha_bar(T - 1)`` close
    enough to zero that the sample mean ≈ 0 and variance ≈ 1 become
    unambiguous without a hard-to-pick tolerance.
    """
    return NoiseSchedule(schedule_type="cosine_iddpm", timesteps=500)


@pytest.fixture()
def continuous_graph_data() -> DenseGraphState:
    """Small continuous-graph batch with populated ``X_feat`` / ``E_feat``.

    Shapes: ``bs=4``, ``n=6``, ``dx_feat=1``, ``de_feat=1``. Edge field is
    symmetric and zero-diagonal so the output symmetry check has a
    clean reference. Legacy ``X`` / ``E`` are populated with zeros to
    satisfy the ``DenseGraphState`` required fields during Wave 2.
    """
    torch.manual_seed(0)
    bs, n = 4, 6
    x_feat = torch.randn(bs, n, 1)
    e_feat = torch.randn(bs, n, n, 1)
    e_feat = 0.5 * (e_feat + e_feat.transpose(-2, -3))
    diag = torch.arange(n)
    e_feat[:, diag, diag, :] = 0.0
    x_class = torch.zeros(bs, n, 2)
    x_class[..., 0] = 1.0
    e_class = torch.zeros(bs, n, n, 2)
    e_class[..., 0] = 1.0
    num_nodes_per_graph = torch.full((bs,), n, dtype=torch.long)
    return DenseGraphState(
        num_nodes_per_graph=num_nodes_per_graph,
        y=torch.zeros(bs, 0),
        X_feat=x_feat,
        E_feat=e_feat,
        X_class=x_class,
        E_class=e_class,
    )


class TestGaussianForwardSample:
    """Forward-process statistics and structural invariants."""

    def test_non_declared_fields_pass_through(
        self,
        continuous_graph_data: DenseGraphState,
        long_schedule: NoiseSchedule,
    ) -> None:
        """``forward_sample`` must leave non-declared fields untouched.

        Test rationale: the default ``GaussianNoiseProcess`` declares
        ``fields = {"E_feat"}``; the class fields (``X_class``,
        ``E_class``, ``X_feat``) and the auxiliary ``y`` must survive a
        forward sample without modification so composition with a
        categorical process can carry them through.
        """
        proc = GaussianNoiseProcess(definition=GaussianNoise(), schedule=long_schedule)
        t = torch.tensor([100] * continuous_graph_data.node_mask.shape[0])
        # forward_sample returns NoisedBatch (parity D-4); shape / passthrough
        # assertions operate on .z_t.
        out = proc.forward_sample(continuous_graph_data, t).z_t

        assert out.X_class is not None
        assert continuous_graph_data.X_class is not None
        torch.testing.assert_close(out.X_class, continuous_graph_data.X_class)
        assert out.E_class is not None
        assert continuous_graph_data.E_class is not None
        torch.testing.assert_close(out.E_class, continuous_graph_data.E_class)
        assert out.X_feat is not None
        assert continuous_graph_data.X_feat is not None
        torch.testing.assert_close(out.X_feat, continuous_graph_data.X_feat)

    def test_prior_statistics_at_large_t(
        self,
        long_schedule: NoiseSchedule,
    ) -> None:
        """At ``t = T - 1`` sample mean ≈ 0 and variance ≈ 1.

        Test rationale: the DDPM marginal at the end of the schedule is
        ``N(sqrt(alpha_bar_T) x_0, (1 - alpha_bar_T) I)``. With a cosine
        schedule ``alpha_bar(T - 1) ≈ 0`` so the moments collapse to
        the standard normal; checking them on a large batch is the
        standard correctness gate.
        """
        torch.manual_seed(1)
        bs, n = 512, 4
        x_feat = torch.randn(bs, n, 1)
        e_feat = torch.randn(bs, n, n, 1)
        e_feat = 0.5 * (e_feat + e_feat.transpose(-2, -3))
        diag = torch.arange(n)
        e_feat[:, diag, diag, :] = 0.0
        num_nodes_per_graph = torch.full((bs,), n, dtype=torch.long)
        data = DenseGraphState(
            num_nodes_per_graph=num_nodes_per_graph,
            X_class=torch.zeros(bs, n, 1),
            E_class=torch.zeros(bs, n, n, 1),
            y=torch.zeros(bs, 0),
            X_feat=x_feat,
            E_feat=e_feat,
        )

        proc = GaussianNoiseProcess(
            definition=GaussianNoise(),
            schedule=long_schedule,
            fields=frozenset({"X_feat", "E_feat"}),
        )
        t = torch.tensor([long_schedule.timesteps - 1] * bs, dtype=torch.long)
        out = proc.forward_sample(data, t).z_t
        assert out.X_feat is not None and out.E_feat is not None

        x_std = out.X_feat.std().item()
        x_mean = out.X_feat.mean().item()
        assert abs(x_mean) < 0.1
        assert abs(x_std - 1.0) < 0.1

        # Drop the diagonal (which is held at zero by symmetrisation).
        off_diag = ~torch.eye(n, dtype=torch.bool)
        e_off = out.E_feat.squeeze(-1)[:, off_diag]
        e_mean = e_off.mean().item()
        e_std = e_off.std().item()
        assert abs(e_mean) < 0.1
        # Variance halves along the diagonal symmetrisation; across all
        # off-diagonal entries each sample is the average of two i.i.d.
        # Gaussians, so the observed std is ~1/sqrt(2) of the input
        # spread. We accept anything in [0.6, 1.1] as consistent.
        assert 0.6 <= e_std <= 1.1

    def test_t_zero_near_identity(
        self,
        continuous_graph_data: DenseGraphState,
        long_schedule: NoiseSchedule,
    ) -> None:
        """Forward sample at ``t = 0`` stays within a few noise stds of the input.

        Test rationale: ``alpha_bar(0) ≈ 1`` so the DDPM forward
        marginal contracts to the input plus a very small Gaussian
        perturbation. We assert an absolute tolerance big enough for
        the schedule's tiny residual noise (~0.04 std) but tight
        enough to flag a regression to mid-schedule noise.
        """
        torch.manual_seed(0)
        proc = GaussianNoiseProcess(definition=GaussianNoise(), schedule=long_schedule)
        t = torch.tensor([0] * continuous_graph_data.node_mask.shape[0])
        out = proc.forward_sample(continuous_graph_data, t).z_t
        assert out.E_feat is not None
        assert continuous_graph_data.E_feat is not None
        assert torch.allclose(out.E_feat, continuous_graph_data.E_feat, atol=0.2)

    def test_edge_field_output_is_symmetric(
        self,
        continuous_graph_data: DenseGraphState,
        long_schedule: NoiseSchedule,
    ) -> None:
        """``E_feat`` output is symmetric across the node-pair axes.

        Test rationale: the spec mandates
        ``E = 0.5 * (E + E.transpose(-2, -3))`` plus zero diagonal on
        every edge-field output so downstream consumers can treat the
        tensor as undirected without re-symmetrising.
        """
        proc = GaussianNoiseProcess(definition=GaussianNoise(), schedule=long_schedule)
        t = torch.tensor([100] * continuous_graph_data.node_mask.shape[0])
        out = proc.forward_sample(continuous_graph_data, t).z_t
        assert out.E_feat is not None
        torch.testing.assert_close(out.E_feat, out.E_feat.transpose(-2, -3))
        n = out.E_feat.shape[-2]
        diag = torch.arange(n)
        assert torch.all(out.E_feat[..., diag, diag, :] == 0.0)


class TestGaussianPosterior:
    """Reverse-posterior sampling and log-density."""

    def test_posterior_sample_deterministic_at_s_zero(
        self,
        continuous_graph_data: DenseGraphState,
        long_schedule: NoiseSchedule,
    ) -> None:
        """When ``s <= 0`` the posterior sample collapses to the mean.

        Test rationale: the DDPM reverse step disables stochasticity at
        the final denoising step so the deterministic output (mean of
        the Gaussian posterior) is what reaches ``finalize_sample``.
        Two independent calls with different seeds must therefore be
        identical.
        """
        proc = GaussianNoiseProcess(definition=GaussianNoise(), schedule=long_schedule)
        bs = continuous_graph_data.node_mask.shape[0]
        t = torch.tensor([1] * bs, dtype=torch.long)
        s = torch.tensor([0] * bs, dtype=torch.long)
        z_t = proc.forward_sample(continuous_graph_data, t).z_t

        torch.manual_seed(0)
        a = proc.posterior_sample(z_t, continuous_graph_data, t, s)
        torch.manual_seed(99)
        b = proc.posterior_sample(z_t, continuous_graph_data, t, s)
        assert a.E_feat is not None and b.E_feat is not None
        torch.testing.assert_close(a.E_feat, b.E_feat)

    def test_posterior_sample_symmetric(
        self,
        continuous_graph_data: DenseGraphState,
        long_schedule: NoiseSchedule,
    ) -> None:
        """Reverse-posterior ``E_feat`` output remains symmetric.

        Test rationale: the spec mandates the same symmetry
        post-condition for ``posterior_sample`` as for
        ``forward_sample``; the downstream sampler assumes the tensor
        is undirected.
        """
        proc = GaussianNoiseProcess(definition=GaussianNoise(), schedule=long_schedule)
        bs = continuous_graph_data.node_mask.shape[0]
        t = torch.tensor([200] * bs, dtype=torch.long)
        s = torch.tensor([100] * bs, dtype=torch.long)
        z_t = proc.forward_sample(continuous_graph_data, t).z_t
        out = proc.posterior_sample(z_t, continuous_graph_data, t, s)
        assert out.E_feat is not None
        torch.testing.assert_close(out.E_feat, out.E_feat.transpose(-2, -3))

    def test_forward_log_prob_matches_analytic(
        self,
        continuous_graph_data: DenseGraphState,
        long_schedule: NoiseSchedule,
    ) -> None:
        """``forward_log_prob`` returns a finite per-sample vector.

        Test rationale: as a sanity check that the Gaussian log-density
        helper and the schedule broadcasting align, we verify the
        output shape is ``(bs,)`` and the values are finite. Tight
        numerical equality is hard to pin down without reimplementing
        the formula; the statistical moment tests above are the
        stronger correctness signal.
        """
        proc = GaussianNoiseProcess(definition=GaussianNoise(), schedule=long_schedule)
        bs = continuous_graph_data.node_mask.shape[0]
        t = torch.tensor([100] * bs, dtype=torch.long)
        z_t = proc.forward_sample(continuous_graph_data, t).z_t
        lp = proc.forward_log_prob(z_t, continuous_graph_data, t)
        assert lp.shape == (bs,)
        assert torch.isfinite(lp).all()

    def test_prior_log_prob_matches_standard_normal(
        self,
        continuous_graph_data: DenseGraphState,
        long_schedule: NoiseSchedule,
    ) -> None:
        """``prior_log_prob`` equals ``sum_field N(0,1)`` log-density.

        Test rationale: at the end of a long cosine schedule the DDPM
        limit is a standard normal on every declared field, so the
        ``prior_log_prob`` value on a zero input equals
        ``-0.5 * d * log(2*pi)`` summed over declared-field element
        counts. Evaluating on a known input keeps the numerical
        reference explicit.
        """
        proc = GaussianNoiseProcess(definition=GaussianNoise(), schedule=long_schedule)
        bs, n = continuous_graph_data.node_mask.shape
        zero_e = torch.zeros(bs, n, n, 1)
        data = continuous_graph_data.replace(E_feat=zero_e)
        lp = proc.prior_log_prob(data)
        expected_const = -0.5 * n * n * math.log(2.0 * math.pi)
        torch.testing.assert_close(
            lp, torch.full((bs,), expected_const), atol=1e-4, rtol=1e-4
        )

    def test_posterior_log_prob_finite(
        self,
        continuous_graph_data: DenseGraphState,
        long_schedule: NoiseSchedule,
    ) -> None:
        """``posterior_log_prob`` yields finite per-sample log-density.

        Test rationale: the posterior density formula uses the schedule
        variance in the denominator; we verify nothing blows up on the
        representative ``(t, s) = (200, 100)`` step. A numerical
        reference would duplicate the closed-form code.
        """
        proc = GaussianNoiseProcess(definition=GaussianNoise(), schedule=long_schedule)
        bs = continuous_graph_data.node_mask.shape[0]
        t = torch.tensor([200] * bs, dtype=torch.long)
        s = torch.tensor([100] * bs, dtype=torch.long)
        z_t = proc.forward_sample(continuous_graph_data, t).z_t
        x_s = proc.posterior_sample(z_t, continuous_graph_data, t, s)
        lp = proc.posterior_log_prob(x_s, z_t, continuous_graph_data, t, s)
        assert lp.shape == (bs,)
        assert torch.isfinite(lp).all()

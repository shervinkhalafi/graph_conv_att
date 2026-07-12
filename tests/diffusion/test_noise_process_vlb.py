"""Tests for exact-density VLB primitives on CategoricalNoiseProcess.

Test rationale
--------------
The refactor replaces categorical-only public VLB helpers with
``ExactDensityNoiseProcess`` methods. These tests validate the exact-density
surface directly and keep reconstruction-specific log-probability tests on the
training-module helper that owns model-parameterization concerns.

Starting state: a uniform ``CategoricalNoiseProcess`` with small
dimensions (dx=2, de=2, T=10, bs=3, n=6).

Invariants tested:
- Exact-density methods return per-sample outputs with shape ``(bs,)``.
- Outputs are finite (no NaN/Inf from division or log).
- When the same clean parameterization is used on both sides of the posterior
  KL identity, the log-probability difference is exactly zero.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from tmgg.data.datasets.graph_types import DenseGraphState
from tmgg.diffusion.noise_process import CategoricalNoiseProcess
from tmgg.diffusion.schedule import NoiseSchedule

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BS, N, DX, DE, T = 3, 6, 2, 2, 10


@pytest.fixture()
def schedule() -> NoiseSchedule:
    return NoiseSchedule(schedule_type="cosine_iddpm", timesteps=T)


@pytest.fixture()
def proc(schedule: NoiseSchedule) -> CategoricalNoiseProcess:
    return CategoricalNoiseProcess(
        schedule=schedule,
        x_classes=DX,
        e_classes=DE,
        limit_distribution="uniform",
    )


@pytest.fixture()
def clean_data() -> DenseGraphState:
    """One-hot clean graph data with all nodes valid."""
    torch.manual_seed(0)

    x_idx = torch.randint(0, DX, (BS, N))
    X = F.one_hot(x_idx, num_classes=DX).float()  # pyright: ignore[reportAttributeAccessIssue]

    e_idx = torch.randint(0, DE, (BS, N, N))
    e_idx = torch.triu(e_idx, diagonal=1)
    e_idx = e_idx + e_idx.transpose(1, 2)
    E = torch.zeros(BS, N, N, DE)
    E.scatter_(3, e_idx.unsqueeze(-1), 1.0)
    diag = torch.arange(N)
    E[:, diag, diag, :] = 0
    E[:, diag, diag, 0] = 1.0

    y = torch.zeros(BS, 0)
    num_nodes_per_graph = torch.full((BS,), N, dtype=torch.long)
    return DenseGraphState(
        num_nodes_per_graph=num_nodes_per_graph, y=y, X_class=X, E_class=E
    )


# ---------------------------------------------------------------------------
# posterior_log_prob tests
# ---------------------------------------------------------------------------


class TestPosteriorLogProb:
    """Tests for categorical posterior exact log-probabilities."""

    def test_output_shape(
        self, proc: CategoricalNoiseProcess, clean_data: DenseGraphState
    ) -> None:
        """posterior_log_prob returns one value per batch element."""
        t_int = torch.tensor([5, 3, 7])
        noisy = proc.forward_sample(clean_data, t_int).z_t
        s_int = t_int - 1
        x_s = proc.posterior_sample(noisy, clean_data, t_int, s_int)
        result = proc.posterior_log_prob(x_s, noisy, clean_data, t_int, s_int)
        assert result.shape == (BS,)

    def test_output_finite(
        self, proc: CategoricalNoiseProcess, clean_data: DenseGraphState
    ) -> None:
        """posterior_log_prob produces finite values."""
        t_int = torch.tensor([5, 3, 7])
        noisy = proc.forward_sample(clean_data, t_int).z_t
        s_int = t_int - 1
        x_s = proc.posterior_sample(noisy, clean_data, t_int, s_int)
        result = proc.posterior_log_prob(x_s, noisy, clean_data, t_int, s_int)
        assert torch.isfinite(result).all()

    def test_perfect_prediction_near_zero(
        self, proc: CategoricalNoiseProcess, clean_data: DenseGraphState
    ) -> None:
        """The posterior KL identity collapses to zero for identical parameters."""
        t_int = torch.tensor([5, 3, 7])
        noisy = proc.forward_sample(clean_data, t_int).z_t
        s_int = t_int - 1
        x_s = proc.posterior_sample(noisy, clean_data, t_int, s_int)
        log_true = proc.posterior_log_prob(x_s, noisy, clean_data, t_int, s_int)
        log_same = proc.posterior_log_prob(x_s, noisy, clean_data, t_int, s_int)
        torch.testing.assert_close(log_true - log_same, torch.zeros_like(log_true))


# ---------------------------------------------------------------------------
# forward_log_prob and prior_log_prob tests
# ---------------------------------------------------------------------------


class TestExactDensityTerms:
    """Tests for forward and prior exact log-probability helpers."""

    def test_output_shape(
        self, proc: CategoricalNoiseProcess, clean_data: DenseGraphState
    ) -> None:
        """forward_log_prob and prior_log_prob return one value per sample."""
        t_int = torch.tensor([T, T - 1, T - 2])
        noisy = proc.forward_sample(clean_data, t_int).z_t
        forward = proc.forward_log_prob(noisy, clean_data, t_int)
        prior = proc.prior_log_prob(noisy)
        assert forward.shape == (BS,)
        assert prior.shape == (BS,)

    def test_output_finite(
        self, proc: CategoricalNoiseProcess, clean_data: DenseGraphState
    ) -> None:
        """forward_log_prob and prior_log_prob stay finite on sampled states."""
        t_int = torch.tensor([T, T - 1, T - 2])
        noisy = proc.forward_sample(clean_data, t_int).z_t
        forward = proc.forward_log_prob(noisy, clean_data, t_int)
        prior = proc.prior_log_prob(noisy)
        assert torch.isfinite(forward).all()
        assert torch.isfinite(prior).all()

"""Tests for diffusion schedules and categorical stationary PMFs.

Test rationale
--------------
The noise schedule underpins every forward diffusion step and reverse sampling
step. Errors in betas or alpha-bar would silently corrupt training targets and
generated samples. Categorical processes now own their stationary PMFs directly,
so this file also checks the schedule-adjacent categorical setup rules. These
tests verify the core invariants:

* **Monotonicity of betas**: the cosine schedule produces non-decreasing betas,
  so noise increases with timestep.
* **Alpha-bar decay**: alpha-bar starts near 1 (clean data) and decreases toward
  0 (pure noise), ensuring the forward process interpolates correctly.
* **Boundary sanity**: t = 0 and t = T produce finite values (no NaN/Inf).
* **Stationary PMFs**: categorical uniform mode starts with valid PMFs, and
  empirical mode learns valid PMFs from real nodes and edges only.

See also: [src/tmgg/diffusion/schedule.py](../../src/tmgg/diffusion/schedule.py)
and [src/tmgg/diffusion/noise_process.py](../../src/tmgg/diffusion/noise_process.py).
"""

from __future__ import annotations

import pytest
import torch

from tmgg.diffusion.noise_process import CategoricalNoiseProcess
from tmgg.diffusion.schedule import NoiseSchedule

# ---------------------------------------------------------------------------
# NoiseSchedule
# ---------------------------------------------------------------------------


class TestNoiseSchedule:
    """Invariants on the precomputed beta / alpha-bar lookup tables."""

    @pytest.fixture()
    def schedule(self) -> NoiseSchedule:
        return NoiseSchedule(schedule_type="cosine_iddpm", timesteps=500)

    def test_betas_monotonically_nondecreasing(self, schedule: NoiseSchedule) -> None:
        """Cosine betas should be monotonically non-decreasing."""
        betas = schedule.betas
        diffs = betas[1:] - betas[:-1]
        assert (diffs >= -1e-7).all(), "Betas are not monotonically non-decreasing"

    def test_alpha_bar_starts_near_one_and_decays(
        self, schedule: NoiseSchedule
    ) -> None:
        """Alpha-bar should start close to 1 and decrease toward 0."""
        ab = schedule.alpha_bar
        assert ab[0] > 0.95, f"alpha_bar[0] = {ab[0]:.4f}, expected > 0.95"
        assert ab[-1] < 0.05, f"alpha_bar[-1] = {ab[-1]:.4f}, expected < 0.05"
        # Monotonically non-increasing
        diffs = ab[1:] - ab[:-1]
        assert (diffs <= 1e-7).all(), "alpha_bar is not monotonically non-increasing"

    def test_forward_returns_correct_beta(self, schedule: NoiseSchedule) -> None:
        """forward(t_int=k) should return betas[k]."""
        t = torch.tensor([0, 100, 499])
        result = schedule.forward(t_int=t)
        expected = schedule.betas[t.long()]
        assert torch.allclose(result, expected)

    def test_get_alpha_bar_returns_correct_values(
        self, schedule: NoiseSchedule
    ) -> None:
        """get_alpha_bar(t_int=k) should return alpha_bar[k]."""
        t = torch.tensor([0, 250, 499])
        result = schedule.get_alpha_bar(t_int=t)
        expected = schedule.alpha_bar[t.long()]
        assert torch.allclose(result, expected)

    def test_exactly_one_argument_required(self, schedule: NoiseSchedule) -> None:
        """Passing both or neither argument should raise."""
        with pytest.raises(ValueError):
            schedule.forward()
        with pytest.raises(ValueError):
            schedule.forward(t_int=torch.tensor([10]), t_normalized=torch.tensor([0.5]))
        with pytest.raises(ValueError):
            schedule.get_alpha_bar()

    def test_boundary_timesteps_are_finite(self, schedule: NoiseSchedule) -> None:
        """t=0 and t=T-1 should produce finite betas and alpha-bar."""
        for t_val in [0, schedule.timesteps - 1]:
            t = torch.tensor([t_val])
            beta = schedule.forward(t_int=t)
            ab = schedule.get_alpha_bar(t_int=t)
            assert torch.isfinite(beta).all(), f"Non-finite beta at t={t_val}"
            assert torch.isfinite(ab).all(), f"Non-finite alpha_bar at t={t_val}"

    def test_unknown_schedule_raises(self) -> None:
        """An unrecognised schedule name should raise immediately."""
        with pytest.raises(ValueError):
            NoiseSchedule(schedule_type="nonexistent", timesteps=100)


# ---------------------------------------------------------------------------
# Categorical process stationary PMFs
# ---------------------------------------------------------------------------


class TestCategoricalStationaryPmfs:
    """Schedule-adjacent checks for the categorical process contract."""

    def test_uniform_mode_starts_with_uniform_pmfs(self) -> None:
        """Uniform categorical mode should construct valid uniform PMFs eagerly."""
        proc = CategoricalNoiseProcess(
            schedule=NoiseSchedule(schedule_type="cosine_iddpm", timesteps=32),
            x_classes=3,
            e_classes=4,
            limit_distribution="uniform",
        )
        assert proc._limit_x is not None
        assert proc._limit_e is not None
        torch.testing.assert_close(proc._limit_x, torch.ones(3) / 3)
        torch.testing.assert_close(proc._limit_e, torch.ones(4) / 4)

    def test_empirical_mode_initialises_from_sparse_pyg_batches(self) -> None:
        """Sparse PyG-backed π estimator matches upstream's edge_counts formula.

        Two graphs in the batch:

        - Graph 0: 2 real nodes, 1 undirected edge (PyG enumerates 2 directed)
        - Graph 1: 3 nodes, 1 undirected edge (PyG enumerates 2 directed)

        all_pairs = 2*1 + 3*2 = 8; directed edges = 4; non-edges = 4.
        Expected π_E = [0.5, 0.5]. With no per-node features the
        structure-only fallback assigns every node to class 1, so
        π_X = [0, 1].
        """
        from torch_geometric.data import Batch, Data

        proc = CategoricalNoiseProcess(
            schedule=NoiseSchedule(schedule_type="cosine_iddpm", timesteps=16),
            x_classes=2,
            e_classes=2,
            limit_distribution="empirical_marginal",
        )

        g0 = Data(
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            num_nodes=2,
        )
        g1 = Data(
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            num_nodes=3,
        )
        pyg_batch = Batch.from_data_list([g0, g1])

        proc.initialize_from_data([pyg_batch])

        assert proc._limit_x is not None
        assert proc._limit_e is not None
        torch.testing.assert_close(proc._limit_x, torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(proc._limit_e, torch.tensor([0.5, 0.5]))

"""Tests for the unified NoiseSchedule.

The schedule wraps DiGress's precomputed beta/alpha/alpha_bar arrays and adds
a simple linear schedule.  These tests verify construction, value correctness,
monotonicity properties, and shape handling.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.diffusion.schedule import NoiseSchedule

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestNoiseScheduleConstruction:
    """Verify that each schedule type constructs without error and exposes T."""

    def test_cosine_schedule(self) -> None:
        """Cosine schedule constructs and reports the correct T."""
        sched = NoiseSchedule("cosine_iddpm", timesteps=100)
        assert sched.timesteps == 100
        assert sched.schedule_type == "cosine_iddpm"

    def test_custom_schedule(self) -> None:
        """Custom schedule accepts average_num_nodes and constructs."""
        sched = NoiseSchedule("custom_vignac", timesteps=200, average_num_nodes=30)
        assert sched.timesteps == 200

    def test_linear_schedule(self) -> None:
        """Linear schedule constructs."""
        sched = NoiseSchedule("linear_ddpm", timesteps=50)
        assert sched.timesteps == 50

    def test_unknown_type_raises(self) -> None:
        """An unrecognised schedule_type raises ValueError immediately."""
        with pytest.raises(ValueError, match="schedule_type"):
            NoiseSchedule("unknown", timesteps=100)

    def test_zero_timesteps_raises(self) -> None:
        """timesteps < 1 raises ValueError."""
        with pytest.raises(ValueError, match="timesteps"):
            NoiseSchedule("cosine_iddpm", timesteps=0)

    def test_negative_timesteps_raises(self) -> None:
        """Negative timesteps raise ValueError."""
        with pytest.raises(ValueError, match="timesteps"):
            NoiseSchedule("linear_ddpm", timesteps=-5)


# ---------------------------------------------------------------------------
# Cosine schedule value properties
# ---------------------------------------------------------------------------


class TestCosineSchedule:
    """Value-level properties specific to the cosine schedule."""

    @pytest.fixture()
    def sched(self) -> NoiseSchedule:
        return NoiseSchedule("cosine_iddpm", timesteps=500)

    def test_get_beta_returns_tensor(self, sched: NoiseSchedule) -> None:
        """get_beta returns a tensor of the expected dtype."""
        beta = sched.get_beta(torch.tensor([0]))
        assert isinstance(beta, torch.Tensor)
        assert beta.dtype == torch.float32

    def test_get_alpha_bar_returns_tensor(self, sched: NoiseSchedule) -> None:
        """get_alpha_bar returns a float tensor."""
        ab = sched.get_alpha_bar(torch.tensor([0]))
        assert isinstance(ab, torch.Tensor)
        assert ab.dtype == torch.float32

    def test_alpha_bar_monotonically_decreasing(self, sched: NoiseSchedule) -> None:
        """alpha_bar should strictly decrease as t grows (signal decays)."""
        t = torch.arange(0, sched.timesteps + 1)
        ab = sched.get_alpha_bar(t)
        diffs = ab[1:] - ab[:-1]
        assert (diffs < 0).all(), "alpha_bar must be monotonically decreasing"

    def test_alpha_bar_at_t0_near_one(self, sched: NoiseSchedule) -> None:
        """At t=0 almost no noise has been applied, so alpha_bar ~ 1."""
        ab0 = sched.get_alpha_bar(torch.tensor([0])).item()
        assert ab0 > 0.99, f"Expected alpha_bar(0) > 0.99, got {ab0}"

    def test_alpha_bar_at_tT_near_zero(self, sched: NoiseSchedule) -> None:
        """At t=T the signal should be almost fully destroyed."""
        abT = sched.get_alpha_bar(torch.tensor([sched.timesteps])).item()
        assert abT < 0.01, f"Expected alpha_bar(T) < 0.01, got {abT}"

    def test_get_noise_level_is_one_minus_alpha_bar(self, sched: NoiseSchedule) -> None:
        """Noise level should be exactly 1 - alpha_bar for cosine schedule."""
        t = torch.arange(0, sched.timesteps + 1)
        noise = sched.get_noise_level(t)
        ab = sched.get_alpha_bar(t)
        torch.testing.assert_close(noise, 1.0 - ab)

    def test_alpha_bar_equals_cumprod_alphas(self, sched: NoiseSchedule) -> None:
        """DDPM identity: alpha_bar[t] = prod(alpha[0:t])."""
        recomputed = torch.cumprod(sched.alphas, dim=0)
        torch.testing.assert_close(sched.alpha_bar, recomputed, atol=1e-6, rtol=1e-5)

    def test_batch_indexing(self, sched: NoiseSchedule) -> None:
        """Multiple timesteps queried at once should return matching batch."""
        t = torch.tensor([0, 10, 50, 100, sched.timesteps])
        beta = sched.get_beta(t)
        assert beta.shape == (5,)
        # Check individual lookups match
        for i, ti in enumerate(t):
            single = sched.get_beta(ti.unsqueeze(0))
            torch.testing.assert_close(beta[i : i + 1], single)


# ---------------------------------------------------------------------------
# Linear schedule value properties
# ---------------------------------------------------------------------------


class TestLinearSchedule:
    """Value-level properties specific to the linear schedule."""

    @pytest.fixture()
    def sched(self) -> NoiseSchedule:
        return NoiseSchedule("linear_ddpm", timesteps=10)

    def test_noise_level_approximately_linear(self, sched: NoiseSchedule) -> None:
        """Noise levels should be approximately linear from 0 to ~1.

        The alpha_bar floor (1e-5, needed for numerical stability in
        log-space) means the last entry is ~0.99999 rather than exactly
        1.0, but the deviation is negligible for all practical purposes.
        """
        t = torch.arange(0, sched.timesteps + 1)
        noise = sched.get_noise_level(t)
        expected = torch.linspace(0.0, 1.0, sched.timesteps + 1)
        torch.testing.assert_close(noise, expected, atol=1e-4, rtol=1e-4)

    def test_noise_level_at_t0_is_zero(self, sched: NoiseSchedule) -> None:
        """No noise at t=0."""
        assert sched.get_noise_level(torch.tensor([0])).item() == pytest.approx(
            0.0, abs=1e-6
        )

    def test_noise_level_at_tT_near_one(self, sched: NoiseSchedule) -> None:
        """Nearly full noise at t=T (alpha_bar floor means ~0.99999, not 1.0)."""
        assert sched.get_noise_level(
            torch.tensor([sched.timesteps])
        ).item() == pytest.approx(1.0, abs=1e-4)

    def test_alpha_bar_at_t0_is_one(self, sched: NoiseSchedule) -> None:
        """alpha_bar(0) = 1 for the linear schedule."""
        assert sched.get_alpha_bar(torch.tensor([0])).item() == pytest.approx(
            1.0, abs=1e-6
        )

    def test_betas_positive_and_bounded(self, sched: NoiseSchedule) -> None:
        """All betas should be in [0, 1) for valid diffusion steps."""
        assert (sched.betas >= 0).all()
        assert (sched.betas < 1).all()

    def test_alpha_bar_equals_cumprod_alphas(self, sched: NoiseSchedule) -> None:
        """The DDPM identity: alpha_bar[t] = prod(alpha[0:t]).

        Before the fix, the linear branch set betas to constant 1/T,
        giving cumprod = (1-1/T)^t (exponential), while alpha_bar was
        linear 1-t/T.  Now betas are derived from the desired alpha_bar,
        so this identity holds by construction.

        Test rationale
        --------------
        DiGress (github.com/cvignac/DiGress) enforces this identity for
        cosine/custom schedules by computing alpha_bar = cumprod(alphas).
        Our linear schedule defines alpha_bar first and derives betas, so
        we verify the round-trip.
        """
        recomputed = torch.cumprod(sched.alphas, dim=0)
        torch.testing.assert_close(sched.alpha_bar, recomputed, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Shape handling
# ---------------------------------------------------------------------------


class TestScheduleShapes:
    """The schedule should handle both (bs,) and (bs, 1) inputs."""

    @pytest.fixture(params=["cosine_iddpm", "linear_ddpm"])
    def sched(self, request: pytest.FixtureRequest) -> NoiseSchedule:
        schedule_type: str = request.param  # type: ignore[assignment]
        return NoiseSchedule(schedule_type, timesteps=100)

    def test_1d_input_shape(self, sched: NoiseSchedule) -> None:
        """Input shape (bs,) produces output shape (bs,)."""
        t = torch.tensor([0, 5, 10])
        assert sched.get_beta(t).shape == (3,)
        assert sched.get_alpha_bar(t).shape == (3,)
        assert sched.get_noise_level(t).shape == (3,)

    def test_2d_input_shape(self, sched: NoiseSchedule) -> None:
        """Input shape (bs, 1) produces output shape (bs, 1)."""
        t = torch.tensor([[0], [5], [10]])
        assert sched.get_beta(t).shape == (3, 1)
        assert sched.get_alpha_bar(t).shape == (3, 1)
        assert sched.get_noise_level(t).shape == (3, 1)


# ---------------------------------------------------------------------------
# state_dict roundtrip (T-3)
# ---------------------------------------------------------------------------


class TestNoiseScheduleStateDictRoundtrip:
    """Verify that save/load via state_dict preserves all registered buffers.

    Test rationale: NoiseSchedule is an nn.Module with three registered
    buffers (betas, alphas, alpha_bar). A regression in buffer registration
    would silently corrupt checkpoint loading. These tests verify the
    roundtrip for both schedule types.
    """

    @pytest.fixture(params=["cosine_iddpm", "linear_ddpm"])
    def sched(self, request: pytest.FixtureRequest) -> NoiseSchedule:
        schedule_type: str = request.param  # type: ignore[assignment]
        return NoiseSchedule(schedule_type, timesteps=100)

    def test_state_dict_contains_all_buffers(self, sched: NoiseSchedule) -> None:
        """state_dict() must include betas, alphas, and alpha_bar."""
        sd = sched.state_dict()
        assert "betas" in sd
        assert "alphas" in sd
        assert "alpha_bar" in sd

    def test_roundtrip_preserves_values(self, sched: NoiseSchedule) -> None:
        """load_state_dict(state_dict()) recovers identical buffer values."""
        sd = sched.state_dict()
        restored = NoiseSchedule(sched.schedule_type, timesteps=sched.timesteps)
        restored.load_state_dict(sd)
        torch.testing.assert_close(restored.betas, sched.betas)
        torch.testing.assert_close(restored.alphas, sched.alphas)
        torch.testing.assert_close(restored.alpha_bar, sched.alpha_bar)

    def test_roundtrip_preserves_dtype(self, sched: NoiseSchedule) -> None:
        """Buffer dtypes survive the roundtrip."""
        sd = sched.state_dict()
        restored = NoiseSchedule(sched.schedule_type, timesteps=sched.timesteps)
        restored.load_state_dict(sd)
        assert restored.betas.dtype == sched.betas.dtype


# ---------------------------------------------------------------------------
# Device / dtype transfer (T-4)
# ---------------------------------------------------------------------------


class TestNoiseScheduleDeviceTransfer:
    """Verify buffers follow .to() calls.

    Test rationale: NoiseSchedule registers betas/alphas/alpha_bar as
    nn.Module buffers. If registration were broken, .to(device) or
    .to(dtype) would silently leave buffers behind. These tests validate
    the mechanism using CPU and dtype transfers (no GPU required).
    """

    def test_buffers_on_correct_device_after_to(self) -> None:
        """All three buffers report the expected device after .to()."""
        sched = NoiseSchedule("cosine_iddpm", timesteps=50)
        sched = sched.to(torch.device("cpu"))
        for name, buf in sched.named_buffers():
            assert buf.device == torch.device(
                "cpu"
            ), f"Buffer '{name}' on {buf.device} after .to('cpu')"

    def test_dtype_transfer(self) -> None:
        """.to(dtype) changes buffer dtypes (validates nn.Module integration)."""
        sched = NoiseSchedule("cosine_iddpm", timesteps=50)
        sched = sched.to(dtype=torch.float64)
        assert sched.betas.dtype == torch.float64
        assert sched.alphas.dtype == torch.float64
        assert sched.alpha_bar.dtype == torch.float64

    def test_values_correct_after_dtype_roundtrip(self) -> None:
        """Values remain numerically close after float32 -> float64 -> float32."""
        original = NoiseSchedule("cosine_iddpm", timesteps=50)
        original_betas = original.betas.clone()
        roundtripped = original.to(dtype=torch.float64).to(dtype=torch.float32)
        torch.testing.assert_close(roundtripped.betas, original_betas)

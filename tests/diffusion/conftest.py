"""Shared fixtures for diffusion tests."""

from __future__ import annotations

import pytest

from tmgg.diffusion.schedule import NoiseSchedule


@pytest.fixture()
def cosine_schedule() -> NoiseSchedule:
    """Cosine schedule with T=50 for noise process tests.

    Enough timesteps for meaningful schedule lookups and posterior
    computation, while still fast.
    """
    return NoiseSchedule(schedule_type="cosine_iddpm", timesteps=50)


@pytest.fixture()
def cosine_schedule_short() -> NoiseSchedule:
    """Cosine schedule with T=5 for fast sampler loop tests."""
    return NoiseSchedule(schedule_type="cosine_iddpm", timesteps=5)

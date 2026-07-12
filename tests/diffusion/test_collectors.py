"""Tests for diffusion metric collectors.

Test Rationale
--------------
StepMetricCollector is a protocol for per-step metric recording during
reverse diffusion. DiffusionLikelihoodCollector is the concrete
implementation for VLB computation. Tests verify the accumulation,
VLB summation, summary statistics, and protocol conformance.
"""

from __future__ import annotations

import pytest

from tmgg.diffusion.collectors import (
    DiffusionLikelihoodCollector,
    StepMetricCollector,
)


class TestDiffusionLikelihoodCollector:
    """Verify DiffusionLikelihoodCollector accumulation and VLB computation."""

    def test_empty_collector_vlb_is_zero(self) -> None:
        c = DiffusionLikelihoodCollector()
        assert c.vlb() == 0.0

    def test_record_accumulates(self) -> None:
        c = DiffusionLikelihoodCollector()
        c.record(10, 9, {"kl": 0.5})
        c.record(9, 8, {"kl": 0.3})
        assert len(c.records) == 2

    def test_vlb_sums_kl(self) -> None:
        c = DiffusionLikelihoodCollector()
        c.record(3, 2, {"kl": 0.1})
        c.record(2, 1, {"kl": 0.2})
        c.record(1, 0, {"kl": 0.3})
        assert c.vlb() == pytest.approx(0.1 + 0.2 + 0.3)

    def test_vlb_ignores_non_kl_keys(self) -> None:
        c = DiffusionLikelihoodCollector()
        c.record(2, 1, {"kl": 0.5, "reconstruction": 1.0})
        assert c.vlb() == 0.5

    def test_vlb_zero_when_no_kl_key(self) -> None:
        c = DiffusionLikelihoodCollector()
        c.record(2, 1, {"reconstruction": 1.0})
        assert c.vlb() == 0.0

    def test_results_contains_vlb_and_num_steps(self) -> None:
        c = DiffusionLikelihoodCollector()
        c.record(2, 1, {"kl": 0.5})
        c.record(1, 0, {"kl": 0.3})
        r = c.results()
        assert r["vlb"] == 0.8
        assert r["num_steps"] == 2.0

    def test_results_contains_mean_per_key(self) -> None:
        c = DiffusionLikelihoodCollector()
        c.record(2, 1, {"kl": 0.4, "reconstruction": 1.0})
        c.record(1, 0, {"kl": 0.6, "reconstruction": 2.0})
        r = c.results()
        assert r["mean_kl"] == 0.5
        assert r["mean_reconstruction"] == 1.5

    def test_results_empty_collector(self) -> None:
        c = DiffusionLikelihoodCollector()
        r = c.results()
        assert r["vlb"] == 0.0
        assert r["num_steps"] == 0.0

    def test_records_returns_copy(self) -> None:
        c = DiffusionLikelihoodCollector()
        c.record(2, 1, {"kl": 0.5})
        records = c.records
        records.clear()
        assert len(c.records) == 1  # original unaffected

    def test_implements_protocol(self) -> None:
        c = DiffusionLikelihoodCollector()
        assert isinstance(c, StepMetricCollector)

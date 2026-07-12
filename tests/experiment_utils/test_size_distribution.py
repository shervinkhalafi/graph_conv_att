"""Tests for the SizeDistribution dataclass.

Test rationale
--------------
``SizeDistribution`` is the compact representation that datamodules use
to communicate graph-size distributions to the sampling pipeline. These
tests verify construction, serialization, sampling, and input validation
so that downstream consumers (``sample_n_nodes``, collation with
variable-size padding) can rely on correct behaviour.

Invariants
~~~~~~~~~~
- ``fixed(n)`` produces a degenerate distribution: single size, probs == [1.0].
- ``from_node_counts`` correctly aggregates frequencies.
- ``to_dict`` / ``from_dict`` round-trips losslessly.
- ``sample`` returns values exclusively from the distribution's ``sizes``.
- Invalid inputs (mismatched lengths, zero sizes, max_size violations)
  raise ``ValueError`` at construction time.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.utils.noising.size_distribution import SizeDistribution


class TestFixed:
    """Degenerate (single-size) distributions."""

    def test_probs(self) -> None:
        """A fixed distribution has probs == [1.0]."""
        d = SizeDistribution.fixed(20)
        assert d.probs == [1.0]

    def test_is_degenerate(self) -> None:
        d = SizeDistribution.fixed(20)
        assert d.is_degenerate

    def test_sample_returns_constant(self) -> None:
        """Sampling from a fixed distribution always returns the same size."""
        d = SizeDistribution.fixed(20)
        result = d.sample(100)
        assert result.shape == (100,)
        assert (result == 20).all()

    def test_max_size(self) -> None:
        d = SizeDistribution.fixed(42)
        assert d.max_size == 42


class TestFromNodeCounts:
    """Construction from raw per-graph node counts."""

    def test_basic(self) -> None:
        """Frequencies should match the input counts."""
        d = SizeDistribution.from_node_counts([8, 8, 10, 12, 12, 12])
        assert d.sizes == (8, 10, 12)
        assert d.counts == (2, 1, 3)
        assert d.max_size == 12

    def test_single_value(self) -> None:
        """All-identical counts should produce a degenerate distribution."""
        d = SizeDistribution.from_node_counts([5, 5, 5])
        assert d.is_degenerate
        assert d.sizes == (5,)
        assert d.counts == (3,)

    def test_empty_raises(self) -> None:
        """An empty sequence is not a valid distribution."""
        with pytest.raises(ValueError, match="non-empty"):
            SizeDistribution.from_node_counts([])


class TestSerialization:
    """to_dict / from_dict round-trip."""

    def test_roundtrip_fixed(self) -> None:
        original = SizeDistribution.fixed(30)
        restored = SizeDistribution.from_dict(original.to_dict())
        assert restored == original

    def test_roundtrip_variable(self) -> None:
        original = SizeDistribution.from_node_counts([5, 10, 10, 15, 15, 15])
        restored = SizeDistribution.from_dict(original.to_dict())
        assert restored == original

    def test_dict_structure(self) -> None:
        """The dict should contain exactly sizes, counts, max_size."""
        d = SizeDistribution.fixed(7).to_dict()
        assert set(d.keys()) == {"sizes", "counts", "max_size"}
        assert d["sizes"] == [7]
        assert d["counts"] == [1]
        assert d["max_size"] == 7


class TestSampling:
    """Sampling produces valid outputs."""

    def test_all_samples_in_support(self) -> None:
        """Every sampled value must be one of the distribution's sizes."""
        d = SizeDistribution.from_node_counts([4, 4, 8, 8, 8, 16])
        result = d.sample(500)
        for val in result.tolist():
            assert val in d.sizes

    def test_dtype_is_long(self) -> None:
        d = SizeDistribution.fixed(10)
        assert d.sample(5).dtype == torch.long

    def test_generator_reproducibility(self) -> None:
        """Using the same generator state produces identical samples."""
        d = SizeDistribution.from_node_counts([3, 3, 7, 7, 7, 11])
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        s1 = d.sample(50, generator=g1)
        s2 = d.sample(50, generator=g2)
        assert torch.equal(s1, s2)


class TestValidation:
    """Invalid inputs should raise ValueError at construction."""

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            SizeDistribution(sizes=(1, 2), counts=(3,), max_size=2)

    def test_zero_size(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SizeDistribution(sizes=(0, 5), counts=(1, 1), max_size=5)

    def test_zero_count(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SizeDistribution(sizes=(5,), counts=(0,), max_size=5)

    def test_max_size_too_small(self) -> None:
        with pytest.raises(ValueError, match="max_size"):
            SizeDistribution(sizes=(10,), counts=(1,), max_size=5)

    def test_unsorted_sizes(self) -> None:
        with pytest.raises(ValueError, match="sorted"):
            SizeDistribution(sizes=(10, 5), counts=(1, 1), max_size=10)

    def test_empty_sizes(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            SizeDistribution(sizes=(), counts=(), max_size=0)


class TestLogProb:
    """log_prob returns correct log-probabilities for queried node counts.

    For the VLB computation, the generative model needs log p(N) — the
    log-probability of each graph's node count under the training size
    distribution. For fixed-size data this is 0.0; for variable-size data
    it reflects the empirical frequency.
    """

    def test_degenerate_matching(self) -> None:
        """Fixed(20).log_prob(20) should be 0.0 (prob = 1.0)."""
        d = SizeDistribution.fixed(20)
        result = d.log_prob(torch.tensor([20]))
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    def test_degenerate_non_matching(self) -> None:
        """Fixed(20).log_prob(15) should be -inf (not in support)."""
        d = SizeDistribution.fixed(20)
        result = d.log_prob(torch.tensor([15]))
        assert result.item() < -60  # log(1e-30) ≈ -69

    def test_variable_distribution(self) -> None:
        """Correct log-probs for a multi-size distribution.

        Distribution: sizes=[5, 10, 15], counts=[2, 3, 5], total=10.
        Probs: [0.2, 0.3, 0.5].
        """
        d = SizeDistribution(sizes=(5, 10, 15), counts=(2, 3, 5), max_size=15)
        result = d.log_prob(torch.tensor([5, 10, 15]))
        expected = torch.log(torch.tensor([0.2, 0.3, 0.5]))
        assert torch.allclose(result, expected, atol=1e-5)

    def test_batch_input(self) -> None:
        """log_prob handles batch of node counts."""
        d = SizeDistribution.fixed(20)
        result = d.log_prob(torch.tensor([20, 20, 20, 20]))
        assert result.shape == (4,)
        assert torch.allclose(result, torch.zeros(4), atol=1e-6)

    def test_mixed_match_and_non_match(self) -> None:
        """Batch with some matching and some non-matching sizes."""
        d = SizeDistribution(sizes=(5, 10), counts=(1, 1), max_size=10)
        result = d.log_prob(torch.tensor([5, 7, 10]))
        assert torch.isclose(result[0], torch.tensor(-0.6931), atol=1e-3)  # log(0.5)
        assert result[1].item() < -60  # 7 not in support
        assert torch.isclose(result[2], torch.tensor(-0.6931), atol=1e-3)  # log(0.5)

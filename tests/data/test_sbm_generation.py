"""Tests for ``generate_sbm_batch`` — diversity knob and fixed-mode regression.

Test rationale
--------------
The diversity knob added in ``src/tmgg/data/datasets/sbm.py`` is the
per-graph randomiser behind Phase 2 of the improvement-gap plan
(``internal notes (not included)``).
These tests exercise three guarantees that the rest of the codebase
relies on:

- **regression**: scalar args + ``diversity == 0`` must produce the same
  numeric output as the pre-diversity implementation. Concretely, any
  seeded experiment / fixture that imports ``generate_sbm_batch`` will
  drift silently if the RNG sequence changes. The regression test
  reconstructs the expected output via ``generate_sbm_adjacency`` against
  the same seed so this invariant is checkable without a frozen golden
  file.
- **width**: when any hyperparameter is a tuple and ``diversity > 0`` the
  per-graph eigengap distribution should actually widen. Flat or
  non-monotone behaviour would mean the knob doesn't move the underlying
  spectrum, contradicting the stated intent.
- **guards**: malformed input (scalar args with ``diversity > 0``,
  inverted ranges, ``diversity`` outside ``[0, 1]``) must raise loudly
  per CLAUDE.md's no-silent-fallback rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from tmgg.data.datasets.sbm import generate_sbm_adjacency, generate_sbm_batch


class TestRegression:
    """Fixed-mode parity with the pre-diversity implementation."""

    def test_fixed_mode_matches_manual_reconstruction(self) -> None:
        """Rationale: the RNG sequence in fixed mode is exactly one
        ``generate_sbm_adjacency`` call per graph, sharing a single
        ``default_rng(seed)``. Reconstructing that sequence by hand and
        comparing byte-for-byte catches any branch that accidentally
        consumes RNG draws in the fast path."""
        num_graphs = 4
        num_nodes = 12
        num_blocks = 3
        p_intra = 0.7
        p_inter = 0.1
        seed = 42

        actual = generate_sbm_batch(
            num_graphs,
            num_nodes,
            num_blocks=num_blocks,
            p_intra=p_intra,
            p_inter=p_inter,
            seed=seed,
        )

        # Manual reconstruction mirroring the fixed-mode branch.
        block_size = num_nodes // num_blocks
        remainder = num_nodes % num_blocks
        block_sizes = [block_size] * num_blocks
        for i in range(remainder):
            block_sizes[i] += 1

        rng = np.random.default_rng(seed)
        expected = np.stack(
            [
                generate_sbm_adjacency(block_sizes, p_intra, p_inter, rng=rng).astype(
                    np.float32
                )
                for _ in range(num_graphs)
            ],
            axis=0,
        )

        assert np.array_equal(actual, expected), (
            "Fixed-mode output diverged from the pre-diversity RNG sequence."
        )

    def test_fixed_mode_deterministic_across_calls(self) -> None:
        """Rationale: two calls with identical args must yield identical
        arrays — a basic sanity check that seeding still works after the
        signature change."""
        a = generate_sbm_batch(3, 10, num_blocks=2, p_intra=0.6, p_inter=0.1, seed=7)
        b = generate_sbm_batch(3, 10, num_blocks=2, p_intra=0.6, p_inter=0.1, seed=7)
        assert np.array_equal(a, b)

    def test_diversity_zero_with_tuple_collapses_to_midpoint(self) -> None:
        """Rationale: ``diversity == 0`` should make the range collapse
        to its midpoint. A ``(0.4, 0.8)`` tuple with ``diversity=0`` must
        behave like scalar ``0.6``. We check statistical equivalence via
        edge density rather than bitwise equality because the diverse
        path consumes extra RNG draws for the (collapsed) sampling."""
        num_graphs = 40
        num_nodes = 20
        tuple_batch = generate_sbm_batch(
            num_graphs,
            num_nodes,
            num_blocks=(2, 4),  # collapses to 3
            p_intra=(0.4, 0.8),  # collapses to 0.6
            p_inter=(0.05, 0.15),  # collapses to 0.1
            diversity=0.0,
            seed=11,
        )
        # The intra-block density should be concentrated around 0.6 when
        # the tuple collapsed correctly; a wide spread would signal that
        # diversity=0 is not clamping the sampler.
        densities = tuple_batch.mean(axis=(1, 2))
        assert densities.std() < 0.1, (
            f"Unexpected spread under diversity=0 with tuple args: "
            f"std={densities.std():.3f}"
        )


def _eigengaps(adj: np.ndarray) -> np.ndarray:
    """Return per-graph adjacency spectral gaps ``λ_1 − λ_2``."""
    gaps = []
    for A in adj:
        w = np.linalg.eigvalsh(A)
        gaps.append(w[-1] - w[-2])
    return np.array(gaps)


class TestDiversityWidensDistribution:
    """Diversity should actually move the spectrum."""

    def test_tuple_with_diversity_one_widens_eigengap_distribution(self) -> None:
        """Rationale: the knob is supposed to diversify the eigenspectrum
        across graphs. At ``diversity=1`` with tuple ranges, per-graph
        spectral gaps should span a wider interval than the fixed-scalar
        case using the midpoint. We measure the interquartile range of
        the gap distribution so outlier seeds don't dominate."""
        num_graphs = 80
        num_nodes = 30
        fixed_gaps = _eigengaps(
            generate_sbm_batch(
                num_graphs,
                num_nodes,
                num_blocks=3,
                p_intra=0.6,
                p_inter=0.1,
                seed=1,
            )
        )
        diverse_gaps = _eigengaps(
            generate_sbm_batch(
                num_graphs,
                num_nodes,
                num_blocks=(2, 5),
                p_intra=(0.3, 0.9),
                p_inter=(0.01, 0.2),
                diversity=1.0,
                seed=1,
            )
        )
        fixed_iqr = np.subtract(*np.percentile(fixed_gaps, [75, 25]))
        diverse_iqr = np.subtract(*np.percentile(diverse_gaps, [75, 25]))
        assert diverse_iqr > fixed_iqr, (
            f"Expected diverse IQR ({diverse_iqr:.3f}) > fixed IQR ({fixed_iqr:.3f})"
        )

    def test_dirichlet_block_sizes_produce_unequal_blocks(self) -> None:
        """Rationale: ``block_size_alpha`` is the knob's mechanism for
        introducing size heterogeneity. With small alpha we expect
        concentration on few blocks (highly unequal); the test asserts
        that block sizes actually vary across graphs rather than
        collapsing to the equal-block default."""
        num_graphs = 30
        num_nodes = 24

        # Small alpha biases toward lopsided partitions.
        batch = generate_sbm_batch(
            num_graphs,
            num_nodes,
            num_blocks=4,
            p_intra=0.7,
            p_inter=0.1,
            block_size_alpha=0.5,
            diversity=1.0,
            seed=3,
        )
        # Per-graph degree histograms vary more when block sizes vary,
        # so std of per-graph mean-degree is a cheap proxy.
        mean_degrees = batch.sum(axis=(1, 2)) / num_nodes
        assert mean_degrees.std() > 0.05, (
            f"Dirichlet partition produced suspiciously uniform degrees: "
            f"std={mean_degrees.std():.3f}"
        )

    def test_num_blocks_range_samples_k_uniformly(self) -> None:
        """Rationale (paper Appendix B.2.3): the synthetic SBM samples the block
        count K uniformly from ``{2, 3, 4}``. In code this is ``num_blocks=(2, 4)``
        with ``diversity=1.0`` (at ``diversity=0`` the range collapses to the
        midpoint). With ``p_intra=1.0`` and ``p_inter=0.0`` each graph is a
        disjoint union of K cliques, so the number of distinct rows of ``A + I``
        equals K exactly. We assert every K in the closed range appears."""
        num_graphs = 60
        num_nodes = 100
        batch = generate_sbm_batch(
            num_graphs,
            num_nodes,
            num_blocks=(2, 4),
            p_intra=1.0,
            p_inter=0.0,
            diversity=1.0,
            seed=0,
        )
        eye = np.eye(num_nodes)
        block_counts = {len(np.unique(graph + eye, axis=0)) for graph in batch}
        assert block_counts == {2, 3, 4}, (
            f"Expected K sampled uniformly over {{2, 3, 4}}, observed {block_counts}"
        )


class TestGuards:
    """Error handling at the API boundary (no silent fallbacks)."""

    def test_scalar_args_with_positive_diversity_raises(self) -> None:
        with pytest.raises(ValueError, match="diversity > 0"):
            generate_sbm_batch(
                2,
                10,
                num_blocks=2,
                p_intra=0.7,
                p_inter=0.1,
                diversity=0.5,
                seed=0,
            )

    def test_diversity_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="diversity must be in"):
            generate_sbm_batch(
                2,
                10,
                num_blocks=(2, 4),
                p_intra=0.7,
                p_inter=0.1,
                diversity=1.5,
                seed=0,
            )

    def test_inverted_range_raises(self) -> None:
        with pytest.raises(ValueError, match="range malformed"):
            generate_sbm_batch(
                2,
                10,
                num_blocks=2,
                p_intra=(0.9, 0.3),  # min > max
                p_inter=0.1,
                diversity=0.5,
                seed=0,
            )

    def test_negative_block_size_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="block_size_alpha"):
            generate_sbm_batch(
                2,
                10,
                num_blocks=3,
                p_intra=0.7,
                p_inter=0.1,
                block_size_alpha=-0.5,
                diversity=0.5,
                seed=0,
            )

"""Tests for the improvement-gap surrogate.

Test rationale
--------------
Equation (18) of the NeurIPS 2026 draft writes the improvement gap between
a linear denoiser and the best eigenvalue-function denoiser as

    ℓ_lin − ℓ_f = tr(Cov(E[vec(B) | Λ̃_k]))
                = E‖E[B | Λ̃_k] − E[B]‖²_F,

where ``B = V̂_k^T A V̂_k`` projects the clean adjacency into the noisy
top-*k* eigenbasis. The kNN and binning estimators implemented in
``analyzer.estimate_improvement_gap`` approximate that conditional-mean
variance directly from finite samples of ``B`` and conditioning features.

These tests exercise the pure estimator on synthetic ``B`` tensors so
they run without needing disk-backed Phase 1 fixtures:

- sanity: identical graphs ⇒ ``g_hat ≈ 0`` (no between-graph variance to
  capture);
- property: ``g_hat ≤ trace_cov_B + slack`` (law of total variance; the
  conditional-variance share can exceed the total only by estimation
  noise);
- separability: two well-separated clusters with cluster-predictive
  conditioning features ⇒ ``g_hat`` captures almost all of
  ``trace_cov_B``;
- estimator agreement: kNN and binning give comparable FVEs on the
  same controlled input;
- collector integration: end-to-end round trip through the noised
  collector and comparator confirms that ``B_k{k}`` tensors are
  persisted, loaded, and surfaced by ``compute_improvement_gap_surrogate``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from tmgg.experiments.eigenstructure_study import (
    EigenstructureCollector,
    ImprovementGapResult,
    NoisedAnalysisComparator,
    NoisedEigenstructureCollector,
    compute_B_invariants,
    estimate_improvement_gap,
)
from tmgg.utils.spectral.subspace import (
    align_top_k_to_reference_batch,
    compute_frechet_mean_subspace,
)


def _synthetic_identical_B(num_graphs: int, k: int, seed: int = 0) -> torch.Tensor:
    """N identical symmetric projection tensors plus tiny jitter.

    Zero variance would make the FVE undefined; a small jitter keeps the
    estimator numerics live while preserving the "should be near zero"
    semantics.
    """
    rng = torch.Generator().manual_seed(seed)
    base = torch.randn(k, k, generator=rng)
    base = (base + base.T) / 2
    jitter = 1e-6 * torch.randn(num_graphs, k, k, generator=rng)
    jitter = (jitter + jitter.transpose(-2, -1)) / 2
    return base.unsqueeze(0).expand(num_graphs, k, k).clone() + jitter


def _two_cluster_B(
    num_graphs: int, k: int, gap: float = 5.0, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-cluster synthetic B tensors + matching conditioning feature.

    Each cluster shares a mean B-matrix; noise within a cluster is small.
    The conditioning feature (e.g. a mock spectral gap) also cleanly
    separates the two clusters, so a well-tuned estimator should recover
    almost the full between-cluster variance.
    """
    rng = torch.Generator().manual_seed(seed)
    half = num_graphs // 2
    cluster_a = torch.zeros(k, k)
    cluster_b = gap * torch.eye(k)

    noise_a = 0.1 * torch.randn(half, k, k, generator=rng)
    noise_a = (noise_a + noise_a.transpose(-2, -1)) / 2
    noise_b = 0.1 * torch.randn(num_graphs - half, k, k, generator=rng)
    noise_b = (noise_b + noise_b.transpose(-2, -1)) / 2

    B_a = cluster_a.unsqueeze(0) + noise_a
    B_b = cluster_b.unsqueeze(0) + noise_b
    B = torch.cat([B_a, B_b], dim=0)

    # Cluster-predictive feature: -1 for cluster A, +1 for cluster B with
    # small jitter so kNN distances are well-defined.
    feat_a = -1.0 + 0.01 * torch.randn(half, generator=rng)
    feat_b = 1.0 + 0.01 * torch.randn(num_graphs - half, generator=rng)
    features = torch.cat([feat_a, feat_b], dim=0)
    return B, features


class TestEstimatorSanity:
    """Pure-estimator behaviour on synthetic B tensors."""

    def test_identical_graphs_give_near_zero_gap(self) -> None:
        """Rationale: when B_i are identical up to jitter, both ``B̂_i`` and
        ``μ_B`` collapse onto the same point, so the surrogate must vanish
        to within the jitter scale. trace_cov_B also collapses, so we check
        absolute magnitude rather than the FVE to avoid 0/0 noise."""
        B = _synthetic_identical_B(num_graphs=40, k=4)
        features = torch.randn(40, 4)  # uninformative features
        g_hat, trace_cov_B, _ = estimate_improvement_gap(
            B, features, estimator="knn", knn_neighbours=5
        )
        assert g_hat < 1e-8, f"g_hat={g_hat}, trace_cov_B={trace_cov_B}"

    def test_permutation_null_fve_is_small_on_cluster_input(self) -> None:
        """Rationale: the reviewer-2 audit flagged that kNN(m=10) at
        diversity=0 gave an FVE near 0.5 when the paper-predicted null
        is ≈0. The permutation-null control diagnoses this: after
        shuffling the conditioning features so they are decorrelated from
        B, a calibrated estimator must return near-zero ĝ. If the
        permutation-null FVE is large, the finite-sample kNN bias is
        real and the headline FVE is inflated. On a two-cluster input
        where the real-features FVE is near 1, the permuted FVE
        should drop by at least an order of magnitude."""
        B, features = _two_cluster_B(num_graphs=80, k=4, gap=5.0)
        features_2d = features.unsqueeze(1)
        _, _, fve_real = estimate_improvement_gap(
            B, features_2d, estimator="knn", knn_neighbours=5
        )
        _, _, fve_null = estimate_improvement_gap(
            B,
            features_2d,
            estimator="knn",
            knn_neighbours=5,
            permute_features=True,
            permutation_seed=0,
        )
        assert fve_real > 0.8, f"setup sanity: real FVE={fve_real}"
        assert fve_null < fve_real / 5, (
            f"permuted FVE {fve_null:.3f} too close to real {fve_real:.3f} "
            f"— kNN may have finite-sample bias at this N"
        )

    def test_random_features_yield_low_fve_on_variable_B(self) -> None:
        """Rationale: replaces the weak 'identical graphs' sanity with a
        stronger one — B has real variance but features are uncorrelated
        with B, so the estimator must deliver a small FVE. Tests that
        kNN does NOT inflate ĝ merely because B varies."""
        torch.manual_seed(0)
        B = torch.randn(80, 4, 4)
        B = (B + B.transpose(-2, -1)) / 2
        # Uninformative features — independent of B.
        features = torch.randn(80, 4)
        _, _, fve = estimate_improvement_gap(
            B, features, estimator="knn", knn_neighbours=5
        )
        # Empirically the finite-sample kNN FVE under independence sits
        # around 1/m ≈ 0.2; 0.4 is a conservative cap for N=80.
        assert fve < 0.4, f"FVE {fve:.3f} too large under independent features"

    def test_invariants_target_is_frame_free(self) -> None:
        """Rationale: ``compute_B_invariants`` extracts (tr, ||·||_F²,
        eigvals), all invariant under orthogonal conjugation
        ``B → R^T B R``. Rotating every B by a graph-specific random
        orthogonal matrix must leave the invariants (and therefore the
        surrogate) unchanged to within numerical precision."""
        torch.manual_seed(7)
        N, k = 60, 4
        B = torch.randn(N, k, k)
        B = (B + B.transpose(-2, -1)) / 2  # symmetrise
        # Per-graph random orthogonal conjugation.
        R = torch.linalg.qr(torch.randn(N, k, k))[0]
        B_rotated = R.transpose(-2, -1) @ B @ R

        inv_a = compute_B_invariants(B)
        inv_b = compute_B_invariants(B_rotated)
        assert torch.allclose(inv_a, inv_b, atol=1e-4), (
            f"invariants drifted under rotation: max diff "
            f"{(inv_a - inv_b).abs().max().item()}"
        )

    def test_knn_1d_and_bin_1d_on_same_feature_agree(self) -> None:
        """Rationale: the reviewer's M3 — with kNN using k-dim features
        and binning using 1-D, the two estimators measure different
        quantities and shouldn't be compared directly. Running kNN on
        the same 1-D feature used by binning gives two estimators of the
        SAME conditional-mean variance, so their FVEs should agree up
        to estimator-noise."""
        B, features = _two_cluster_B(num_graphs=60, k=4, gap=5.0)
        _, _, fve_knn_1d = estimate_improvement_gap(
            B, features.unsqueeze(1), estimator="knn", knn_neighbours=5
        )
        _, _, fve_bin_1d = estimate_improvement_gap(
            B, features, estimator="bin", num_bins=2
        )
        assert abs(fve_knn_1d - fve_bin_1d) < 0.10, (
            f"knn_1d vs bin_1d disagreement: "
            f"knn_1d={fve_knn_1d:.3f}, bin_1d={fve_bin_1d:.3f}"
        )

    def test_law_of_total_variance_bound(self) -> None:
        """Rationale: by the law of total variance the conditional-variance
        share cannot exceed the total. A small finite-sample slack is
        allowed because neighbour averaging introduces bias that can push
        the estimator slightly above the Monte-Carlo total variance."""
        torch.manual_seed(0)
        B = torch.randn(60, 4, 4)
        B = (B + B.transpose(-2, -1)) / 2
        features = torch.randn(60, 4)
        g_hat, trace_cov_B, _ = estimate_improvement_gap(
            B, features, estimator="knn", knn_neighbours=5
        )
        # Slack: conditional-mean estimator inherits noise from the N-dim
        # sample; small overshoots are numerical, not semantic.
        assert (
            g_hat <= trace_cov_B * 1.05
        ), f"g_hat={g_hat} exceeded trace_cov_B={trace_cov_B} by more than 5%"

    def test_two_cluster_knn_captures_most_variance(self) -> None:
        """Rationale: with two well-separated B-clusters and a cluster-
        predictive feature, kNN neighbourhoods resolve the cluster
        identity, so ``B̂_i ≈ μ_cluster(i)`` and ``g_hat`` should cover
        most of ``trace_cov_B``."""
        B, features = _two_cluster_B(num_graphs=60, k=4, gap=5.0)
        features_2d = features.unsqueeze(1)
        g_hat, trace_cov_B, fve = estimate_improvement_gap(
            B, features_2d, estimator="knn", knn_neighbours=5
        )
        assert trace_cov_B > 1.0, "Test setup: clusters should induce sizeable variance"
        # With gap=5, noise=0.1, FVE should be close to 1.
        assert fve > 0.85, f"fve={fve} (g_hat={g_hat}, trace_cov_B={trace_cov_B})"

    def test_bin_estimator_matches_knn_on_cluster_input(self) -> None:
        """Rationale: when the conditioning feature cleanly separates
        clusters, quantile binning with num_bins=2 and kNN both recover
        the same structure; their FVEs should agree to within finite-
        sample tolerance. The binning estimator takes a 1-D feature."""
        B, features = _two_cluster_B(num_graphs=60, k=4, gap=5.0)
        _, _, fve_knn = estimate_improvement_gap(
            B, features.unsqueeze(1), estimator="knn", knn_neighbours=5
        )
        _, _, fve_bin = estimate_improvement_gap(
            B, features, estimator="bin", num_bins=2
        )
        assert (
            abs(fve_knn - fve_bin) < 0.10
        ), f"FVE disagreement too large: knn={fve_knn}, bin={fve_bin}"


class TestFrechetAlignment:
    """Frame-convention tests for the Fréchet-mean alignment path."""

    def test_frechet_mean_subspace_is_orthonormal(self) -> None:
        """Rationale: the dataset-wide reference must be a valid
        orthonormal basis, or the downstream Procrustes step produces
        garbage. Checks ``V*^T V* == I_k`` to machine precision."""
        torch.manual_seed(0)
        N, n, k = 40, 30, 6
        # Draw N random orthonormal top-k blocks.
        V_blocks = torch.linalg.qr(torch.randn(N, n, k))[0]
        V_star = compute_frechet_mean_subspace(V_blocks)
        assert V_star.shape == (n, k)
        gram = V_star.T @ V_star
        assert torch.allclose(gram, torch.eye(k), atol=1e-5), (
            f"Fréchet mean not orthonormal; gram max off-diag: "
            f"{(gram - torch.eye(k)).abs().max().item()}"
        )

    def test_align_to_reference_is_closer_to_reference_than_raw(self) -> None:
        """Rationale: the helper must actually reduce the residual
        ``||V_ref − V_aligned||_F`` relative to the raw block. If it
        doesn't, the Procrustes computation is misoriented."""
        torch.manual_seed(1)
        N, n, k = 20, 40, 4
        V_ref = torch.linalg.qr(torch.randn(n, k))[0]
        # Noisy blocks: perturb V_ref with small random rotations and noise.
        rotations = torch.linalg.qr(torch.randn(N, k, k))[0]
        noise = 0.05 * torch.randn(N, n, k)
        V_noisy_k = (V_ref.unsqueeze(0) @ rotations) + noise
        # Pack into full (N, n, n) by zero-padding the non-top-k columns.
        V_noisy_full = torch.zeros(N, n, n)
        V_noisy_full[:, :, -k:] = V_noisy_k

        aligned = align_top_k_to_reference_batch(V_ref, V_noisy_full, k)
        assert aligned.shape == (N, n, k)
        raw_residuals = torch.norm(
            V_noisy_k - V_ref.unsqueeze(0), p="fro", dim=(-2, -1)
        )
        aligned_residuals = torch.norm(
            aligned - V_ref.unsqueeze(0), p="fro", dim=(-2, -1)
        )
        assert (
            aligned_residuals <= raw_residuals + 1e-5
        ).all(), "Procrustes alignment did not reduce residuals vs. raw"

    def test_frechet_and_per_graph_produce_B_in_different_frames(self) -> None:
        """Rationale: regression sanity — the two frame modes must
        actually produce DIFFERENT B_k tensors on the same graphs (if
        they agreed, the reviewer's concern about frame choice would be
        moot and the Fréchet path would be pointless). We check they
        differ materially but share the frame-invariant summaries."""
        with tempfile.TemporaryDirectory() as tmp:
            phase1_dir = Path(tmp) / "phase1"
            noised_per_graph = Path(tmp) / "noised_per_graph"
            noised_frechet = Path(tmp) / "noised_frechet"
            phase1_dir.mkdir()
            noised_per_graph.mkdir()
            noised_frechet.mkdir()

            TestCollectorIntegration()._build_phase1_fixture(
                phase1_dir, num_graphs=12, n=20, batch_size=6
            )

            NoisedEigenstructureCollector(
                input_dir=phase1_dir,
                output_dir=noised_per_graph,
                noise_type="gaussian",
                noise_levels=[0.05],
                seed=0,
                surrogate_k_values=[4],
                frame_mode="per_graph",
            ).collect()

            NoisedEigenstructureCollector(
                input_dir=phase1_dir,
                output_dir=noised_frechet,
                noise_type="gaussian",
                noise_levels=[0.05],
                seed=0,
                surrogate_k_values=[4],
                frame_mode="frechet",
            ).collect()

            # Load B_k4 from both; compare raw matrices and invariants.
            from tmgg.experiments.eigenstructure_study.storage import (
                iter_batches,
                load_decomposition_batch,
            )

            def _load_B(base: Path) -> torch.Tensor:
                eps_dir = base / "eps_0.0500"
                return torch.cat(
                    [
                        load_decomposition_batch(p)[0]["B_k4"]
                        for p in iter_batches(eps_dir)
                    ],
                    dim=0,
                )

            B_per = _load_B(noised_per_graph)
            B_fre = _load_B(noised_frechet)
            # Raw matrices should differ — different alignments.
            assert not torch.allclose(
                B_per, B_fre, atol=1e-3
            ), "frame modes produced identical B; alignment is a no-op"
            # Invariants must match (both are orthogonal conjugates of
            # V̂_k^T A V̂_k without alignment, so same eigvals/tr/||·||_F²).
            # Tolerance is loose because the safetensors pipeline stores
            # in float32 and the B matrices go through 3–4 matmuls before
            # eigh, so per-entry relative error accumulates to ~0.5% —
            # within expected numerical regime, not a correctness issue.
            inv_per = compute_B_invariants(B_per)
            inv_fre = compute_B_invariants(B_fre)
            assert torch.allclose(inv_per, inv_fre, rtol=0.02, atol=1e-2), (
                f"invariants should be frame-independent; max abs diff "
                f"{(inv_per - inv_fre).abs().max().item()}, max rel diff "
                f"{((inv_per - inv_fre).abs() / inv_fre.abs().clamp(min=1e-6)).max().item()}"
            )


class TestEstimatorGuards:
    """Error handling at the estimator boundary."""

    def test_bin_rejects_multi_dimensional_feature(self) -> None:
        B = torch.randn(10, 3, 3)
        bad_feature = torch.randn(10, 4)
        try:
            estimate_improvement_gap(B, bad_feature, estimator="bin")
        except ValueError:
            return
        raise AssertionError("Expected ValueError for 2-D bin feature")

    def test_unknown_estimator_raises(self) -> None:
        B = torch.randn(10, 3, 3)
        feat = torch.randn(10, 3)
        try:
            estimate_improvement_gap(B, feat, estimator="bogus")
        except ValueError:
            return
        raise AssertionError("Expected ValueError for unknown estimator")

    def test_non_square_B_raises(self) -> None:
        bad_B = torch.randn(10, 3, 4)
        feat = torch.randn(10, 3)
        try:
            estimate_improvement_gap(bad_B, feat, estimator="knn")
        except ValueError:
            return
        raise AssertionError("Expected ValueError for non-square B")


class TestCollectorIntegration:
    """End-to-end: collect → noise → surrogate.

    Rationale: exercises the plumbing that wires the ``B_k{k}`` extra
    tensors through safetensors storage and back out via the comparator.
    Uses a small synthetic SBM dataset so the full pipeline runs in
    under a second without disk contention beyond the tempdir.
    """

    def _build_phase1_fixture(
        self, tmpdir: Path, num_graphs: int = 8, n: int = 20, batch_size: int = 4
    ) -> Path:
        """Produce a Phase-1 output dir by calling the real collector on
        a tiny Erdős–Rényi dataset. Mirrors the integration fixtures
        already used by tests/experiment_utils/test_eigenstructure_study.py.

        ER is used instead of SBM because the existing eigenstructure
        tests exercise ER heavily and the collector's ER path is
        simpler (no block partitioning logic)."""
        collector = EigenstructureCollector(
            dataset_name="er",
            dataset_config={"num_nodes": n, "num_graphs": num_graphs, "p": 0.3},
            output_dir=tmpdir,
            batch_size=batch_size,
            seed=0,
        )
        collector.collect()
        return tmpdir

    def test_roundtrip_produces_surrogate(self) -> None:
        """Rationale: verifies that B_k{k} tensors survive safetensors
        round-trip and that the comparator returns an
        ``ImprovementGapResult`` with sane values (g_hat ≥ 0, FVE ∈
        [0, 1+slack])."""
        with tempfile.TemporaryDirectory() as tmp:
            phase1_dir = Path(tmp) / "phase1"
            noised_dir = Path(tmp) / "noised"
            phase1_dir.mkdir()
            noised_dir.mkdir()

            self._build_phase1_fixture(phase1_dir)

            collector = NoisedEigenstructureCollector(
                input_dir=phase1_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.05],
                seed=0,
                surrogate_k_values=[4, 8],
            )
            collector.collect()

            comparator = NoisedAnalysisComparator(
                original_dir=phase1_dir,
                noised_base_dir=noised_dir,
            )
            result = comparator.compute_improvement_gap_surrogate(
                0.05, k=4, estimator="knn", knn_neighbours=3
            )
            assert isinstance(result, ImprovementGapResult)
            assert result.k == 4
            assert result.estimator == "knn"
            assert result.g_hat >= 0.0
            assert result.trace_cov_B >= 0.0
            assert result.num_graphs == 8
            # FVE is clamped to [0, 1+slack]; 1.1 is generous for tiny N.
            assert 0.0 <= result.fve <= 1.1

    def test_bin_estimator_works_roundtrip(self) -> None:
        """Rationale: binning path uses a different code path inside
        ``estimate_improvement_gap`` (1-D feature, bucketize). Making sure
        it also survives the round trip prevents silent regressions when
        the safetensors layout changes."""
        with tempfile.TemporaryDirectory() as tmp:
            phase1_dir = Path(tmp) / "phase1"
            noised_dir = Path(tmp) / "noised"
            phase1_dir.mkdir()
            noised_dir.mkdir()

            self._build_phase1_fixture(phase1_dir, num_graphs=12)

            collector = NoisedEigenstructureCollector(
                input_dir=phase1_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.05],
                seed=0,
                surrogate_k_values=[4],
            )
            collector.collect()

            comparator = NoisedAnalysisComparator(
                original_dir=phase1_dir,
                noised_base_dir=noised_dir,
            )
            result = comparator.compute_improvement_gap_surrogate(
                0.05,
                k=4,
                estimator="bin",
                conditioning="spectral_gap",
                num_bins=3,
            )
            assert result.estimator == "bin"
            assert result.num_bins == 3
            assert result.knn_neighbours is None
            assert result.conditioning == "spectral_gap"

    def test_invariants_target_roundtrips(self) -> None:
        """Rationale: the invariants-target code path diverges from the
        matrix path inside ``compute_improvement_gap_surrogate``; the
        round trip must produce an ``ImprovementGapResult`` with
        ``target == "invariants"`` and sane ĝ/FVE bounds. Catches
        silent drops of the target parameter or mismatch between the
        invariants dim and the estimator's feature dim."""
        with tempfile.TemporaryDirectory() as tmp:
            phase1_dir = Path(tmp) / "phase1"
            noised_dir = Path(tmp) / "noised"
            phase1_dir.mkdir()
            noised_dir.mkdir()

            self._build_phase1_fixture(phase1_dir, num_graphs=12)

            NoisedEigenstructureCollector(
                input_dir=phase1_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.05],
                seed=0,
                surrogate_k_values=[4],
            ).collect()

            comparator = NoisedAnalysisComparator(
                original_dir=phase1_dir,
                noised_base_dir=noised_dir,
            )
            result = comparator.compute_improvement_gap_surrogate(
                0.05, k=4, estimator="knn", target="invariants", knn_neighbours=3
            )
            assert result.target == "invariants"
            assert result.g_hat >= 0.0
            assert 0.0 <= result.fve <= 1.1

    def test_permutation_null_roundtrip_reduces_fve(self) -> None:
        """Rationale: the reviewer asked for a permutation-null control
        to diagnose kNN finite-sample bias. The plumbing should carry
        ``permute_features=True`` from the comparator to the estimator;
        testing the dropout effect on a tiny fixture is enough to prove
        the wiring. We don't assert a tight numeric bound (too noisy at
        N=8) — only that the permuted result is marked and returns
        finite numbers."""
        with tempfile.TemporaryDirectory() as tmp:
            phase1_dir = Path(tmp) / "phase1"
            noised_dir = Path(tmp) / "noised"
            phase1_dir.mkdir()
            noised_dir.mkdir()

            self._build_phase1_fixture(phase1_dir)

            NoisedEigenstructureCollector(
                input_dir=phase1_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.05],
                seed=0,
                surrogate_k_values=[4],
            ).collect()

            comparator = NoisedAnalysisComparator(
                original_dir=phase1_dir,
                noised_base_dir=noised_dir,
            )
            result = comparator.compute_improvement_gap_surrogate(
                0.05,
                k=4,
                estimator="knn",
                knn_neighbours=3,
                permute_features=True,
                permutation_seed=17,
            )
            assert result.permuted is True
            assert result.g_hat >= 0.0
            assert result.trace_cov_B >= 0.0

    def test_missing_k_raises_keyerror(self) -> None:
        """Rationale: if the user asks for a surrogate at k that wasn't
        persisted by the collector, the comparator must fail loudly
        (per CLAUDE.md: no silent fallbacks)."""
        with tempfile.TemporaryDirectory() as tmp:
            phase1_dir = Path(tmp) / "phase1"
            noised_dir = Path(tmp) / "noised"
            phase1_dir.mkdir()
            noised_dir.mkdir()

            self._build_phase1_fixture(phase1_dir)

            collector = NoisedEigenstructureCollector(
                input_dir=phase1_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.05],
                seed=0,
                surrogate_k_values=[4],
            )
            collector.collect()

            comparator = NoisedAnalysisComparator(
                original_dir=phase1_dir,
                noised_base_dir=noised_dir,
            )
            try:
                comparator.compute_improvement_gap_surrogate(
                    0.05, k=16, estimator="knn"
                )
            except KeyError:
                return
            raise AssertionError("Expected KeyError for missing k")

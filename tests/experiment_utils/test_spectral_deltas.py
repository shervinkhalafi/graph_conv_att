"""Tests for spectral delta metrics module.

This module tests compute_spectral_deltas (including its reduce='mean' mode)
which provides training-time spectral analysis between clean and
noisy/denoised graphs.

Testing Strategy:
- Unit tests with known graph structures (complete, path, disconnected)
- Numerical stability tests for edge cases (small gaps, zero connectivity)
- Batch consistency tests (batched vs per-graph computation)
- Output type and shape validation

Key Invariants:
- Identical graphs produce near-zero deltas
- Subspace distance is bounded by 2*sqrt(k)
- Disconnected graphs have zero algebraic connectivity
- 2D and 3D inputs produce appropriately shaped outputs
"""

from __future__ import annotations

import math

import torch

from tmgg.utils.spectral.spectral_deltas import (
    compute_spectral_deltas,
)


def create_complete_graph(n: int) -> torch.Tensor:
    """Create adjacency matrix for complete graph K_n."""
    return torch.ones(n, n) - torch.eye(n)


def create_path_graph(n: int) -> torch.Tensor:
    """Create adjacency matrix for path graph P_n."""
    A = torch.zeros(n, n)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def create_disconnected_graph(n: int) -> torch.Tensor:
    """Create adjacency matrix for two disconnected cliques of size n/2."""
    A = torch.zeros(n, n)
    half = n // 2
    # First clique
    A[:half, :half] = 1.0
    for i in range(half):
        A[i, i] = 0.0
    # Second clique
    A[half:, half:] = 1.0
    for i in range(half, n):
        A[i, i] = 0.0
    return A


def create_random_symmetric(
    n: int, density: float = 0.3, seed: int = 42
) -> torch.Tensor:
    """Create random symmetric adjacency matrix."""
    torch.manual_seed(seed)
    A = torch.rand(n, n)
    A = (A + A.T) / 2
    A = ((1 - density) < A).float()
    A.fill_diagonal_(0)
    return A


class TestComputeSpectralDeltas:
    """Tests for compute_spectral_deltas function."""

    def test_identical_graphs_produce_zero_deltas(self):
        """Identical graphs should have near-zero deltas for all metrics.

        Rationale: When A_clean == A_other, all spectral properties are
        identical, so deltas should be zero within numerical precision.
        """
        A = create_random_symmetric(10, seed=123)
        A_batch = A.unsqueeze(0)

        deltas = compute_spectral_deltas(A_batch, A_batch.clone())

        assert torch.allclose(deltas["eigengap_delta"], torch.tensor([0.0]), atol=1e-6)
        assert torch.allclose(deltas["alg_conn_delta"], torch.tensor([0.0]), atol=1e-6)
        assert torch.allclose(
            deltas["eigenvalue_drift"], torch.tensor([0.0]), atol=1e-6
        )
        assert torch.allclose(
            deltas["subspace_distance"], torch.tensor([0.0]), atol=1e-6
        )

    def test_2d_input_handling(self):
        """Single graph (n,n) input should return shape (1,) tensors.

        Rationale: For consistent downstream handling, even single-graph inputs
        return 1D tensors (shape (1,)) rather than scalars.
        """
        A = create_random_symmetric(8, seed=456)
        A_other = create_random_symmetric(8, seed=789)

        deltas = compute_spectral_deltas(A, A_other)

        for key, val in deltas.items():
            assert val.shape == (
                1,
            ), f"{key} should have shape (1,) for 2D input, got {val.shape}"
            assert torch.isfinite(val), f"{key} should be finite"

    def test_3d_batch_input_handling(self):
        """Batched (batch,n,n) input should return (batch,) shaped tensors.

        Rationale: Standard use case with batch dimension should produce
        per-graph delta values.
        """
        batch_size, n = 5, 10
        A_clean = torch.stack(
            [create_random_symmetric(n, seed=i) for i in range(batch_size)]
        )
        A_other = torch.stack(
            [create_random_symmetric(n, seed=i + 100) for i in range(batch_size)]
        )

        deltas = compute_spectral_deltas(A_clean, A_other)

        for key, val in deltas.items():
            assert val.shape == (batch_size,), f"{key} should have shape (batch,)"
            assert torch.all(
                torch.isfinite(val)
            ), f"{key} should have all finite values"

    def test_batch_consistency(self):
        """Batched result should match per-graph computation.

        Rationale: Computing deltas for a batch should give the same results
        as computing each graph individually.
        """
        batch_size, n = 3, 8
        A_clean = torch.stack(
            [create_random_symmetric(n, seed=i) for i in range(batch_size)]
        )
        A_other = torch.stack(
            [create_random_symmetric(n, seed=i + 50) for i in range(batch_size)]
        )

        # Batched computation
        deltas_batch = compute_spectral_deltas(A_clean, A_other)

        # Per-graph computation
        for i in range(batch_size):
            deltas_single = compute_spectral_deltas(A_clean[i], A_other[i])
            for key in deltas_batch:
                assert torch.allclose(
                    deltas_batch[key][i], deltas_single[key], atol=1e-5
                ), f"Mismatch in {key} for graph {i}"

    def test_complete_graph_known_values(self):
        """Complete graph has known spectrum: n-1 (mult. 1), -1 (mult. n-1).

        Rationale: For K_n, eigenvalues are n-1 (once) and -1 (n-1 times).
        The spectral gap is (n-1) - (-1) = n. Slightly perturbing the graph
        should show small deltas.
        """
        n = 10
        K_n = create_complete_graph(n).unsqueeze(0)

        # Verify spectrum of complete graph
        eigs = torch.linalg.eigvalsh(K_n[0])
        assert torch.allclose(eigs[-1], torch.tensor(float(n - 1)), atol=1e-5)
        assert torch.allclose(eigs[:-1], torch.tensor(-1.0), atol=1e-5)

        # Small perturbation should give small deltas
        K_n_perturbed = K_n.clone()
        K_n_perturbed[0, 0, 1] -= 0.1
        K_n_perturbed[0, 1, 0] -= 0.1

        deltas = compute_spectral_deltas(K_n, K_n_perturbed)

        # Deltas should be small but nonzero
        assert abs(deltas["eigengap_delta"].item()) < 0.1
        assert abs(deltas["eigenvalue_drift"].item()) < 0.1

    def test_subspace_distance_bounded(self):
        """Subspace distance should be <= 2*sqrt(k) for any graphs.

        Rationale: The projection Frobenius norm ||P1 - P2||_F is bounded
        by 2*sqrt(k) where P is a rank-k projection matrix. This holds because
        ||P||_F = sqrt(rank(P)) for any projection matrix.
        """
        n, k = 15, 4
        A1 = create_random_symmetric(n, seed=111).unsqueeze(0)
        A2 = create_random_symmetric(n, seed=222).unsqueeze(0)

        deltas = compute_spectral_deltas(A1, A2, k=k)

        max_bound = 2 * math.sqrt(k)
        assert deltas["subspace_distance"].item() <= max_bound + 1e-5

    def test_numerical_stability_small_gap(self):
        """Graphs with near-zero spectral gap shouldn't produce inf/nan.

        Rationale: Division by spectral gap uses epsilon (1e-10) to avoid
        division by zero. This test verifies numerical stability.
        """
        # Create graph with repeated eigenvalues (small gap)
        n = 6
        # Regular graph has more clustered eigenvalues
        A = torch.ones(n, n)
        A.fill_diagonal_(0)
        A = A.unsqueeze(0)

        deltas = compute_spectral_deltas(A, A + 0.001 * torch.randn(1, n, n))

        for key, val in deltas.items():
            assert torch.all(torch.isfinite(val)), f"{key} should be finite"
            assert not torch.any(torch.isnan(val)), f"{key} should not be NaN"

    def test_disconnected_graph_zero_connectivity(self):
        """Disconnected graphs have zero algebraic connectivity.

        Rationale: The algebraic connectivity (Fiedler value, lambda_2 of
        Laplacian) is 0 iff the graph is disconnected.
        """
        n = 10
        A_disconnected = create_disconnected_graph(n).unsqueeze(0)

        # Compute Laplacian eigenvalues
        from tmgg.utils.spectral.laplacian import (
            compute_laplacian,
        )

        L = compute_laplacian(A_disconnected)
        eigs = torch.linalg.eigvalsh(L)

        # lambda_2 should be near 0 for disconnected graph
        assert eigs[0, 1].item() < 1e-5, "Disconnected graph should have lambda_2 ≈ 0"

        # Comparing two disconnected graphs
        A_disconnected2 = create_disconnected_graph(n).unsqueeze(0)
        A_disconnected2 = A_disconnected2 + 0.01 * torch.randn_like(A_disconnected2)
        A_disconnected2 = (A_disconnected2 + A_disconnected2.transpose(-2, -1)) / 2
        A_disconnected2.clamp_(0, 1)

        deltas = compute_spectral_deltas(A_disconnected, A_disconnected2)

        # Should still produce finite results
        assert torch.all(torch.isfinite(deltas["alg_conn_delta"]))

    def test_different_k_values(self):
        """Different k values should affect subspace_distance only.

        Rationale: The k parameter only affects subspace comparison;
        other metrics use full spectrum.
        """
        n = 12
        A1 = create_random_symmetric(n, seed=333).unsqueeze(0)
        A2 = create_random_symmetric(n, seed=444).unsqueeze(0)

        deltas_k2 = compute_spectral_deltas(A1, A2, k=2)
        deltas_k4 = compute_spectral_deltas(A1, A2, k=4)
        deltas_k8 = compute_spectral_deltas(A1, A2, k=8)

        # Non-subspace metrics should be identical
        assert torch.allclose(deltas_k2["eigengap_delta"], deltas_k4["eigengap_delta"])
        assert torch.allclose(deltas_k4["eigengap_delta"], deltas_k8["eigengap_delta"])
        assert torch.allclose(
            deltas_k2["eigenvalue_drift"], deltas_k8["eigenvalue_drift"]
        )

        # Subspace distance may differ with k
        # (not testing specific values, just that it's computed)
        assert deltas_k2["subspace_distance"].shape == (1,)


class TestComputeSpectralDeltasReduceMean:
    """Tests for compute_spectral_deltas with reduce='mean'."""

    def test_returns_float_values(self):
        """reduce='mean' should return dict of floats, not tensors.

        Rationale: reduce='mean' is designed for logging, which expects
        Python floats, not tensors.
        """
        batch_size, n = 4, 8
        A_clean = torch.stack(
            [create_random_symmetric(n, seed=i) for i in range(batch_size)]
        )
        A_other = torch.stack(
            [create_random_symmetric(n, seed=i + 100) for i in range(batch_size)]
        )

        summary = compute_spectral_deltas(A_clean, A_other, reduce="mean")

        for key, val in summary.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"

    def test_matches_deltas_mean(self):
        """reduce='mean' values should match mean of unreduced results.

        Rationale: reduce='mean' averages over the batch dimension, so
        values must match manual .mean().item() on per-graph tensors.
        """
        batch_size, n = 5, 10
        A_clean = torch.stack(
            [create_random_symmetric(n, seed=i) for i in range(batch_size)]
        )
        A_other = torch.stack(
            [create_random_symmetric(n, seed=i + 200) for i in range(batch_size)]
        )

        deltas = compute_spectral_deltas(A_clean, A_other)
        summary = compute_spectral_deltas(A_clean, A_other, reduce="mean")

        for key in deltas:
            expected_mean = deltas[key].mean().item()
            assert abs(summary[key] - expected_mean) < 1e-6, f"Mismatch in {key}"

    def test_2d_input_returns_floats(self):
        """2D input with reduce='mean' should still return float dict.

        Rationale: Even with single graph (which produces scalar tensors),
        reduce='mean' should return Python floats.
        """
        n = 8
        A_clean = create_random_symmetric(n, seed=555)
        A_other = create_random_symmetric(n, seed=666)

        summary = compute_spectral_deltas(A_clean, A_other, reduce="mean")

        for key, val in summary.items():
            assert isinstance(val, float), f"{key} should be float for 2D input"

    def test_expected_keys(self):
        """reduce='mean' should contain all four expected metric keys.

        Rationale: Verify the API contract - all metrics should be present.
        """
        A = create_random_symmetric(8, seed=777).unsqueeze(0)

        summary = compute_spectral_deltas(A, A, reduce="mean")

        expected_keys = {
            "eigengap_delta",
            "alg_conn_delta",
            "eigenvalue_drift",
            "subspace_distance",
        }
        assert set(summary.keys()) == expected_keys

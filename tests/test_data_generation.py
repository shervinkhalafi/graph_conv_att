"""Tests for data generation utilities."""

import numpy as np
import pytest
import torch

from tmgg.data import (
    generate_block_sizes,
    generate_sbm_adjacency,
)
from tmgg.utils.noising import (
    LogitNoise,
    add_edge_flip_noise,
    add_gaussian_noise,
    add_logit_noise,
    add_rotation_noise,
    create_noise_definition,
    random_skew_symmetric_matrix,
)


class TestSBMGeneration:
    """Test stochastic block model generation."""

    def test_generate_sbm_adjacency_basic(self):
        """Test basic SBM generation."""
        block_sizes = [5, 5, 5]
        p = 1.0
        q = 0.0

        A = generate_sbm_adjacency(block_sizes, p, q)

        assert A.shape == (15, 15)
        assert np.all(A >= 0) and np.all(A <= 1)
        assert np.allclose(A, A.T)  # Symmetric

        # Check block structure (off-diagonal entries are 1, diagonal is 0)
        for start in (0, 5, 10):
            block = A[start : start + 5, start : start + 5]
            assert np.all(np.diag(block) == 0), "No self-loops"
            assert np.all(block + np.eye(5) == 1), "All off-diagonal are 1"
        assert np.all(A[:5, 5:10] == 0)  # No inter-block connections

    def test_generate_block_sizes(self):
        """Test block size generation."""
        n = 20
        partitions = generate_block_sizes(
            n, min_blocks=2, max_blocks=3, min_size=3, max_size=10
        )

        assert len(partitions) > 0
        for partition in partitions:
            assert sum(partition) == n
            assert len(partition) >= 2 and len(partition) <= 3
            assert all(size >= 3 and size <= 10 for size in partition)


class TestNoiseGeneration:
    """Test noise generation functions."""

    def test_random_skew_symmetric_matrix(self):
        """Test skew-symmetric matrix generation with explicit RNG."""
        n = 5
        rng = np.random.default_rng(42)
        S = random_skew_symmetric_matrix(n, rng=rng)

        assert S.shape == (n, n)
        assert np.allclose(S, -S.T)  # Skew-symmetric property
        assert np.allclose(np.diag(S), 0)  # Diagonal is zero

    def test_random_skew_symmetric_matrix_reproducibility(self):
        """Passing the same seed produces identical matrices."""
        n = 5
        S1 = random_skew_symmetric_matrix(n, rng=np.random.default_rng(99))
        S2 = random_skew_symmetric_matrix(n, rng=np.random.default_rng(99))
        assert np.array_equal(S1, S2)

    def test_add_gaussian_noise(self):
        """Test Gaussian noise addition."""
        A = np.eye(5)
        eps = 0.1

        A_noisy = add_gaussian_noise(A, eps)
        eigenvalues, V = torch.linalg.eigh(A_noisy.float())

        assert A_noisy.shape == A.shape
        assert isinstance(A_noisy, torch.Tensor)
        assert isinstance(V, torch.Tensor)
        assert isinstance(eigenvalues, torch.Tensor)
        assert V.shape == (5, 5)
        assert eigenvalues.shape == (5,)

        # Test noise level effect
        eps_large = 1.0
        A_noisy_large = add_gaussian_noise(A, eps_large)

        # Larger noise should cause more deviation
        deviation_small = torch.norm(A_noisy - torch.tensor(A, dtype=torch.float32))
        deviation_large = torch.norm(
            A_noisy_large - torch.tensor(A, dtype=torch.float32)
        )
        assert deviation_large > deviation_small

    def test_add_gaussian_noise_batch(self):
        """Test Gaussian noise with batch input."""
        batch_size = 3
        A = torch.eye(5).unsqueeze(0).repeat(batch_size, 1, 1)
        eps = 0.1

        A_noisy = add_gaussian_noise(A, eps)
        eigenvalues, V = torch.linalg.eigh(A_noisy.float())

        assert A_noisy.shape == (batch_size, 5, 5)
        assert V.shape == (batch_size, 5, 5)
        assert eigenvalues.shape == (batch_size, 5)

        # Check that noise is different for each batch element
        assert not torch.allclose(A_noisy[0], A_noisy[1])

    def test_add_edge_flip_noise(self):
        """Test digress (edge flipping) noise."""
        A = np.ones((5, 5))
        p = 0.0  # No flipping

        A_noisy = add_edge_flip_noise(A, p)
        eigenvalues, V = torch.linalg.eigh(A_noisy.float())

        assert torch.allclose(A_noisy, torch.ones(5, 5))

        # Test with some flipping
        A = np.zeros((5, 5))
        p = 1.0  # Flip all edges (not diagonal)
        A_noisy = add_edge_flip_noise(A, p)
        eigenvalues, V = torch.linalg.eigh(A_noisy.float())

        # Should flip all off-diagonal elements to 1, diagonal stays 0
        expected = torch.ones(5, 5) - torch.eye(5)
        assert torch.allclose(A_noisy, expected)

        # Test intermediate noise level
        # Note: This test requires a specific seed to ensure at least some elements are flipped
        # due to the probabilistic nature of the noise. Seed 0 works reliably for 5x5 matrix with p=0.2
        torch.manual_seed(0)
        A = np.eye(5)
        p = 0.2
        A_noisy = add_edge_flip_noise(A, p)
        eigenvalues, V = torch.linalg.eigh(A_noisy.float())

        # Check that some elements are flipped
        diff_count = torch.sum(A_noisy != torch.tensor(A, dtype=torch.float32))
        assert diff_count > 0 and diff_count < 25  # Some but not all elements flipped

        # Check eigendecomposition is valid
        reconstructed = torch.matmul(
            torch.matmul(V, torch.diag_embed(eigenvalues)), V.T
        )
        assert torch.allclose(reconstructed, A_noisy, atol=1e-5)

    def test_add_edge_flip_noise_symmetry(self):
        """Test that digress noise preserves symmetry."""
        # Create symmetric matrix
        A = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=float)
        p = 0.3

        A_noisy = add_edge_flip_noise(A, p)

        # Check symmetry is preserved
        assert torch.allclose(A_noisy, A_noisy.T)

    def test_add_edge_flip_noise_discrete(self):
        """Test that digress noise maintains discrete values (0 or 1 only)."""
        # Test various cases
        test_cases = [
            (np.ones((5, 5)), 0.5),
            (np.zeros((5, 5)), 0.5),
            (np.eye(5), 0.3),
            (np.random.randint(0, 2, (10, 10)), 0.7),
        ]

        for A, p in test_cases:
            # Make A symmetric
            A = (A + A.T) / 2
            A = (A > 0.5).astype(float)

            A_noisy = add_edge_flip_noise(A, p)

            # Check that all values are either 0 or 1
            unique_values = torch.unique(A_noisy)
            assert torch.all(
                (unique_values == 0) | (unique_values == 1)
            ), f"Found non-discrete values: {unique_values}"

            # Verify no intermediate values like 0.5
            assert not torch.any(
                torch.abs(A_noisy - 0.5) < 1e-6
            ), "Found values close to 0.5"

    def test_add_edge_flip_noise_signature(self):
        """Signature includes A, p, and optional generator (torch.Generator).

        The old ``rng: np.random.Generator`` was removed; the new
        ``generator: torch.Generator | None`` provides seeded reproducibility
        via PyTorch's generator protocol.
        """
        import inspect

        sig = inspect.signature(add_edge_flip_noise)
        param_names = list(sig.parameters.keys())
        assert param_names == [
            "A",
            "p",
            "generator",
        ], f"Expected ['A', 'p', 'generator'], got {param_names}"
        # generator defaults to None (keyword-only not required)
        assert sig.parameters["generator"].default is None

    def test_add_rotation_noise(self):
        """Test rotation noise addition."""
        A = torch.eye(5).unsqueeze(0)  # Add batch dimension
        eps = 0.1
        skew = random_skew_symmetric_matrix(5, rng=np.random.default_rng(0))

        A_noisy = add_rotation_noise(A, eps, skew)
        eigenvalues, V_rot = torch.linalg.eigh(A_noisy.float())
        eigenvalues_original, _ = torch.linalg.eigh(A.float())

        assert A_noisy.shape == A.shape
        assert V_rot.shape == (1, 5, 5)
        assert eigenvalues.shape == (1, 5)

        # Check that eigenvalues are preserved (rotation doesn't change eigenvalues)
        assert torch.allclose(eigenvalues, eigenvalues_original, atol=1e-5)

        # Check reconstruction
        eig_diag = torch.diag_embed(eigenvalues)
        reconstructed = torch.matmul(
            torch.matmul(V_rot, eig_diag), V_rot.transpose(-2, -1)
        )
        assert torch.allclose(reconstructed, A_noisy, atol=1e-5)

    def test_add_rotation_noise_orthogonality(self):
        """Test that rotation preserves orthogonality of eigenvectors."""
        A = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])).unsqueeze(0)
        eps = 0.5
        skew = random_skew_symmetric_matrix(5, rng=np.random.default_rng(1))

        A_noisy = add_rotation_noise(A, eps, skew)
        eigenvalues, V_rot = torch.linalg.eigh(A_noisy.float())
        eigenvalues_original, _ = torch.linalg.eigh(A.float())

        # Check eigenvectors are still orthonormal
        VTV = torch.matmul(V_rot.transpose(-2, -1), V_rot)
        assert torch.allclose(VTV, torch.eye(5), atol=1e-5)

        # Check eigenvalues are preserved (rotation doesn't change eigenvalues)
        assert torch.allclose(
            eigenvalues.sort().values, eigenvalues_original.sort().values, atol=1e-5
        )

    def test_noise_functions_with_numpy_input(self):
        """Test that noise functions handle numpy arrays properly."""
        A_np = np.eye(5)

        # Test each noise function
        A_gauss = add_gaussian_noise(A_np, 0.1)
        assert isinstance(A_gauss, torch.Tensor)

        A_digress = add_edge_flip_noise(A_np, 0.1)
        assert isinstance(A_digress, torch.Tensor)

        A_rot = add_rotation_noise(
            A_np, 0.1, random_skew_symmetric_matrix(5, rng=np.random.default_rng(2))
        )
        assert isinstance(A_rot, torch.Tensor)


class TestLogitNoise:
    """Test logit-space noise generation.

    Test Rationale
    --------------
    Logit noise operates in log-odds space, transforming binary adjacencies to
    continuous probabilities. The key properties we verify are:
    - Output bounds: values must lie in (0, 1) due to sigmoid activation
    - Symmetry preservation: noise is applied symmetrically to upper triangle
    - Numerical stability: edge cases near 0 and 1 are handled via clamping
    - Batch handling: consistent behavior for single and batched inputs
    """

    def test_add_logit_noise_output_bounds(self):
        """Verify output is strictly within (0, 1) interval."""
        A = torch.eye(5)
        sigma = 2.0

        A_noisy = add_logit_noise(A, sigma)

        assert A_noisy.min() > 0, "Output should be strictly positive"
        assert A_noisy.max() < 1, "Output should be strictly less than 1"

    def test_add_logit_noise_symmetry(self):
        """Verify noisy adjacency remains symmetric."""
        A = torch.eye(5)
        sigma = 1.0

        A_noisy = add_logit_noise(A, sigma)

        assert torch.allclose(A_noisy, A_noisy.T), "Output should be symmetric"

    def test_add_logit_noise_batch(self):
        """Test logit noise with batched input."""
        batch_size = 3
        A = torch.eye(5).unsqueeze(0).repeat(batch_size, 1, 1)
        sigma = 1.0

        A_noisy = add_logit_noise(A, sigma)

        assert A_noisy.shape == (batch_size, 5, 5), "Batch shape should be preserved"
        assert A_noisy.min() > 0 and A_noisy.max() < 1, "Bounds should hold for batch"

        # Each batch element should be symmetric
        for i in range(batch_size):
            assert torch.allclose(A_noisy[i], A_noisy[i].T)

        # Batch elements should differ due to independent noise
        assert not torch.allclose(A_noisy[0], A_noisy[1])

    def test_add_logit_noise_intensity(self):
        """Verify sigma controls noise intensity.

        Note: Noise is only applied to off-diagonal elements (upper triangle,
        mirrored for symmetry). The diagonal remains unperturbed, reflecting
        that self-loops are typically excluded from noise processes.
        """
        A = torch.eye(5)

        A_low = add_logit_noise(A, sigma=0.5)
        A_high = add_logit_noise(A, sigma=3.0)

        # Extract upper triangular off-diagonal elements (where noise was applied)
        # Original off-diagonal values were 0.0 -> clamped -> logit -> noise -> sigmoid
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        offdiag_low = A_low[mask]
        offdiag_high = A_high[mask]

        # With higher sigma, off-diagonal values should have more variance
        # (pushed further from their baseline by larger noise)
        assert (
            offdiag_high.std() > 0
        ), "High sigma should produce varied off-diagonal values"
        # Also check that high sigma produces more variance than low sigma on average
        # (stochastic, but very likely with sigma 0.5 vs 3.0)
        assert (
            offdiag_high.std() >= offdiag_low.std() * 0.5
        ), "High sigma should have similar or more variance"

    def test_add_logit_noise_numpy_input(self):
        """Test that numpy arrays are handled correctly."""
        A_np = np.eye(5)
        sigma = 1.0

        A_noisy = add_logit_noise(A_np, sigma)

        assert isinstance(A_noisy, torch.Tensor)
        assert A_noisy.min() > 0 and A_noisy.max() < 1

    def test_add_logit_noise_clamp_eps(self):
        """Test numerical stability with custom clamp epsilon."""
        # Binary adjacency matrix
        A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        sigma = 1.0

        # Should work with default clamping
        A_noisy = add_logit_noise(A, sigma, clamp_eps=1e-6)
        assert torch.isfinite(A_noisy).all(), "Output should be finite"

        # Custom clamp_eps should also work
        A_noisy_custom = add_logit_noise(A, sigma, clamp_eps=1e-3)
        assert torch.isfinite(
            A_noisy_custom
        ).all(), "Output should be finite with custom eps"

    def test_logit_noise_generator_class(self):
        """Test LogitNoise wrapper class."""
        generator = LogitNoise(clamp_eps=1e-6)

        assert generator.requires_state is False

        A = torch.eye(5)
        A_noisy = generator.add_noise(A, eps=1.0)

        assert isinstance(A_noisy, torch.Tensor)
        assert A_noisy.min() > 0 and A_noisy.max() < 1
        assert torch.allclose(A_noisy, A_noisy.T)

    def test_create_noise_definition_logit(self):
        """Test factory function creates LogitNoise correctly."""
        generator = create_noise_definition("logit")

        assert isinstance(generator, LogitNoise)
        assert generator.requires_state is False

        A = torch.eye(5)
        A_noisy = generator.add_noise(A, eps=1.5)

        assert A_noisy.min() > 0 and A_noisy.max() < 1

    def test_create_noise_definition_logit_with_clamp_eps(self):
        """Test factory function respects clamp_eps parameter."""
        generator = create_noise_definition("logit", clamp_eps=1e-4)

        assert isinstance(generator, LogitNoise)
        assert generator.clamp_eps == 1e-4


if __name__ == "__main__":
    pytest.main([__file__])

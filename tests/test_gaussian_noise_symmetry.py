"""Regression test: Gaussian noise must be symmetric before downstream processing.

Rationale: Prior to this fix, add_gaussian_noise added independent noise to
each (i,j) and (j,i). Downstream symmetrization via (A+A^T)/2 then halved
the variance, making effective noise eps/sqrt(2) instead of eps. This 29%
bias made Gaussian noise experiments incomparable to other noise types at
matched eps values. The fix generates upper-triangular noise and mirrors it.
"""

import torch

from tmgg.utils.noising.noise import add_gaussian_noise


def test_gaussian_noise_is_symmetric():
    """Noise matrix (A_noisy - A) must be symmetric entry-wise."""
    torch.manual_seed(42)
    A = torch.zeros(10, 10)
    A_noisy = add_gaussian_noise(A, eps=1.0)
    noise = A_noisy - A
    assert torch.allclose(noise, noise.T), (
        "Gaussian noise is not symmetric. "
        f"Max asymmetry: {(noise - noise.T).abs().max().item():.6f}"
    )


def test_gaussian_noise_is_symmetric_batched():
    """Batched noise must also be symmetric."""
    torch.manual_seed(42)
    A = torch.zeros(4, 10, 10)
    A_noisy = add_gaussian_noise(A, eps=1.0)
    noise = A_noisy - A
    noise_T = noise.transpose(-2, -1)
    assert torch.allclose(noise, noise_T), (
        "Batched Gaussian noise is not symmetric. "
        f"Max asymmetry: {(noise - noise_T).abs().max().item():.6f}"
    )


def test_gaussian_noise_variance_matches_eps():
    """Off-diagonal noise variance should be eps^2, not eps^2/2."""
    torch.manual_seed(0)
    n = 50
    eps = 1.0
    # Run many samples to get stable variance estimate
    variances = []
    for seed in range(100):
        torch.manual_seed(seed)
        A = torch.zeros(n, n)
        A_noisy = add_gaussian_noise(A, eps=eps)
        noise = A_noisy - A
        # Extract upper-triangular off-diagonal entries
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        off_diag = noise[mask]
        variances.append(off_diag.var().item())

    mean_var = sum(variances) / len(variances)
    # Should be close to eps^2 = 1.0, not eps^2/2 = 0.5
    assert mean_var > 0.8, (
        f"Noise variance {mean_var:.3f} is too low (expected ~{eps**2:.1f}). "
        "Noise may be getting halved by symmetrization."
    )


def test_gaussian_noise_diagonal_is_zero_noise():
    """Diagonal should receive zero noise (self-loops don't get perturbed)."""
    torch.manual_seed(42)
    A = torch.eye(10)
    A_noisy = add_gaussian_noise(A, eps=1.0)
    # Diagonal noise should be zero (we only add off-diagonal noise)
    diag_noise = (A_noisy - A).diagonal()
    assert torch.allclose(
        diag_noise, torch.zeros_like(diag_noise)
    ), f"Diagonal received noise: {diag_noise}"

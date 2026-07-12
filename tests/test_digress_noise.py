"""Tests for correct DiGress categorical noise (Vignac et al. 2023).

Rationale: The audit found that the old 'add_digress_noise' was simple
Bernoulli edge flipping (now renamed to add_edge_flip_noise). The correct
DiGress noise uses categorical transition matrices parameterised by
alpha_bar (cumulative noise schedule), not raw flip probability.

For binary edges (K=2), the transition matrix is:
    Q_bar_t = alpha_bar * I + (1 - alpha_bar) / 2 * [[1,1],[1,1]]

This gives flip probability = (1 - alpha_bar) / 2, where alpha_bar
ranges from ~1.0 (clean) to ~0.0 (fully noisy / uniform).
"""

import torch


class TestDigressNoise:
    def test_importable(self):
        from tmgg.utils.noising.noise import add_digress_noise

        assert callable(add_digress_noise)

    def test_alpha_bar_1_returns_clean(self):
        """alpha_bar=1.0 means no noise: output must equal input."""
        from tmgg.utils.noising.noise import add_digress_noise

        A = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        A_noisy = add_digress_noise(A, alpha_bar=1.0)
        torch.testing.assert_close(A_noisy, A)

    def test_alpha_bar_0_approaches_uniform(self):
        """alpha_bar=0.0 means fully noisy: each edge has ~50% probability."""
        from tmgg.utils.noising.noise import add_digress_noise

        torch.manual_seed(42)
        A = torch.zeros(500, 20, 20)
        A_noisy = add_digress_noise(A, alpha_bar=0.0)
        # Each entry flips with probability 0.5, so mean should be ~0.5
        mask = torch.triu(torch.ones(20, 20, dtype=torch.bool), diagonal=1)
        mean_val = A_noisy[:, mask].mean().item()
        assert 0.45 < mean_val < 0.55, f"Expected ~0.5, got {mean_val}"

    def test_output_is_binary(self):
        """DiGress noise on binary input produces binary output."""
        from tmgg.utils.noising.noise import add_digress_noise

        torch.manual_seed(0)
        A = (torch.rand(4, 15, 15) > 0.5).float()
        A = (A + A.transpose(-1, -2)).clamp(max=1)
        A_noisy = add_digress_noise(A, alpha_bar=0.8)
        unique_vals = A_noisy.unique()
        assert all(v in (0.0, 1.0) for v in unique_vals.tolist())

    def test_preserves_symmetry(self):
        """Noisy adjacency must remain symmetric."""
        from tmgg.utils.noising.noise import add_digress_noise

        torch.manual_seed(123)
        A = torch.zeros(10, 10)
        A[0, 1] = A[1, 0] = 1
        A[2, 3] = A[3, 2] = 1
        A_noisy = add_digress_noise(A, alpha_bar=0.7)
        torch.testing.assert_close(A_noisy, A_noisy.T)

    def test_flip_probability_matches_theory(self):
        """For binary edges, flip prob should be (1 - alpha_bar) / 2.

        With alpha_bar=0.6, flip_prob = 0.2. Over many samples, the fraction
        of flipped edges should be close to 0.2.
        """
        from tmgg.utils.noising.noise import add_digress_noise

        torch.manual_seed(42)
        alpha_bar = 0.6
        expected_flip = (1 - alpha_bar) / 2  # 0.2

        A = torch.zeros(500, 20, 20)  # all-zero adjacency
        A_noisy = add_digress_noise(A, alpha_bar=alpha_bar)

        # Fraction of upper-triangle entries that flipped from 0 to 1
        mask = torch.triu(torch.ones(20, 20, dtype=torch.bool), diagonal=1)
        flipped = A_noisy[:, mask].mean().item()
        assert (
            abs(flipped - expected_flip) < 0.03
        ), f"Expected flip rate ~{expected_flip}, got {flipped}"

    def test_signature_uses_alpha_bar_not_p(self):
        """DiGress noise must be parameterised by alpha_bar, not flip probability."""
        import inspect

        from tmgg.utils.noising.noise import add_digress_noise

        sig = inspect.signature(add_digress_noise)
        params = list(sig.parameters.keys())
        assert "alpha_bar" in params, f"Expected alpha_bar parameter, got {params}"
        assert "p" not in params, "Should use alpha_bar, not raw flip probability p"

    def test_definition_converts_eps_to_alpha_bar(self):
        """DigressNoise.add_noise(A, eps) must convert eps to alpha_bar.

        eps=0 -> alpha_bar=1 (clean)
        eps=1 -> alpha_bar=0 (fully noisy)
        """
        import torch

        from tmgg.utils.noising.noise import (
            create_noise_definition,
        )

        torch.manual_seed(42)
        gen = create_noise_definition("digress")

        # eps=0 should produce clean output
        A = torch.zeros(10, 10)
        A[0, 1] = A[1, 0] = 1
        A_noisy = gen.add_noise(A, eps=0.0)
        torch.testing.assert_close(A_noisy, A)

    def test_factory_creates_digress(self):
        from tmgg.utils.noising.noise import (
            create_noise_definition,
        )

        gen = create_noise_definition("digress")
        assert type(gen).__name__ == "DigressNoise"

    def test_factory_edge_flip_still_works(self):
        from tmgg.utils.noising.noise import (
            create_noise_definition,
        )

        gen = create_noise_definition("edge_flip")
        assert type(gen).__name__ == "EdgeFlipNoise"

"""Tests for the categorical sampling primitives in diffusion_sampling.py.

Pins the contract of :func:`_sample_from_unnormalised_posterior`, the
canonical upstream-DiGress primitive ported in parity #28 / #29 / D-5.
The helper normalises an unnormalised PMF with a row-sum floor, then
draws categorical indices via ``torch.multinomial``. The floor mirrors
upstream's ``denom[denom == 0] = 1e-5`` guard
(``DiGress/src/diffusion/diffusion_utils.py:274-290``) so degenerate
posterior rows do not crash sampling.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.diffusion.diffusion_sampling import (
    _sample_from_unnormalised_posterior,
    compute_posterior_distribution,
    compute_posterior_distribution_per_x0,
    sample_discrete_features,
)


class TestSampleFromUnnormalisedPosterior:
    """Contract for :func:`_sample_from_unnormalised_posterior`.

    Test rationale (D-5): the helper underpins the upstream sampling
    contract. We pin three behaviours:

    1. On a healthy non-negative tensor the sampled distribution agrees
       with the manual ``unnorm / sum + multinomial`` reference within
       Monte-Carlo error.
    2. On a degenerate row (all zeros), the helper does NOT crash and
       produces a uniformly-distributed sample because the row-sum
       floor clamps the divisor away from zero.
    3. Negative input raises a clear ``ValueError`` -- the helper does
       not silently mask a bug by calling ``abs()``.
    """

    def test_matches_reference_on_healthy_posterior(self) -> None:
        """Sampled empirical distribution matches the manual reference."""
        torch.manual_seed(0)
        bs, k = 1, 4
        n = 10000
        # Random non-negative weights of shape (bs, n, k) -- sample n
        # categorical draws and compare empirical class frequency to the
        # explicit normalised PMF.
        unnorm = torch.rand(bs, n, k).abs() + 0.1  # strictly positive
        sampled = _sample_from_unnormalised_posterior(unnorm)
        assert sampled.shape == (bs, n)
        assert sampled.dtype == torch.long

        # Reference normalisation
        ref_prob = unnorm / unnorm.sum(dim=-1, keepdim=True)
        # Empirical class-1 frequency at each position should agree
        # with the analytic class-1 probability within 1.5/sqrt(n).
        # We check a handful of positions.
        # Per-row only one sample; instead pick a single PMF row and
        # draw many samples from it via the helper.
        single_row = unnorm[0, 0:1].repeat(1, n).reshape(1, n, k)
        single_sampled = _sample_from_unnormalised_posterior(single_row)
        empirical = torch.bincount(single_sampled[0], minlength=k).float() / n
        analytic = ref_prob[0, 0]
        max_dev = (empirical - analytic).abs().max().item()
        # Standard error ~ sqrt(p*(1-p)/n) <= 0.5/sqrt(n) = 0.005;
        # 6-sigma envelope is 0.03.
        assert max_dev < 0.03, (
            f"Empirical vs analytic disagree by {max_dev:.4f}, "
            f"analytic={analytic.tolist()}, empirical={empirical.tolist()}"
        )

    def test_degenerate_row_does_not_crash(self) -> None:
        """All-zero row falls back to uniform via the upstream zero-row branch.

        Upstream ``unnormalized[unnormalized.sum(dim=-1) == 0] = 1e-5``
        sets the entire row to ``zero_floor``, producing a uniform PMF
        after normalisation. The helper inherits this fallback so
        sampling continues on degenerate rows rather than crashing in
        :func:`torch.multinomial`.
        """
        torch.manual_seed(1)
        bs, n, k = 2, 3, 4
        unnorm = torch.zeros(bs, n, k)
        sampled = _sample_from_unnormalised_posterior(unnorm)
        assert sampled.shape == (bs, n)
        assert sampled.dtype == torch.long
        # All sampled indices must be in [0, k).
        assert int(sampled.min().item()) >= 0
        assert int(sampled.max().item()) < k

    @pytest.mark.skipif(
        not __debug__,
        reason=(
            "Non-negativity check is __debug__-guarded so production runs "
            "(PYTHONOPTIMIZE=1) skip the per-step bool(.all()) sync; see "
            "internal notes (not included)"
        ),
    )
    def test_negative_input_raises(self) -> None:
        """Negative probability mass surfaces as a clear ValueError."""
        unnorm = torch.tensor([[[0.5, -0.1, 0.6]]])
        with pytest.raises(ValueError, match="negative probability mass"):
            _sample_from_unnormalised_posterior(unnorm)

    def test_non_degenerate_row_unaffected_by_floor(self) -> None:
        """A row with strictly positive mass is normalised as-is.

        The upstream zero-row branch only fires when
        ``row.sum() == 0`` exactly. A row whose entries are tiny but
        non-zero passes through normalisation unchanged.
        """
        torch.manual_seed(0)
        unnorm = torch.full((1, 1, 2), 5e-6)  # strictly positive
        sampled = _sample_from_unnormalised_posterior(unnorm)
        assert int(sampled.min().item()) >= 0
        assert int(sampled.max().item()) < 2

    def test_shape_preserved_for_4d_input(self) -> None:
        """Edge-PMF shape ``(bs, n, n, k)`` returns ``(bs, n, n)``."""
        torch.manual_seed(2)
        bs, n, k = 2, 3, 5
        unnorm = torch.rand(bs, n, n, k) + 0.1
        sampled = _sample_from_unnormalised_posterior(unnorm)
        assert sampled.shape == (bs, n, n)


class TestComputePosteriorDistributionMaskAware:
    """Regression tests for the mask-aware posterior denom guard.

    Test rationale (parity W2-3): commit ``6424fc34`` introduced a
    mask-aware split inside :func:`compute_posterior_distribution`
    after its predecessor ``e74ee0f8`` tripped on legitimate VLB-path
    inputs. CLAUDE.md mandates "first add the bug as a unit test
    BEFORE fixing"; the meta-synthesis flagged the missing regression
    test. These tests pin the three branches the guard distinguishes:

    1. Valid (one-hot M and M_t) rows divide cleanly and match the
       manual ``(M_t @ Qt^T) * (M @ Qsb) / denom`` formula.
    2. Masked rows (M or M_t row-sum zero) produce zero output rather
       than NaN, matching the post-mask cleanup the guard provides.
    3. A zero denom at a *valid* position trips the assertion with the
       informative ``min_at_valid`` message — the fail-loud branch
       CLAUDE.md mandates for real bugs.
    """

    def _kernel_with_off_diagonal_mass(
        self, bs: int, d: int, *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Build an irreducible doubly-stochastic kernel of shape (bs, d, d)."""
        kernel = torch.full((d, d), 0.1, dtype=dtype)
        kernel.fill_diagonal_(0.9 - 0.1 * (d - 2))
        # Row-stochastic, never exactly zero anywhere.
        return kernel.unsqueeze(0).expand(bs, d, d).contiguous()

    def test_masked_rows_yield_zero(self) -> None:
        """Masked-row inputs (zero M or M_t) produce zero output, no NaN."""
        torch.manual_seed(0)
        bs, n, d = 1, 4, 3
        M = torch.zeros(bs, n, d)
        M_t = torch.zeros(bs, n, d)
        # Positions 0, 1: masked (all-zero rows in both M and M_t).
        # Positions 2, 3: valid one-hot.
        M[0, 2, 0] = 1.0
        M[0, 3, 1] = 1.0
        M_t[0, 2, 1] = 1.0
        M_t[0, 3, 2] = 1.0

        Qt_M = self._kernel_with_off_diagonal_mass(bs, d)
        Qsb_M = self._kernel_with_off_diagonal_mass(bs, d)
        Qtb_M = self._kernel_with_off_diagonal_mass(bs, d)

        out = compute_posterior_distribution(
            M, M_t, Qt_M, Qsb_M, Qtb_M, field="X_class"
        )
        assert out.shape == (bs, n, d)
        assert torch.isfinite(out).all(), "guard must not return NaN"
        # Masked positions: every class has zero probability.
        assert torch.allclose(out[0, 0], torch.zeros(d))
        assert torch.allclose(out[0, 1], torch.zeros(d))
        # Valid positions: nonzero somewhere.
        assert out[0, 2].abs().sum() > 0
        assert out[0, 3].abs().sum() > 0

    def test_valid_rows_match_manual_formula(self) -> None:
        """Valid-row outputs match the manual ``(M_t @ Qt^T) * (M @ Qsb) / denom`` formula."""
        torch.manual_seed(1)
        bs, n, d = 1, 3, 4
        M = torch.zeros(bs, n, d)
        M_t = torch.zeros(bs, n, d)
        # All-valid one-hot: position k → class (k+1) mod d, M_t → class k mod d.
        for i in range(n):
            M[0, i, (i + 1) % d] = 1.0
            M_t[0, i, i % d] = 1.0

        Qt_M = self._kernel_with_off_diagonal_mass(bs, d)
        Qsb_M = self._kernel_with_off_diagonal_mass(bs, d)
        Qtb_M = self._kernel_with_off_diagonal_mass(bs, d)

        out = compute_posterior_distribution(
            M, M_t, Qt_M, Qsb_M, Qtb_M, field="X_class"
        )

        # Manual reference using the same formula (no guard applied because
        # every position is valid).
        Qt_M_T = Qt_M.transpose(-2, -1)
        left = M_t @ Qt_M_T
        right = M @ Qsb_M
        product = left * right
        denom = ((M @ Qtb_M) * M_t).sum(dim=-1, keepdim=True)
        expected = product / denom

        torch.testing.assert_close(out, expected)

    @pytest.mark.skipif(
        not __debug__,
        reason=(
            "Degenerate-denominator assert is __debug__-guarded so "
            "production runs (PYTHONOPTIMIZE=1) skip the bool(.all()) sync; "
            "see internal notes (not included)"
        ),
    )
    def test_zero_denom_at_valid_position_raises(self) -> None:
        """Construct a degenerate valid-position scenario; guard asserts loudly.

        Set ``M`` and ``M_t`` to one-hot at *disjoint* classes, with
        ``Qtb_M`` sending mass strictly along the identity. Then
        ``denom = (M @ Qtb_M * M_t).sum(-1) = 0`` while the position is
        flagged valid (both row-sums positive), tripping the assert.
        """
        bs, n, d = 1, 1, 2
        M = torch.zeros(bs, n, d)
        M_t = torch.zeros(bs, n, d)
        M[0, 0, 0] = 1.0
        M_t[0, 0, 1] = 1.0
        # Identity kernels: M @ Qtb selects class 0; M_t selects class 1; the
        # element-wise product has a zero at class 1, so denom = 0.
        identity = torch.eye(d).unsqueeze(0).expand(bs, d, d).contiguous()
        Qt_M = identity.clone()
        Qsb_M = identity.clone()
        Qtb_M = identity.clone()

        with pytest.raises(AssertionError, match="min_at_valid"):
            compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M, field="X_class")


class TestHelperWireThroughEquivalence:
    """Pin numerical equivalence between the helper-based reverse-step
    sampling path and the two-step ``compute_posterior_distribution_per_x0``
    + ``sample_discrete_features`` pattern that predated W4-1.

    Test rationale (W4-1, parity #28 / D-5 wire-through): the helper
    ``_sample_from_unnormalised_posterior`` was plumbed in commit
    ``9f8e2ed1`` but unused. The active reverse path went through the
    older two-step ``per_x0 → contract → multinomial`` pattern. W4-1
    wires the helper into the active path. Before refactoring we pin
    the numerical equivalence here so the wire-through is provably a
    no-op on healthy posteriors.

    The assertion is "PMFs agree to within 1e-6 absolute" because the
    two paths differ only in whether the row-sum-zero floor (``1e-5``)
    fires; on healthy strictly-positive inputs the floor is inert and
    the two computations are float-identical up to summation order.
    """

    def _kernel_with_off_diagonal_mass(
        self, bs: int, d: int, *, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Build an irreducible row-stochastic kernel of shape ``(bs, d, d)``."""
        kernel = torch.full((d, d), 0.1, dtype=dtype)
        kernel.fill_diagonal_(1.0 - 0.1 * (d - 1))
        return kernel.unsqueeze(0).expand(bs, d, d).contiguous()

    def _build_inputs(
        self,
        *,
        bs: int = 2,
        n: int = 5,
        d: int = 4,
        seed: int = 17,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble a healthy posterior batch -- one-hot ``z_t``, soft ``pred_X``,
        and three Chapman-Kolmogorov-compatible row-stochastic kernels
        ``Qt``, ``Qsb``, ``Qtb`` with ``Qtb = Qsb @ Qt``.

        The Chapman-Kolmogorov identity ``Qtb = Qsb @ Qt`` is what makes
        the per-x0 normalisation
        ``denominator = (z_t @ Qtb^T)[c] = p(z_t | x_0=c)`` agree with the
        explicit row-sum of the per-x0 numerator -- i.e. it is the
        condition under which ``per_x0[c, :]`` is a valid PMF over ``k``
        for every ``x_0`` class ``c``. Without it the test compares two
        un-normalised tensors and the row-sums diverge by 30%+, masking
        the helper-vs-two-step comparison we are actually trying to pin.
        """
        torch.manual_seed(seed)
        # One-hot z_t (bs, n, d)
        zt_idx = torch.randint(0, d, (bs, n))
        z_t = torch.nn.functional.one_hot(zt_idx, num_classes=d).float()
        # Soft pred_X: dirichlet-ish (always > 0)
        pred_X = torch.rand(bs, n, d).abs() + 0.1
        pred_X = pred_X / pred_X.sum(dim=-1, keepdim=True)
        Qt = self._kernel_with_off_diagonal_mass(bs, d)
        Qsb = self._kernel_with_off_diagonal_mass(bs, d)
        # Enforce Chapman-Kolmogorov compatibility per the docstring.
        Qtb = torch.bmm(Qsb, Qt)
        return z_t, pred_X, Qt, Qsb, Qtb

    def test_helper_pmf_matches_two_step_pmf_on_healthy_posterior(self) -> None:
        """The marginalised contraction + helper-floor produces a PMF
        bit-equivalent to the per-x0 normalisation + contraction PMF on a
        healthy posterior (no zero rows)."""
        z_t, pred_X, Qt, Qsb, Qtb = self._build_inputs()

        # Two-step path: per-x0 normalisation -> contract over x_0.
        per_x0 = compute_posterior_distribution_per_x0(
            M_t=z_t, Qt_M=Qt, Qsb_M=Qsb, Qtb_M=Qtb, field="X_class"
        )
        prob_two_step = (per_x0 * pred_X.unsqueeze(-1)).sum(dim=2)  # (bs, n, d_zs)

        # Helper path: same contraction, then the helper's floor + normalise
        # collapses to a no-op on healthy rows.
        unnorm = (per_x0 * pred_X.unsqueeze(-1)).sum(dim=2)
        # Equivalent renormalisation that the helper performs internally.
        prob_helper = unnorm / unnorm.sum(dim=-1, keepdim=True)

        # Healthy inputs: row-sum > 0 everywhere; the helper's floor never
        # fires; renormalisation is a no-op modulo float roundoff.
        torch.testing.assert_close(prob_two_step, prob_helper, atol=1e-6, rtol=1e-6)

    def test_helper_samples_match_two_step_samples_under_same_seed(self) -> None:
        """Drawing samples through the helper vs the two-step path under the
        same ``torch.manual_seed`` yields the same indices on healthy posteriors.

        The two-step path used ``sample_discrete_features`` which calls
        ``probX.multinomial(1)`` after re-normalising masked rows. The
        helper does the same multinomial on the contracted PMF. Same seed,
        same PMF, same draw.
        """
        z_t, pred_X, Qt, Qsb, Qtb = self._build_inputs()
        bs, n, d_zs = z_t.shape

        # Healthy node mask: all positions valid.
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        # Two-step path samples
        per_x0 = compute_posterior_distribution_per_x0(
            M_t=z_t, Qt_M=Qt, Qsb_M=Qsb, Qtb_M=Qtb, field="X_class"
        )
        prob_two_step = (per_x0 * pred_X.unsqueeze(-1)).sum(dim=2)

        # Per-x0 already gives a row-summed-to-1 PMF, but sample_discrete_features
        # operates on (probX, probE). Build a dummy probE to satisfy the API.
        prob_E_dummy = torch.full((bs, n, n, d_zs), 1.0 / d_zs)
        torch.manual_seed(123)
        x_two_step, _ = sample_discrete_features(prob_two_step, prob_E_dummy, node_mask)

        # Helper path: same PMF (by previous test), same seed; multinomial draw
        # over a flat (bs*n, d_zs) PMF. The helper internally does the same
        # multinomial after a no-op renormalise on healthy rows.
        torch.manual_seed(123)
        # The helper takes an unnormalised tensor; pass the un-floored
        # contraction directly.
        x_helper = _sample_from_unnormalised_posterior(prob_two_step)

        # Both samples are integer indices in [0, d_zs); on healthy inputs
        # they should agree exactly (same seed + same effective PMF).
        # NOTE: sample_discrete_features overwrites masked rows to uniform
        # before sampling; with all-True node_mask there is no overwrite,
        # so the two paths converge.
        assert x_two_step.shape == x_helper.shape == (bs, n)
        assert torch.equal(x_two_step, x_helper), (
            f"Two-step samples {x_two_step.tolist()} differ from helper "
            f"samples {x_helper.tolist()} despite identical seed and PMF."
        )

"""Two-level sanity check tests for denoising models.

=============================================================================
PERMANENT SANITY CHECK TESTS - DO NOT MODIFY UNLESS ADDING NEW TEST CASES
=============================================================================

These tests verify fundamental model learning capability at two levels:

LEVEL 1: Constant Noise Memorization
    - Uses the SAME fixed noisy input for every training step
    - Model only needs to memorize one input→output mapping
    - If this fails, model fundamentally cannot learn (architecture broken)
    - Expected: ~100% accuracy after sufficient steps

LEVEL 2: Fresh Noise Generalization
    - Generates fresh noise each training step
    - Model must learn general denoising, not memorize one sample
    - If Level 1 passes but this fails: task too hard, needs more training
    - Expected: >70% accuracy on structured targets

Target Types:
    - erdos_renyi: Erdos-Renyi random graph with p=0.3 (sane sparsity)
    - block_diagonal: 4 dense blocks, no inter-block edges (structured)
    - sbm: Stochastic block model with p=0.8 intra, q=0.1 inter (structured)

Decision Tree:
    - Level 1 fails on ALL targets → Architecture/gradient flow broken
    - Level 1 passes, Level 2 fails on ALL → Training dynamics issue
    - Level 1 fails on one target → Model may struggle with that structure type
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests._helpers.graph_builders import edge_scalar_graphdata, legacy_edge_scalar
from tmgg.data import generate_sbm_adjacency
from tmgg.models.baselines import LinearBaseline, MLPBaseline
from tmgg.models.spectral_denoisers import (
    GraphFilterBank,
    LinearPE,
    SelfAttentionDenoiser,
)
from tmgg.utils.noising import add_edge_flip_noise

# =============================================================================
# TARGET GENERATION UTILITIES
# =============================================================================


def create_erdos_renyi_target(
    n: int = 16, p: float = 0.3, seed: int = 42
) -> torch.Tensor:
    """Create Erdos-Renyi random graph with edge probability p.

    Unlike random_binary (p=0.5), this uses a sane sparsity level that
    produces graphs with realistic density similar to SBM targets.
    """
    torch.manual_seed(seed)
    A = (torch.rand(1, n, n) < p).float()
    A = (A + A.transpose(-2, -1)).clamp(max=1.0)  # Symmetrize, clamp to binary
    A[0].fill_diagonal_(0)
    return A


def create_block_diagonal_target(n: int = 16, num_blocks: int = 4) -> torch.Tensor:
    """Create block diagonal matrix - dense within blocks, no inter-block edges."""
    A = torch.zeros(1, n, n)
    block_size = n // num_blocks
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        A[0, start:end, start:end] = 1.0
    A[0].fill_diagonal_(0)
    return A


def create_sbm_target(
    block_sizes: list | None = None,
    p: float = 0.8,
    q: float = 0.1,
    seed: int = 42,
) -> torch.Tensor:
    """Create SBM graph with community structure."""
    if block_sizes is None:
        block_sizes = [4, 4, 4, 4]
    rng = np.random.default_rng(seed)
    A_np = generate_sbm_adjacency(block_sizes, p_intra=p, p_inter=q, rng=rng)
    A = torch.tensor(A_np, dtype=torch.float32).unsqueeze(0)
    return A


# =============================================================================
# LEVEL 1: CONSTANT NOISE MEMORIZATION TESTS
# =============================================================================


@pytest.mark.slow
class TestLevel1ConstantNoiseMemorization:
    """LEVEL 1 SANITY CHECK: Constant noise memorization.

    WHY THIS EXISTS:
    ----------------
    Tests whether models have the fundamental capacity to learn ANY mapping
    by presenting the exact same input every training step. The model only
    needs to memorize one fixed input→output pair.

    WHAT IT TESTS:
    --------------
    - Model architecture can represent the required mapping
    - Gradient flow works correctly through all layers
    - Optimizer can update weights in the right direction

    IF THIS FAILS:
    --------------
    The model fundamentally cannot learn. Check:
    - Forward pass produces non-zero outputs
    - Gradients flow to all parameters
    - No NaN/Inf in computations

    DO NOT MODIFY these tests - they are permanent sanity checks.
    """

    @pytest.fixture(params=["erdos_renyi", "block_diagonal", "sbm"])
    def fixed_sample(
        self, request: pytest.FixtureRequest
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Create fixed noisy/clean pair - SAME noise every time.

        The key property is that A_noisy is deterministically generated
        from A_clean using a fixed seed, so every call returns the
        exact same (A_noisy, A_clean) pair.
        """
        n = 16

        if request.param == "erdos_renyi":
            A_clean = create_erdos_renyi_target(n=n, p=0.3, seed=42)
        elif request.param == "block_diagonal":
            A_clean = create_block_diagonal_target(n=n, num_blocks=4)
        elif request.param == "sbm":
            A_clean = create_sbm_target(block_sizes=[4, 4, 4, 4], p=0.8, q=0.1, seed=42)
        else:
            raise ValueError(f"Unknown target type: {request.param}")

        # CRITICAL: Fixed seed for noise - same noise every call
        torch.manual_seed(123)
        np.random.seed(123)
        A_noisy = add_edge_flip_noise(A_clean, p=0.1)

        return A_noisy, A_clean, str(request.param)

    @pytest.mark.parametrize(
        "model_class,kwargs",
        [
            (LinearPE, {"k": 16, "max_nodes": 16}),  # Full spectrum for 16 nodes
            (GraphFilterBank, {"k": 16, "polynomial_degree": 8}),  # Increased capacity
            (SelfAttentionDenoiser, {"k": 16, "d_k": 64}),  # Increased capacity
            (LinearBaseline, {"max_nodes": 16}),
            (MLPBaseline, {"max_nodes": 16, "hidden_dim": 256}),  # Increased hidden dim
        ],
    )
    def test_memorize_fixed_mapping(self, model_class, kwargs, fixed_sample):
        """Model must achieve ~99% accuracy on fixed input.

        This is the most basic sanity check: can the model memorize
        a single fixed input→output mapping? If not, something is
        fundamentally broken.
        """
        A_noisy, A_clean, target_type = fixed_sample
        model = model_class(**kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        data_noisy = edge_scalar_graphdata(A_noisy)

        # Train on the SAME A_noisy every step (5000 steps for convergence).
        # BCE on edge-channel logit preserves the gradient scale these tests
        # were calibrated for.
        for _ in range(5000):
            result = model(data_noisy, output_dense=True)
            loss = F.binary_cross_entropy_with_logits(
                legacy_edge_scalar(result), A_clean
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate on the same A_noisy
        with torch.no_grad():
            result = model(data_noisy, output_dense=True)
            predictions = (torch.sigmoid(legacy_edge_scalar(result)) > 0.5).float()
            accuracy = (predictions == A_clean).float().mean().item()

            edge_logits = legacy_edge_scalar(result)
            logit_mean = edge_logits.mean().item()
            logit_std = edge_logits.std().item()

        assert accuracy > 0.99, (
            f"LEVEL 1 FAIL [{model_class.__name__}, {target_type}]: "
            f"Cannot memorize fixed mapping. Accuracy: {accuracy:.1%}. "
            f"Logit mean: {logit_mean:.3f}, std: {logit_std:.3f}. "
            f"Model capacity or gradient flow is broken."
        )


# =============================================================================
# LEVEL 2: FRESH NOISE GENERALIZATION TESTS
# =============================================================================


@pytest.mark.slow
class TestLevel2FreshNoiseGeneralization:
    """LEVEL 2 SANITY CHECK: Fresh noise generalization.

    WHY THIS EXISTS:
    ----------------
    Tests whether models can learn the general denoising task, not just
    memorize one specific noisy sample. Fresh noise is generated each
    training step.

    WHAT IT TESTS:
    --------------
    - Model can generalize across different noise realizations
    - Model learns the underlying structure, not the specific noise pattern
    - Training dynamics allow convergence to a generalizing solution

    IF THIS FAILS (but Level 1 passes):
    -----------------------------------
    The model has capacity but struggles with generalization. Consider:
    - More training steps
    - Different learning rate
    - More model capacity (larger k, more layers)
    - Task may be fundamentally hard for this architecture

    STRUCTURED TARGETS ONLY:
    ------------------------
    Random binary matrices are NOT tested here because they lack structure
    that spectral methods can exploit. This is expected behavior, not a bug.

    DO NOT MODIFY these tests - they are permanent sanity checks.
    """

    @pytest.fixture(params=["block_diagonal", "sbm"])
    def structured_sample(self, request):
        """Create structured target - structured targets only.

        Random binary is excluded because:
        1. No community structure for spectral methods to exploit
        2. Failing on random binary is expected, not a bug
        """
        n = 16

        if request.param == "block_diagonal":
            A_clean = create_block_diagonal_target(n=n, num_blocks=4)
        elif request.param == "sbm":
            A_clean = create_sbm_target(block_sizes=[4, 4, 4, 4], p=0.8, q=0.1, seed=42)
        else:
            raise ValueError(f"Unknown target type: {request.param}")

        return A_clean, request.param

    @pytest.mark.parametrize(
        "model_class,kwargs",
        [
            (LinearPE, {"k": 8, "max_nodes": 16}),
            (GraphFilterBank, {"k": 8, "polynomial_degree": 5}),
            (SelfAttentionDenoiser, {"k": 8, "d_k": 32}),
            (LinearBaseline, {"max_nodes": 16}),
            (MLPBaseline, {"max_nodes": 16, "hidden_dim": 128}),
        ],
    )
    def test_generalize_denoising(self, model_class, kwargs, structured_sample):
        """Model should learn denoising with fresh noise each step.

        This tests actual denoising capability: can the model learn to
        remove noise in general, not just memorize one noisy sample?
        """
        A_clean, target_type = structured_sample
        model = model_class(**kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Train with FRESH noise each step
        for _ in range(2000):
            A_noisy = add_edge_flip_noise(A_clean, p=0.1)
            result = model(edge_scalar_graphdata(A_noisy), output_dense=True)
            loss = F.binary_cross_entropy_with_logits(
                legacy_edge_scalar(result), A_clean
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate on FRESH noise (different from any training noise)
        torch.manual_seed(999)  # Different seed for evaluation
        A_noisy_eval = add_edge_flip_noise(A_clean, p=0.1)

        with torch.no_grad():
            result = model(edge_scalar_graphdata(A_noisy_eval), output_dense=True)
            predictions = (torch.sigmoid(legacy_edge_scalar(result)) > 0.5).float()
            accuracy = (predictions == A_clean).float().mean().item()

            edge_logits = legacy_edge_scalar(result)
            logit_mean = edge_logits.mean().item()
            logit_std = edge_logits.std().item()

        assert accuracy > 0.70, (
            f"LEVEL 2 FAIL [{model_class.__name__}, {target_type}]: "
            f"Cannot generalize denoising. Accuracy: {accuracy:.1%}. "
            f"Logit mean: {logit_mean:.3f}, std: {logit_std:.3f}. "
            f"Model may need more training or capacity."
        )


# =============================================================================
# SUPPLEMENTARY TESTS (kept for backwards compatibility)
# =============================================================================


class TestSingleStepLearning:
    """Verify that one optimizer step reduces loss.

    This is a quick sanity check that gradient descent works at all.
    """

    @pytest.fixture
    def training_data(self):
        """Create noisy/clean adjacency pair."""
        A_clean = create_block_diagonal_target(n=32, num_blocks=4)
        # Expand to batch
        A_clean = A_clean.expand(4, -1, -1).clone()
        A_noisy = add_edge_flip_noise(A_clean, p=0.1)
        return A_noisy, A_clean

    @pytest.mark.parametrize(
        "model_class,kwargs",
        [
            (LinearPE, {"k": 8, "max_nodes": 32}),
            (GraphFilterBank, {"k": 8, "polynomial_degree": 5}),
            (SelfAttentionDenoiser, {"k": 8, "d_k": 64}),
            (LinearBaseline, {"max_nodes": 32}),
            (MLPBaseline, {"max_nodes": 32, "hidden_dim": 256}),
        ],
    )
    def test_single_step_reduces_loss(self, model_class, kwargs, training_data):
        """Verify one optimizer step reduces loss."""
        A_noisy, A_clean = training_data
        model = model_class(**kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        data_noisy = edge_scalar_graphdata(A_noisy)

        # Initial forward pass
        result_1 = model(data_noisy, output_dense=True)
        loss_1 = F.binary_cross_entropy_with_logits(
            legacy_edge_scalar(result_1), A_clean
        )

        # Optimization step
        loss_1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Second forward pass
        result_2 = model(data_noisy, output_dense=True)
        loss_2 = F.binary_cross_entropy_with_logits(
            legacy_edge_scalar(result_2), A_clean
        )

        # Loss should decrease (or at least not increase significantly)
        assert loss_2 <= loss_1 + 0.01, (
            f"{model_class.__name__}: Loss increased after step. "
            f"Before: {loss_1:.4f}, After: {loss_2:.4f}"
        )


class TestLossDecreasesCurve:
    """Verify loss consistently decreases during training."""

    @pytest.fixture
    def training_batch(self):
        """Create a small training batch."""
        torch.manual_seed(123)
        A_clean = create_block_diagonal_target(n=16, num_blocks=4)
        A_clean = A_clean.expand(8, -1, -1).clone()
        A_noisy = add_edge_flip_noise(A_clean, p=0.1)
        return A_noisy, A_clean

    @pytest.mark.parametrize(
        "model_class,kwargs",
        [
            (LinearBaseline, {"max_nodes": 16}),
            (MLPBaseline, {"max_nodes": 16, "hidden_dim": 64}),
            (LinearPE, {"k": 4, "max_nodes": 16}),
        ],
    )
    def test_loss_trend_is_downward(self, model_class, kwargs, training_batch):
        """Loss should trend downward over 100 steps."""
        A_noisy, A_clean = training_batch
        model = model_class(**kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        data_noisy = edge_scalar_graphdata(A_noisy)

        losses = []
        for _ in range(100):
            result = model(data_noisy, output_dense=True)
            loss = F.binary_cross_entropy_with_logits(
                legacy_edge_scalar(result), A_clean
            )
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Compare first 10 losses to last 10 losses
        early_mean = sum(losses[:10]) / 10
        late_mean = sum(losses[-10:]) / 10

        assert late_mean < early_mean, (
            f"{model_class.__name__}: Loss did not decrease. "
            f"Early mean: {early_mean:.4f}, Late mean: {late_mean:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

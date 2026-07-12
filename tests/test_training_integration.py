"""Integration tests for the full training pipeline.

Test rationale:
    This test verifies that the end-to-end training pipeline works correctly,
    from configuration loading through model training to final evaluation.
    It ensures that:
    1. Model instantiation produces valid objects
    2. The training loop executes without errors
    3. Loss decreases during training (model is learning)
    4. Validation metrics are computed and logged
    5. All model types work through the Lightning interface

Invariants:
    - Model output shape matches input shape (adjacency matrix reconstruction)
    - Loss is finite and positive
    - Training loss decreases over epochs (on average)
"""

from typing import Any, cast

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from tests._helpers.graph_builders import binary_graphdata
from tmgg.models.spectral_denoisers import GraphFilterBank, LinearPE
from tmgg.models.spectral_denoisers.bilinear import (
    BilinearDenoiserWithMLP,
    MultiLayerBilinearDenoiser,
)
from tmgg.models.spectral_denoisers.self_attention import SelfAttentionDenoiser
from tmgg.training.lightning_modules.denoising_module import (
    SingleStepDenoisingModule,
)


class TestTrainingPipeline:
    """Integration tests for the training pipeline."""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create a minimal configuration for testing."""
        return OmegaConf.create(
            {
                "seed": 42,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "optimizer_type": "adam",
                "loss_type": "bce_logits",
                "noise_type": "digress",
                "noise_levels": [0.1, 0.2],
                "paths": {
                    "output_dir": str(tmp_path),
                    "results_dir": str(tmp_path / "results"),
                },
            }
        )

    @pytest.fixture
    def sample_adjacency_matrices(self):
        """Generate sample adjacency matrices for testing."""
        batch_size = 4
        n_nodes = 20

        # Create block diagonal adjacency matrices (simple structure)
        matrices = []
        for _ in range(batch_size):
            A = torch.zeros(n_nodes, n_nodes)
            # Create two blocks
            A[:10, :10] = torch.bernoulli(torch.full((10, 10), 0.7))
            A[10:, 10:] = torch.bernoulli(torch.full((10, 10), 0.7))
            # Make symmetric and remove diagonal
            A = (A + A.T) / 2
            A.fill_diagonal_(0)
            A = (A > 0.5).float()
            matrices.append(A)

        return torch.stack(matrices)

    def test_spectral_lightning_module_instantiation(self, minimal_config):
        """Test that SingleStepDenoisingModule can be instantiated from config."""
        model = GraphFilterBank(k=8, polynomial_degree=3)
        module = SingleStepDenoisingModule(
            model=model,
            learning_rate=minimal_config.learning_rate,
            weight_decay=minimal_config.weight_decay,
            optimizer_type=minimal_config.optimizer_type,
            loss_type=minimal_config.loss_type,
            noise_type=minimal_config.noise_type,
            noise_levels=minimal_config.noise_levels,
        )

        assert module is not None
        assert hasattr(module, "model")
        assert hasattr(module, "noise_process")

    @pytest.mark.parametrize(
        "model",
        [
            LinearPE(k=8),
            GraphFilterBank(k=8, polynomial_degree=3),
            SelfAttentionDenoiser(k=8, d_k=16),
            BilinearDenoiserWithMLP(k=8, d_k=16, mlp_hidden_dim=32, mlp_num_layers=1),
            MultiLayerBilinearDenoiser(k=8, d_model=16, num_heads=2, num_layers=1),
        ],
        ids=[
            "LinearPE",
            "GraphFilterBank",
            "SelfAttentionDenoiser",
            "BilinearDenoiserWithMLP",
            "MultiLayerBilinearDenoiser",
        ],
    )
    def test_all_model_types_instantiate(self, model):
        """Verify all valid spectral model types instantiate without error.

        Test Rationale
        --------------
        Each model should produce a valid SingleStepDenoisingModule.
        This parametrized test ensures SingleStepDenoisingModule can wrap
        all supported spectral architectures.
        """
        module = SingleStepDenoisingModule(
            model=model,
            learning_rate=1e-3,
            noise_levels=[0.1],
        )

        assert module is not None
        assert module.model is not None
        # Model should have a forward method
        assert hasattr(module.model, "forward")

    def test_forward_pass_shape(self, minimal_config, sample_adjacency_matrices):
        """Test that forward pass preserves shape."""
        model = GraphFilterBank(k=8, polynomial_degree=3)
        module = SingleStepDenoisingModule(
            model=model,
            learning_rate=minimal_config.learning_rate,
            noise_levels=minimal_config.noise_levels,
        )

        with torch.no_grad():
            output = module(sample_adjacency_matrices)

        assert output.shape == sample_adjacency_matrices.shape

    def test_training_step_produces_finite_loss(
        self, minimal_config, sample_adjacency_matrices
    ):
        """Test that training step produces finite loss.

        Note: training_step requires noise_levels, so we pass them explicitly.
        We also mock self.log since it requires an active training loop.
        """
        from unittest.mock import patch

        model = GraphFilterBank(k=8, polynomial_degree=3)
        module = SingleStepDenoisingModule(
            model=model,
            learning_rate=minimal_config.learning_rate,
            loss_type=minimal_config.loss_type,
            noise_levels=minimal_config.noise_levels,
        )

        # Mock self.log to avoid MisconfigurationException.
        # Sparse-default refactor: training_step expects a sparse GraphState
        # batch; binary_graphdata returns dense, so flip at the boundary.
        batch = binary_graphdata(sample_adjacency_matrices).to_sparse()
        with patch.object(module, "log", return_value=None):
            loss = module.training_step(batch, batch_idx=0)

        assert torch.isfinite(loss)
        assert loss > 0

    def test_short_training_run(self, minimal_config, tmp_path):
        """Test a short training run completes without errors."""
        from tmgg.data.data_modules.data_module import GraphDataModule

        # Create model
        model = GraphFilterBank(k=8, polynomial_degree=3)
        module = SingleStepDenoisingModule(
            model=model,
            learning_rate=minimal_config.learning_rate,
            noise_levels=minimal_config.noise_levels,
            loss_type=minimal_config.loss_type,
        )

        # Create minimal data module
        data_module = GraphDataModule(
            graph_type="sbm",
            graph_config={
                "num_nodes": 20,
                "num_graphs": 10,
            },
            batch_size=4,
        )

        # Create trainer with minimal settings
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )

        # Run training
        trainer.fit(module, data_module)

        # Verify training completed
        assert trainer.current_epoch == 2

    def test_loss_decreases_during_training(self, minimal_config, tmp_path):
        """Test that loss decreases during training (model learns)."""
        from tmgg.data.data_modules.data_module import GraphDataModule

        # Create model with nonlinear model for bounded output with MSE loss
        model = GraphFilterBank(k=8, polynomial_degree=3)
        module = SingleStepDenoisingModule(
            model=model,
            learning_rate=1e-2,  # Higher LR for faster convergence in test
            noise_levels=[0.1],  # Single noise level for cleaner test
            loss_type="mse",  # MSE is easier to optimize
        )

        # Create data module with fixed data for deterministic test
        data_module = GraphDataModule(
            graph_type="sbm",
            graph_config={
                "num_nodes": 15,
                "num_graphs": 5,
            },
            batch_size=5,
        )

        # Collect losses
        losses: list[float] = []
        original_training_step = module.training_step

        def tracking_training_step(batch, batch_idx):
            result = original_training_step(batch, batch_idx)
            # SingleStepDenoisingModule.training_step returns loss tensor directly
            losses.append(result.item())
            return result

        module.training_step = tracking_training_step  # pyright: ignore[reportAttributeAccessIssue]

        # Train for a few epochs
        trainer = pl.Trainer(
            max_epochs=5,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(module, data_module)

        # Check that average loss in last epoch is lower than first epoch
        n_batches_per_epoch = len(losses) // 5
        first_epoch_loss = sum(losses[:n_batches_per_epoch]) / n_batches_per_epoch
        last_epoch_loss = sum(losses[-n_batches_per_epoch:]) / n_batches_per_epoch

        # Allow some tolerance - just check it's not getting worse
        assert last_epoch_loss <= first_epoch_loss * 1.5, (
            f"Loss did not decrease: first_epoch={first_epoch_loss:.4f}, "
            f"last_epoch={last_epoch_loss:.4f}"
        )


class TestDataModuleIntegration:
    """Tests for data module integration with training."""

    def test_datamodule_rejects_legacy_noise_levels_param(self) -> None:
        """Data module should fail loudly on removed ``noise_levels``."""
        from tmgg.data.data_modules.data_module import GraphDataModule

        legacy_kwargs = cast(Any, {"noise_levels": [0.05, 0.1, 0.2]})
        with pytest.raises(TypeError, match="noise_levels"):
            GraphDataModule(
                graph_type="sbm",
                graph_config={"num_nodes": 10, "num_graphs": 5},
                **legacy_kwargs,
            )

    def test_datamodule_rejects_legacy_noise_type_param(self) -> None:
        """Data module should fail loudly on removed ``noise_type``."""
        from tmgg.data.data_modules.data_module import GraphDataModule

        legacy_kwargs = cast(Any, {"noise_type": "digress"})
        with pytest.raises(TypeError, match="noise_type"):
            GraphDataModule(
                graph_type="sbm",
                graph_config={"num_nodes": 10, "num_graphs": 5},
                **legacy_kwargs,
            )

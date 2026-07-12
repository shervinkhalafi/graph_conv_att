# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""Single-step and full-flow tests for baseline denoising via SingleStepDenoisingModule.

Test rationale
--------------
Prior to these tests, ``lin_mlp_baseline_denoising`` had zero training
coverage -- only an import smoke test (``tests/test_baseline_runner.py``).
These tests verify that both baseline architectures ("linear" and "mlp")
produce valid loss through the full SingleStepDenoisingModule path and survive a
complete train-validate-test cycle via the Trainer.

Invariants
~~~~~~~~~~
- ``training_step`` produces a finite, positive scalar loss for both
  model types.
- Gradients propagate to model parameters after backward.
- A 1-epoch ``Trainer.fit`` followed by ``Trainer.test`` completes
  without error on a tiny SBM dataset.
"""

from unittest.mock import patch

import pytest
import pytorch_lightning as pl
import torch

from tests._helpers.graph_builders import binary_graphdata
from tmgg.data.data_modules.data_module import GraphDataModule
from tmgg.data.datasets.graph_types import GraphData
from tmgg.training.lightning_modules.denoising_module import (
    SingleStepDenoisingModule,
)

N_NODES = 16


@pytest.fixture
def sample_adjacency() -> torch.Tensor:
    """4 binary symmetric adjacency matrices with block structure, 16 nodes."""
    batch = torch.zeros(4, N_NODES, N_NODES)
    half = N_NODES // 2
    batch[:, :half, :half] = torch.bernoulli(torch.full((4, half, half), 0.7))
    batch[:, half:, half:] = torch.bernoulli(torch.full((4, half, half), 0.5))
    batch = (batch + batch.transpose(-2, -1)) / 2
    batch = (batch > 0.5).float()
    batch.diagonal(dim1=-2, dim2=-1).zero_()
    return batch


@pytest.fixture
def sample_batch(sample_adjacency: torch.Tensor) -> GraphData:
    """Sparse ``GraphState`` batch wrapping the sample adjacency matrices.

    Production datamodules emit sparse batches per the 2026-05-07
    sparse-default refactor; ``SingleStepDenoisingModule.training_step``
    consumes the same carrier shape.
    """
    return binary_graphdata(sample_adjacency).to_sparse()


@pytest.fixture
def data_module() -> GraphDataModule:
    return GraphDataModule(
        graph_type="sbm",
        graph_config={"num_nodes": N_NODES, "num_graphs": 8},
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )


def _make_module(model_type: str) -> SingleStepDenoisingModule:
    """Create a small baseline module for CPU tests."""
    from tmgg.models.baselines import LinearBaseline, MLPBaseline

    if model_type == "linear":
        model = LinearBaseline(max_nodes=N_NODES)
    else:
        model = MLPBaseline(max_nodes=N_NODES, hidden_dim=32, num_layers=1)
    return SingleStepDenoisingModule(
        model=model,
        learning_rate=1e-3,
        noise_levels=[0.1, 0.2],
    )


class TestBaselineSingleStep:
    """Lightweight checks that training_step runs and produces valid output."""

    @pytest.mark.parametrize("model_type", ["linear", "mlp"])
    def test_training_step_finite_loss(
        self,
        model_type: str,
        sample_batch: GraphData,
    ) -> None:
        """training_step should return a finite positive loss."""
        module = _make_module(model_type)

        with patch.object(module, "log", return_value=None):
            loss = module.training_step(sample_batch, batch_idx=0)

        assert torch.isfinite(loss)
        assert loss > 0

    @pytest.mark.parametrize("model_type", ["linear", "mlp"])
    def test_gradients_flow(
        self,
        model_type: str,
        sample_batch: GraphData,
    ) -> None:
        """Gradients should propagate to model parameters after backward."""
        module = _make_module(model_type)

        with patch.object(module, "log", return_value=None):
            loss = module.training_step(sample_batch, batch_idx=0)

        loss.backward()
        params_with_grad = [p for p in module.model.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0, "No gradients flowed to model parameters"


class TestBaselineFullFlow:
    """Full train -> validation -> test flow through the Lightning Trainer."""

    @pytest.mark.parametrize("model_type", ["linear", "mlp"])
    def test_train_val_test_completes(
        self,
        model_type: str,
        data_module: GraphDataModule,
    ) -> None:
        """One epoch of fit + test should complete without error.

        Starting state: fresh module and datamodule (SBM, 16 nodes, 8 graphs).
        Invariant: Trainer completes fit and test without exceptions.
        """
        module = _make_module(model_type)

        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            limit_test_batches=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, data_module)
        trainer.test(module, data_module)

        assert trainer.current_epoch == 1

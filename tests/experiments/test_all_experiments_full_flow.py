"""Tiny-model full-flow tests: train -> validation -> test for every experiment type.

Test rationale
--------------
Prior to these tests, no test in the suite called ``trainer.test()``, leaving
``test_step`` and ``on_test_epoch_end`` completely unverified. Several
experiment types also lacked any ``trainer.fit()`` coverage through their
LightningModule (only raw model tests existed). This file ensures every
experiment type survives a complete 1-batch train/val/test cycle with a
deliberately tiny model and dataset, catching wiring bugs (wrong shapes,
missing attributes, broken callbacks) without the cost of real training.

Invariants
~~~~~~~~~~
- ``trainer.fit()`` and ``trainer.test()`` complete without exceptions.
- Each test finishes within ~15 seconds on CPU.

Each experiment type gets its own test class so failures are immediately
attributable to a specific experiment.
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl

from tmgg.data.data_modules.data_module import GraphDataModule
from tmgg.training.lightning_modules.denoising_module import (
    SingleStepDenoisingModule,
)

N_NODES = 16
N_GRAPHS = 8
BATCH_SIZE = 4
NOISE_LEVELS = [0.1, 0.2]


def _tiny_trainer() -> pl.Trainer:
    """Trainer configured for a single tiny batch through train/val/test."""
    return pl.Trainer(
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


@pytest.fixture
def denoising_data_module() -> GraphDataModule:
    """Tiny SBM data module shared by all denoising experiments."""
    return GraphDataModule(
        graph_type="sbm",
        graph_config={"num_nodes": N_NODES, "num_graphs": N_GRAPHS},
        batch_size=BATCH_SIZE,
        num_workers=0,
    )


# ── Denoising experiments ─────────────────────────────────────────────


class TestSpectralArchFullFlow:
    """spectral_arch_denoising: filter_bank through the full Lightning path."""

    def test_train_val_test(self, denoising_data_module: GraphDataModule) -> None:
        from tmgg.models.spectral_denoisers import GraphFilterBank

        model = GraphFilterBank(k=4, polynomial_degree=3)
        module = SingleStepDenoisingModule(
            model=model,
            learning_rate=1e-3,
            noise_levels=NOISE_LEVELS,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, denoising_data_module)
        trainer.test(module, denoising_data_module)
        assert trainer.current_epoch == 1


class TestGNNFullFlow:
    """gnn_denoising: GNN model through the full Lightning path."""

    def test_train_val_test(self, denoising_data_module: GraphDataModule) -> None:
        from tmgg.models.gnn import GNN

        model = GNN(
            num_layers=1,
            num_terms=2,
            feature_dim_in=8,
            feature_dim_out=4,
        )
        module = SingleStepDenoisingModule(
            model=model,
            learning_rate=1e-3,
            noise_levels=NOISE_LEVELS,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, denoising_data_module)
        trainer.test(module, denoising_data_module)
        assert trainer.current_epoch == 1


class TestGNNTransformerFullFlow:
    """gnn_transformer_denoising: hybrid GNN+Transformer through Lightning."""

    def test_train_val_test(self, denoising_data_module: GraphDataModule) -> None:
        from tmgg.models.hybrid import create_sequential_model

        model = create_sequential_model(
            gnn_config={
                "num_layers": 1,
                "num_terms": 2,
                "feature_dim_in": 8,
                "feature_dim_out": 4,
            },
            transformer_config={
                "num_heads": 2,
                "num_layers": 1,
            },
        )
        module = SingleStepDenoisingModule(
            model=model,
            learning_rate=1e-3,
            noise_levels=NOISE_LEVELS,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, denoising_data_module)
        trainer.test(module, denoising_data_module)
        assert trainer.current_epoch == 1


class TestDigressFullFlow:
    """digress_denoising: GraphTransformer through the full Lightning path."""

    def test_train_val_test(self, denoising_data_module: GraphDataModule) -> None:
        from tmgg.models.digress.transformer_model import GraphTransformer

        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": 2, "E": 1, "y": 0},
            hidden_mlp_dims={"X": 16, "E": 8, "y": 16},
            hidden_dims={"dx": 16, "de": 8, "dy": 16, "n_head": 2},
            output_dims={"X": 0, "E": 1, "y": 0},
        )
        module = SingleStepDenoisingModule(
            model=model,
            loss_type="mse",
            noise_type="digress",
            learning_rate=1e-3,
            noise_levels=NOISE_LEVELS,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, denoising_data_module)
        trainer.test(module, denoising_data_module)
        assert trainer.current_epoch == 1


class TestBaselineFullFlow:
    """lin_mlp_baseline_denoising: both linear and mlp through Lightning."""

    @pytest.mark.parametrize("model_type", ["linear", "mlp"])
    def test_train_val_test(
        self, model_type: str, denoising_data_module: GraphDataModule
    ) -> None:
        from tmgg.models.baselines import LinearBaseline, MLPBaseline

        if model_type == "linear":
            model = LinearBaseline(max_nodes=N_NODES)
        else:
            model = MLPBaseline(max_nodes=N_NODES, hidden_dim=32, num_layers=1)

        module = SingleStepDenoisingModule(
            model=model,
            learning_rate=1e-3,
            noise_levels=NOISE_LEVELS,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, denoising_data_module)
        trainer.test(module, denoising_data_module)
        assert trainer.current_epoch == 1


# ── Generative experiments ─────────────────────────────────────────────


class TestDiffusionModuleGaussianFullFlow:
    """DiffusionModule with continuous noise: full train/val/test cycle.

    Test rationale: verifies that DiffusionModule with ContinuousNoiseProcess
    and ContinuousSampler can complete a full Lightning lifecycle.
    """

    def test_train_val_test(self) -> None:
        from tmgg.data.data_modules.multigraph_data_module import MultiGraphDataModule
        from tmgg.diffusion.noise_process import ContinuousNoiseProcess
        from tmgg.diffusion.sampler import ContinuousSampler
        from tmgg.diffusion.schedule import NoiseSchedule
        from tmgg.evaluation.graph_evaluator import (
            GraphEvaluator,
        )
        from tmgg.models.spectral_denoisers.self_attention import SelfAttentionDenoiser
        from tmgg.training.lightning_modules.diffusion_module import (
            DiffusionModule,
        )
        from tmgg.utils.noising.noise import DigressNoise

        schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=5)
        noise_process = ContinuousNoiseProcess(
            definition=DigressNoise(), schedule=schedule
        )
        sampler = ContinuousSampler()
        evaluator = GraphEvaluator(eval_num_samples=4, kernel="gaussian", sigma=1.0)

        module = DiffusionModule(
            model=SelfAttentionDenoiser(k=4, d_k=16),
            noise_process=noise_process,
            sampler=sampler,
            noise_schedule=schedule,
            evaluator=evaluator,
            loss_type="mse",
            num_nodes=N_NODES,
            eval_every_n_steps=1,
            learning_rate=1e-3,
        )
        datamodule = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=N_NODES,
            num_graphs=N_GRAPHS,
            batch_size=BATCH_SIZE,
            graph_config={"num_blocks": 2, "p_in": 0.7, "p_out": 0.1},
            seed=42,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, datamodule)
        trainer.test(module, datamodule)
        assert trainer.current_epoch == 1


class TestDiscreteGenerativeFullFlow:
    """discrete_diffusion_generative: categorical diffusion through Lightning.

    Uses DiffusionModule + CategoricalNoiseProcess + CategoricalSampler,
    categorical diffusion through Lightning.
    """

    def test_train_val_test(self) -> None:
        from tmgg.data.data_modules.synthetic_categorical import (
            SyntheticCategoricalDataModule,
        )
        from tmgg.diffusion.noise_process import CategoricalNoiseProcess
        from tmgg.diffusion.sampler import CategoricalSampler
        from tmgg.diffusion.schedule import NoiseSchedule
        from tmgg.evaluation.graph_evaluator import (
            GraphEvaluator,
        )
        from tmgg.models.digress.transformer_model import GraphTransformer
        from tmgg.training.lightning_modules.diffusion_module import (
            DiffusionModule,
        )

        diffusion_steps = 5
        dx, de = 2, 2

        schedule = NoiseSchedule("cosine_iddpm", timesteps=diffusion_steps)
        noise_process = CategoricalNoiseProcess(
            schedule=schedule,
            x_classes=dx,
            e_classes=de,
            limit_distribution="uniform",
        )
        sampler = CategoricalSampler()
        evaluator = GraphEvaluator(
            eval_num_samples=4,
            kernel="gaussian",
            sigma=1.0,
        )

        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": dx, "E": de, "y": 0},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 16, "dy": 16, "n_head": 2},
            output_dims={"X": dx, "E": de, "y": 0},
            use_timestep=True,
        )
        module = DiffusionModule(
            model=model,
            noise_process=noise_process,
            sampler=sampler,
            noise_schedule=schedule,
            evaluator=evaluator,
            loss_type="cross_entropy",
            num_nodes=N_NODES,
            eval_every_n_steps=1,
        )
        datamodule = SyntheticCategoricalDataModule(
            num_nodes=N_NODES,
            num_graphs=N_GRAPHS,
            batch_size=BATCH_SIZE,
            seed=42,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, datamodule)
        trainer.test(module, datamodule)
        assert trainer.current_epoch == 1

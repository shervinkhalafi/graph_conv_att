"""Tests for discrete diffusion Hydra runner and config.

Testing strategy
----------------
Verifies that the YAML configuration parses correctly, the runner's
utility functions behave as expected, and that the full Hydra-driven
training loop can execute a minimal 2-step run. The fast tests
(config parsing, checkpoint detection) run in CI; the full end-to-end
training test is marked slow.

Key invariants tested:
- The YAML config parses without OmegaConf syntax errors
- All required _target_ classes resolve to importable types
- _is_training_complete returns False for missing/incomplete checkpoints
- A 2-step programmatic training run completes without errors
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tmgg.training.orchestration.run_experiment import (
    _is_training_complete,
)

EXP_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "tmgg" / "experiments" / "exp_configs"
)

MODEL_INSTANTIATION_CASES = [
    ("base_config_discrete_diffusion_generative", []),
    (
        "base_config_discrete_diffusion_generative",
        ["models/discrete@model=discrete_small"],
    ),
    (
        "base_config_discrete_diffusion_generative",
        ["models/discrete@model=discrete_sbm_eigenvec"],
    ),
    (
        "base_config_discrete_diffusion_generative",
        ["models/discrete@model=discrete_sbm_official"],
    ),
    ("base_config_gaussian_diffusion", []),
]


def _base_overrides(tmp_path: Path) -> list[str]:
    """Return Hydra overrides that remove runtime-only dependencies."""
    return [
        f"paths.output_dir={tmp_path}",
        f"paths.results_dir={tmp_path}/results",
        f"hydra.run.dir={tmp_path}",
        "~logger",
    ]


def _compose_hydra_config(
    tmp_path: Path,
    config_name: str,
    overrides: list[str] | None = None,
):
    """Compose an experiment config with minimal test-time overrides."""
    with initialize_config_dir(version_base=None, config_dir=str(EXP_CONFIG_DIR)):
        return compose(
            config_name=config_name,
            overrides=_base_overrides(tmp_path) + (overrides or []),
        )


def _compose_config(tmp_path: Path) -> dict[str, Any]:
    """Compose the discrete diffusion config via Hydra and return as dict.

    Uses Hydra's ``compose`` API so that defaults (model configs, base
    infra) are resolved, matching what the runner sees at launch time.
    Overrides disable loggers and S3-dependent interpolations that
    require runtime state.
    """
    cfg = _compose_hydra_config(
        tmp_path=tmp_path,
        config_name="base_config_discrete_diffusion_generative",
    )
    result = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(result, dict)
    return result  # pyright: ignore[reportReturnType]


# ------------------------------------------------------------------
# Config loading tests
# ------------------------------------------------------------------


class TestConfigLoads:
    """Verify the Hydra-composed config contains expected structure.

    After the migration to DiffusionModule, the model section contains
    model (Hydra _target_ instantiation), noise_process, sampler,
    noise_schedule, and evaluator sub-configs.
    """

    @pytest.fixture(autouse=True)
    def _clear_hydra(self) -> Generator[None, None, None]:
        """Clear Hydra global state before each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.fixture()
    def cfg(self, tmp_path: Path) -> dict[str, Any]:
        """Hydra-composed discrete diffusion config as a plain dict."""
        return _compose_config(tmp_path)

    def test_yaml_parses(self) -> None:
        """Raw YAML should load as valid OmegaConf without syntax errors."""
        raw = OmegaConf.load(
            EXP_CONFIG_DIR / "base_config_discrete_diffusion_generative.yaml"
        )
        assert OmegaConf.is_config(raw)

    def test_has_model_section(self, cfg: dict[str, Any]) -> None:
        """Model section must contain _target_ pointing to DiffusionModule."""
        assert "model" in cfg
        model_cfg = cfg["model"]
        assert "_target_" in model_cfg
        assert model_cfg["_target_"].endswith("DiffusionModule")

    def test_has_model_and_model_name(self, cfg: dict[str, Any]) -> None:
        """Model section must contain a nested model _target_ for the GraphTransformer."""
        model_cfg = cfg["model"]
        assert "model" in model_cfg
        assert model_cfg["model"]["_target_"].endswith("GraphTransformer")
        assert "model_name" in model_cfg

    def test_has_noise_process(self, cfg: dict[str, Any]) -> None:
        """Model section must contain noise_process targeting CategoricalNoiseProcess."""
        model_cfg = cfg["model"]
        assert "noise_process" in model_cfg
        assert model_cfg["noise_process"]["_target_"].endswith(
            "CategoricalNoiseProcess"
        )
        assert model_cfg["noise_process"]["limit_distribution"] == "empirical_marginal"
        assert "schedule" in model_cfg["noise_process"]
        assert "noise_schedule" not in model_cfg["noise_process"]

    def test_has_sampler(self, cfg: dict[str, Any]) -> None:
        """Sampler config should now be a pure target-only wrapper."""
        model_cfg = cfg["model"]
        assert "sampler" in model_cfg
        assert model_cfg["sampler"]["_target_"].endswith("CategoricalSampler")
        assert set(model_cfg["sampler"].keys()) == {"_target_"}

    def test_has_noise_schedule(self, cfg: dict[str, Any]) -> None:
        """Model section must contain noise_schedule targeting NoiseSchedule."""
        model_cfg = cfg["model"]
        assert "noise_schedule" in model_cfg
        assert model_cfg["noise_schedule"]["_target_"].endswith("NoiseSchedule")

    def test_has_evaluator(self, cfg: dict[str, Any]) -> None:
        """Model section must contain evaluator targeting GraphEvaluator."""
        model_cfg = cfg["model"]
        assert "evaluator" in model_cfg
        assert model_cfg["evaluator"]["_target_"].endswith("GraphEvaluator")

    def test_has_data_section(self, cfg: dict[str, Any]) -> None:
        """Data section must target SyntheticCategoricalDataModule."""
        assert "data" in cfg
        assert cfg["data"]["_target_"].endswith("SyntheticCategoricalDataModule")

    def test_callbacks_monitor_nll(self, cfg: dict[str, Any]) -> None:
        """Callbacks must monitor val/epoch_NLL, not val/loss."""
        callbacks = cfg["callbacks"]
        assert callbacks["early_stopping"]["monitor"] == "val/epoch_NLL"
        assert callbacks["checkpoint"]["monitor"] == "val/epoch_NLL"

    def test_target_classes_importable(self, cfg: dict[str, Any]) -> None:
        """Key _target_ strings must resolve to importable Python classes."""
        targets = [
            cfg["model"]["_target_"],
            cfg["model"]["noise_process"]["_target_"],
            cfg["model"]["noise_schedule"]["_target_"],
            cfg["model"]["evaluator"]["_target_"],
            cfg["data"]["_target_"],
        ]

        for target in targets:
            module_path, class_name = target.rsplit(".", 1)
            mod = __import__(module_path, fromlist=[class_name])
            assert hasattr(mod, class_name), f"{target} not importable"


# ------------------------------------------------------------------
# Utility function tests
# ------------------------------------------------------------------


class TestIsTrainingComplete:
    """Checkpoint completion detection logic."""

    def test_missing_checkpoint(self, tmp_path: Path) -> None:
        """Returns False for nonexistent checkpoint."""
        assert _is_training_complete(tmp_path / "missing.ckpt", 100) is False

    def test_incomplete_checkpoint(self, tmp_path: Path) -> None:
        """Returns False when global_step < max_steps."""
        ckpt_path = tmp_path / "last.ckpt"
        torch.save({"global_step": 50}, ckpt_path)
        assert _is_training_complete(ckpt_path, 100) is False

    def test_complete_checkpoint(self, tmp_path: Path) -> None:
        """Returns True when global_step >= max_steps."""
        ckpt_path = tmp_path / "last.ckpt"
        torch.save({"global_step": 100}, ckpt_path)
        assert _is_training_complete(ckpt_path, 100) is True


class TestConfigInstantiationChecklist:
    """Every supported diffusion config should compose and instantiate."""

    @pytest.fixture(autouse=True)
    def _clear_hydra(self) -> Generator[None, None, None]:
        """Clear Hydra global state before each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    @pytest.mark.parametrize(
        ("config_name", "overrides"),
        MODEL_INSTANTIATION_CASES,
    )
    def test_model_instantiates_for_supported_diffusion_configs(
        self,
        tmp_path: Path,
        config_name: str,
        overrides: list[str],
    ) -> None:
        """Checklist configs must compose and instantiate without overrides."""
        cfg = _compose_hydra_config(
            tmp_path=tmp_path,
            config_name=config_name,
            overrides=overrides,
        )

        instantiate(cfg.model)

    def test_over_complete_checkpoint(self, tmp_path: Path) -> None:
        """Returns True when global_step exceeds max_steps."""
        ckpt_path = tmp_path / "last.ckpt"
        torch.save({"global_step": 200}, ckpt_path)
        assert _is_training_complete(ckpt_path, 100) is True


# ------------------------------------------------------------------
# End-to-end training (Hydra-free, programmatic)
# ------------------------------------------------------------------


class TestEndToEndTraining:
    """Run a complete 2-step training via the programmatic API.

    This bypasses Hydra to avoid subprocess overhead while still
    verifying that all components wire together: datamodule -> model ->
    trainer -> validation with VLB.
    """

    def test_two_step_training(self) -> None:
        """A 2-step training run should complete and log val/epoch_NLL."""
        import pytorch_lightning as pl

        from tmgg.data.data_modules.synthetic_categorical import (
            SyntheticCategoricalDataModule,
        )
        from tmgg.diffusion.noise_process import CategoricalNoiseProcess
        from tmgg.diffusion.sampler import CategoricalSampler
        from tmgg.diffusion.schedule import NoiseSchedule
        from tmgg.evaluation.graph_evaluator import (
            GraphEvaluator,
        )
        from tmgg.training.lightning_modules.diffusion_module import (
            DiffusionModule,
        )

        diffusion_steps = 10
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

        from tmgg.models.digress.transformer_model import GraphTransformer

        dm = SyntheticCategoricalDataModule(
            num_nodes=8, num_graphs=32, batch_size=4, seed=42
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
            num_nodes=8,
            eval_every_n_steps=1,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=2,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, dm)

        logged = trainer.logged_metrics
        assert "val/epoch_NLL" in logged, f"Missing val/epoch_NLL in {list(logged)}"

"""Tests for Modal runner.

Test rationale:
    The Modal runner orchestrates experiment execution on cloud GPUs
    via the CLI transport (modal_run_cli). These tests verify runner
    initialization, deployment checks, spawn mechanics, and result
    handling.

Invariants:
    - ModalRunner.spawn_experiment returns SpawnedTask with valid run_id
    - Deployment check fails fast when the Modal app is not deployed
    - spawn() passes (cmd, config_yaml, run_id) to the Modal function
"""

from unittest.mock import MagicMock, patch

import modal.exception
import pytest

from tmgg.modal.app import DEFAULT_TIMEOUTS, GPU_CONFIGS
from tmgg.modal.runner import (
    ExperimentResult,
    ModalNotDeployedError,
    ModalRunner,
    ModalSpawnedTask,
    check_modal_deployment,
    create_runner,
)


class TestDeploymentCheck:
    """Tests for check_modal_deployment() and ModalNotDeployedError.

    Test rationale:
        Pre-deployment verification prevents cryptic errors when users
        try to run experiments before deploying the Modal app. The check
        should fail fast with clear instructions.
    """

    def test_raises_when_function_not_found(self):
        """Should raise ModalNotDeployedError when function lookup fails."""
        mock_fn = MagicMock()
        mock_fn.hydrate.side_effect = modal.exception.NotFoundError(
            "Function not found"
        )

        with patch("modal.Function") as mock_Function:
            mock_Function.from_name.return_value = mock_fn

            with pytest.raises(ModalNotDeployedError) as exc_info:
                check_modal_deployment()

        assert "not deployed" in str(exc_info.value)
        assert "mise run modal-deploy" in str(exc_info.value)

    def test_succeeds_when_function_deployed(self):
        """Should not raise when function is accessible."""
        mock_fn = MagicMock()
        mock_fn.hydrate.return_value = None  # Success

        with patch("modal.Function") as mock_Function:
            mock_Function.from_name.return_value = mock_fn

            # Should not raise
            check_modal_deployment()

            # Verify it tried to look up and hydrate
            mock_Function.from_name.assert_called_with("graph-denoising", "modal_run_cli")
            mock_fn.hydrate.assert_called_once()

    def test_error_message_includes_instructions(self):
        """Error message should include deployment command."""
        mock_fn = MagicMock()
        mock_fn.hydrate.side_effect = modal.exception.NotFoundError("Not found")

        with patch("modal.Function") as mock_Function:
            mock_Function.from_name.return_value = mock_fn

            with pytest.raises(ModalNotDeployedError) as exc_info:
                check_modal_deployment()

        error_msg = str(exc_info.value)
        assert "uv run modal deploy" in error_msg
        assert "graph-denoising" in error_msg


class TestModalRunner:
    """Tests for ModalRunner experiment execution."""

    def test_runner_initialization(self):
        """ModalRunner should initialize with gpu_type."""
        runner = ModalRunner(gpu_type="fast", skip_deployment_check=True)

        assert runner.gpu_type == "fast"

    def test_runner_checks_deployment_by_default(self):
        """ModalRunner should call check_modal_deployment by default."""
        mock_fn = MagicMock()
        mock_fn.hydrate.side_effect = modal.exception.NotFoundError("Not deployed")

        with patch("modal.Function") as mock_Function:
            mock_Function.from_name.return_value = mock_fn

            with pytest.raises(ModalNotDeployedError):
                ModalRunner(gpu_type="standard")

    def test_runner_skips_deployment_check_when_requested(self):
        """ModalRunner should skip deployment check when skip_deployment_check=True."""
        runner = ModalRunner(gpu_type="standard", skip_deployment_check=True)

        assert runner.gpu_type == "standard"

    def test_create_runner_factory(self):
        """create_runner should create ModalRunner with default settings."""
        mock_fn = MagicMock()
        mock_fn.hydrate.return_value = None

        with patch("modal.Function") as mock_Function:
            mock_Function.from_name.return_value = mock_fn

            runner = create_runner(gpu_type="standard")

            assert runner.gpu_type == "standard"


class TestExperimentResult:
    """Tests for ExperimentResult dataclass structure."""

    def test_result_from_dict(self):
        """ExperimentResult should be constructable from result dict."""
        result_dict = {
            "run_id": "test-123",
            "config": {"lr": 1e-4},
            "metrics": {"best_val_loss": 0.5},
            "checkpoint_path": "s3://bucket/checkpoints/test-123/model.ckpt",
            "status": "completed",
            "duration_seconds": 3600.0,
        }

        result = ExperimentResult(**result_dict)

        assert result.run_id == "test-123"
        assert result.metrics["best_val_loss"] == 0.5
        assert result.status == "completed"

    def test_result_with_error(self):
        """Failed experiments should capture error message."""
        result = ExperimentResult(
            run_id="failed-run",
            config={},
            metrics={},
            status="failed",
            error_message="CUDA out of memory",
            duration_seconds=120.0,
        )

        assert result.status == "failed"
        assert result.error_message is not None and "CUDA" in result.error_message


class TestGPUConfigs:
    """Tests for GPU configuration constants."""

    def test_gpu_configs_defined(self):
        """GPU_CONFIGS should define expected tiers."""
        assert "standard" in GPU_CONFIGS
        assert "fast" in GPU_CONFIGS

    def test_default_timeouts_match_gpu_tiers(self):
        """DEFAULT_TIMEOUTS should have entries for GPU tiers."""
        # All GPU configs should have corresponding timeouts
        for tier in GPU_CONFIGS:
            assert tier in DEFAULT_TIMEOUTS, f"Missing timeout for {tier}"

    def test_timeouts_are_reasonable(self):
        """Timeouts should be in reasonable range (10 min to 24 hours)."""
        for tier, timeout in DEFAULT_TIMEOUTS.items():
            assert timeout >= 600, f"Timeout for {tier} too short"
            assert timeout <= 86400, f"Timeout for {tier} too long"


class TestModalSpawnedTask:
    """Tests for ModalSpawnedTask dataclass.

    Test rationale:
        ModalSpawnedTask is the handle returned by spawn_experiment() that allows
        tracking experiment status. It must be serializable (for persistence)
        and contain all necessary tracking information. It implements the
        ModalSpawnedTask dataclass.
    """

    def test_spawned_task_creation(self):
        """ModalSpawnedTask should be creatable with required fields."""
        task = ModalSpawnedTask(run_id="test-123", gpu_tier="standard")

        assert task.run_id == "test-123"
        assert task.gpu_tier == "standard"
        assert task.function_call is None  # Optional
        assert isinstance(task, ModalSpawnedTask)  # Implements protocol

    def test_spawned_task_with_function_call(self):
        """ModalSpawnedTask should accept optional function_call handle."""
        mock_call = MagicMock()
        task = ModalSpawnedTask(
            run_id="test-456",
            gpu_tier="fast",
            function_call=mock_call,
        )

        assert task.function_call is mock_call


class TestSpawnMethods:
    """Tests for ModalRunner spawn methods.

    Test rationale:
        spawn_experiment() and spawn_sweep() enable fire-and-forget execution.
        They must return ModalSpawnedTask handles with valid run_ids and call
        Modal's spawn() with the CLI transport arguments (cmd, config_yaml, run_id).
    """

    @pytest.fixture
    def runner(self):
        """Create a ModalRunner with deployment check skipped."""
        return ModalRunner(
            gpu_type="standard",
            skip_deployment_check=True,
        )

    def test_spawn_experiment_returns_spawned_task(self, runner):
        """spawn_experiment should return SpawnedTask with run_id."""
        from omegaconf import OmegaConf

        config = OmegaConf.create(
            {
                "_cli_cmd": "tmgg-discrete-gen",
                "model": {"hidden_dim": 64},
                "paths": {"output_dir": "/tmp", "results_dir": "/tmp"},
            }
        )

        mock_fn_call = MagicMock()
        with patch("modal.Function") as mock_Function:
            mock_Function.from_name.return_value.spawn.return_value = mock_fn_call

            task = runner.spawn_experiment(config)

        assert isinstance(task, ModalSpawnedTask)
        assert task.run_id is not None
        assert len(task.run_id) == 8  # UUID[:8]
        assert task.gpu_tier == "standard"

    def test_spawn_experiment_passes_cli_args(self, runner):
        """spawn() should receive (cmd, config_yaml, run_id) positional args."""
        from omegaconf import OmegaConf

        config = OmegaConf.create(
            {
                "_cli_cmd": "tmgg-discrete-gen",
                "model": {"hidden_dim": 64},
            }
        )

        mock_fn_call = MagicMock()
        with patch("modal.Function") as mock_Function:
            mock_spawn = mock_Function.from_name.return_value.spawn
            mock_spawn.return_value = mock_fn_call

            task = runner.spawn_experiment(config)

        # Verify spawn was called with (cmd, config_yaml, run_id)
        mock_spawn.assert_called_once()
        call_args = mock_spawn.call_args[0]
        assert call_args[0] == "tmgg-discrete-gen"
        assert "hidden_dim: 64" in call_args[1]  # config_yaml
        assert call_args[2] == task.run_id

    def test_spawn_sweep_returns_list_of_spawned_tasks(self, runner):
        """spawn_sweep should return list of ModalSpawnedTask handles."""
        from omegaconf import OmegaConf

        configs = [
            OmegaConf.create(
                {
                    "_cli_cmd": "tmgg-discrete-gen",
                    "model": {"hidden_dim": dim},
                }
            )
            for dim in [32, 64, 128]
        ]

        mock_fn_call = MagicMock()
        with patch("modal.Function") as mock_Function:
            mock_Function.from_name.return_value.spawn.return_value = mock_fn_call

            tasks = runner.spawn_sweep(configs)

        assert len(tasks) == 3
        assert all(isinstance(t, ModalSpawnedTask) for t in tasks)
        # Each task should have a unique run_id
        run_ids = [t.run_id for t in tasks]
        assert len(set(run_ids)) == 3

    def test_spawn_experiment_tracks_active_runs(self, runner):
        """spawn_experiment should track spawned tasks in _active_runs."""
        from omegaconf import OmegaConf

        config = OmegaConf.create(
            {
                "_cli_cmd": "tmgg-discrete-gen",
                "model": {"hidden_dim": 64},
            }
        )

        mock_fn_call = MagicMock()
        with patch("modal.Function") as mock_Function:
            mock_Function.from_name.return_value.spawn.return_value = mock_fn_call

            task = runner.spawn_experiment(config)

        assert task.run_id in runner._active_runs
        assert runner._active_runs[task.run_id] is task


@pytest.mark.modal
class TestModalSmoke:
    """Smoke tests that submit real experiments to Modal (T4 debug tier).

    One representative architecture per base config type. Exercises the full
    path: config composition -> YAML serialization -> Modal container ->
    CLI subprocess -> result return.

    Skipped by default. Run with ``pytest -m modal -v --tb=long``.
    Requires the ``your-modal-profile`` Modal profile with ``graph-denoising`` deployed.
    """

    @pytest.fixture(autouse=True)
    def _require_modal(self, require_modal_profile):
        pass

    SMOKE_CONFIGS = [
        ("base_config_spectral_arch", "models/spectral/linear_pe", "spectral"),
    ]

    @pytest.mark.parametrize(
        "base_config,arch,desc",
        SMOKE_CONFIGS,
        ids=[c[2] for c in SMOKE_CONFIGS],
    )
    def test_modal_single_step(self, base_config, arch, desc, tmp_path):
        """Submit max_steps=1 experiment to Modal debug tier, verify completion."""
        import uuid
        from pathlib import Path

        import modal
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        # 1. Compose base config
        exp_configs = (
            Path(__file__).parent.parent.parent
            / "src"
            / "tmgg"
            / "experiments"
            / "exp_configs"
        )
        overrides = [
            f"paths.output_dir={tmp_path}",
            f"paths.results_dir={tmp_path}/results",
            "trainer.max_steps=2",
            "trainer.accelerator=auto",
            "~logger",
            "data.batch_size=2",
            "data.num_workers=0",
            "seed=42",
            f"hydra.run.dir={tmp_path}",
        ]

        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(exp_configs)):
            cfg = compose(config_name=base_config, overrides=overrides)

        # 2. Load and merge architecture (replaces deleted config_builder)
        import yaml

        arch_yaml = exp_configs / f"{arch}.yaml"
        with open(arch_yaml) as f:
            arch_config = yaml.safe_load(f)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(cfg_dict, dict)
        base_model = cfg_dict.get("model", {})
        merged_model = {**base_model, **arch_config}
        cfg_dict["model"] = merged_model

        # 3. Serialize to YAML and submit via modal_run_cli_debug
        run_id = f"smoke_{desc}_{uuid.uuid4().hex[:6]}"
        config_yaml = OmegaConf.to_yaml(OmegaConf.create(cfg_dict), resolve=True)

        fn = modal.Function.from_name("graph-denoising", "modal_run_cli_debug")
        result = fn.remote("tmgg-discrete-gen", config_yaml, run_id)

        # 4. Validate
        assert result["status"] == "completed", (
            f"Expected completed, got {result['status']}: "
            f"{result.get('error', 'no error')}"
        )
        assert result["run_id"] == run_id

        GlobalHydra.instance().clear()

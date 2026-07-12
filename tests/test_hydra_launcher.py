"""Tests for the custom Hydra launcher plugin.

Test rationale
--------------
TmggLauncher is a custom Hydra launcher plugin that dispatches jobs to ModalRunner
via the CloudRunner interface. These tests verify that the launcher:

- Raises RuntimeError when use_modal is not set (LocalRunner was removed)
- Selects ModalRunner when use_modal=True
- Dispatches all jobs as a single batch via run_sweep (not run_experiment per job)
- Converts ExperimentResult to Hydra's JobReturn format
- Handles job failures appropriately (fail-fast, no graceful fallback)

Assumptions:
    Tests use mocked CloudRunner implementations to avoid actual experiment execution.
    The launcher receives Hydra-style config and job_overrides.
    All tests run without actual GPU or cloud resources.

Invariants:
    setup() must be called before launch().
    ModalRunner is used when use_modal=True.
    Without use_modal=True, TmggLauncher raises RuntimeError.
    launch() calls run_sweep exactly once with all configs collected upfront.
    Each job override sequence produces one ExperimentResult and one JobReturn.
"""

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hydra.core.utils import JobReturn, JobStatus
from omegaconf import OmegaConf

from tmgg.modal.runner import ExperimentResult


class TestTmggLauncherRunnerSelection:
    """Tests for runner selection based on configuration."""

    def test_raises_without_modal(self) -> None:
        """RuntimeError when use_modal is not True.

        TmggLauncher no longer supports local execution (LocalRunner was removed).
        Users should use Hydra's built-in launcher for local sweeps.
        """
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        launcher = TmggLauncher()
        config = OmegaConf.create(
            {
                "hydra": {"launcher": {"use_modal": False}},
            }
        )

        with pytest.raises(RuntimeError, match="TmggLauncher requires use_modal=True"):
            launcher.setup(
                hydra_context=MagicMock(),
                task_function=MagicMock(),
                config=config,
            )

    def test_raises_when_modal_not_configured(self) -> None:
        """RuntimeError when launcher config omits use_modal entirely."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        launcher = TmggLauncher()
        config = OmegaConf.create(
            {
                "hydra": {"launcher": {}},
            }
        )

        with pytest.raises(RuntimeError, match="TmggLauncher requires use_modal=True"):
            launcher.setup(
                hydra_context=MagicMock(),
                task_function=MagicMock(),
                config=config,
            )

    def test_selects_modal_runner_when_configured(self) -> None:
        """ModalRunner is selected when use_modal=True."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        launcher = TmggLauncher()
        config = OmegaConf.create(
            {
                "hydra": {
                    "launcher": {
                        "use_modal": True,
                        "use_ray": False,
                        "gpu_type": "a10g",
                    }
                },
            }
        )

        with patch(
            "tmgg.hydra_plugins.tmgg_launcher.launcher.ModalRunner"
        ) as mock_modal_cls:
            mock_modal_cls.return_value = MagicMock()
            launcher.setup(
                hydra_context=MagicMock(),
                task_function=MagicMock(),
                config=config,
            )

        mock_modal_cls.assert_called_once_with(gpu_type="a10g")


class TestTmggLauncherJobExecution:
    """Tests for job execution via the launcher."""

    @pytest.fixture
    def mock_runner(self) -> MagicMock:
        """Create a mock CloudRunner that returns successful results via run_sweep."""
        runner = MagicMock()
        # run_sweep returns a list; default returns one result per call.
        # Individual tests override this when they need a specific number of results.
        runner.run_sweep.return_value = [
            ExperimentResult(
                run_id="test-123",
                config={"model": "test"},
                metrics={"val_loss": 0.5},
                status="completed",
                duration_seconds=10.0,
            )
        ]
        return runner

    @pytest.fixture
    def configured_launcher(self, mock_runner: MagicMock) -> Any:
        """Create a TmggLauncher with mocked ModalRunner."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        launcher = TmggLauncher()
        config = OmegaConf.create(
            {
                "hydra": {"launcher": {"use_modal": True, "gpu_type": "debug"}},
                "model": "baseline",
                "paths": {
                    "output_dir": "/tmp/test",
                    "results_dir": "/tmp/test/results",
                },
            }
        )

        with patch(
            "tmgg.hydra_plugins.tmgg_launcher.launcher.ModalRunner",
            return_value=mock_runner,
        ):
            launcher.setup(
                hydra_context=MagicMock(),
                task_function=MagicMock(),
                config=config,
            )

        return launcher

    def test_launch_single_job(
        self, configured_launcher: Any, mock_runner: MagicMock
    ) -> None:
        """Single job execution produces one JobReturn via run_sweep."""
        job_overrides: Sequence[Sequence[str]] = [["model=test_model"]]

        results = configured_launcher.launch(job_overrides, initial_job_idx=0)

        assert len(results) == 1
        assert isinstance(results[0], JobReturn)
        mock_runner.run_sweep.assert_called_once()
        mock_runner.run_experiment.assert_not_called()

    def test_launch_multirun_jobs(
        self, configured_launcher: Any, mock_runner: MagicMock
    ) -> None:
        """Multiple job overrides produce multiple JobReturns via a single run_sweep call."""
        job_overrides: Sequence[Sequence[str]] = [
            ["model=model_a", "lr=0.01"],
            ["model=model_b", "lr=0.01"],
            ["model=model_a", "lr=0.001"],
        ]

        # run_sweep must return one result per config
        mock_runner.run_sweep.return_value = [
            ExperimentResult(
                run_id=f"test-{i}",
                config={"model": "test"},
                metrics={"val_loss": 0.5},
                status="completed",
                duration_seconds=10.0,
            )
            for i in range(3)
        ]

        results = configured_launcher.launch(job_overrides, initial_job_idx=0)

        assert len(results) == 3
        assert all(isinstance(r, JobReturn) for r in results)
        mock_runner.run_sweep.assert_called_once()
        mock_runner.run_experiment.assert_not_called()

    def test_job_idx_preserved(
        self, configured_launcher: Any, mock_runner: MagicMock
    ) -> None:
        """Job index is correctly assigned starting from initial_job_idx."""
        job_overrides: Sequence[Sequence[str]] = [["model=a"], ["model=b"]]

        mock_runner.run_sweep.return_value = [
            ExperimentResult(
                run_id=f"test-{i}",
                config={"model": "test"},
                metrics={"val_loss": 0.5},
                status="completed",
                duration_seconds=10.0,
            )
            for i in range(2)
        ]

        results = configured_launcher.launch(job_overrides, initial_job_idx=5)

        # Job indices are stored in return_value as _job_idx
        assert results[0].return_value["_job_idx"] == 5
        assert results[1].return_value["_job_idx"] == 6


class TestJobReturnConversion:
    """Tests for ExperimentResult to JobReturn conversion."""

    def test_successful_result_converts_to_completed_status(self) -> None:
        """Completed ExperimentResult maps to JobStatus.COMPLETED."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        experiment_result = ExperimentResult(
            run_id="abc123",
            config={"model": "test"},
            metrics={"val_loss": 0.25},
            status="completed",
            duration_seconds=60.0,
        )

        job_return = TmggLauncher._result_to_job_return(
            experiment_result,
            overrides=["model=test"],
            idx=0,
        )

        assert job_return.status == JobStatus.COMPLETED
        assert job_return.return_value["val_loss"] == 0.25
        assert job_return.return_value["_job_idx"] == 0

    def test_failed_result_converts_to_failed_status(self) -> None:
        """Failed ExperimentResult maps to JobStatus.FAILED."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        experiment_result = ExperimentResult(
            run_id="abc123",
            config={"model": "test"},
            metrics={},
            status="failed",
            error_message="CUDA out of memory",
            duration_seconds=5.0,
        )

        job_return = TmggLauncher._result_to_job_return(
            experiment_result,
            overrides=["model=test"],
            idx=0,
        )

        assert job_return.status == JobStatus.FAILED
        # For failed jobs, access _return_value directly (return_value raises)
        assert job_return._return_value["_job_idx"] == 0


class TestLauncherErrorHandling:
    """Tests for error handling during job execution."""

    def test_runner_exception_raises(self) -> None:
        """Runner exception crashes the launcher (fail-fast policy)."""
        from tmgg.hydra_plugins.tmgg_launcher import TmggLauncher

        launcher = TmggLauncher()
        config = OmegaConf.create(
            {
                "hydra": {"launcher": {"use_modal": True, "gpu_type": "debug"}},
                "model": "baseline",
                "paths": {
                    "output_dir": "/tmp/test",
                    "results_dir": "/tmp/test/results",
                },
            }
        )

        failing_runner = MagicMock()
        failing_runner.run_sweep.side_effect = RuntimeError("Experiment failed")

        with patch(
            "tmgg.hydra_plugins.tmgg_launcher.launcher.ModalRunner",
            return_value=failing_runner,
        ):
            launcher.setup(
                hydra_context=MagicMock(),
                task_function=MagicMock(),
                config=config,
            )

        # Per project conventions: no graceful fallback, crash loudly
        with pytest.raises(RuntimeError, match="Experiment failed"):
            launcher.launch([["model=test"]], initial_job_idx=0)


class TestLauncherConfiguration:
    """Tests for launcher configuration via Hydra ConfigStore."""

    def test_launcher_config_dataclass_exists(self) -> None:
        """TmggLauncherConf dataclass is properly defined."""
        from tmgg.hydra_plugins.tmgg_launcher.config import TmggLauncherConf

        # Verify required attributes exist with correct defaults
        config = TmggLauncherConf()
        assert config.use_modal is False
        assert config.gpu_type == "debug"
        assert config.timeout_seconds == 3600
        assert not hasattr(config, "parallelism")

    def test_launcher_registered_in_config_store(self) -> None:
        """Launcher config is registered in Hydra's ConfigStore."""
        from hydra.core.config_store import ConfigStore

        # Import the module to trigger registration
        import tmgg.hydra_plugins.tmgg_launcher  # noqa: F401

        cs = ConfigStore.instance()

        # The config should be registered under hydra/launcher group
        configs = cs.list("hydra/launcher")
        assert "tmgg.yaml" in configs

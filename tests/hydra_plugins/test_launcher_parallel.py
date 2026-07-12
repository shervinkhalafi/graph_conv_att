"""Verify TmggLauncher dispatches via run_sweep (parallel) not run_experiment (sequential).

Test rationale
--------------
The launcher previously called ``run_experiment()`` in a sequential for loop,
negating Modal's parallelism. After the fix, ``launch()`` collects all resolved
configs and passes them to ``run_sweep()`` in a single batch, enabling parallel
dispatch for ModalRunner.
"""

from unittest.mock import MagicMock

from omegaconf import OmegaConf

from tmgg.modal.runner import ExperimentResult


def test_launch_calls_run_sweep_not_run_experiment():
    """launch() must call run_sweep once, not run_experiment N times."""
    from tmgg.hydra_plugins.tmgg_launcher.launcher import TmggLauncher

    launcher = TmggLauncher()
    launcher._config = OmegaConf.create(
        {
            "seed": 1,
            "hydra": {"launcher": {"gpu_type": "debug"}},
            "paths": {"output_dir": "/tmp/test", "results_dir": "/tmp/test/results"},
        }
    )

    mock_runner = MagicMock()
    mock_runner.run_sweep.return_value = [
        ExperimentResult(run_id="r1", config={}, metrics={}, status="completed"),
        ExperimentResult(run_id="r2", config={}, metrics={}, status="completed"),
    ]
    launcher._runner = mock_runner

    overrides = [["model=a"], ["model=b"]]
    results = launcher.launch(overrides, initial_job_idx=0)

    mock_runner.run_sweep.assert_called_once()
    mock_runner.run_experiment.assert_not_called()
    assert len(results) == 2

"""Utility functions for integration tests.

This module provides helpers for running CLI commands as subprocesses with
timeout handling and output validation. Designed for testing experiment
runners without risking Hydra state conflicts.
"""

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def run_cli_command(
    cmd: Sequence[str],
    timeout: int = 120,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a CLI command with timeout, returning captured output.

    Parameters
    ----------
    cmd
        Command and arguments to execute.
    timeout
        Maximum execution time in seconds. Defaults to 120s.
    cwd
        Working directory for the command. Defaults to current directory.
    env
        Optional subprocess environment. When omitted, inherits the current
        process environment unchanged.

    Returns
    -------
    subprocess.CompletedProcess
        Result with returncode, stdout, and stderr.

    Raises
    ------
    subprocess.TimeoutExpired
        If command exceeds timeout.
    """
    result = subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
        env=env,
    )
    return result


def assert_cli_success(result: subprocess.CompletedProcess[str]) -> None:
    """Assert that a CLI command completed without Python exceptions.

    Parameters
    ----------
    result
        Completed subprocess result from run_cli_command.

    Raises
    ------
    AssertionError
        If the command failed or stderr contains exception traces.
    """
    # Check exit code
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed with exit code {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # Check for Python exceptions in stderr
    exception_markers = [
        "Traceback (most recent call last)",
        "Error:",
        "Exception:",
        "raise ",
    ]
    stderr_lower = result.stderr.lower()
    for marker in exception_markers:
        if marker.lower() in stderr_lower:
            # Allow certain expected log messages
            if "error" in marker.lower() and (
                "error_rate" in stderr_lower or "no error" in stderr_lower
            ):
                continue
            raise AssertionError(
                f"Found exception marker '{marker}' in stderr.\n"
                f"stderr:\n{result.stderr}"
            )


def assert_training_success(result: subprocess.CompletedProcess[str]) -> None:
    """Backward-compatible alias for training-focused smoke tests."""
    assert_cli_success(result)


def get_test_subprocess_env(scratch_dir: Path) -> dict[str, str]:
    """Return a stable subprocess environment for tiny smoke tests.

    Test rationale
    --------------
    Many CLI entrypoints import matplotlib transitively. Pointing
    ``MPLCONFIGDIR`` at a writable test-local directory avoids warnings and
    expensive cache regeneration under the user's home directory.
    """
    mpl_dir = scratch_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(mpl_dir)
    return env


def get_quick_training_overrides(
    output_dir: Path,
    data_module: str = "denoising",
) -> list[str]:
    """Generate Hydra overrides for a minimal training run.

    These overrides configure the experiment to complete in seconds by:
    - Limiting to 2 training steps
    - Using the local CSV logger instead of external services
    - Using minimal data
    - Running on CPU

    Parameters
    ----------
    output_dir
        Directory for experiment outputs.
    data_module
        DataModule category. Determines which data-reduction overrides
        are safe to apply:
        ``"denoising"`` — GraphDataModule (has ``samples_per_graph``,
        SBM partition params);
        ``"single_graph"`` — SingleGraphDataModule (has
        ``num_train_samples``, ``num_val_samples``, ``num_test_samples``);
        ``"generative"`` — MultiGraphDataModule or
        SyntheticCategoricalDataModule (has ``num_graphs``).

    Returns
    -------
    list[str]
        Hydra override strings.
    """
    overrides = [
        f"paths.output_dir={output_dir}",
        f"paths.results_dir={output_dir}/results",
        "trainer.max_steps=2",
        "trainer.val_check_interval=1",
        "trainer.accelerator=cpu",
        # Note: enable_checkpointing and enable_progress_bar must stay true
        # because create_callbacks unconditionally adds ModelCheckpoint and
        # StepProgressBar, which conflict with False settings
        # Use a local-only logger profile; the default callback stack includes
        # LearningRateMonitor, which Lightning rejects when no logger exists.
        "base/logger=csv",
        # Data reduction - common keys that exist in all DataModules
        "data.batch_size=2",
        "data.num_workers=0",
        # Set hydra run dir to output dir
        f"hydra.run.dir={output_dir}",
    ]

    if data_module == "denoising":
        overrides += [
            "++data.samples_per_graph=4",
            "++data.graph_config.num_train_partitions=2",
            "++data.graph_config.num_test_partitions=2",
            "++data.graph_config.num_graphs=4",
        ]
    elif data_module == "single_graph":
        overrides += [
            "++data.num_train_samples=8",
            "++data.num_val_samples=4",
            "++data.num_test_samples=4",
        ]
    elif data_module == "generative":
        overrides += [
            "data.num_graphs=20",
        ]

    return overrides


def build_runner_command(
    runner: str,
    overrides: list[str],
) -> list[str]:
    """Build a command to run an experiment runner with overrides.

    Parameters
    ----------
    runner
        Name of the CLI runner (e.g., 'tmgg-spectral-arch').
    overrides
        List of Hydra override strings.

    Returns
    -------
    list[str]
        Full command including 'uv run' prefix.
    """
    return [sys.executable, "-m", "uv", "run", runner, *overrides]

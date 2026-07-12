"""Integration smokes for non-training experiment CLIs.

Test rationale
--------------
These tests cover the remaining public experiment surfaces that do not fit the
plain "train for two steps" pattern:

- Hydra study runners with explicit phase chaining
- direct structural-study CLIs
- discrete checkpoint evaluation against a real tiny checkpoint

The invariants are simple: each command exits cleanly and writes the expected
artifact for the code path it exercises.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from .integration_utils import get_quick_training_overrides


def _assert_cli_success(result: subprocess.CompletedProcess[str]) -> None:
    """Mirror the subprocess success checks used by the shared integration helpers."""
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed with exit code {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    exception_markers = [
        "Traceback (most recent call last)",
        "Error:",
        "Exception:",
        "raise ",
    ]
    stderr_lower = result.stderr.lower()
    for marker in exception_markers:
        if marker.lower() in stderr_lower:
            if "error" in marker.lower() and (
                "error_rate" in stderr_lower or "no error" in stderr_lower
            ):
                continue
            raise AssertionError(
                f"Found exception marker '{marker}' in stderr.\n"
                f"stderr:\n{result.stderr}"
            )


def _build_test_env(scratch_dir: Path) -> dict[str, str]:
    """Provide a writable matplotlib cache directory for subprocess tests."""
    mpl_dir = scratch_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(mpl_dir)
    return env


def _run_cli_smoke(
    cmd: list[str],
    *,
    scratch_dir: Path,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    """Run one CLI command under the shared smoke-test environment."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_build_test_env(scratch_dir),
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - exercised on failure
        pytest.fail(
            f"Command timed out after {timeout}s.\n"
            f"stdout: {exc.stdout}\n"
            f"stderr: {exc.stderr}"
        )
        raise AssertionError("pytest.fail should have aborted execution") from exc

    _assert_cli_success(result)
    return result


def _tiny_discrete_training_overrides(output_dir: Path) -> list[str]:
    """Return a minimal discrete training config that still writes a checkpoint."""
    overrides = get_quick_training_overrides(output_dir, data_module="generative")
    overrides.extend(
        [
            "data.num_nodes=12",
            "model.noise_schedule.timesteps=10",
            "model.evaluator.eval_num_samples=4",
            "models/discrete@model=discrete_small",
            "model.model.n_layers=2",
            "model.model.hidden_dims.dx=64",
            "model.model.hidden_dims.de=16",
            "model.model.hidden_dims.dy=16",
            "model.model.hidden_dims.n_head=2",
            "model.model.hidden_dims.dim_ffX=64",
            "model.model.hidden_dims.dim_ffE=16",
            "model.model.hidden_dims.dim_ffy=64",
            "model.model.hidden_mlp_dims.X=64",
            "model.model.hidden_mlp_dims.E=32",
            "model.model.hidden_mlp_dims.y=32",
        ]
    )
    return overrides


@pytest.mark.integration
@pytest.mark.slow
class TestHydraStudyRunners:
    """Smoke tests for the Hydra phase runners."""

    def test_eigenstructure_exp_executes_full_tiny_pipeline(
        self,
        tmp_path: Path,
    ) -> None:
        """The Hydra eigenstructure runner should complete all shipped phases."""
        output_dir = tmp_path / "eigenstructure_exp"
        common = [
            f"paths.output_dir={output_dir}",
            f"hydra.run.dir={output_dir}",
            "dataset.name=sbm",
            "dataset.config.num_nodes=8",
            "dataset.config.num_partitions=6",
            "dataset.config.p_intra=0.8",
            "dataset.config.p_inter=0.1",
            "batch_size=3",
        ]

        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-eigenstructure-exp",
                "phase=collect",
                *common,
            ],
            scratch_dir=tmp_path,
        )
        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-eigenstructure-exp",
                "phase=analyze",
                "analysis.subspace_k=0",
                *common,
            ],
            scratch_dir=tmp_path,
        )
        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-eigenstructure-exp",
                "phase=noised",
                "noise.type=gaussian",
                "noise.levels=[0.1]",
                *common,
            ],
            scratch_dir=tmp_path,
        )
        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-eigenstructure-exp",
                "phase=compare",
                "comparison.subspace_k=2",
                "comparison.procrustes_k_values=[1,2]",
                "comparison.compute_covariance_evolution=true",
                *common,
            ],
            scratch_dir=tmp_path,
        )
        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-eigenstructure-exp",
                "phase=covariance",
                *common,
            ],
            scratch_dir=tmp_path,
        )

        assert (output_dir / "original" / "manifest.json").exists()
        assert (output_dir / "analysis" / "analysis.json").exists()
        assert (output_dir / "noised" / "eps_0.1000" / "manifest.json").exists()
        assert (output_dir / "comparison" / "comparison.json").exists()
        assert (output_dir / "comparison" / "covariance_evolution.json").exists()
        assert (output_dir / "covariance" / "covariance.json").exists()

    def test_embedding_study_exp_runs_and_analyzes_tiny_dataset(
        self,
        tmp_path: Path,
    ) -> None:
        """The Hydra embedding-study runner should complete run and analyze."""
        output_dir = tmp_path / "embedding_exp"
        common = [
            f"paths.output_dir={output_dir}",
            f"hydra.run.dir={output_dir}",
            "datasets=[sbm]",
            "num_graphs=3",
            "num_nodes=12",
            "methods=symmetric",
            "fitter=spectral",
        ]

        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-embedding-study-exp",
                "phase=run",
                *common,
            ],
            scratch_dir=tmp_path,
        )
        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-embedding-study-exp",
                "phase=analyze",
                *common,
            ],
            scratch_dir=tmp_path,
        )

        assert (output_dir / "embedding_study.json").exists()
        assert (output_dir / "embeddings.safetensors").exists()


@pytest.mark.integration
@pytest.mark.slow
class TestDirectStudyCLIs:
    """Smoke tests for the direct Click/argparse CLIs."""

    def test_eigenstructure_cli_executes_full_tiny_pipeline(
        self,
        tmp_path: Path,
    ) -> None:
        """The direct eigenstructure CLI should cover all command surfaces."""
        original_dir = tmp_path / "eigen_cli_original"
        analysis_dir = tmp_path / "eigen_cli_analysis"
        noised_dir = tmp_path / "eigen_cli_noised"
        comparison_dir = tmp_path / "eigen_cli_comparison"
        covariance_dir = tmp_path / "eigen_cli_covariance"

        dataset_config = (
            '{"num_nodes": 8, "p_intra": 0.8, "p_inter": 0.1, "num_partitions": 6}'
        )

        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-eigenstructure",
                "collect",
                "--dataset",
                "sbm",
                "--dataset-config",
                dataset_config,
                "--output-dir",
                str(original_dir),
                "--batch-size",
                "3",
            ],
            scratch_dir=tmp_path,
        )
        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-eigenstructure",
                "analyze",
                "--input-dir",
                str(original_dir),
                "--output-dir",
                str(analysis_dir),
                "--subspace-k",
                "0",
            ],
            scratch_dir=tmp_path,
        )
        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-eigenstructure",
                "noised",
                "--input-dir",
                str(original_dir),
                "--output-dir",
                str(noised_dir),
                "--noise-type",
                "gaussian",
                "--noise-levels",
                "0.1",
            ],
            scratch_dir=tmp_path,
        )
        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-eigenstructure",
                "compare",
                "--original-dir",
                str(original_dir),
                "--noised-dir",
                str(noised_dir),
                "--output-dir",
                str(comparison_dir),
                "--subspace-k",
                "2",
                "--procrustes-k",
                "1,2",
            ],
            scratch_dir=tmp_path,
        )
        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-eigenstructure",
                "covariance",
                "--original-dir",
                str(original_dir),
                "--noised-dir",
                str(noised_dir),
                "--output-dir",
                str(covariance_dir),
            ],
            scratch_dir=tmp_path,
        )

        assert (original_dir / "manifest.json").exists()
        assert (analysis_dir / "analysis.json").exists()
        assert (noised_dir / "eps_0.1000" / "manifest.json").exists()
        assert (comparison_dir / "comparison.json").exists()
        assert (covariance_dir / "covariance_evolution.json").exists()

    def test_embedding_study_cli_runs_and_analyzes_tiny_dataset(
        self,
        tmp_path: Path,
    ) -> None:
        """The direct embedding-study CLI should cover run and analyze."""
        output_dir = tmp_path / "embedding_cli"
        results_path = output_dir / "embedding_study.json"

        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-embedding-study",
                "run",
                "--datasets",
                "sbm",
                "--output",
                str(output_dir),
                "--methods",
                "symmetric",
                "--fitter",
                "spectral",
                "--num-graphs",
                "3",
                "--num-nodes",
                "12",
            ],
            scratch_dir=tmp_path,
        )
        _run_cli_smoke(
            [
                "uv",
                "run",
                "tmgg-embedding-study",
                "analyze",
                "--input",
                str(results_path),
            ],
            scratch_dir=tmp_path,
        )

        assert results_path.exists()
        assert (output_dir / "embeddings.safetensors").exists()


@pytest.mark.integration
@pytest.mark.slow
class TestDiscreteEvaluateCLI:
    """Smoke test for evaluating a real tiny discrete checkpoint."""

    def test_discrete_eval_cli_consumes_real_tiny_checkpoint(
        self,
        tmp_path: Path,
    ) -> None:
        """Train a tiny discrete model, then evaluate it through the real CLI."""
        train_dir = tmp_path / "discrete_train"
        eval_path = tmp_path / "discrete_eval.json"

        train_cmd = [
            "uv",
            "run",
            "tmgg-discrete-gen",
            *_tiny_discrete_training_overrides(train_dir),
        ]
        _run_cli_smoke(train_cmd, scratch_dir=tmp_path)

        checkpoint_path = train_dir / "checkpoints" / "last.ckpt"
        assert checkpoint_path.exists()

        eval_cmd = [
            "uv",
            "run",
            "tmgg-discrete-eval",
            "--checkpoint",
            str(checkpoint_path),
            "--dataset",
            "sbm",
            "--num-samples",
            "4",
            "--num-nodes",
            "12",
            "--device",
            "cpu",
            "--output",
            str(eval_path),
        ]
        _run_cli_smoke(eval_cmd, scratch_dir=tmp_path)

        payload = json.loads(eval_path.read_text())
        assert payload["checkpoint_name"] == "last.ckpt"
        assert payload["num_generated"] == 4
        assert "mmd_results" in payload

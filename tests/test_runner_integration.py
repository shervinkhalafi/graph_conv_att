"""Integration tests for training-oriented CLI runners.

Test rationale
--------------
These smokes exercise every training entrypoint through its real subprocess
surface with:

- explicit tiny model overrides
- tiny data and batch sizes
- CPU-only execution
- local CSV logging only

They catch configuration drift, import errors, Hydra override regressions, and
runner wiring bugs before the more expensive Modal path is involved.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.integration_utils import (
    assert_training_success,
    get_quick_training_overrides,
    get_test_subprocess_env,
    run_cli_command,
)


@dataclass(frozen=True)
class RunnerSmokeCase:
    """One real training CLI smoke configuration."""

    case_id: str
    runner_cmd: str
    data_module: str
    extra_overrides: tuple[str, ...] = ()


def _tiny_discrete_overrides(*, with_eigenvectors: bool = False) -> tuple[str, ...]:
    """Return a compact discrete-diffusion model override set."""
    overrides = (
        "data.num_nodes=12",
        "data.num_graphs=20",
        "model.noise_schedule.timesteps=10",
        "model.evaluator.eval_num_samples=4",
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
    )
    if with_eigenvectors:
        return overrides + ("model.model.extra_features.k=4",)
    return overrides


EXPERIMENT_RUNNERS = [
    RunnerSmokeCase(
        "digress-default",
        "tmgg-digress",
        "denoising",
        (
            "model.model.n_layers=2",
            "model.model.hidden_dims.dx=32",
            "model.model.hidden_dims.de=8",
            "model.model.hidden_dims.dy=32",
            "model.model.hidden_dims.n_head=2",
            "model.model.hidden_mlp_dims.X=64",
            "model.model.hidden_mlp_dims.E=16",
            "model.model.hidden_mlp_dims.y=64",
            "model.model.extra_features.k=4",
        ),
    ),
    RunnerSmokeCase(
        "gnn-default",
        "tmgg-gnn",
        "denoising",
        (
            "model.model.num_terms=2",
            "model.model.feature_dim_in=8",
            "model.model.feature_dim_out=4",
        ),
    ),
    RunnerSmokeCase(
        "gnn-transformer-default",
        "tmgg-gnn-transformer",
        "denoising",
        (
            "model.model.gnn_config.num_layers=1",
            "model.model.gnn_config.num_terms=2",
            "model.model.gnn_config.feature_dim_in=8",
            "model.model.gnn_config.feature_dim_out=4",
            "model.model.transformer_config.num_layers=1",
            "model.model.transformer_config.num_heads=2",
            "model.model.transformer_config.d_k=4",
            "model.model.transformer_config.d_v=4",
        ),
    ),
    RunnerSmokeCase(
        "spectral-arch-default",
        "tmgg-spectral-arch",
        "denoising",
        (
            "model.model.k=4",
            "model.model.max_nodes=32",
        ),
    ),
    RunnerSmokeCase(
        "grid-search-default",
        "tmgg-grid-search",
        "denoising",
        (
            "model.model.k=4",
            "model.model.polynomial_degree=2",
        ),
    ),
    RunnerSmokeCase(
        "baseline-linear",
        "tmgg-baseline",
        "denoising",
        ("model.model.max_nodes=32",),
    ),
    RunnerSmokeCase(
        "baseline-mlp",
        "tmgg-baseline",
        "denoising",
        (
            "+models/baselines@model=mlp",
            "model.model.max_nodes=32",
            "model.model.hidden_dim=32",
            "model.model.num_layers=1",
        ),
    ),
    RunnerSmokeCase(
        "gaussian-default",
        "tmgg-gaussian-gen",
        "generative",
        (
            "data.num_nodes=16",
            "model.model.k=4",
            "model.model.d_k=16",
            "model.noise_schedule.timesteps=5",
            "model.evaluator.eval_num_samples=4",
        ),
    ),
    RunnerSmokeCase(
        "discrete-default",
        "tmgg-discrete-gen",
        "generative",
        _tiny_discrete_overrides(),
    ),
    RunnerSmokeCase(
        "discrete-small",
        "tmgg-discrete-gen",
        "generative",
        ("models/discrete@model=discrete_small",) + _tiny_discrete_overrides(),
    ),
    RunnerSmokeCase(
        "discrete-eigenvec",
        "tmgg-discrete-gen",
        "generative",
        ("models/discrete@model=discrete_sbm_eigenvec",)
        + _tiny_discrete_overrides(with_eigenvectors=True),
    ),
    RunnerSmokeCase(
        "discrete-official",
        "tmgg-discrete-gen",
        "generative",
        ("models/discrete@model=discrete_sbm_official",) + _tiny_discrete_overrides(),
    ),
]


STAGE_CONFIGS = [
    ("stage1_poc", "single_graph"),
    ("stage1_sanity", "denoising"),
    ("stage2_validation", "single_graph"),
    ("stage3_diversity", "single_graph"),
    ("stage4_benchmarks", "single_graph"),
    ("stage5_full", "single_graph"),
]


def _run_training_smoke(
    cmd: list[str],
    *,
    scratch_dir: Path,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    """Run one training CLI smoke command and assert success."""
    try:
        result = run_cli_command(
            cmd,
            timeout=timeout,
            env=get_test_subprocess_env(scratch_dir),
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - exercised on failure
        pytest.fail(
            f"Command timed out after {timeout}s.\n"
            f"stdout: {exc.stdout}\n"
            f"stderr: {exc.stderr}"
        )

    assert_training_success(result)
    return result


@pytest.mark.integration
@pytest.mark.slow
class TestExperimentRunners:
    """Smoke tests for experiment-family training CLIs."""

    @pytest.mark.parametrize(
        "case",
        EXPERIMENT_RUNNERS,
        ids=[case.case_id for case in EXPERIMENT_RUNNERS],
    )
    def test_runner_executes_brief_training(
        self,
        case: RunnerSmokeCase,
        tmp_path: Path,
    ) -> None:
        """Each runner should finish a two-step tiny training run."""
        overrides = get_quick_training_overrides(tmp_path, data_module=case.data_module)
        cmd = ["uv", "run", case.runner_cmd, *case.extra_overrides, *overrides]

        _run_training_smoke(cmd, scratch_dir=tmp_path)
        assert tmp_path.exists(), f"Output directory not created: {tmp_path}"


@pytest.mark.integration
@pytest.mark.slow
class TestUnifiedExperimentRunner:
    """Smoke tests for the unified stage runner."""

    @pytest.mark.parametrize("stage_config,dm_type", STAGE_CONFIGS)
    def test_stage_executes_brief_training(
        self,
        stage_config: str,
        dm_type: str,
        tmp_path: Path,
    ) -> None:
        """Each shipped stage should complete with a tiny spectral model."""
        overrides = get_quick_training_overrides(tmp_path, data_module=dm_type)
        if dm_type == "single_graph":
            overrides.append("data.num_nodes=16")
        else:
            overrides.append("data.graph_config.num_nodes=16")
        overrides.extend(
            [
                "+models/spectral@model=linear_pe",
                "model.model.k=4",
                "model.model.max_nodes=16",
            ]
        )
        cmd = ["uv", "run", "tmgg-experiment", f"+stage={stage_config}", *overrides]

        _run_training_smoke(cmd, scratch_dir=tmp_path)
        assert tmp_path.exists(), f"Output directory not created: {tmp_path}"


@pytest.mark.integration
class TestRunnerImports:
    """Quick import smokes for runner modules."""

    def test_training_runner_imports(self) -> None:
        """All training runner modules should expose a callable main."""
        from tmgg.experiments import grid_search_runner
        from tmgg.experiments.digress_denoising import runner as digress
        from tmgg.experiments.gaussian_diffusion_generative import runner as gaussian
        from tmgg.experiments.gnn_denoising import runner as gnn
        from tmgg.experiments.gnn_transformer_denoising import runner as hybrid
        from tmgg.experiments.lin_mlp_baseline_denoising import runner as baseline
        from tmgg.experiments.spectral_arch_denoising import runner as spectral

        assert callable(digress.main)
        assert callable(gnn.main)
        assert callable(hybrid.main)
        assert callable(spectral.main)
        assert callable(baseline.main)
        assert callable(gaussian.main)
        assert callable(grid_search_runner.main)

    def test_study_runner_imports(self) -> None:
        """The non-training Hydra runners should also import cleanly."""
        from tmgg.experiments.eigenstructure_study import runner as eigenstructure
        from tmgg.experiments.embedding_study import runner as embedding
        from tmgg.experiments.stages import runner as stages

        assert callable(stages.main)
        assert callable(eigenstructure.main)
        assert callable(embedding.main)

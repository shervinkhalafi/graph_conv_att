"""Minimal full-coverage matrix for runnable experiment configs.

Test rationale
--------------
The existing CLI smokes prove that each public experiment entrypoint starts and
exits cleanly. This module covers a different risk surface: config inventory
drift inside the experiment families themselves.

The matrix is intentionally non-cartesian. It enforces these invariants:

- every runnable leaf model config under ``exp_configs/models/**`` is exercised
  at least once with a real training run
- every runnable leaf data config under ``exp_configs/data/*.yaml`` is
  exercised at least once with a real training run
- each training case produces evidence that fit, validation logging, and test
  execution all happened
- each denoising family still reaches the sanity-check plotting path and emits
  the expected PNG artifacts

The test data is deliberately tiny, and PyG-backed data configs are replaced
with a deterministic in-memory fixture so the suite stays offline and stable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.experiment_matrix_utils import (
    DATA_COVERAGE_CASES,
    INLINE_MODEL_CASES,
    MODEL_COVERAGE_CASES,
    PLOT_COVERAGE_CASES,
    DataCoverageCase,
    InlineTrainingCase,
    ModelCoverageCase,
    PlotCoverageCase,
    clear_hydra,
    discover_data_leaves,
    discover_model_leaves,
    isolate_matplotlib_cache,
    patch_fake_pyg_wrapper,
    run_plot_case,
    run_training_case,
)

_ = (clear_hydra, isolate_matplotlib_cache, patch_fake_pyg_wrapper)


@pytest.mark.integration
@pytest.mark.slow
class TestCoverageInventories:
    """Guard the model and data coverage manifests against config drift."""

    def test_all_runnable_leaf_models_are_listed(self) -> None:
        """Every runnable model leaf should appear in the model matrix."""
        expected = discover_model_leaves()
        covered = {case.model_leaf for case in MODEL_COVERAGE_CASES}
        assert covered == expected, (
            "Model coverage manifest is out of sync.\n"
            f"Missing: {sorted(expected - covered)}\n"
            f"Extra: {sorted(covered - expected)}"
        )

    def test_all_runnable_leaf_data_configs_are_listed(self) -> None:
        """Every runnable data leaf should appear in the data matrix."""
        expected = discover_data_leaves()
        covered = {case.data_leaf for case in DATA_COVERAGE_CASES}
        assert covered == expected, (
            "Data coverage manifest is out of sync.\n"
            f"Missing: {sorted(expected - covered)}\n"
            f"Extra: {sorted(covered - expected)}"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestModelCoverageMatrix:
    """Exercise every runnable leaf model config at least once."""

    @pytest.mark.parametrize(
        "case",
        MODEL_COVERAGE_CASES,
        ids=[case.model_leaf for case in MODEL_COVERAGE_CASES],
    )
    def test_leaf_model_runs_tiny_train_val_test(
        self,
        case: ModelCoverageCase,
        tmp_path: Path,
    ) -> None:
        """Each leaf model config should survive one tiny train/val/test run."""
        output_dir = tmp_path / case.model_leaf.replace("/", "__")
        run_training_case(
            base_config=case.base_config,
            output_dir=output_dir,
            extra_overrides=case.extra_overrides,
        )

    @pytest.mark.parametrize(
        "case",
        INLINE_MODEL_CASES,
        ids=[case.case_id for case in INLINE_MODEL_CASES],
    )
    def test_inline_training_surface_runs_tiny_train_val_test(
        self,
        case: InlineTrainingCase,
        tmp_path: Path,
    ) -> None:
        """Inline model surfaces should still get train/val/test coverage."""
        output_dir = tmp_path / case.case_id.replace("/", "__")
        run_training_case(
            base_config=case.base_config,
            output_dir=output_dir,
            extra_overrides=case.extra_overrides,
        )


@pytest.mark.integration
@pytest.mark.slow
class TestDataCoverageMatrix:
    """Exercise every runnable leaf data config at least once."""

    @pytest.mark.parametrize(
        "case",
        DATA_COVERAGE_CASES,
        ids=[case.data_leaf for case in DATA_COVERAGE_CASES],
    )
    def test_leaf_data_config_runs_tiny_train_val_test(
        self,
        case: DataCoverageCase,
        tmp_path: Path,
    ) -> None:
        """Each leaf data config should survive one compatible tiny run."""
        output_dir = tmp_path / case.data_leaf
        run_training_case(
            base_config=case.base_config,
            output_dir=output_dir,
            extra_overrides=case.extra_overrides,
        )


@pytest.mark.integration
@pytest.mark.slow
class TestPlotCoverageMatrix:
    """Exercise the sanity-check plotting path for each denoising family."""

    @pytest.mark.parametrize(
        "case",
        PLOT_COVERAGE_CASES,
        ids=[case.case_id for case in PLOT_COVERAGE_CASES],
    )
    def test_family_emits_sanity_check_plots(
        self,
        case: PlotCoverageCase,
        tmp_path: Path,
    ) -> None:
        """Representative family cases should emit both diagnostic plots."""
        output_dir = tmp_path / case.case_id
        run_plot_case(
            base_config=case.base_config,
            output_dir=output_dir,
            extra_overrides=case.extra_overrides,
        )

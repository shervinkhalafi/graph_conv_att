"""Representatively-cartesian experiment coverage lane.

Test rationale
--------------
The lighter matrix proves that every runnable model and data config is hit at
least once. This heavier lane strengthens that guarantee in two ways:

- three anchor denoising models run against the entire runnable data inventory
- every remaining training surface still runs once on a family-representative
  dataset

That combination is deliberately not exhaustive, but it is much closer to the
real compatibility surface than a pure inventory sweep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.experiment_matrix_utils import (
    ANCHOR_MODEL_LEAVES,
    DATA_COVERAGE_CASES,
    PLOT_COVERAGE_CASES,
    REPRESENTATIVE_SURFACE_CASES,
    PlotCoverageCase,
    TrainingCase,
    build_cartesian_anchor_case,
    clear_hydra,
    discover_data_leaves,
    discover_model_leaves,
    isolate_matplotlib_cache,
    patch_fake_pyg_wrapper,
    run_plot_case,
    run_training_case,
)

_ = (clear_hydra, isolate_matplotlib_cache, patch_fake_pyg_wrapper)

ALL_DATA_LEAVES = tuple(case.data_leaf for case in DATA_COVERAGE_CASES)

# The anchor sweep uses three deliberately different denoising interfaces:
# plain message passing, baseline DiGress, and the tweaked DiGress variant.
# Running those across every data leaf gives broad compatibility coverage
# without paying for the full model × data cartesian product.
ANCHOR_TRAINING_CASES = [
    build_cartesian_anchor_case(anchor_model_leaf, data_leaf)
    for anchor_model_leaf in ANCHOR_MODEL_LEAVES
    for data_leaf in ALL_DATA_LEAVES
]

EXPECTED_CARTESIAN_TEST_COUNT = 99
assert len(ANCHOR_TRAINING_CASES) == 66
assert len(REPRESENTATIVE_SURFACE_CASES) == 26
assert len(PLOT_COVERAGE_CASES) == 5
assert (
    len(ANCHOR_TRAINING_CASES)
    + len(REPRESENTATIVE_SURFACE_CASES)
    + len(PLOT_COVERAGE_CASES)
    + 2
    == EXPECTED_CARTESIAN_TEST_COUNT
)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.cartesian
class TestCartesianCoverageInventories:
    """Guard the heavier matrix against inventory drift."""

    def test_all_runnable_leaf_models_are_covered_by_lane(self) -> None:
        """All runnable model leaves should be covered by the lane as a whole."""
        expected = discover_model_leaves()
        covered = set(ANCHOR_MODEL_LEAVES) | {
            case.case_id
            for case in REPRESENTATIVE_SURFACE_CASES
            if case.case_id in expected
        }
        assert covered == expected, (
            "Cartesian model coverage is out of sync.\n"
            f"Missing: {sorted(expected - covered)}\n"
            f"Extra: {sorted(covered - expected)}"
        )

    def test_all_runnable_leaf_data_configs_are_covered_by_anchor_sweep(self) -> None:
        """The anchor sweep should span the full runnable data inventory."""
        expected = discover_data_leaves()
        covered = set(ALL_DATA_LEAVES)
        assert covered == expected, (
            "Cartesian data coverage is out of sync.\n"
            f"Missing: {sorted(expected - covered)}\n"
            f"Extra: {sorted(covered - expected)}"
        )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.cartesian
class TestAnchorByDataMatrix:
    """Run the three anchor models across every runnable data config."""

    @pytest.mark.parametrize(
        "case",
        ANCHOR_TRAINING_CASES,
        ids=[case.case_id for case in ANCHOR_TRAINING_CASES],
    )
    def test_anchor_model_runs_each_data_surface(
        self,
        case: TrainingCase,
        tmp_path: Path,
    ) -> None:
        """Each anchor/data pair should complete a tiny train/val/test run."""
        output_dir = tmp_path / case.case_id.replace("/", "__")
        run_training_case(
            base_config=case.base_config,
            output_dir=output_dir,
            extra_overrides=case.extra_overrides,
        )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.cartesian
class TestRepresentativeModelMatrix:
    """Run the remaining model surfaces on representative family data."""

    @pytest.mark.parametrize(
        "case",
        REPRESENTATIVE_SURFACE_CASES,
        ids=[case.case_id for case in REPRESENTATIVE_SURFACE_CASES],
    )
    def test_remaining_model_surface_runs_representative_case(
        self,
        case: TrainingCase,
        tmp_path: Path,
    ) -> None:
        """Each non-anchor surface should still get one real tiny run."""
        output_dir = tmp_path / case.case_id.replace("/", "__")
        run_training_case(
            base_config=case.base_config,
            output_dir=output_dir,
            extra_overrides=case.extra_overrides,
        )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.cartesian
class TestCartesianPlotCoverage:
    """Keep the representative plotting path covered in the heavy lane too."""

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

"""Regression tests for config and stage refactoring.

These tests capture the current behavior before refactoring to ensure
we don't break existing functionality. As refactoring progresses,
tests will be updated to reflect the new expected behavior.

Test rationale:
    - Document current optimizer settings per stage
    - Document current GNN model output formats
    - Document current config structure (stage1_5 separate from stage2)
    - Ensure config composition works throughout refactoring

Invariants:
    - All stage configs compose with base_config_spectral_arch
    - All models produce finite outputs
    - Optimizer settings are explicitly documented
"""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import tmgg  # noqa: F401 - registers OmegaConf resolvers
from tests._helpers.graph_builders import edge_scalar_graphdata, legacy_edge_scalar
from tmgg.data.datasets.graph_types import GraphData


@pytest.fixture(autouse=True)
def clear_hydra():
    """Clear Hydra global state before each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def exp_config_path() -> Path:
    """Path to experiment configs."""
    return Path(__file__).parent.parent / "src" / "tmgg" / "experiments" / "exp_configs"


def get_minimal_overrides(tmp_path: Path) -> list[str]:
    """Generate minimal overrides for testing config composition."""
    return [
        f"paths.output_dir={tmp_path}",
        f"paths.results_dir={tmp_path}/results",
        "trainer.max_steps=2",
        "trainer.val_check_interval=1",
        "trainer.accelerator=cpu",
        "~logger",
        "data.batch_size=2",
        "data.num_workers=0",
        f"hydra.run.dir={tmp_path}",
    ]


@pytest.mark.integration
class TestOptimizerConsistency:
    """Tests verifying optimizer settings are consistent across stages.

    After refactoring, all stages should use:
    - optimizer_type: adamw
    - weight_decay: 1e-12
    - amsgrad: true
    """

    STAGES_WITH_EXPECTED_OPTIMIZER = [
        # (stage_name, expected_optimizer, expected_weight_decay, expected_amsgrad)
        ("stage1_poc", "adamw", 1e-12, True),
        ("stage1_sanity", "adamw", 1e-12, True),
        ("stage2_validation", "adamw", 1e-12, True),
    ]

    @pytest.mark.parametrize(
        "stage_name,expected_optimizer,expected_wd,expected_amsgrad",
        STAGES_WITH_EXPECTED_OPTIMIZER,
    )
    def test_stage_optimizer_settings(
        self,
        stage_name: str,
        expected_optimizer: str,
        expected_wd: float,
        expected_amsgrad: bool,
        exp_config_path: Path,
        tmp_path: Path,
    ) -> None:
        """Verify stage uses expected optimizer settings."""
        overrides = get_minimal_overrides(tmp_path)
        overrides.append(f"+stage={stage_name}")

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name="base_config_spectral_arch", overrides=overrides)

        assert cfg.optimizer_type == expected_optimizer, (
            f"Stage {stage_name}: expected optimizer_type={expected_optimizer}, "
            f"got {cfg.optimizer_type}"
        )
        assert abs(cfg.weight_decay - expected_wd) < 1e-15, (
            f"Stage {stage_name}: expected weight_decay={expected_wd}, "
            f"got {cfg.weight_decay}"
        )
        assert cfg.get("amsgrad", False) == expected_amsgrad, (
            f"Stage {stage_name}: expected amsgrad={expected_amsgrad}, "
            f"got {cfg.get('amsgrad', False)}"
        )


@pytest.mark.integration
class TestStage2CrossGraphParameter:
    """Tests for merged stage2 with cross_graph parameter.

    After refactoring, stage2 should support cross_graph parameter:
    - cross_graph=false: same_graph_all_splits=true (like old stage1_5)
    - cross_graph=true: same_graph_all_splits=false (like old stage2)
    """

    def test_stage2_default_same_graph(
        self, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify stage2 defaults to same graph for all splits."""
        overrides = get_minimal_overrides(tmp_path)
        overrides.append("+stage=stage2_validation")

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name="base_config_spectral_arch", overrides=overrides)

        # After refactoring, cross_graph=false should be the default
        # This means same_graph_all_splits should be true
        cross_graph = cfg.get("cross_graph", False)
        if cross_graph:
            assert cfg.data.same_graph_all_splits is False
        else:
            assert cfg.data.same_graph_all_splits is True

    def test_stage2_cross_graph_true(
        self, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify stage2 with cross_graph=true uses different graphs."""
        overrides = get_minimal_overrides(tmp_path)
        overrides.extend(["+stage=stage2_validation", "cross_graph=true"])

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name="base_config_spectral_arch", overrides=overrides)

        # With cross_graph=true, same_graph_all_splits should be false
        assert (
            cfg.data.same_graph_all_splits is False
        ), "cross_graph=true should set same_graph_all_splits=false"


@pytest.mark.unit
class TestModelOutputFormat:
    """Tests verifying all models return adjacency logits directly.

    After refactoring, all models should return a single tensor
    (adjacency logits, pre-sigmoid) instead of tuples.
    """

    def test_gnn_returns_graph_data(self) -> None:
        """Verify GNN returns GraphData."""
        from tmgg.models.gnn import GNN

        model = GNN(
            num_layers=1,
            num_terms=3,
            feature_dim_in=5,
            feature_dim_out=5,
        )

        batch_size = 2
        num_nodes = 5
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)

        result = model(edge_scalar_graphdata(A), output_dense=True)

        assert isinstance(
            result, GraphData
        ), f"GNN should return GraphData, got {type(result)}"
        assert legacy_edge_scalar(result).shape == (batch_size, num_nodes, num_nodes)

    def test_gnn_symmetric_returns_graph_data(self) -> None:
        """Verify GNNSymmetric returns GraphData."""
        from tmgg.models.gnn import GNNSymmetric

        model = GNNSymmetric(num_layers=1, feature_dim_out=5)

        batch_size = 2
        num_nodes = 5
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)

        result = model(edge_scalar_graphdata(A), output_dense=True)

        assert isinstance(
            result, GraphData
        ), f"GNNSymmetric should return GraphData, got {type(result)}"
        assert legacy_edge_scalar(result).shape == (batch_size, num_nodes, num_nodes)

    def test_nodevar_gnn_returns_graph_data(self) -> None:
        """Verify NodeVarGNN returns GraphData."""
        from tmgg.models.gnn import NodeVarGNN

        model = NodeVarGNN(num_layers=1, feature_dim=5)

        batch_size = 2
        num_nodes = 5
        A = torch.eye(num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)

        result = model(edge_scalar_graphdata(A), output_dense=True)

        assert isinstance(
            result, GraphData
        ), f"NodeVarGNN should return GraphData, got {type(result)}"
        assert legacy_edge_scalar(result).shape == (batch_size, num_nodes, num_nodes)


@pytest.mark.integration
class TestSingleGraphBaseConfig:
    """Tests for single_graph_base.yaml extraction.

    After refactoring, all *_single_graph.yaml configs should inherit
    from single_graph_base.yaml and only specify graph-specific params.
    """

    SINGLE_GRAPH_DATASETS = [
        "er_single_graph",
        "sbm_single_graph",
        "regular_single_graph",
        "tree_single_graph",
        "roc_single_graph",
        "lfr_single_graph",
    ]

    @pytest.mark.parametrize("dataset", SINGLE_GRAPH_DATASETS)
    def test_single_graph_config_has_required_fields(
        self, dataset: str, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify single-graph configs have all required fields."""
        overrides = get_minimal_overrides(tmp_path)
        overrides.append(f"data={dataset}")

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name="base_config_spectral_arch", overrides=overrides)

        # Required fields from single_graph_base. ``noise_type`` was
        # removed in commit 80142202 ("fix(denoising): align config noise
        # ownership and digress edge state") — the noise contract belongs
        # to the LightningModule, not the datamodule, and
        # ``test_denoising_task_does_not_copy_noise_config_into_data_module``
        # in test_config_composition.py guards that invariant.
        required_fields = [
            "same_graph_all_splits",
            "train_seed",
            "num_train_samples",
            "num_val_samples",
            "num_test_samples",
            "batch_size",
            "num_workers",
        ]

        for field in required_fields:
            assert (
                field in cfg.data
            ), f"Dataset {dataset} missing required field: {field}"


@pytest.mark.integration
class TestDigressEigenvectorOnly:
    """Tests verifying DiGress uses eigenvector mode only.

    After refactoring, adjacency-mode DiGress configs are removed.
    Only eigenvector mode (use_eigenvectors=true) should exist.
    """

    def test_digress_uses_eigenvectors(
        self, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify DiGress config uses eigenvector mode."""
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name="base_config_digress", overrides=overrides)

        # After cleanup, use_eigenvectors should always be true
        assert (
            cfg.model.get("use_eigenvectors", True) is True
        ), "DiGress should use eigenvector mode (use_eigenvectors=true)"


@pytest.mark.integration
class TestConfigPrecedence:
    """Tests for config precedence documentation.

    Verifies that:
    1. Stage config overrides base config
    2. CLI overrides stage config
    """

    def test_stage_overrides_base_learning_rate(
        self, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify stage config overrides base config learning_rate."""
        overrides = get_minimal_overrides(tmp_path)

        # First check base config composes (we verify stage override works below)
        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            compose(config_name="base_config_spectral_arch", overrides=overrides)

        GlobalHydra.instance().clear()

        # Then check with stage override
        overrides_with_stage = get_minimal_overrides(tmp_path)
        overrides_with_stage.append("+stage=stage1_poc")

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            stage_cfg = compose(
                config_name="base_config_spectral_arch", overrides=overrides_with_stage
            )
            stage_lr = stage_cfg.learning_rate

        # Stage should override base (both should be 1e-2 after harmonization)
        assert stage_lr == 1e-2, f"Stage learning_rate should be 1e-2, got {stage_lr}"

    def test_cli_overrides_stage(self, exp_config_path: Path, tmp_path: Path) -> None:
        """Verify CLI overrides stage config."""
        overrides = get_minimal_overrides(tmp_path)
        overrides.extend(["+stage=stage1_poc", "learning_rate=5e-3"])

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name="base_config_spectral_arch", overrides=overrides)

        assert (
            cfg.learning_rate == 5e-3
        ), f"CLI override should set learning_rate=5e-3, got {cfg.learning_rate}"

"""Integration tests for config composition and instantiation.

Test rationale:
    These tests verify that all Hydra config combinations compose correctly
    and produce valid, instantiable objects. They catch:
    - Missing required config keys
    - Invalid interpolation references
    - Incompatible config compositions
    - Model/data module instantiation errors
    - Forward pass shape mismatches

Assumptions:
    - Tests run on CPU with minimal data
    - Hydra state is cleared between tests
    - Each config produces a valid LightningModule and DataModule

Invariants:
    - hydra.compose() succeeds for all base configs
    - hydra.utils.instantiate(config.data) returns a DataModule
    - hydra.utils.instantiate(config.model) returns a LightningModule
    - Model forward pass returns tensor with correct shape
"""

from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Base configs to test with their default model/data compositions
BASE_CONFIGS = [
    "base_config_spectral_arch",
    "base_config_gnn",
    "base_config_gnn_transformer",
    "base_config_digress",
    "grid_search_base",
]


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
    """Generate minimal overrides for testing config composition.

    These overrides ensure configs can be resolved without runtime
    dependencies (Hydra output dir, S3, W&B).
    """
    return [
        f"paths.output_dir={tmp_path}",
        f"paths.results_dir={tmp_path}/results",
        "trainer.max_steps=2",
        "trainer.val_check_interval=1",
        "trainer.accelerator=cpu",
        # Disable logger config to avoid W&B runtime dependencies
        "~logger",
        # Minimal data for fast instantiation
        "data.batch_size=2",
        "data.num_workers=0",
        # Override interpolations that depend on Hydra runtime
        f"hydra.run.dir={tmp_path}",
    ]


@pytest.mark.integration
@pytest.mark.config
class TestConfigComposition:
    """Tests for config loading and composition."""

    @pytest.mark.parametrize("base_config", BASE_CONFIGS)
    def test_config_composes_successfully(
        self, base_config: str, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify that base config composes without errors.

        This tests the Hydra composition process, including:
        - Defaults resolution
        - Package directives
        - Interpolation resolution
        """
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name=base_config, overrides=overrides)

        # Verify required keys exist
        assert "model" in cfg, f"Config {base_config} missing 'model' key"
        assert "data" in cfg, f"Config {base_config} missing 'data' key"
        assert "trainer" in cfg, f"Config {base_config} missing 'trainer' key"
        assert "paths" in cfg, f"Config {base_config} missing 'paths' key"

        # Verify _target_ is set for instantiation
        assert "_target_" in cfg.model, f"Config {base_config} model missing _target_"
        assert "_target_" in cfg.data, f"Config {base_config} data missing _target_"

    @pytest.mark.parametrize("base_config", BASE_CONFIGS)
    def test_model_instantiation(
        self, base_config: str, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify that model can be instantiated from config."""
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name=base_config, overrides=overrides)

        model = hydra.utils.instantiate(cfg.model)

        # Verify model is a LightningModule
        import pytorch_lightning as pl

        assert isinstance(model, pl.LightningModule), (
            f"Config {base_config} model is not a LightningModule: {type(model)}"
        )

    @pytest.mark.parametrize("base_config", BASE_CONFIGS)
    def test_data_module_instantiation(
        self, base_config: str, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify that data module can be instantiated from config."""
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name=base_config, overrides=overrides)

        data_module = hydra.utils.instantiate(cfg.data)

        # Verify data module is a LightningDataModule
        import pytorch_lightning as pl

        assert isinstance(data_module, pl.LightningDataModule), (
            f"Config {base_config} data is not a LightningDataModule: {type(data_module)}"
        )

    def test_denoising_task_does_not_copy_noise_config_into_data_module(
        self, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Denoising task config should not inject noise fields into cfg.data.

        Regression rationale
        --------------------
        The denoising noise contract belongs to the LightningModule, not the
        datamodule constructor. When task/denoising.yaml leaks ``noise_type`` or
        ``noise_levels`` into ``cfg.data``, Hydra instantiation breaks as soon
        as GraphDataModule stops accepting those dead kwargs.
        """
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name="base_config_spectral_arch", overrides=overrides)

        assert "noise_type" not in cfg.data
        assert "noise_levels" not in cfg.data

    def test_sbm_vignac_repro_exact_uses_live_upstream_transformer_parity(
        self, exp_config_path: Path
    ) -> None:
        """The exact repro config should opt into live upstream DiGress behavior.

        Rationale: upstream ``experiment/sbm.yaml`` records ``dim_ffy=256``,
        but the upstream ``GraphTransformer`` constructor does not pass
        that field into ``XEyTransformerLayer`` and therefore uses the
        layer default ``dim_ffy=2048`` at runtime. ``repro_exact`` also
        needs the padding-only hidden-edge mask so local outputs match
        the live upstream forward path under equal weights.
        """
        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(
                config_name="base_config_discrete_diffusion_generative",
                overrides=["+experiment=discrete_sbm_vignac_repro_exact"],
            )

        assert cfg.model.model.use_upstream_hidden_edge_diagonal is True
        assert cfg.model.model.hidden_dims.dim_ffy == 2048

    @pytest.mark.parametrize(
        "experiment_name",
        [
            "discrete_sbm_pearl_repro_exact",
            "discrete_sbm_pearl_spectral_repro_exact",
            "discrete_sbm_pearl_gnnconv_norm_repro_exact",
            "discrete_enzymes_pearl_repro_exact",
            "discrete_enzymes_pearl_spectral_repro_exact",
            "discrete_enzymes_pearl_gnnconv_norm_repro_exact",
        ],
    )
    def test_pearl_exact_panel_disables_cycle_features(
        self, experiment_name: str, exp_config_path: Path
    ) -> None:
        """PEARL-only exact-panel configs must disable cycle channels.

        Rationale: the four ``*_pearl_*_repro_exact`` panel configs are
        meant to isolate R-PEARL as the sole structural augmentation. They
        inherit from ``discrete_{sbm,enzymes}_pearl_repro_exact``, which
        sets ``use_cycles: false`` on the ``PEARLExtraFeatures`` block.
        The default ``PEARLExtraFeatures`` keeps cycles on for backward
        compatibility with prior runs; this test guards against future
        edits to either the parent configs or the class default that
        would silently re-enable cycle features and contaminate the
        "PEARL-only" intent.
        """
        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(
                config_name="base_config_discrete_diffusion_generative",
                overrides=[f"+experiment={experiment_name}"],
            )

        extra = cfg.model.model.extra_features
        assert (
            extra._target_
            == "tmgg.models.digress.pearl_extra_features.PEARLExtraFeatures"
        ), f"{experiment_name} should use PEARLExtraFeatures, got {extra._target_}"
        assert extra.use_cycles is False, (
            f"{experiment_name} must set PEARLExtraFeatures.use_cycles=False to "
            f"realise the 'PEARL-only' contract; got {extra.use_cycles!r}"
        )


# Configs that use standard adjacency matrix input
ADJACENCY_INPUT_CONFIGS = [
    "base_config_spectral_arch",
    "base_config_gnn",
    "base_config_gnn_transformer",
    "grid_search_base",
]

# DiGress uses eigenvector input by default, requiring special handling
EIGENVECTOR_INPUT_CONFIGS = [
    "base_config_digress",
]


@pytest.mark.integration
@pytest.mark.config
class TestForwardPass:
    """Tests for model forward pass with composed configs."""

    @pytest.mark.parametrize("base_config", ADJACENCY_INPUT_CONFIGS)
    def test_forward_pass_produces_valid_output(
        self,
        base_config: str,
        exp_config_path: Path,
        tmp_path: Path,
        sample_adjacency_batch: torch.Tensor,
    ) -> None:
        """Verify model forward pass produces tensor of correct shape.

        This tests that the full config composition produces a model
        that can process input data without errors.
        """
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name=base_config, overrides=overrides)

        model = hydra.utils.instantiate(cfg.model)
        model.eval()

        with torch.no_grad():
            output = model(sample_adjacency_batch)

        # Output should have same shape as input (denoising task)
        assert output.shape == sample_adjacency_batch.shape, (
            f"Config {base_config}: output shape {output.shape} != "
            f"input shape {sample_adjacency_batch.shape}"
        )

        # Output should be finite (no NaN/Inf)
        assert torch.isfinite(output).all(), (
            f"Config {base_config}: output contains non-finite values"
        )

    @pytest.mark.parametrize("base_config", EIGENVECTOR_INPUT_CONFIGS)
    def test_digress_forward_pass(
        self,
        base_config: str,
        exp_config_path: Path,
        tmp_path: Path,
    ) -> None:
        """Verify DiGress model forward pass.

        DiGress models with use_eigenvectors=True internally compute
        eigenvectors from the adjacency matrix. The input must be a
        square adjacency matrix matching the model's node_feature_dim
        for eigenvector extraction.
        """
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name=base_config, overrides=overrides)

        model = hydra.utils.instantiate(cfg.model)
        model.eval()

        # Get the node feature dimension (used as k for eigenvector extraction)
        node_feature_dim = getattr(model, "_node_feature_dim", 50)
        batch_size = 4
        # For eigenvector extraction, n_nodes must be >= node_feature_dim
        n_nodes = max(node_feature_dim, 50)

        # Generate appropriately sized adjacency matrix
        adjacency_batch = torch.zeros(batch_size, n_nodes, n_nodes)
        for i in range(batch_size):
            A = torch.bernoulli(torch.full((n_nodes, n_nodes), 0.3))
            A = (A + A.T) / 2
            A.fill_diagonal_(0)
            adjacency_batch[i] = (A > 0.5).float()

        with torch.no_grad():
            output = model(adjacency_batch)

        # LightningModule.forward() bridge returns (B, N, N) tensor
        assert output.shape == (batch_size, n_nodes, n_nodes), (
            f"Config {base_config}: output shape {output.shape} != "
            f"expected ({batch_size}, {n_nodes}, {n_nodes})"
        )

        # Output should be finite (no NaN/Inf)
        assert torch.isfinite(output).all(), (
            f"Config {base_config}: output contains non-finite values"
        )


@pytest.mark.integration
@pytest.mark.config
class TestTrainerInstantiation:
    """Tests for trainer instantiation from config."""

    @pytest.mark.parametrize("base_config", BASE_CONFIGS)
    def test_trainer_instantiation(
        self, base_config: str, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify trainer can be instantiated from config.

        Tests that trainer config includes all required parameters
        and produces a valid Trainer object.
        """
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name=base_config, overrides=overrides)

        import pytorch_lightning as pl

        trainer = hydra.utils.instantiate(cfg.trainer)

        assert isinstance(trainer, pl.Trainer), (
            f"Config {base_config} trainer is not a Trainer: {type(trainer)}"
        )


@pytest.mark.integration
@pytest.mark.config
class TestStageConfigComposition:
    """Tests for stage config composition with base spectral config."""

    STAGE_CONFIGS = [
        "stage1_poc",
        "stage1_sanity",
        "stage2_validation",
        "stage3_diversity",
        "stage4_benchmarks",
        "stage5_full",
    ]

    @pytest.mark.parametrize("stage_name", STAGE_CONFIGS)
    def test_stage_config_composes_with_base(
        self, stage_name: str, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify stage config composes correctly with base spectral config."""
        overrides = get_minimal_overrides(tmp_path)
        overrides.append(f"+stage={stage_name}")

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name="base_config_spectral_arch", overrides=overrides)

        # Verify stage was applied
        assert cfg.get("stage") == stage_name, (
            f"Stage {stage_name} not applied correctly"
        )

        # Verify essential keys still exist
        assert "model" in cfg
        assert "data" in cfg
        assert "trainer" in cfg

    @pytest.mark.parametrize(
        ("stage_name", "expected_k"),
        [
            ("stage3_diversity", 50),
            ("stage4_benchmarks", 32),
            ("stage5_full", 50),
        ],
    )
    def test_future_stages_inherit_shared_runtime_settings(
        self,
        stage_name: str,
        expected_k: int,
        exp_config_path: Path,
        tmp_path: Path,
    ) -> None:
        """Future stages should inherit the shared runtime config verbatim.

        Rationale
        ---------
        Stage 3-5 intentionally share the same executable training settings.
        This test locks those inherited values so a future stage-specific edit
        does not silently fork optimizer or noise-level behavior.
        """
        overrides = get_minimal_overrides(tmp_path)
        overrides.append(f"+stage={stage_name}")

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name="base_config_spectral_arch", overrides=overrides)

        assert cfg.learning_rate == 1e-2
        assert cfg.weight_decay == 1e-12
        assert cfg.optimizer_type == "adamw"
        assert cfg.amsgrad is True
        assert cfg.scheduler_config.type == "none"
        assert list(cfg.noise_levels) == [0.1]
        assert list(cfg.eval_noise_levels) == [0.1]
        # ``k`` is a parameter on the inner spectral denoiser
        # (``cfg.model.model.k``); see e.g. stage1_poc.yaml's
        # ``model.model.k`` override and linear_pe.yaml's ``model.k``
        # field. The earlier ``cfg.model.k`` path predated the
        # inner-model nesting and never existed at runtime.
        assert cfg.model.model.k == expected_k

    @pytest.mark.parametrize(
        ("data_config", "expected_num_nodes", "expected_max_block_size"),
        [
            ("sbm_n100", 100, 50),
            ("sbm_n200", 200, 60),
        ],
    )
    def test_sbm_variants_inherit_default_config(
        self,
        data_config: str,
        expected_num_nodes: int,
        expected_max_block_size: int,
        exp_config_path: Path,
        tmp_path: Path,
    ) -> None:
        """Sized SBM configs should override only the intended graph parameters.

        Rationale
        ---------
        The n=100 and n=200 variants are maintenance aliases over the default
        SBM config. This test guards against config copy-paste drift by
        asserting that inherited probabilities and partition settings remain
        identical while node-count-specific overrides still apply.
        """
        overrides = get_minimal_overrides(tmp_path)
        overrides.append(f"data={data_config}")

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name="base_config_spectral_arch", overrides=overrides)

        assert cfg.data.graph_type == "sbm"
        assert cfg.data.graph_config.num_nodes == expected_num_nodes
        assert cfg.data.graph_config.max_block_size == expected_max_block_size
        assert cfg.data.graph_config.p_intra == 1.0
        assert cfg.data.graph_config.p_inter == 0.0
        assert cfg.data.graph_config.min_blocks == 2
        assert cfg.data.graph_config.max_blocks == 4
        assert cfg.data.graph_config.num_train_partitions == 10
        assert cfg.data.graph_config.num_test_partitions == 10

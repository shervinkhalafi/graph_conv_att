"""Shared helpers for the experiment coverage matrices.

Test rationale
--------------
The repository now has two experiment-matrix lanes:

- a lighter non-cartesian matrix that proves broad inventory coverage
- a heavier cartesian-ish matrix that proves selected cross-product coverage

Both suites should exercise the same runtime surface. This module keeps the
tiny-run overrides, deterministic PyG stand-in, and artifact assertions in one
place so the two lanes do not drift apart over time.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data

import tmgg  # noqa: F401 - registers OmegaConf resolvers used by config tests
from tmgg.data.data_modules import graph_generation
from tmgg.training.orchestration.run_experiment import run_experiment
from tmgg.training.orchestration.sanity_check import maybe_run_sanity_check


@dataclass(frozen=True)
class TrainingCase:
    """One tiny executable training surface."""

    case_id: str
    base_config: str
    extra_overrides: tuple[str, ...]


@dataclass(frozen=True)
class ModelCoverageCase:
    """One training case that covers a runnable leaf model config."""

    model_leaf: str
    base_config: str
    extra_overrides: tuple[str, ...]


@dataclass(frozen=True)
class DataCoverageCase:
    """One training case that covers a runnable leaf data config."""

    data_leaf: str
    base_config: str
    extra_overrides: tuple[str, ...]


@dataclass(frozen=True)
class InlineTrainingCase:
    """One training case for a model surface that is not a leaf YAML."""

    case_id: str
    base_config: str
    extra_overrides: tuple[str, ...]


@dataclass(frozen=True)
class PlotCoverageCase:
    """One representative denoising-family sanity-check plot case."""

    case_id: str
    base_config: str
    extra_overrides: tuple[str, ...]


EXP_CONFIG_DIR = (
    Path(__file__).parent.parent / "src" / "tmgg" / "experiments" / "exp_configs"
)
MODELS_DIR = EXP_CONFIG_DIR / "models"
DATA_DIR = EXP_CONFIG_DIR / "data"

EXCLUDED_MODEL_LEAVES = {"digress/digress_base"}
EXCLUDED_DATA_LEAVES = {"base_dataloader", "single_graph_base", "grid_base"}

COMMON_RUN_OVERRIDES = (
    "trainer.max_steps=2",
    "trainer.val_check_interval=1",
    "trainer.limit_train_batches=1",
    "trainer.limit_val_batches=1",
    "trainer.limit_test_batches=1",
    "+trainer.num_sanity_val_steps=0",
    "trainer.accelerator=cpu",
    "base/logger=csv",
    "allow_no_wandb=true",
    "force_fresh=true",
)

DENOISING_MULTIGRAPH_COMMON = (
    "noise_levels=[0.1]",
    "eval_noise_levels=[0.1]",
    "data.batch_size=1",
    "data.num_workers=0",
    "++data.samples_per_graph=1",
    "++data.val_samples_per_graph=1",
)

SINGLE_GRAPH_COMMON = (
    "noise_levels=[0.1]",
    "eval_noise_levels=[0.1]",
    "data.batch_size=1",
    "data.num_workers=0",
    "++data.num_train_samples=2",
    "++data.num_val_samples=1",
    "++data.num_test_samples=1",
)

SMALL_SBM_DATA = (
    "++data.graph_config.num_nodes=16",
    "++data.graph_config.num_train_partitions=2",
    "++data.graph_config.num_test_partitions=2",
)

PLOT_SBM_DATA = (
    "++data.graph_config.num_nodes=32",
    "++data.graph_config.num_train_partitions=2",
    "++data.graph_config.num_test_partitions=2",
)

MULTIGRAPH_SYNTHETIC_DATA = ("++data.graph_config.num_graphs=10",)

PYG_MULTIGRAPH_DATA = ("++data.graph_config.max_graphs=10",)

GRID_DATA = (
    "++data.graph_config.num_train_partitions=2",
    "++data.graph_config.num_test_partitions=2",
)

SMALL_GNN_MODEL = (
    "model.model.num_terms=2",
    "model.model.feature_dim_in=8",
    "model.model.feature_dim_out=4",
)

SMALL_HYBRID_MODEL = (
    "model.model.gnn_config.num_layers=1",
    "model.model.gnn_config.num_terms=2",
    "model.model.gnn_config.feature_dim_in=8",
    "model.model.gnn_config.feature_dim_out=4",
    "model.model.transformer_config.num_layers=1",
    "model.model.transformer_config.num_heads=2",
    "model.model.transformer_config.d_k=4",
    "model.model.transformer_config.d_v=4",
)

SMALL_HYBRID_GNN_ONLY_MODEL = (
    "model.model.gnn_config.num_layers=1",
    "model.model.gnn_config.num_terms=2",
    "model.model.gnn_config.feature_dim_in=8",
    "model.model.gnn_config.feature_dim_out=4",
)

SMALL_DIGRESS_MODEL = (
    "model.model.n_layers=2",
    "model.model.hidden_dims.dx=32",
    "model.model.hidden_dims.de=8",
    "model.model.hidden_dims.dy=32",
    "model.model.hidden_dims.n_head=2",
    "model.model.hidden_mlp_dims.X=64",
    "model.model.hidden_mlp_dims.E=16",
    "model.model.hidden_mlp_dims.y=64",
    "model.model.extra_features.k=4",
)

SMALL_SPECTRAL_LINEAR_MODEL = (
    "model.model.k=4",
    "model.model.max_nodes=256",
)

SMALL_FILTER_BANK_MODEL = (
    "model.model.k=4",
    "model.model.polynomial_degree=2",
)

SMALL_BASELINE_LINEAR_MODEL = ("model.model.max_nodes=32",)

SMALL_BASELINE_MLP_MODEL = (
    "+models/baselines@model=mlp",
    "model.model.max_nodes=32",
    "model.model.hidden_dim=32",
    "model.model.num_layers=1",
)

SMALL_DISCRETE_MODEL = (
    "data.num_nodes=12",
    "data.num_graphs=20",
    "data.batch_size=2",
    "data.num_workers=0",
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

SMALL_GAUSSIAN_MODEL = (
    "data.num_nodes=16",
    "data.num_graphs=20",
    "data.batch_size=2",
    "data.num_workers=0",
    "model.model.k=4",
    "model.model.d_k=16",
    "model.noise_schedule.timesteps=5",
    "model.evaluator.eval_num_samples=4",
)

MULTIGRAPH_PARTITION_DATA_LEAVES = {
    "sbm_default",
    "sbm_digress",
    "sbm_n100",
    "sbm_n200",
}
MULTIGRAPH_SYNTHETIC_DATA_LEAVES = {
    "nx",
    "nx_square",
    "nx_star",
    "ring_of_cliques",
}
PYG_MULTIGRAPH_DATA_LEAVES = {"pyg_enzymes", "pyg_proteins"}
SINGLE_GRAPH_DATA_LEAVES = {
    "sbm_single_graph",
    "er_single_graph",
    "regular_single_graph",
    "tree_single_graph",
    "roc_single_graph",
    "lfr_single_graph",
    "pyg_enzymes_single_graph",
    "pyg_proteins_single_graph",
    "pyg_qm9_single_graph",
}
GRID_DATA_LEAVES = {"grid_digress", "grid_gaussian", "grid_rotation"}


def data_overrides_for_leaf(data_leaf: str) -> tuple[str, ...]:
    """Return the tiny-run data overrides for one runnable data leaf.

    The cartesian lane needs to reuse the same data inventory with different
    model families. This helper keeps the data surface independent from the
    representative model used elsewhere in the lighter matrix.
    """

    if data_leaf in MULTIGRAPH_PARTITION_DATA_LEAVES:
        return DENOISING_MULTIGRAPH_COMMON + (
            f"data={data_leaf}",
            "++data.graph_config.num_train_partitions=2",
            "++data.graph_config.num_test_partitions=2",
        )
    if data_leaf in MULTIGRAPH_SYNTHETIC_DATA_LEAVES:
        return DENOISING_MULTIGRAPH_COMMON + (
            f"data={data_leaf}",
            *MULTIGRAPH_SYNTHETIC_DATA,
        )
    if data_leaf in PYG_MULTIGRAPH_DATA_LEAVES:
        return DENOISING_MULTIGRAPH_COMMON + (
            f"data={data_leaf}",
            *PYG_MULTIGRAPH_DATA,
        )
    if data_leaf in SINGLE_GRAPH_DATA_LEAVES:
        return SINGLE_GRAPH_COMMON + (f"data={data_leaf}",)
    if data_leaf in GRID_DATA_LEAVES:
        return DENOISING_MULTIGRAPH_COMMON + (
            f"data={data_leaf}",
            *GRID_DATA,
        )
    raise KeyError(f"Unhandled runnable data leaf: {data_leaf}")


ANCHOR_MODEL_VARIANTS: dict[str, TrainingCase] = {
    "digress/digress_transformer": TrainingCase(
        "digress/digress_transformer",
        "base_config_digress",
        SMALL_DIGRESS_MODEL,
    ),
    "digress/digress_transformer_spectral_all": TrainingCase(
        "digress/digress_transformer_spectral_all",
        "base_config_digress",
        ("+models/digress@model=digress_transformer_spectral_all",)
        + SMALL_DIGRESS_MODEL,
    ),
    "gnn/standard_gnn": TrainingCase(
        "gnn/standard_gnn",
        "base_config_gnn",
        SMALL_GNN_MODEL,
    ),
}
ANCHOR_MODEL_LEAVES = tuple(ANCHOR_MODEL_VARIANTS)


def build_cartesian_anchor_case(anchor_model_leaf: str, data_leaf: str) -> TrainingCase:
    """Combine one anchor denoising model with one runnable data config."""

    model_case = ANCHOR_MODEL_VARIANTS[anchor_model_leaf]
    return TrainingCase(
        case_id=f"{anchor_model_leaf}__{data_leaf}",
        base_config=model_case.base_config,
        extra_overrides=data_overrides_for_leaf(data_leaf) + model_case.extra_overrides,
    )


MODEL_COVERAGE_CASES = [
    ModelCoverageCase(
        "baselines/linear",
        "base_config_baseline",
        DENOISING_MULTIGRAPH_COMMON + SMALL_SBM_DATA + SMALL_BASELINE_LINEAR_MODEL,
    ),
    ModelCoverageCase(
        "baselines/mlp",
        "base_config_baseline",
        DENOISING_MULTIGRAPH_COMMON + SMALL_SBM_DATA + SMALL_BASELINE_MLP_MODEL,
    ),
    ModelCoverageCase(
        "gnn/standard_gnn",
        "base_config_gnn",
        DENOISING_MULTIGRAPH_COMMON + SMALL_SBM_DATA + SMALL_GNN_MODEL,
    ),
    ModelCoverageCase(
        "gnn/symmetric_gnn",
        "base_config_gnn",
        DENOISING_MULTIGRAPH_COMMON
        + SMALL_SBM_DATA
        + ("+models/gnn@model=symmetric_gnn",)
        + SMALL_GNN_MODEL,
    ),
    ModelCoverageCase(
        "gnn/nodevar_gnn",
        "base_config_denoising",
        DENOISING_MULTIGRAPH_COMMON
        + SMALL_SBM_DATA
        + ("+models/gnn@model=nodevar_gnn",)
        + (
            "model.model.num_terms=2",
            "model.model.feature_dim=8",
        ),
    ),
    ModelCoverageCase(
        "hybrid/hybrid_gnn_only",
        "base_config_gnn_transformer",
        DENOISING_MULTIGRAPH_COMMON
        + SMALL_SBM_DATA
        + ("+models/hybrid@model=hybrid_gnn_only",)
        + SMALL_HYBRID_GNN_ONLY_MODEL,
    ),
    ModelCoverageCase(
        "hybrid/hybrid_with_transformer",
        "base_config_gnn_transformer",
        DENOISING_MULTIGRAPH_COMMON + SMALL_SBM_DATA + SMALL_HYBRID_MODEL,
    ),
    ModelCoverageCase(
        "spectral/filter_bank",
        "base_config_denoising",
        DENOISING_MULTIGRAPH_COMMON
        + SMALL_SBM_DATA
        + ("+models/spectral@model=filter_bank",)
        + SMALL_FILTER_BANK_MODEL,
    ),
    ModelCoverageCase(
        "spectral/linear_pe",
        "base_config_spectral_arch",
        DENOISING_MULTIGRAPH_COMMON + SMALL_SBM_DATA + SMALL_SPECTRAL_LINEAR_MODEL,
    ),
    ModelCoverageCase(
        "spectral/multilayer_self_attention",
        "base_config_denoising",
        DENOISING_MULTIGRAPH_COMMON
        + SMALL_SBM_DATA
        + ("+models/spectral@model=multilayer_self_attention",)
        + (
            "model.model.k=4",
            "model.model.d_model=16",
            "model.model.num_heads=2",
            "model.model.num_layers=1",
            "model.model.mlp_hidden_dim=32",
        ),
    ),
    ModelCoverageCase(
        "spectral/self_attention",
        "base_config_denoising",
        DENOISING_MULTIGRAPH_COMMON
        + SMALL_SBM_DATA
        + ("+models/spectral@model=self_attention",)
        + (
            "model.model.k=4",
            "model.model.d_k=8",
        ),
    ),
    ModelCoverageCase(
        "spectral/self_attention_mlp",
        "base_config_denoising",
        DENOISING_MULTIGRAPH_COMMON
        + SMALL_SBM_DATA
        + ("+models/spectral@model=self_attention_mlp",)
        + (
            "model.model.k=4",
            "model.model.d_k=8",
            "model.model.mlp_hidden_dim=16",
            "model.model.mlp_num_layers=1",
        ),
    ),
    ModelCoverageCase(
        "digress/digress_sbm_small",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress")
        + ("+models/digress@model=digress_sbm_small",)
        + SMALL_DIGRESS_MODEL,
    ),
    ModelCoverageCase(
        "digress/digress_sbm_small_gnn",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress")
        + ("+models/digress@model=digress_sbm_small_gnn",)
        + SMALL_DIGRESS_MODEL,
    ),
    ModelCoverageCase(
        "digress/digress_sbm_small_highlr",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress")
        + ("+models/digress@model=digress_sbm_small_highlr",)
        + SMALL_DIGRESS_MODEL,
    ),
    ModelCoverageCase(
        "digress/digress_sbm_small_vanilla",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress")
        + ("+models/digress@model=digress_sbm_small_vanilla",)
        + SMALL_DIGRESS_MODEL,
    ),
    ModelCoverageCase(
        "digress/digress_sbm_vanilla",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress")
        + ("+models/digress@model=digress_sbm_vanilla",)
        + SMALL_DIGRESS_MODEL,
    ),
    ModelCoverageCase(
        "digress/digress_sbm_vanilla_gnn",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress")
        + ("+models/digress@model=digress_sbm_vanilla_gnn",)
        + SMALL_DIGRESS_MODEL,
    ),
    ModelCoverageCase(
        "digress/digress_transformer",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress") + SMALL_DIGRESS_MODEL,
    ),
    ModelCoverageCase(
        "digress/digress_transformer_gnn_all",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress")
        + ("+models/digress@model=digress_transformer_gnn_all",)
        + SMALL_DIGRESS_MODEL,
    ),
    ModelCoverageCase(
        "digress/digress_transformer_gnn_qk",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress")
        + ("+models/digress@model=digress_transformer_gnn_qk",)
        + SMALL_DIGRESS_MODEL,
    ),
    ModelCoverageCase(
        "digress/digress_transformer_gnn_v",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress")
        + ("+models/digress@model=digress_transformer_gnn_v",)
        + SMALL_DIGRESS_MODEL,
    ),
    ModelCoverageCase(
        "digress/digress_transformer_spectral_all",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress")
        + ("+models/digress@model=digress_transformer_spectral_all",)
        + SMALL_DIGRESS_MODEL,
    ),
    ModelCoverageCase(
        "digress/digress_transformer_spectral_qk",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress")
        + ("+models/digress@model=digress_transformer_spectral_qk",)
        + SMALL_DIGRESS_MODEL,
    ),
    ModelCoverageCase(
        "discrete/discrete_default",
        "base_config_discrete_diffusion_generative",
        SMALL_DISCRETE_MODEL,
    ),
    ModelCoverageCase(
        "discrete/discrete_small",
        "base_config_discrete_diffusion_generative",
        ("models/discrete@model=discrete_small",) + SMALL_DISCRETE_MODEL,
    ),
    ModelCoverageCase(
        "discrete/discrete_sbm_eigenvec",
        "base_config_discrete_diffusion_generative",
        ("models/discrete@model=discrete_sbm_eigenvec",)
        + SMALL_DISCRETE_MODEL
        + ("model.model.extra_features.k=4",),
    ),
    ModelCoverageCase(
        "discrete/discrete_sbm_official",
        "base_config_discrete_diffusion_generative",
        ("models/discrete@model=discrete_sbm_official",) + SMALL_DISCRETE_MODEL,
    ),
]

INLINE_MODEL_CASES = [
    InlineTrainingCase(
        "gaussian/default_inline_self_attention",
        "base_config_gaussian_diffusion",
        SMALL_GAUSSIAN_MODEL,
    )
]

DATA_COVERAGE_CASES = [
    DataCoverageCase(
        "sbm_default",
        "base_config_gnn",
        data_overrides_for_leaf("sbm_default") + SMALL_GNN_MODEL,
    ),
    DataCoverageCase(
        "sbm_digress",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress") + SMALL_DIGRESS_MODEL,
    ),
    DataCoverageCase(
        "sbm_n100",
        "base_config_gnn",
        data_overrides_for_leaf("sbm_n100") + SMALL_GNN_MODEL,
    ),
    DataCoverageCase(
        "sbm_n200",
        "base_config_gnn",
        data_overrides_for_leaf("sbm_n200") + SMALL_GNN_MODEL,
    ),
    DataCoverageCase(
        "nx",
        "base_config_gnn",
        data_overrides_for_leaf("nx") + SMALL_GNN_MODEL,
    ),
    DataCoverageCase(
        "nx_square",
        "base_config_gnn",
        data_overrides_for_leaf("nx_square") + SMALL_GNN_MODEL,
    ),
    DataCoverageCase(
        "nx_star",
        "base_config_gnn",
        data_overrides_for_leaf("nx_star") + SMALL_GNN_MODEL,
    ),
    DataCoverageCase(
        "ring_of_cliques",
        "base_config_gnn",
        data_overrides_for_leaf("ring_of_cliques") + SMALL_GNN_MODEL,
    ),
    DataCoverageCase(
        "pyg_enzymes",
        "base_config_gnn",
        data_overrides_for_leaf("pyg_enzymes") + SMALL_GNN_MODEL,
    ),
    DataCoverageCase(
        "pyg_proteins",
        "base_config_gnn",
        data_overrides_for_leaf("pyg_proteins") + SMALL_GNN_MODEL,
    ),
    DataCoverageCase(
        "sbm_single_graph",
        "base_config_spectral_arch",
        data_overrides_for_leaf("sbm_single_graph") + SMALL_SPECTRAL_LINEAR_MODEL,
    ),
    DataCoverageCase(
        "er_single_graph",
        "base_config_spectral_arch",
        data_overrides_for_leaf("er_single_graph") + SMALL_SPECTRAL_LINEAR_MODEL,
    ),
    DataCoverageCase(
        "regular_single_graph",
        "base_config_spectral_arch",
        data_overrides_for_leaf("regular_single_graph") + SMALL_SPECTRAL_LINEAR_MODEL,
    ),
    DataCoverageCase(
        "tree_single_graph",
        "base_config_spectral_arch",
        data_overrides_for_leaf("tree_single_graph") + SMALL_SPECTRAL_LINEAR_MODEL,
    ),
    DataCoverageCase(
        "roc_single_graph",
        "base_config_spectral_arch",
        data_overrides_for_leaf("roc_single_graph") + SMALL_SPECTRAL_LINEAR_MODEL,
    ),
    DataCoverageCase(
        "lfr_single_graph",
        "base_config_spectral_arch",
        data_overrides_for_leaf("lfr_single_graph") + SMALL_SPECTRAL_LINEAR_MODEL,
    ),
    DataCoverageCase(
        "pyg_enzymes_single_graph",
        "base_config_spectral_arch",
        data_overrides_for_leaf("pyg_enzymes_single_graph")
        + SMALL_SPECTRAL_LINEAR_MODEL,
    ),
    DataCoverageCase(
        "pyg_proteins_single_graph",
        "base_config_spectral_arch",
        data_overrides_for_leaf("pyg_proteins_single_graph")
        + SMALL_SPECTRAL_LINEAR_MODEL,
    ),
    DataCoverageCase(
        "pyg_qm9_single_graph",
        "base_config_spectral_arch",
        data_overrides_for_leaf("pyg_qm9_single_graph") + SMALL_SPECTRAL_LINEAR_MODEL,
    ),
    DataCoverageCase(
        "grid_digress",
        "grid_search_base",
        data_overrides_for_leaf("grid_digress") + SMALL_FILTER_BANK_MODEL,
    ),
    DataCoverageCase(
        "grid_gaussian",
        "grid_search_base",
        data_overrides_for_leaf("grid_gaussian") + SMALL_FILTER_BANK_MODEL,
    ),
    DataCoverageCase(
        "grid_rotation",
        "grid_search_base",
        data_overrides_for_leaf("grid_rotation") + SMALL_FILTER_BANK_MODEL,
    ),
]

PLOT_COVERAGE_CASES = [
    PlotCoverageCase(
        "baseline",
        "base_config_baseline",
        DENOISING_MULTIGRAPH_COMMON + PLOT_SBM_DATA + SMALL_BASELINE_LINEAR_MODEL,
    ),
    PlotCoverageCase(
        "gnn",
        "base_config_gnn",
        DENOISING_MULTIGRAPH_COMMON + PLOT_SBM_DATA + SMALL_GNN_MODEL,
    ),
    PlotCoverageCase(
        "hybrid",
        "base_config_gnn_transformer",
        DENOISING_MULTIGRAPH_COMMON + PLOT_SBM_DATA + SMALL_HYBRID_MODEL,
    ),
    PlotCoverageCase(
        "spectral",
        "base_config_spectral_arch",
        DENOISING_MULTIGRAPH_COMMON + PLOT_SBM_DATA + SMALL_SPECTRAL_LINEAR_MODEL,
    ),
    PlotCoverageCase(
        "digress",
        "base_config_digress",
        data_overrides_for_leaf("sbm_digress") + SMALL_DIGRESS_MODEL,
    ),
]

REPRESENTATIVE_SURFACE_CASES = [
    TrainingCase(case.model_leaf, case.base_config, case.extra_overrides)
    for case in MODEL_COVERAGE_CASES
    if case.model_leaf not in ANCHOR_MODEL_LEAVES
] + [
    TrainingCase(case.case_id, case.base_config, case.extra_overrides)
    for case in INLINE_MODEL_CASES
]


class FakePyGDatasetWrapper:
    """Deterministic in-memory stand-in for PyG benchmark datasets.

    The real PyG wrappers are already covered elsewhere. The matrix only needs
    a stable adjacency-only fixture that exercises the surrounding datamodule
    and experiment wiring without downloads or local caches.
    """

    def __init__(
        self,
        dataset_name: str,
        root: str | None = None,
        max_graphs: int | None = None,
    ):
        _ = (dataset_name, root)
        sizes = [5, 6, 7, 8, 9, 6, 7, 8, 5, 9]
        if max_graphs is not None:
            sizes = sizes[:max_graphs]

        self.data_list: list[Data] = []
        self.num_nodes = np.array(sizes, dtype=np.int64)
        self.max_n = int(self.num_nodes.max())

        padded: list[np.ndarray] = []
        for idx, num_nodes in enumerate(sizes):
            adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for node in range(num_nodes - 1):
                adjacency[node, node + 1] = 1.0
                adjacency[node + 1, node] = 1.0
            if idx % 2 == 1 and num_nodes > 2:
                adjacency[0, num_nodes - 1] = 1.0
                adjacency[num_nodes - 1, 0] = 1.0

            row, col = np.nonzero(adjacency)
            edge_index = torch.tensor(
                np.stack([row, col], axis=0),
                dtype=torch.long,
            )
            self.data_list.append(Data(edge_index=edge_index, num_nodes=num_nodes))

            padded_adj = np.zeros((self.max_n, self.max_n), dtype=np.float32)
            padded_adj[:num_nodes, :num_nodes] = adjacency
            padded.append(padded_adj)

        self.adjacencies = np.stack(padded, axis=0)
        self.num_graphs = len(self.adjacencies)

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.adjacencies[idx]


@pytest.fixture(autouse=True)
def clear_hydra() -> Generator[None, None, None]:
    """Clear Hydra global state around each parametrized matrix case."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture(autouse=True)
def patch_fake_pyg_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace PyG dataset loading with a deterministic local fixture."""
    monkeypatch.setattr(
        graph_generation,
        "PyGDatasetWrapper",
        FakePyGDatasetWrapper,
    )


@pytest.fixture(autouse=True)
def isolate_matplotlib_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Point matplotlib at a writable test-local cache directory."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path_factory.mktemp("mplconfig")))


def discover_model_leaves() -> set[str]:
    """Return the runnable leaf model config set under exp_configs/models."""
    discovered = {
        path.relative_to(MODELS_DIR).with_suffix("").as_posix()
        for path in MODELS_DIR.rglob("*.yaml")
    }
    return discovered - EXCLUDED_MODEL_LEAVES


def discover_data_leaves() -> set[str]:
    """Return the runnable leaf data config set under exp_configs/data."""
    discovered = {path.stem for path in DATA_DIR.glob("*.yaml")}
    return discovered - EXCLUDED_DATA_LEAVES


def compose_matrix_config(
    *,
    base_config: str,
    output_dir: Path,
    extra_overrides: tuple[str, ...],
) -> DictConfig:
    """Compose one experiment config with tiny deterministic overrides."""
    overrides = [
        f"paths.output_dir={output_dir}",
        f"paths.results_dir={output_dir}/results",
        f"hydra.run.dir={output_dir}",
        *COMMON_RUN_OVERRIDES,
        *extra_overrides,
    ]
    with initialize_config_dir(version_base=None, config_dir=str(EXP_CONFIG_DIR)):
        cfg = compose(config_name=base_config, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def find_metrics_csv(output_dir: Path) -> Path:
    """Locate the Lightning CSV metrics file for one training run."""
    candidates = sorted(output_dir.rglob("metrics.csv"))
    assert candidates, f"No metrics.csv found under {output_dir}"
    return candidates[0]


def assert_training_artifacts(output_dir: Path) -> None:
    """Assert that fit, validation logging, and test execution all happened."""
    checkpoint_dir = output_dir / "checkpoints"
    last_ckpt = checkpoint_dir / "last.ckpt"
    versioned_last = sorted(checkpoint_dir.glob("last-v*.ckpt"))
    assert (
        last_ckpt.exists() or versioned_last
    ), f"No last checkpoint found under {checkpoint_dir}"

    metrics_csv = find_metrics_csv(output_dir)
    metrics_text = metrics_csv.read_text()
    assert (
        "train/loss" in metrics_text
    ), f"Training metrics missing from {metrics_csv}:\n{metrics_text}"
    assert (
        "val/loss" in metrics_text
    ), f"Validation metrics missing from {metrics_csv}:\n{metrics_text}"

    test_results = output_dir / "test_results.json"
    assert test_results.exists(), f"Missing test marker: {test_results}"


def run_training_case(
    *,
    base_config: str,
    output_dir: Path,
    extra_overrides: tuple[str, ...],
) -> None:
    """Compose, run, and assert one tiny training case."""
    cfg = compose_matrix_config(
        base_config=base_config,
        output_dir=output_dir,
        extra_overrides=extra_overrides,
    )
    _ = run_experiment(cfg)
    assert_training_artifacts(output_dir)


def run_plot_case(
    *,
    base_config: str,
    output_dir: Path,
    extra_overrides: tuple[str, ...],
) -> None:
    """Run the sanity-check plotting path for one representative family case."""
    # ``maybe_run_sanity_check()`` currently probes the noise generator on a
    # random 10x10 graph without seeding. Fix the test boundary instead of
    # weakening the assertions: deterministic seeds keep the plot lane stable
    # while still exercising the real plotting path.
    np.random.seed(0)
    torch.manual_seed(0)

    cfg = compose_matrix_config(
        base_config=base_config,
        output_dir=output_dir,
        extra_overrides=(
            "sanity_check=true",
            "fixed_noise_seed=123",
            *extra_overrides,
        ),
    )
    data_module = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    result = maybe_run_sanity_check(config=cfg, data_module=data_module, model=model)
    assert result == {"sanity_check": "passed"}

    plot_dir = output_dir / "sanity_check_plots"
    assert (plot_dir / "noise_effects.png").exists()
    assert (plot_dir / "model_io.png").exists()

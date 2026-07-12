"""Graph data modules and datasets.

X_class / E_class synthesis convention: see
``internal notes (not included)``.
"""

from __future__ import annotations

from .data_modules import (
    BaseGraphDataModule,
    GraphDataModule,
    MultiGraphDataModule,
    SingleGraphDataModule,
    SyntheticCategoricalDataModule,
)
from .datasets import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphDistribution,
    GraphState,
    PyGDatasetWrapper,
    SyntheticGraphDataset,
    collapse_to_indices,
    generate_block_sizes,
    generate_sbm_adjacency,
    generate_sbm_batch,
    state_to_dense_logits,
    state_to_dense_sample,
)

__all__ = [
    # Graph types — abstract base + 4 concrete leaves
    "GraphData",
    "GraphState",
    "GraphDistribution",
    "DenseGraphState",
    "DenseGraphDistribution",
    "collapse_to_indices",
    "state_to_dense_logits",
    "state_to_dense_sample",
    # Data Modules
    "BaseGraphDataModule",
    "MultiGraphDataModule",
    "GraphDataModule",
    "SingleGraphDataModule",
    "SyntheticCategoricalDataModule",
    # Datasets
    "SyntheticGraphDataset",
    "PyGDatasetWrapper",
    # SBM functions
    "generate_block_sizes",
    "generate_sbm_adjacency",
    "generate_sbm_batch",
]

"""Graph dataset definitions, types, and generation utilities."""

from __future__ import annotations

from .graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphDistribution,
    GraphState,
    collapse_to_indices,
    state_to_dense_logits,
    state_to_dense_sample,
)
from .pyg_datasets import PyGDatasetWrapper
from .sbm import generate_block_sizes, generate_sbm_adjacency, generate_sbm_batch
from .synthetic_graphs import SyntheticGraphDataset

__all__ = [
    "GraphData",
    "GraphState",
    "GraphDistribution",
    "DenseGraphState",
    "DenseGraphDistribution",
    "collapse_to_indices",
    "state_to_dense_logits",
    "state_to_dense_sample",
    "SyntheticGraphDataset",
    "PyGDatasetWrapper",
    "generate_block_sizes",
    "generate_sbm_adjacency",
    "generate_sbm_batch",
]

"""Lightning DataModules for graph learning experiments."""

from __future__ import annotations

from .base_data_module import BaseGraphDataModule
from .data_module import GraphDataModule
from .multigraph_data_module import MultiGraphDataModule
from .single_graph_data_module import SingleGraphDataModule
from .synthetic_categorical import SyntheticCategoricalDataModule

__all__ = [
    "BaseGraphDataModule",
    "MultiGraphDataModule",
    "GraphDataModule",
    "SingleGraphDataModule",
    "SyntheticCategoricalDataModule",
]

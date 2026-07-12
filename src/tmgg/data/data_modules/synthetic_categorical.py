"""SyntheticCategoricalDataModule for discrete diffusion experiments.

Generates synthetic graphs as adjacency matrices, stores them as PyG
``Data`` objects (matching the parent ``MultiGraphDataModule``), and
provides train/val/test DataLoaders that collate into sparse ``GraphState``
batches.

Graph generation and index splitting are handled by the
``MultiGraphDataModule`` superclass.
"""

# pyright: reportUnknownMemberType=false
# pyright: reportExplicitAny=false
# torch.from_numpy().float() and config dict Any parameters have
# incomplete type stubs.

from __future__ import annotations

from typing import Any, override

from torch_geometric.data import Data

from tmgg.data.data_modules.multigraph_data_module import (
    MultiGraphDataModule,
)
from tmgg.utils.noising.size_distribution import SizeDistribution


class SyntheticCategoricalDataModule(MultiGraphDataModule):
    """Data module that generates synthetic graphs in categorical format.

    Generates graphs as adjacency matrices using the shared
    ``_generate_adjacencies()`` from ``MultiGraphDataModule``, converts
    each to a PyG ``Data`` object, and splits into train/val/test sets.
    DataLoaders collate into sparse ``GraphState`` batches via the parent's
    :class:`~tmgg.data.data_modules.multigraph_data_module.GraphDataCollator`.

    The categorical representation uses ``dx=2`` (node present / absent)
    and ``de=2`` (edge present / absent), matching the DiGress convention
    for synthetic binary graphs.

    Parameters
    ----------
    graph_type
        Graph type to generate. ``"sbm"`` uses the SBM generator directly;
        all other types delegate to ``SyntheticGraphDataset``.
    num_nodes
        Number of nodes per graph.
    num_graphs
        Total number of graphs to generate across all splits.
    train_ratio
        Fraction of graphs allocated to the training split.
    val_ratio
        Fraction allocated to validation. The remainder goes to test.
    batch_size
        Batch size for all DataLoaders.
    num_workers
        Number of DataLoader worker processes.
    seed
        Random seed for graph generation and splitting.
    graph_config
        Extra keyword arguments forwarded to the graph generator.
    """

    train_ratio: float
    val_ratio: float

    def __init__(
        self,
        graph_type: str = "sbm",
        num_nodes: int = 50,
        num_graphs: int = 1000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        seed: int = 42,
        graph_config: dict[str, Any] | None = None,
        **_metadata: object,
    ) -> None:
        # ``**_metadata`` absorbs informational config keys (notably
        # ``eval_meta``) that the data block exposes for downstream
        # Hydra interpolation into the evaluator. They are consumed by
        # other subtrees of the resolved config and have no effect on
        # data generation.
        super().__init__(
            graph_type=graph_type,
            num_nodes=num_nodes,
            num_graphs=num_graphs,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            graph_config=graph_config,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @override
    def setup(self, stage: str | None = None) -> None:
        """Generate graphs, convert to PyG Data, and split the dataset.

        Delegates graph generation and storage to the parent, which populates
        ``_train_data`` / ``_val_data`` / ``_test_data`` as ``list[Data]``.
        Idempotent: calling setup multiple times is safe.
        """
        super().setup(stage)

    # ------------------------------------------------------------------
    # Size distribution
    # ------------------------------------------------------------------

    @override
    def get_size_distribution(self, split: str | None = None) -> SizeDistribution:
        """Return the graph size distribution for a split or the whole dataset.

        After ``setup()``, computes the empirical distribution from the
        stored per-graph node counts. Before ``setup()`` (or if the
        requested split is empty), falls back to the degenerate
        distribution at ``num_nodes``.

        Parameters
        ----------
        split
            ``"train"``, ``"val"``, ``"test"``, or ``None`` for all
            splits combined.

        Returns
        -------
        SizeDistribution
        """
        split_map: dict[str | None, list[Data] | None] = {
            "train": self._train_data,
            "val": self._val_data,
            "test": self._test_data,
        }

        if split is None:
            all_data = (
                (self._train_data or [])
                + (self._val_data or [])
                + (self._test_data or [])
            )
        elif split in split_map:
            all_data = split_map[split] or []
        else:
            raise ValueError(
                f"Unknown split {split!r}. Expected 'train', 'val', 'test', or None."
            )

        if not all_data:
            return SizeDistribution.fixed(self.num_nodes)

        node_counts = [int(d.num_nodes) for d in all_data]  # type: ignore[arg-type]
        return SizeDistribution.from_node_counts(node_counts)

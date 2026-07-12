"""Abstract base for all graph data modules in TMGG.

Defines the minimal contract that every graph data module must satisfy,
independent of the graph representation (adjacency matrices vs.
categorical one-hot features) or the generation protocol (multi-graph
vs. single-graph). Concrete generation and splitting logic lives in
``MultiGraphDataModule`` and the leaf classes.
"""

# pyright: reportExplicitAny=false
# DataLoader/Dataset generic parameters and config dicts require Any
# until PyTorch provides complete generic stubs.

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, override

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from tmgg.utils.noising.size_distribution import SizeDistribution


class BaseGraphDataModule(pl.LightningDataModule, abc.ABC):
    """Abstract base class for all graph data modules.

    Provides the shared DataLoader configuration (batch size, workers,
    pin_memory) and a concrete ``_make_dataloader`` factory. Subclasses
    implement generation, splitting, setup, and dataloaders.

    Parameters
    ----------
    batch_size
        Batch size for all dataloaders.
    num_workers
        Number of dataloader worker processes.
    pin_memory
        Whether to pin memory in DataLoaders for faster GPU transfer.
    prefetch_factor
        Per-worker batches to prefetch ahead. PyTorch default is 2;
        4 hides typical step latency without much memory overhead. Has
        no effect when ``num_workers == 0`` (PyTorch ignores it). See
        ``an internal validation harness (not included)``.
    seed
        Random seed for graph generation and splitting.
    """

    batch_size: int
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    seed: int
    graph_type: str
    num_nodes: int

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        seed: int = 42,
        drop_last_train: bool = True,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        # ``drop_last_train`` defaults to ``True`` so the train loader
        # emits only full batches. The trailing partial batch is the
        # dominant source of dynamic-shape recompiles under
        # ``torch.compile`` (a 128-graph fixture at bs=12 yields a
        # size-8 tail every epoch). Validation and test loaders never
        # drop the tail — losing eval samples would skew metrics.
        self.drop_last_train = drop_last_train

    # ------------------------------------------------------------------
    # DataLoader factory
    # ------------------------------------------------------------------

    def _make_dataloader(
        self,
        dataset: Dataset[Any],
        shuffle: bool,
        collate_fn: Callable[..., Any] | None = None,
    ) -> DataLoader[Any]:
        """Create a DataLoader with the module's shared configuration.

        Parameters
        ----------
        dataset
            The dataset to wrap.
        shuffle
            Whether to shuffle each epoch.
        collate_fn
            Custom collation function. ``None`` uses the default
            torch collation.

        Returns
        -------
        DataLoader
            Configured with ``self.batch_size``, ``self.num_workers``,
            ``self.pin_memory``, and ``persistent_workers`` when workers
            are present. ``prefetch_factor`` only forwarded when workers
            are spawned (PyTorch raises if you pass it with
            ``num_workers == 0``).
        """
        kwargs: dict[str, Any] = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
            # Train loader (shuffle=True) drops the tail to keep batch
            # shapes stable for compile/CUDA-graph capture. Val/test
            # loaders preserve every sample so metrics aren't biased.
            drop_last=shuffle and self.drop_last_train,
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(dataset, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @override
    @abc.abstractmethod
    def setup(self, stage: str | None = None) -> None:
        """Prepare datasets for the given stage.

        Parameters
        ----------
        stage : str or None
            One of ``"fit"``, ``"validate"``, ``"test"``, ``"predict"``,
            or None (all stages).
        """
        ...

    @override
    @abc.abstractmethod
    def train_dataloader(self) -> DataLoader[Any]:
        """Return the training dataloader."""
        ...

    @override
    @abc.abstractmethod
    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation dataloader."""
        ...

    @override
    @abc.abstractmethod
    def test_dataloader(self) -> DataLoader[Any]:
        """Return the test dataloader."""
        ...

    @abc.abstractmethod
    def train_dataloader_raw_pyg(self) -> DataLoader[Any]:
        """Return a training dataloader yielding *raw* PyG ``Batch`` objects.

        Sits one layer below ``train_dataloader``: bypasses the
        ``GraphState`` collator so consumers that need the raw PyG
        ``Batch`` representation (notably the upstream-parity edge /
        node count helpers in :mod:`tmgg.data.utils.edge_counts` and
        :meth:`CategoricalNoiseProcess.initialize_from_data`) can
        iterate the same training data with the source-shaped
        per-edge / per-node tensors. Kept distinct from the GraphState
        path so the empirical-π estimator stays a direct line-by-line
        port of upstream DiGress's sparse counters.

        Marked ``@abc.abstractmethod`` so a subclass that omits the
        override fails at instantiation time (``TypeError: Can't
        instantiate abstract class ... with abstract method
        train_dataloader_raw_pyg``) rather than at config-preflight
        time on a Modal worker. Datamodules whose training data is not
        PyG-backed should still implement this and raise
        ``NotImplementedError`` from the body, so the failure mode is
        explicit at the leaf class rather than inherited from the base.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete utility methods
    # ------------------------------------------------------------------

    def get_size_distribution(self, split: str | None = None) -> SizeDistribution:
        """Return the graph-size distribution for a dataset split.

        The base contract assumes fixed-size graphs and therefore returns
        a degenerate distribution at ``self.num_nodes`` regardless of
        *split*. Variable-size datamodules should override this method.

        Parameters
        ----------
        split
            Split selector. Accepted for interface compatibility with
            variable-size subclasses and ignored by the fixed-size base
            implementation.

        Returns
        -------
        SizeDistribution
            Degenerate distribution at ``self.num_nodes``.
        """
        _ = split
        return SizeDistribution.fixed(self.num_nodes)

    def get_reference_graphs(self, stage: str, max_graphs: int) -> list[Any]:
        """Extract up to *max_graphs* from a dataset split as ``DenseGraphState``.

        Iterates the appropriate dataloader (which now yields sparse
        ``GraphState`` batches under the sparse-default convention),
        densifies each batch via ``GraphState.to_dense`` with the
        no-edge-one-hot fill, then slices along the leading dim,
        returning a flat list of per-graph (single-instance)
        ``DenseGraphState`` carriers with a leading batch dim of 1 so
        existing downstream consumers that read ``X_class[0]``,
        ``E_class[0]``, ``node_mask[0]`` continue to work.

        Per spec ``internal notes (not included)``
        the return type is per-graph ``DenseGraphState`` rather than
        ``list[nx.Graph]``: the dense state is the universal transport
        format for the whole evaluation pipeline. Callers that need
        NetworkX graphs convert at the consumption site via
        :meth:`DenseGraphState.to_networkx` (cheap, local, lossless).

        Parameters
        ----------
        stage : str
            ``"val"`` or ``"test"``.
        max_graphs : int
            Maximum number of graphs to return.

        Returns
        -------
        list[DenseGraphState]
            Up to *max_graphs* per-graph ``DenseGraphState`` from the
            requested split.

        Raises
        ------
        ValueError
            If *stage* is not ``"val"`` or ``"test"``.
        """
        from tmgg.data.datasets.graph_types import (
            DenseGraphState,
            state_to_dense_sample,
        )

        if stage == "val":
            loader = self.val_dataloader()
        elif stage == "test":
            loader = self.test_dataloader()
        else:
            raise ValueError(f"stage must be 'val' or 'test', got {stage!r}")

        graphs: list[Any] = []
        for batch in loader:
            # The collator now emits a sparse ``GraphState``. Densify so
            # the per-graph slicing convention below (single-batch
            # leading dim of 1, indexable per-position fields) keeps
            # working. Phase 7+ will migrate downstream consumers to
            # take a sparse view directly.
            dense_batch = state_to_dense_sample(batch)
            bs = int(dense_batch.num_nodes_per_graph.shape[0])
            for i in range(bs):
                if len(graphs) >= max_graphs:
                    return graphs
                # Keep a leading batch dim of 1 on every field so callers
                # built around the old ``X_class[0]``/``E_class[0]``
                # indexing convention keep working unchanged.
                graphs.append(
                    DenseGraphState(
                        num_nodes_per_graph=dense_batch.num_nodes_per_graph[i : i + 1],
                        X_class=dense_batch.X_class[i : i + 1]
                        if dense_batch.X_class is not None
                        else None,
                        X_feat=dense_batch.X_feat[i : i + 1]
                        if dense_batch.X_feat is not None
                        else None,
                        E_class=dense_batch.E_class[i : i + 1]
                        if dense_batch.E_class is not None
                        else None,
                        E_feat=dense_batch.E_feat[i : i + 1]
                        if dense_batch.E_feat is not None
                        else None,
                        y=dense_batch.y[i : i + 1],
                    )
                )
        return graphs

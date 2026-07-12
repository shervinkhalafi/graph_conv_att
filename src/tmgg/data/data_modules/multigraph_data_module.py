"""Base for datamodules that generate multiple graphs and split by ratio.

Sits between ``BaseGraphDataModule`` (shared config + DataLoader factory) and
specialized leaf classes (denoising ``GraphDataModule``, discrete generative
``SyntheticCategoricalDataModule``). Provides graph generation, index-based
splitting, SBM partition-aware splitting, and a default setup that converts
adjacency arrays to PyG ``Data`` objects served via DataLoaders.

Usable directly (e.g. as the gaussian diffusion generative datamodule) or
subclassed when a different data representation is needed.
"""

# pyright: reportExplicitAny=false
# DataLoader/Dataset generic parameters and config dicts require Any
# until PyTorch provides complete generic stubs.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from tmgg.data.datasets.graph_types import GraphState
from tmgg.utils.noising.size_distribution import SizeDistribution

from .base_data_module import BaseGraphDataModule
from .graph_generation import generate_multigraph_split


def _adjacencies_to_pyg(adjs: np.ndarray) -> list[Data]:
    """Convert numpy adjacency matrices to PyG Data objects.

    Parameters
    ----------
    adjs
        Adjacency matrices, shape ``(N, n, n)``.

    Returns
    -------
    list[Data]
        One ``Data`` per graph with COO ``edge_index``.
    """
    from torch_geometric.utils import dense_to_sparse

    result: list[Data] = []
    for i in range(len(adjs)):
        adj_t = torch.from_numpy(adjs[i]).float()
        edge_index, _ = dense_to_sparse(adj_t)
        result.append(Data(edge_index=edge_index, num_nodes=adj_t.shape[0]))
    return result


@dataclass(frozen=True)
class GraphDataCollator:
    """Hot-path collator producing sparse ``GraphState`` batches.

    Wired into every ``train_dataloader`` / ``val_dataloader`` /
    ``test_dataloader`` across all datamodules. Fires once per batch on
    the worker side, returns the sparse representation the noise
    process + loss now consume by default:

    - ``GraphState.edge_index`` — sparse COO edge list (active edges
      only).
    - ``GraphState.edge_class`` — per-edge one-hot ``(E, C_e)``.
    - ``GraphState.x_class`` — per-node one-hot ``(N, C_x)`` when the
      input ``Data`` carries categorical atom labels; ``None`` for
      structure-only datasets (SBM / Planar).
    - ``GraphState.batch`` — PyG batch vector mapping each node back
      to its source graph.

    Dense-internal models (the GraphTransformer in particular) request
    ``GraphState.to_dense(...)`` themselves; the static-pad ceiling
    that used to live on this collator has moved into
    ``DenseGraphState.from_pyg_batch`` / ``GraphState.to_dense`` call
    sites in those models. Keeping the sparse representation as the
    canonical wire format avoids paying the dense materialisation cost
    on every batch even for callers that never look at the dense view.

    Pickles cleanly across multi-worker DataLoader subprocesses
    (module-level ``frozen=True`` dataclass with primitive fields).

    Parameters
    ----------
    n_max_static
        Retained for source compatibility with downstream model code
        that still constructs the collator with this kwarg. Inert at
        the collator boundary — ``GraphState`` carries no padding
        ceiling. Static-pad is now requested at ``to_dense(n_max=...)``
        time by the dense consumer. Will be removed once Phase 6
        migrates the model wiring.
    num_atom_types_x
        Optional explicit width for the per-node ``x_class`` one-hot
        when the input ``Data.x`` carries integer atom-class indices.
        Forwarded verbatim to :meth:`GraphState.from_pyg_batch`.
        Required for molecular-path correctness — without it, batches
        that miss a rare atom class end up with a narrower ``x_class``
        than batches that include it. ``None`` (default) preserves the
        structural SPECTRE-SBM / SPECTRE-Planar behaviour and lets
        :meth:`from_pyg_batch` infer the width from data when ``x`` is
        present.
    num_bond_types_e
        Optional explicit width for the per-edge ``edge_class`` one-hot
        when the input carries integer bond-class indices on
        ``edge_attr``. Forwarded verbatim to
        :meth:`GraphState.from_pyg_batch`. Required for molecular-path
        correctness — without it, batches that miss a rare bond class
        (e.g. AROMATIC) end up with a narrower ``edge_class`` than
        batches that include it. ``None`` (default) preserves the
        legacy 2-class ``[no-edge, edge]`` encoding used by
        SPECTRE-SBM / SPECTRE-Planar when ``edge_attr`` is absent.

    See Also
    --------
    RawPyGCollator
        Pre-sparse-default counterpart that returned the raw PyG
        ``Batch``. Used by the noise process's empirical-marginals
        estimator. Different consumer, different representation,
        intentionally not unified — see that class's docstring for the
        parity-port reasoning.
    """

    n_max_static: int | None = None
    num_atom_types_x: int | None = None
    num_bond_types_e: int | None = None

    def __call__(self, data_list: list[Data]) -> GraphState:
        from typing import cast

        from torch_geometric.data import Batch
        from torch_geometric.data.data import BaseData

        batch = Batch.from_data_list(cast(list[BaseData], data_list))
        return GraphState.from_pyg_batch(
            batch,
            num_atom_types_x=self.num_atom_types_x,
            num_bond_types_e=self.num_bond_types_e,
        )


@dataclass(frozen=True)
class RawPyGCollator:
    """Preprocessing-pass collator returning the sparse PyG ``Batch``.

    Off the hot training path. Wired into ``train_dataloader_raw_pyg``
    on every datamodule and consumed exactly once per run before
    training starts, by the noise process's empirical-stationary-
    marginals estimator (:meth:`NoiseProcess.initialize_from_data`,
    ``noise_process.py:1089``). That method walks the training set
    once, summing per-class node and edge counts via the sparse
    helpers in :mod:`tmgg.data.utils.edge_counts`, then sets the
    noise process's limit distribution to the empirical π.

    Why sparse rather than dense:

    - The π estimator counts entries in the explicit edge list. The
      densified ``E_class`` would force it to deduce edges from
      one-hot indicators, with a per-position lookup overhead and an
      ambiguous treatment of padded positions.
    - Upstream parity: this is a direct port of DiGress's
      ``AbstractDatasetInfos.edge_counts`` / ``node_types``
      (``digress-upstream-readonly/src/datasets/abstract_dataset.py:34-72``),
      which operates on the sparse representation. Mirroring the
      shape of computation makes parity audits tractable.
    - When :class:`GraphDataCollator` runs in static-pad mode
      (``n_max_static=200`` on SPECTRE), padded zero-edges would
      pollute the empirical edge histogram if the estimator went
      through the dense view. The sparse view is naturally
      pad-invariant.

    Two call sites today:

    - ``DiffusionModule.setup`` (``diffusion_module.py:525``) — fires
      once per fit cycle when the noise process needs data init.
    - ``modal/_functions.py:122`` — the Modal config-preflight
      subprocess, which sanity-checks that init succeeds inside the
      container before launching the real run.

    Stateless today; kept as a ``frozen=True`` dataclass so every
    datamodule's dataloader-construction site uses a uniform
    ``collate_fn=SomeCollator()`` idiom rather than a function-or-
    instance mix. Future state (e.g. a "skip empty batches" parity
    flag) can be added as fields without changing call sites.

    See Also
    --------
    GraphDataCollator
        Dense counterpart used on the hot training path.
    """

    def __call__(self, data_list: list[Data]):
        from typing import cast

        from torch_geometric.data import Batch
        from torch_geometric.data.data import BaseData

        return Batch.from_data_list(cast(list[BaseData], data_list))


class _ListDataset(Dataset[Data]):
    """Thin wrapper making a list indexable as a Dataset."""

    def __init__(self, data: list[Data]) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Data:
        return self._data[idx]


class MultiGraphDataModule(BaseGraphDataModule):
    """Datamodule that generates N graphs and splits by ratio.

    Extends ``BaseGraphDataModule`` with graph generation parameters and
    concrete methods for adjacency generation, train/val/test splitting,
    and SBM partition-aware generation.

    The default ``setup()`` generates adjacency matrices, converts them to
    PyG ``Data`` objects (COO edge_index), and serves them via DataLoaders
    that collate into sparse ``GraphState`` batches. Subclasses that need a
    different representation (e.g. categorical one-hot) override ``setup()``
    and the dataloaders.

    Parameters
    ----------
    graph_type
        Graph type to generate (``"sbm"``, ``"er"``, ``"tree"``, etc.).
    num_nodes
        Number of nodes per graph.
    num_graphs
        Total number of graphs to generate across all splits.
    train_ratio
        Fraction of graphs for training.
    val_ratio
        Fraction of graphs for validation. Remainder goes to test.
    graph_config
        Extra keyword arguments forwarded to the graph generator.
    batch_size, num_workers, pin_memory, seed
        Passed to ``BaseGraphDataModule``.
    """

    graph_type: str
    num_nodes: int
    num_graphs: int
    train_ratio: float
    val_ratio: float
    graph_config: dict[str, Any]

    def __init__(
        self,
        graph_type: str = "sbm",
        num_nodes: int = 50,
        num_graphs: int = 1000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        graph_config: dict[str, Any] | None = None,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            seed=seed,
        )
        self.graph_type = graph_type
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.graph_config = graph_config or {}

        self.save_hyperparameters()

        # Populated by setup()
        self._train_data: list[Data] | None = None
        self._val_data: list[Data] | None = None
        self._test_data: list[Data] | None = None

    # ------------------------------------------------------------------
    # Graph generation
    # ------------------------------------------------------------------

    def _generate_and_split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate train, val, and test adjacencies through the shared generator."""
        return generate_multigraph_split(
            graph_type=self.graph_type,
            num_nodes=self.num_nodes,
            num_graphs=self.num_graphs,
            graph_config=self.graph_config,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # Size distribution
    # ------------------------------------------------------------------

    @override
    def get_size_distribution(self, split: str | None = None) -> SizeDistribution:
        """Return the graph size distribution for a split or the whole dataset.

        Parameters
        ----------
        split
            ``"train"``, ``"val"``, ``"test"``, or ``None`` for the
            whole dataset (pre-split).

        Returns
        -------
        SizeDistribution
            Default implementation returns a degenerate distribution
            (all graphs have ``num_nodes`` nodes). Subclasses handling
            variable-size data should override this method.
        """
        return SizeDistribution.fixed(self.num_nodes)

    # ------------------------------------------------------------------
    # Default setup and DataLoaders (PyG Data storage)
    # ------------------------------------------------------------------

    @override
    def setup(self, stage: str | None = None) -> None:
        """Generate graphs, split, and convert to PyG Data objects.

        Idempotent: calling setup multiple times is safe.

        Parameters
        ----------
        stage
            Lightning stage (``"fit"``, ``"test"``, etc.) or None.
        """
        if self._train_data is not None:
            return

        train, val, test = self._generate_and_split()

        self._train_data = _adjacencies_to_pyg(train)
        self._val_data = _adjacencies_to_pyg(val)
        self._test_data = _adjacencies_to_pyg(test)

    @override
    def train_dataloader(self) -> DataLoader[GraphState]:
        """Create training dataloader."""
        if self._train_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._train_data),
            shuffle=True,
            collate_fn=GraphDataCollator(),
        )

    @override
    def val_dataloader(self) -> DataLoader[GraphState]:
        """Create validation dataloader."""
        if self._val_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._val_data),
            shuffle=False,
            collate_fn=GraphDataCollator(),
        )

    @override
    def test_dataloader(self) -> DataLoader[GraphState]:
        """Create test dataloader."""
        if self._test_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._test_data),
            shuffle=False,
            collate_fn=GraphDataCollator(),
        )

    @override
    def train_dataloader_raw_pyg(self) -> DataLoader[Any]:
        """Return the training loader without the GraphState collator.

        Used by the upstream-parity sparse π estimator. Shuffle is off so
        the iteration is deterministic across the (single) preprocessing
        pass; the dense ``train_dataloader`` keeps shuffling on for
        actual training.
        """
        if self._train_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._train_data),
            shuffle=False,
            collate_fn=RawPyGCollator(),
        )

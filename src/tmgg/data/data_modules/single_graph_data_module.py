"""Single-graph data module for denoising experiments.

Implements the single-graph training protocol where a model is trained on
a single graph with multiple noise realizations, testing generalization to
new noise samples and unseen graph instances from the same distribution.
"""

from typing import Any, override

import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from tmgg.data.datasets.graph_types import GraphState

from .base_data_module import BaseGraphDataModule
from .graph_generation import generate_single_graph
from .multigraph_data_module import (
    GraphDataCollator,
    RawPyGCollator,
    _adjacencies_to_pyg,
    _ListDataset,
)


class SingleGraphDataModule(BaseGraphDataModule):
    """Lightning DataModule for single-graph training protocol.

    Supports two modes controlled by `same_graph_all_splits`:

    When `same_graph_all_splits=True` (Stage 1 protocol):
    - All splits (train/val/test) use the identical graph G
    - Only noise varies across samples
    - Tests generalization to new noise realizations on a known structure

    When `same_graph_all_splits=False` (default, Stage 2+ protocol):
    - Training uses graph G, validation/test use different graphs G', G''
    - Tests generalization to both new noise and unseen graph structures

    Parameters
    ----------
    graph_type : str
        Type of graph to generate. Supported types:
        - ``"sbm"``: Stochastic Block Model
        - ``"erdos_renyi"``: Erdos-Renyi random graphs
        - ``"regular"``: d-regular graphs
        - ``"tree"``: Random trees
        - ``"ring_of_cliques"``: NetworkX ring of cliques
        - ``"lfr"``: LFR benchmark graphs (community structure)
        - ``"pyg_enzymes"``, ``"pyg_qm9"``, ``"pyg_proteins"``: PyG datasets
    num_nodes : int
        Number of nodes.
    graph_config : dict or None
        Additional parameters for graph generation (e.g., ``p`` for ER,
        ``d`` for regular, SBM parameters).
    num_train_samples : int
        Number of training samples per epoch (noise realizations).
    num_val_samples : int
        Number of validation samples.
    num_test_samples : int
        Number of test samples.
    batch_size : int
        Batch size for dataloaders.
    num_workers : int
        Number of dataloader workers.
    train_seed : int
        Seed for generating the training graph.
    val_test_seed : int
        Seed for generating validation/test graphs (used only when
        ``same_graph_all_splits=False``).
    same_graph_all_splits : bool
        If True, val/test use the same graph as training (Stage 1 protocol).
        If False, val/test use different graphs (Stage 2+ protocol).

    Examples
    --------
    >>> dm = SingleGraphDataModule(
    ...     graph_type="sbm",
    ...     num_nodes=50,
    ...     graph_config={"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3},
    ...     num_train_samples=1000,
    ...     num_val_samples=100,
    ...     num_test_samples=100,
    ...     batch_size=16,
    ... )
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()
    """

    graph_type: str
    num_nodes: int
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int
    train_seed: int
    val_test_seed: int
    same_graph_all_splits: bool
    graph_config: dict[str, Any]
    _train_data: list[Data] | None
    _val_data: list[Data] | None
    _test_data: list[Data] | None
    _actual_num_nodes: int

    def __init__(
        self,
        graph_type: str = "sbm",
        num_nodes: int = 50,
        graph_config: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
        num_train_samples: int = 1000,
        num_val_samples: int = 100,
        num_test_samples: int = 100,
        batch_size: int = 16,
        num_workers: int = 0,
        prefetch_factor: int = 4,
        train_seed: int = 42,
        val_test_seed: int = 123,
        same_graph_all_splits: bool = False,
        eval_meta: object = None,
    ):
        # ``eval_meta`` is informational metadata (typically ``{p_intra,
        # p_inter}`` for SBM evaluators) attached to the data namespace by
        # upstream config blocks for Hydra interpolation. Accepted as an
        # explicit parameter (rather than ``**kwargs``) so the
        # legacy-kwarg rejection contract stays intact: unknown kwargs
        # like ``noise_levels`` / ``noise_type`` still raise ``TypeError``.
        # Captured into ``self.hparams`` by ``save_hyperparameters()`` below.
        _ = eval_meta
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            seed=train_seed,
        )
        self.save_hyperparameters()

        self.graph_type = graph_type.lower()
        self.num_nodes = num_nodes
        self.graph_config = graph_config or {}
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.train_seed = train_seed
        self.val_test_seed = val_test_seed
        self.same_graph_all_splits = same_graph_all_splits

        self._train_data = None
        self._val_data = None
        self._test_data = None
        self._actual_num_nodes = num_nodes

    def _generate_graph(self, seed: int) -> npt.NDArray[np.float32]:
        """Generate one graph through the shared graph generator."""
        generated = generate_single_graph(
            graph_type=self.graph_type,
            num_nodes=self.num_nodes,
            graph_config=self.graph_config,
            seed=seed,
        )
        self._actual_num_nodes = generated.num_nodes
        return generated.adjacency

    def _adj_to_data(self, adj: npt.NDArray[np.float32]) -> Data:
        """Convert a single adjacency matrix to a PyG Data object.

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix of shape ``(n, n)``.

        Returns
        -------
        Data
            PyG Data object with COO ``edge_index`` and ``num_nodes``.
        """
        return _adjacencies_to_pyg(adj[np.newaxis])[0]

    @override
    def setup(self, stage: str | None = None) -> None:
        """Generate graphs for train/val/test.

        Parameters
        ----------
        stage : str, optional
            Stage: "fit", "validate", "test", or None (all).
        """
        # Generate training graph
        if stage in (None, "fit"):
            train_adj = self._generate_graph(self.train_seed)
            train_data = self._adj_to_data(train_adj)
            self._train_data = [train_data] * self.num_train_samples

        if self.same_graph_all_splits:
            # Stage 1 protocol: all splits use identical graph, only noise varies
            if stage in (None, "fit", "validate"):
                if self._train_data is None:
                    # setup called with stage="validate" only — generate train graph
                    train_adj = self._generate_graph(self.train_seed)
                    train_data = self._adj_to_data(train_adj)
                else:
                    train_data = self._train_data[0]
                self._val_data = [train_data] * self.num_val_samples
            if stage in (None, "test"):
                if self._train_data is None:
                    train_adj = self._generate_graph(self.train_seed)
                    train_data = self._adj_to_data(train_adj)
                else:
                    train_data = self._train_data[0]
                self._test_data = [train_data] * self.num_test_samples
        else:
            # Stage 2+ protocol: different graphs for val/test
            if stage in (None, "fit", "validate"):
                val_adj = self._generate_graph(self.val_test_seed)
                val_data = self._adj_to_data(val_adj)
                self._val_data = [val_data] * self.num_val_samples
            if stage in (None, "test"):
                test_adj = self._generate_graph(self.val_test_seed + 1000)
                test_data = self._adj_to_data(test_adj)
                self._test_data = [test_data] * self.num_test_samples

    @override
    def train_dataloader(self) -> DataLoader[GraphState]:
        """Return training dataloader."""
        if self._train_data is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self._make_dataloader(
            _ListDataset(self._train_data),
            shuffle=True,
            collate_fn=GraphDataCollator(),
        )

    @override
    def val_dataloader(self) -> DataLoader[GraphState]:
        """Return validation dataloader."""
        if self._val_data is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self._make_dataloader(
            _ListDataset(self._val_data),
            shuffle=False,
            collate_fn=GraphDataCollator(),
        )

    @override
    def test_dataloader(self) -> DataLoader[GraphState]:
        """Return test dataloader."""
        if self._test_data is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self._make_dataloader(
            _ListDataset(self._test_data),
            shuffle=False,
            collate_fn=GraphDataCollator(),
        )

    @override
    def train_dataloader_raw_pyg(self) -> DataLoader[object]:
        """Raw PyG ``Batch`` training loader for the parity-port π estimator."""
        if self._train_data is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self._make_dataloader(
            _ListDataset(self._train_data),
            shuffle=False,
            collate_fn=RawPyGCollator(),
        )

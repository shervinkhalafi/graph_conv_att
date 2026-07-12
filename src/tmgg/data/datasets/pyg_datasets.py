"""PyTorch Geometric dataset wrappers for graph denoising experiments.

This module provides wrappers around PyTorch Geometric datasets (QM9 plus
selected TU datasets) that extract adjacency matrices and handle variable
graph sizes through padding.
"""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data


class PyGDatasetWrapper(Dataset[np.ndarray]):
    """Torch Dataset wrapper that extracts padded adjacency matrices from PyG.

    Handles variable-size graphs by padding to the maximum size in the dataset.
    Node features are ignored as per experiment design. The wrapper keeps the
    native PyG ``Data`` objects in ``data_list`` for callers that need them,
    but its dataset surface is adjacency-only.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset key, e.g. "qm9", "enzymes", "proteins",
        "collab", "deezer_ego_nets", "imdb_binary", or "reddit_binary".
    root : str or Path, optional
        Root directory for downloading/storing datasets.
        Default is "~/.pyg_data".
    max_graphs : int, optional
        Maximum number of graphs to load. Default is None (all graphs).

    Attributes
    ----------
    adjacencies : np.ndarray
        Padded adjacency matrices of shape (num_graphs, max_n, max_n).
    num_nodes : np.ndarray
        Actual number of nodes per graph (before padding).
    max_n : int
        Maximum number of nodes across all graphs.
    data_list : list[Data]
        Original PyG ``Data`` objects (one per graph, same order as
        ``adjacencies``), each holding ``edge_index`` and ``num_nodes``.
    """

    VALID_DATASETS = {
        "qm9",
        "enzymes",
        "proteins",
        "collab",
        "deezer_ego_nets",
        "imdb_binary",
        "reddit_binary",
    }
    TU_DATASET_NAME_MAP = {
        "enzymes": "ENZYMES",
        "proteins": "PROTEINS",
        "collab": "COLLAB",
        "deezer_ego_nets": "deezer_ego_nets",
        "imdb_binary": "IMDB-BINARY",
        "reddit_binary": "REDDIT-BINARY",
    }

    def __init__(
        self,
        dataset_name: str,
        root: str | None = None,
        max_graphs: int | None = None,
    ):
        if dataset_name.lower() not in self.VALID_DATASETS:
            raise ValueError(
                f"dataset_name must be one of {self.VALID_DATASETS}, "
                f"got '{dataset_name}'"
            )

        self.dataset_name = dataset_name.lower()
        self.root = Path(root) if root else Path.home() / ".pyg_data"
        self.max_graphs = max_graphs

        # Load dataset
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load PyG dataset and extract adjacency matrices."""
        try:
            from torch_geometric.datasets import QM9, TUDataset
            from torch_geometric.utils import to_dense_adj
        except ImportError as e:
            raise ImportError(
                "PyTorch Geometric is required for benchmark datasets. "
                "Install with: pip install torch-geometric"
            ) from e

        # Load appropriate dataset
        if self.dataset_name == "qm9":
            dataset = QM9(root=str(self.root / "QM9"))
        elif self.dataset_name in self.TU_DATASET_NAME_MAP:
            dataset = TUDataset(
                root=str(self.root), name=self.TU_DATASET_NAME_MAP[self.dataset_name]
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Limit number of graphs if specified
        if self.max_graphs is not None:
            dataset = dataset[: self.max_graphs]

        # Extract adjacency matrices and preserve native Data objects
        adjacencies = []
        num_nodes = []
        self.data_list: list[Data] = []

        for i in range(len(dataset)):
            data = dataset[i]  # pyright: ignore[reportArgumentType]
            # Get number of nodes - use getattr for PyG Data dynamic attributes
            n = getattr(data, "num_nodes")  # noqa: B009
            num_nodes.append(n)

            # Convert edge_index to dense adjacency
            edge_index = getattr(data, "edge_index")  # noqa: B009
            A = to_dense_adj(edge_index, max_num_nodes=n).squeeze(0)
            A_np = A.numpy().astype(np.float32)
            A_np = (A_np + A_np.T) / 2
            A_np = (A_np > 0).astype(np.float32)
            adjacencies.append(A_np)

            # Preserve the native Data object (edge_index + num_nodes only)
            self.data_list.append(Data(edge_index=edge_index, num_nodes=n))

        self.num_nodes = np.array(num_nodes)
        self.max_n = int(self.num_nodes.max())

        # Pad all adjacencies to max size
        padded = []
        for A_raw in adjacencies:
            n = A_raw.shape[0]
            if n < self.max_n:
                # Pad with zeros
                padded_A = np.zeros((self.max_n, self.max_n), dtype=np.float32)
                padded_A[:n, :n] = A_raw
                padded.append(padded_A)
            else:
                padded.append(A_raw)

        self.adjacencies = np.stack(padded, axis=0)
        self.num_graphs = len(self.adjacencies)

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.adjacencies[idx]

    def get_adjacency_matrices(self) -> np.ndarray:
        """Return all padded adjacency matrices.

        Returns
        -------
        np.ndarray
            Adjacency matrices of shape (num_graphs, max_n, max_n).
        """
        return self.adjacencies

    def get_node_counts(self) -> np.ndarray:
        """Return actual node counts for each graph.

        Returns
        -------
        np.ndarray
            Number of nodes per graph (before padding).
        """
        return self.num_nodes

    def get_masks(self) -> np.ndarray:
        """Get node masks indicating valid (non-padded) regions.

        Returns
        -------
        np.ndarray
            Boolean masks of shape (num_graphs, max_n) where True indicates
            a valid node.
        """
        indices = np.arange(self.max_n)
        return indices[np.newaxis, :] < self.num_nodes[:, np.newaxis]

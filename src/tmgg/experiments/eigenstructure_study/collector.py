"""Eigenstructure collector for graph datasets.

Collects eigendecompositions of both adjacency and Laplacian matrices
for all graphs in a dataset, storing results in safetensors format.
"""

# pyright: reportAttributeAccessIssue=false
# torch.nn.functional.pad has incomplete type stubs.

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from tmgg.utils.spectral.laplacian import compute_laplacian

from .storage import save_dataset_manifest, save_decomposition_batch

# Supported dataset types organized by loader mechanism
SYNTHETIC_TYPES = {
    "regular",
    "tree",
    "lfr",
    "erdos_renyi",
    "er",
    "watts_strogatz",
    "ws",
    "random_geometric",
    "rg",
    "configuration_model",
    "cm",
    "ring_of_cliques",
    "lollipop",
    "circular_ladder",
    "star",
    "square_grid",
    "triangle_grid",
}

PYG_TYPES = {"qm9", "enzymes", "proteins", "collab"}


class EigenstructureCollector:
    """
    Collect eigendecompositions for all graphs in a dataset.

    Processes graphs in batches, computing both adjacency and Laplacian
    decompositions, and stores results in safetensors format.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Supported values:
        - "sbm": Stochastic Block Model
        - "regular", "tree", "lfr", "erdos_renyi"/"er",
          "watts_strogatz"/"ws", "random_geometric"/"rg",
          "configuration_model"/"cm", "ring_of_cliques",
          "lollipop", "circular_ladder", "star",
          "square_grid", "triangle_grid": Synthetic graph types
        - "qm9", "enzymes", "proteins": PyG benchmark datasets
    dataset_config : dict
        Configuration for the dataset loader.
    output_dir : Path
        Directory to write output files.
    batch_size : int
        Number of graphs to process per batch.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: dict[str, Any],
        output_dir: Path,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name.lower()
        self.dataset_config = dataset_config
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect(self) -> None:
        """Run the collection process."""
        adjacency_matrices = self._load_all_adjacencies()
        num_graphs = len(adjacency_matrices)
        logger.info(f"Loaded {num_graphs} graphs from {self.dataset_name}")

        batch_index = 0
        for start_idx in range(0, num_graphs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_graphs)
            batch_matrices = adjacency_matrices[start_idx:end_idx]

            self._process_batch(batch_matrices, batch_index, start_idx)
            batch_index += 1
            logger.info(
                f"Processed batch {batch_index}, graphs {start_idx}-{end_idx - 1}"
            )

        save_dataset_manifest(
            self.output_dir,
            self.dataset_name,
            self.dataset_config,
            num_graphs,
            batch_index,
            self.seed,
        )
        logger.info(f"Collection complete. Manifest saved to {self.output_dir}")

    def _load_all_adjacencies(self) -> list[torch.Tensor]:
        """Load all adjacency matrices from the configured dataset."""
        if self.dataset_name == "sbm":
            return self._load_sbm()
        elif self.dataset_name in SYNTHETIC_TYPES:
            return self._load_synthetic()
        elif self.dataset_name in PYG_TYPES:
            return self._load_pyg()
        else:
            raise ValueError(
                f"Unknown dataset: {self.dataset_name}. "
                f"Supported: sbm, {SYNTHETIC_TYPES}, {PYG_TYPES}"
            )

    def _load_sbm(self) -> list[torch.Tensor]:
        """Load SBM graphs."""
        from tmgg.data.datasets.sbm import (
            generate_block_sizes,
            generate_sbm_adjacency,
        )

        config = self.dataset_config
        num_nodes = config["num_nodes"]
        p_intra = config.get("p_intra", 1.0)
        p_inter = config.get("p_inter", 0.0)

        if "block_sizes" in config:
            # Fixed partition
            partitions = [config["block_sizes"]]
            num_graphs = config.get("num_graphs", 1)
            partitions = partitions * num_graphs
        else:
            # Random partitions
            num_partitions = config.get("num_partitions", 100)
            partitions = generate_block_sizes(
                num_nodes,
                min_blocks=config.get("min_blocks", 2),
                max_blocks=config.get("max_blocks", 4),
                min_size=config.get("min_block_size", 2),
                max_size=config.get("max_block_size", 15),
            )
            if len(partitions) < num_partitions:
                logger.warning(
                    f"Only {len(partitions)} valid partitions found, "
                    f"requested {num_partitions}"
                )
            partitions = partitions[:num_partitions]

        matrices = []
        for partition in partitions:
            A = generate_sbm_adjacency(partition, p_intra, p_inter)
            matrices.append(torch.from_numpy(A).float())

        return matrices

    def _load_synthetic(self) -> list[torch.Tensor]:
        """Load synthetic graphs."""
        from tmgg.data.datasets.synthetic_graphs import (
            SyntheticGraphDataset,
        )

        graph_type = SyntheticGraphDataset.TYPE_ALIASES.get(
            self.dataset_name, self.dataset_name
        )

        config = self.dataset_config
        n = config.get("num_nodes", 50)
        num_graphs = config.get("num_graphs", 100)

        # Extract type-specific kwargs
        type_kwargs = {}
        if graph_type == "regular":
            type_kwargs["d"] = config.get("d", 3)
        elif graph_type == "erdos_renyi":
            type_kwargs["p"] = config.get("p", 0.1)
        elif graph_type == "watts_strogatz":
            type_kwargs["k"] = config.get("k", 4)
            type_kwargs["p"] = config.get("p", 0.3)
        elif graph_type == "random_geometric":
            type_kwargs["radius"] = config.get("radius", 0.3)
        elif graph_type == "configuration_model":
            type_kwargs["degree_sequence"] = config.get("degree_sequence")
        elif graph_type == "lfr":
            for key in ["tau1", "tau2", "mu", "average_degree", "min_community"]:
                if key in config:
                    type_kwargs[key] = config[key]
        elif graph_type == "ring_of_cliques":
            type_kwargs["num_cliques"] = config.get("num_cliques", 4)
            type_kwargs["clique_size"] = config.get("clique_size", 5)
        elif graph_type == "lollipop":
            type_kwargs["cluster_size"] = config.get("cluster_size", n // 2)
            type_kwargs["path_length"] = config.get("path_length", n - n // 2)

        dataset = SyntheticGraphDataset(
            graph_type=graph_type,
            num_nodes=n,
            num_graphs=num_graphs,
            seed=self.seed,
            **type_kwargs,
        )

        return [torch.from_numpy(A).float() for A in dataset.adjacencies]

    def _load_pyg(self) -> list[torch.Tensor]:
        """Load PyG benchmark datasets."""
        from tmgg.data.datasets.pyg_datasets import PyGDatasetWrapper

        config = self.dataset_config
        dataset = PyGDatasetWrapper(
            dataset_name=self.dataset_name,
            root=config.get("root"),
            max_graphs=config.get("max_graphs"),
        )

        return [torch.from_numpy(A).float() for A in dataset.adjacencies]

    def _process_batch(
        self,
        batch_matrices: list[torch.Tensor],
        batch_index: int,
        global_start_idx: int,
    ) -> None:
        """Compute decompositions for a batch and save."""
        eig_adj_vals_list: list[torch.Tensor] = []
        eig_adj_vecs_list: list[torch.Tensor] = []
        eig_lap_vals_list: list[torch.Tensor] = []
        eig_lap_vecs_list: list[torch.Tensor] = []
        metadata_list: list[dict[str, Any]] = []

        for i, A in enumerate(batch_matrices):
            # Compute Laplacian
            L = compute_laplacian(A)

            # Compute decompositions
            vals_a, vecs_a = torch.linalg.eigh(A.float())
            vals_l, vecs_l = torch.linalg.eigh(L.float())

            eig_adj_vals_list.append(vals_a)
            eig_adj_vecs_list.append(vecs_a)
            eig_lap_vals_list.append(vals_l)
            eig_lap_vecs_list.append(vecs_l)

            # Compute edge count from adjacency (symmetric, so divide by 2)
            edge_count = int(A.sum().item() / 2)

            metadata_list.append(
                {
                    "dataset_name": self.dataset_name,
                    "graph_index": global_start_idx + i,
                    "node_count": int(A.shape[0]),
                    "edge_count": edge_count,
                }
            )

        # Stack tensors - handle potential size differences by padding
        max_n = max(A.shape[0] for A in batch_matrices)
        padded_adj = self._pad_matrices(batch_matrices, max_n)
        padded_vals_a = self._pad_vectors(eig_adj_vals_list, max_n)
        padded_vecs_a = self._pad_matrices(eig_adj_vecs_list, max_n)
        padded_vals_l = self._pad_vectors(eig_lap_vals_list, max_n)
        padded_vecs_l = self._pad_matrices(eig_lap_vecs_list, max_n)

        save_decomposition_batch(
            self.output_dir,
            batch_index,
            padded_vals_a,
            padded_vecs_a,
            padded_vals_l,
            padded_vecs_l,
            padded_adj,
            metadata_list,
        )

    def _pad_matrices(
        self, matrices: list[torch.Tensor], target_size: int
    ) -> torch.Tensor:
        """Pad matrices to uniform size and stack."""
        padded = []
        for M in matrices:
            n = M.shape[0]
            if n < target_size:
                pad_size = target_size - n
                M_padded = torch.nn.functional.pad(
                    M, (0, pad_size, 0, pad_size), value=0.0
                )
                padded.append(M_padded)
            else:
                padded.append(M)
        return torch.stack(padded)

    def _pad_vectors(
        self, vectors: list[torch.Tensor], target_size: int
    ) -> torch.Tensor:
        """Pad vectors to uniform size and stack."""
        padded = []
        for v in vectors:
            n = v.shape[0]
            if n < target_size:
                pad_size = target_size - n
                v_padded = torch.nn.functional.pad(v, (0, pad_size), value=0.0)
                padded.append(v_padded)
            else:
                padded.append(v)
        return torch.stack(padded)

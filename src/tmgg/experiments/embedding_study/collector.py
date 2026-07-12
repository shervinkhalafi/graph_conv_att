"""Collector for embedding dimension studies across datasets.

Runs dimension search on multiple graphs and collects results into
a structured dataset for analysis. Stats are saved to JSON, embeddings
to safetensors.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file as save_safetensors

from .embeddings.dimension_search import (
    DimensionResult,
    DimensionSearcher,
    EmbeddingType,
)


@dataclass
class GraphMetadata:
    """Metadata about a graph in the study.

    Attributes
    ----------
    dataset_name
        Name of the source dataset.
    graph_idx
        Index within the dataset.
    num_nodes
        Number of nodes.
    num_edges
        Number of edges.
    density
        Edge density (edges / possible edges).
    """

    dataset_name: str
    graph_idx: int
    num_nodes: int
    num_edges: int
    density: float


@dataclass
class GraphEmbeddingStudy:
    """Complete embedding study results for a single graph.

    Attributes
    ----------
    metadata
        Graph metadata.
    results
        Dimension results for each embedding type tried.
    dim_results
        Full DimensionResult objects (with embeddings) for each type.
    adjacency
        Original adjacency matrix.
    """

    metadata: GraphMetadata
    results: dict[str, dict[str, Any]]
    dim_results: dict[str, DimensionResult]
    adjacency: torch.Tensor


class EmbeddingCollector:
    """Collects embedding dimension statistics across datasets.

    Runs dimension search for multiple graphs and embedding types,
    storing stats to JSON and embeddings to safetensors.
    """

    def __init__(
        self,
        searcher: DimensionSearcher | None = None,
        output_dir: Path | str | None = None,
        embedding_types: list[EmbeddingType] | None = None,
    ) -> None:
        """Initialize collector.

        Parameters
        ----------
        searcher
            Dimension searcher to use. Creates default if None.
        output_dir
            Directory for saving results. If None, results are only
            kept in memory.
        embedding_types
            Which embedding types to study. All types if None.
        """
        self.searcher = searcher or DimensionSearcher()
        self.output_dir = Path(output_dir) if output_dir else None
        self.embedding_types = embedding_types or list(EmbeddingType)
        self.studies: list[GraphEmbeddingStudy] = []

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_graph(
        self,
        adjacency: torch.Tensor,
        dataset_name: str,
        graph_idx: int,
    ) -> GraphEmbeddingStudy:
        """Process a single graph and collect embedding dimensions.

        Parameters
        ----------
        adjacency
            Adjacency matrix of shape (n, n).
        dataset_name
            Name of the source dataset.
        graph_idx
            Index of this graph in the dataset.

        Returns
        -------
        GraphEmbeddingStudy
            Complete study results for this graph (including embeddings).
        """
        n = adjacency.shape[0]
        num_edges = int((adjacency.sum() / 2).item())  # Undirected
        max_edges = n * (n - 1) // 2
        density = num_edges / max_edges if max_edges > 0 else 0.0

        metadata = GraphMetadata(
            dataset_name=dataset_name,
            graph_idx=graph_idx,
            num_nodes=n,
            num_edges=num_edges,
            density=density,
        )

        results: dict[str, dict[str, Any]] = {}
        dim_results: dict[str, DimensionResult] = {}

        for etype in self.embedding_types:
            dim_result = self.searcher.find_min_dimension(adjacency, etype)
            dim_results[etype.value] = dim_result

            # Stats for JSON (excluding tensor embeddings)
            results[etype.value] = {
                "min_dimension": dim_result.min_dimension,
                "final_fnorm": dim_result.final_fnorm,
                "final_accuracy": dim_result.final_accuracy,
                "converged": dim_result.converged,
                "fitter_type": dim_result.fitter_type,
                "search_steps": len(dim_result.search_history),
            }

        study = GraphEmbeddingStudy(
            metadata=metadata,
            results=results,
            dim_results=dim_results,
            adjacency=adjacency,
        )
        self.studies.append(study)
        return study

    def process_dataset(
        self,
        adjacency_matrices: list[torch.Tensor] | torch.Tensor,
        dataset_name: str,
        max_graphs: int | None = None,
    ) -> list[GraphEmbeddingStudy]:
        """Process all graphs in a dataset.

        Parameters
        ----------
        adjacency_matrices
            List of adjacency matrices or stacked tensor of shape (N, n, n).
        dataset_name
            Name of the dataset.
        max_graphs
            Maximum number of graphs to process. All if None.

        Returns
        -------
        list
            Study results for each processed graph.
        """
        if isinstance(adjacency_matrices, torch.Tensor):
            if adjacency_matrices.dim() == 3:
                matrices = [
                    adjacency_matrices[i] for i in range(len(adjacency_matrices))
                ]
            else:
                matrices = [adjacency_matrices]
        else:
            matrices = adjacency_matrices

        if max_graphs:
            matrices = matrices[:max_graphs]

        results = []
        for idx, adj in enumerate(matrices):
            study = self.process_graph(adj, dataset_name, idx)
            results.append(study)

        return results

    def save_results(
        self,
        stats_filename: str = "embedding_study.json",
        embeddings_filename: str = "embeddings.safetensors",
    ) -> tuple[Path, Path]:
        """Save stats to JSON and embeddings to safetensors.

        Parameters
        ----------
        stats_filename
            Output filename for JSON stats.
        embeddings_filename
            Output filename for safetensors embeddings.

        Returns
        -------
        tuple[Path, Path]
            Paths to (stats_file, embeddings_file).
        """
        if self.output_dir is None:
            raise ValueError("No output_dir specified")

        stats_path = self.output_dir / stats_filename
        embeddings_path = self.output_dir / embeddings_filename

        # Build JSON stats
        stats_data = []
        for study in self.studies:
            stats_data.append(
                {
                    "metadata": asdict(study.metadata),
                    "results": study.results,
                }
            )

        with open(stats_path, "w") as f:
            json.dump(stats_data, f, indent=2)

        # Build safetensors dict
        # Keys: {dataset}_{graph_idx:04d}_{embedding_type}_{X|Y|adj}
        tensors: dict[str, torch.Tensor] = {}

        for study in self.studies:
            ds = study.metadata.dataset_name
            idx = study.metadata.graph_idx
            prefix = f"{ds}_{idx:04d}"

            # Save adjacency
            tensors[f"{prefix}_adjacency"] = study.adjacency

            # Save embeddings for each type
            for etype_name, dim_result in study.dim_results.items():
                if dim_result.embeddings is None:
                    continue

                emb = dim_result.embeddings
                key_prefix = f"{prefix}_{etype_name}"

                tensors[f"{key_prefix}_X"] = emb.X
                if emb.Y is not None:
                    tensors[f"{key_prefix}_Y"] = emb.Y

        save_safetensors(tensors, embeddings_path)

        return stats_path, embeddings_path

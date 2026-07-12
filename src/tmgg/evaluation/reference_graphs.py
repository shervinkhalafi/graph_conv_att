"""Reference graph generation for MMD evaluation.

Provides ``generate_reference_graphs`` for constructing synthetic reference
datasets (SBM, Erdos-Renyi, etc.) used in MMD evaluation. The CLI
evaluation workflow lives in
``tmgg.experiments.discrete_diffusion_generative.evaluate_cli``.
"""

from __future__ import annotations

from typing import Any

import torch


def generate_reference_graphs(
    dataset_type: str,
    num_graphs: int,
    num_nodes: int,
    seed: int = 42,
    **kwargs: Any,
) -> list[torch.Tensor]:
    """Generate reference graphs from a synthetic dataset.

    Parameters
    ----------
    dataset_type
        Type of dataset. Supported: "sbm", "erdos_renyi", "watts_strogatz",
        "regular", "tree".
    num_graphs
        Number of graphs to generate.
    num_nodes
        Number of nodes per graph.
    seed
        Random seed.
    **kwargs
        Additional parameters for the dataset generator.

    Returns
    -------
    list[torch.Tensor]
        List of adjacency matrices.
    """
    from tmgg.data.datasets.synthetic_graphs import (
        SyntheticGraphDataset,
    )

    if dataset_type == "sbm":
        from tmgg.data.datasets.sbm import generate_sbm_batch

        p_intra = kwargs.get("p_intra", kwargs.get("p", 0.5))
        p_inter = kwargs.get("p_inter", kwargs.get("q", 0.1))
        num_blocks = kwargs.get("num_blocks", 2)
        batch = generate_sbm_batch(
            num_graphs,
            num_nodes,
            num_blocks=num_blocks,
            p_intra=p_intra,
            p_inter=p_inter,
            seed=seed,
        )
        return list(torch.from_numpy(batch).float())

    # SyntheticGraphDataset resolves aliases (e.g. "er" → "erdos_renyi") internally
    dataset = SyntheticGraphDataset(
        graph_type=dataset_type,
        num_nodes=num_nodes,
        num_graphs=num_graphs,
        seed=seed,
        **kwargs,
    )

    return [torch.from_numpy(dataset[i]).float() for i in range(len(dataset))]

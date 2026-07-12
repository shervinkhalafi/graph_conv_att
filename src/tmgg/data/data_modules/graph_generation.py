"""Shared graph-generation helpers for datamodules.

Centralizes the production graph-source dispatch so datamodules do not
carry separate ``graph_type`` switch trees for SBM, synthetic graphs, and
PyG-backed datasets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger
from torch_geometric.data import Data

from tmgg.data.datasets.pyg_datasets import PyGDatasetWrapper
from tmgg.data.datasets.sbm import (
    generate_block_sizes,
    generate_sbm_adjacency,
    generate_sbm_batch,  # pyright: ignore[reportAttributeAccessIssue]  # runtime-verified
)
from tmgg.data.datasets.synthetic_graphs import SyntheticGraphDataset

from ._split import split_indices

PYG_BENCHMARK_DATASETS = frozenset(
    {
        "qm9",
        "enzymes",
        "proteins",
        "collab",
        "deezer_ego_nets",
        "imdb_binary",
        "reddit_binary",
    }
)


@dataclass(frozen=True)
class GeneratedGraph:
    """Single generated graph plus its realized node count."""

    adjacency: npt.NDArray[np.float32]
    num_nodes: int


def uses_pyg_dataset_split(graph_type: str) -> bool:
    """Return whether ``graph_type`` should load a split from a PyG dataset."""

    return graph_type.lower() in PYG_BENCHMARK_DATASETS


def uses_single_pyg_graph(graph_type: str) -> bool:
    """Return whether ``graph_type`` denotes a single-graph PyG source."""

    return graph_type.lower().startswith("pyg_")


def _resolve_sbm_batch_kwargs(
    graph_config: dict[str, Any],
    *,
    default_num_blocks: int,
) -> tuple[int | tuple[int, int], float | None, float]:
    """Resolve batch-path SBM kwargs from ``graph_config``.

    Returns ``(num_blocks, block_size_alpha, diversity)`` for
    :func:`generate_sbm_batch`. ``num_blocks`` becomes an inclusive
    ``(min_blocks, max_blocks)`` range when those keys are present (so the block
    count K is sampled per graph, e.g. K ∈ {2, 3, 4} for the paper's synthetic
    SBM); otherwise the scalar ``num_blocks`` is used. ``block_size_alpha`` is
    the symmetric-Dirichlet block-size concentration (paper Appendix B.2.3,
    Eq. 28). ``diversity`` defaults to ``1.0`` whenever a range or a
    ``block_size_alpha`` is in play — the diverse generation mode requires it
    (see :func:`generate_sbm_batch`) — and to ``0.0`` otherwise, preserving the
    legacy fixed-mode RNG stream for scalar, no-α configs.
    """
    block_size_alpha = graph_config.get("block_size_alpha")

    num_blocks: int | tuple[int, int]
    if "num_blocks" in graph_config:
        num_blocks = graph_config["num_blocks"]
    elif "min_blocks" in graph_config and "max_blocks" in graph_config:
        num_blocks = (graph_config["min_blocks"], graph_config["max_blocks"])
    else:
        num_blocks = default_num_blocks

    varies = block_size_alpha is not None or isinstance(num_blocks, tuple)
    diversity: float = graph_config.get("diversity", 1.0 if varies else 0.0)
    return num_blocks, block_size_alpha, diversity


def generate_adjacency_batch(
    graph_type: str,
    *,
    num_nodes: int,
    num_graphs: int,
    graph_config: dict[str, Any],
    seed: int,
) -> npt.NDArray[np.float32]:
    """Generate a dense adjacency batch for SBM or synthetic graphs."""

    normalized_graph_type = graph_type.lower()

    if normalized_graph_type == "sbm":
        num_blocks, block_size_alpha, diversity = _resolve_sbm_batch_kwargs(
            graph_config, default_num_blocks=2
        )
        return generate_sbm_batch(
            num_graphs=num_graphs,
            num_nodes=num_nodes,
            num_blocks=num_blocks,
            p_intra=graph_config.get("p_intra", 0.7),
            p_inter=graph_config.get("p_inter", 0.1),
            block_size_alpha=block_size_alpha,
            diversity=diversity,
            seed=seed,
        )

    extra = {
        key: value
        for key, value in graph_config.items()
        if key not in {"num_nodes", "num_graphs", "seed"}
    }
    dataset = SyntheticGraphDataset(
        graph_type=normalized_graph_type,
        num_nodes=num_nodes,
        num_graphs=num_graphs,
        seed=seed,
        **extra,
    )
    return dataset.get_adjacency_matrices()


def generate_single_graph(
    graph_type: str,
    *,
    num_nodes: int,
    graph_config: dict[str, Any],
    seed: int,
) -> GeneratedGraph:
    """Generate one graph from an SBM, synthetic, or PyG source."""

    normalized_graph_type = graph_type.lower()

    if normalized_graph_type == "sbm":
        num_blocks, block_size_alpha, diversity = _resolve_sbm_batch_kwargs(
            graph_config, default_num_blocks=3
        )
        adjacency = generate_sbm_batch(
            num_graphs=1,
            num_nodes=num_nodes,
            num_blocks=num_blocks,
            p_intra=graph_config.get("p_intra", 0.7),
            p_inter=graph_config.get("p_inter", 0.05),
            block_size_alpha=block_size_alpha,
            diversity=diversity,
            seed=seed,
        )[0]
        return GeneratedGraph(adjacency=adjacency, num_nodes=int(adjacency.shape[0]))

    if uses_single_pyg_graph(normalized_graph_type):
        return _load_single_pyg_graph(
            graph_type=normalized_graph_type,
            graph_config=graph_config,
            seed=seed,
        )

    dataset = SyntheticGraphDataset(
        graph_type=normalized_graph_type,
        num_nodes=num_nodes,
        num_graphs=1,
        seed=seed,
        **graph_config,
    )
    adjacency = dataset.get_adjacency_matrices()[0]
    return GeneratedGraph(adjacency=adjacency, num_nodes=int(adjacency.shape[0]))


def load_pyg_dataset_split(
    graph_type: str,
    *,
    graph_config: dict[str, Any],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Data], list[Data], list[Data]]:
    """Load a PyG dataset and split it into train, val, and test lists."""

    wrapper = PyGDatasetWrapper(
        dataset_name=graph_type.lower(),
        root=graph_config.get("root"),
        max_graphs=graph_config.get("max_graphs"),
    )

    train_idx, val_idx, test_idx = split_indices(
        len(wrapper), train_ratio, val_ratio, graph_config.get("seed", seed)
    )
    return (
        [wrapper.data_list[index] for index in train_idx],
        [wrapper.data_list[index] for index in val_idx],
        [wrapper.data_list[index] for index in test_idx],
    )


def generate_multigraph_split(
    graph_type: str,
    *,
    num_nodes: int,
    num_graphs: int,
    graph_config: dict[str, Any],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate and split adjacency batches for multigraph datamodules."""

    normalized_graph_type = graph_type.lower()
    partition_mode = _resolve_sbm_partition_mode(
        graph_type=normalized_graph_type,
        graph_config=graph_config,
    )

    # Per-graph Dirichlet block sizes (paper α, ``block_size_alpha``) are
    # incompatible with fixed-partition enumeration: the enumerated/fixed path
    # builds explicit integer partitions via ``generate_sbm_adjacency`` and never
    # calls ``generate_sbm_batch``. When α is requested it therefore wins and
    # generation takes the batch path, emitting ``num_graphs`` graphs.
    uses_block_size_alpha = (
        normalized_graph_type == "sbm"
        and graph_config.get("block_size_alpha") is not None
    )
    if (
        normalized_graph_type == "sbm"
        and partition_mode in {"enumerated", "fixed"}
        and not uses_block_size_alpha
    ):
        return _generate_partitioned_sbm_split(
            num_nodes=num_nodes,
            graph_config=graph_config,
            seed=seed,
        )

    if uses_block_size_alpha and partition_mode in {"enumerated", "fixed"}:
        logger.warning(
            "SBM graph_config sets block_size_alpha={} with partition_mode={!r}; "
            "per-graph Dirichlet block sizes override fixed-partition enumeration, "
            "so generation uses the batch path ({} graphs) and ignores "
            "num_train_partitions/num_test_partitions.",
            graph_config.get("block_size_alpha"),
            partition_mode,
            num_graphs,
        )

    adjacencies = generate_adjacency_batch(
        graph_type=normalized_graph_type,
        num_nodes=num_nodes,
        num_graphs=num_graphs,
        graph_config=graph_config,
        seed=seed,
    )
    train_idx, val_idx, test_idx = split_indices(
        len(adjacencies), train_ratio, val_ratio, seed
    )
    return adjacencies[train_idx], adjacencies[val_idx], adjacencies[test_idx]


def _load_single_pyg_graph(
    *,
    graph_type: str,
    graph_config: dict[str, Any],
    seed: int,
) -> GeneratedGraph:
    """Load one graph from a PyG dataset and trim any wrapper padding."""

    dataset = PyGDatasetWrapper(
        dataset_name=graph_type.removeprefix("pyg_"),
        root=graph_config.get("root"),
    )

    graph_idx: int | None = graph_config.get("graph_idx")
    if graph_idx is None:
        rng = np.random.default_rng(seed)
        graph_idx = int(rng.integers(0, len(dataset)))

    if graph_idx >= len(dataset):
        raise ValueError(
            f"graph_idx {graph_idx} out of range for dataset with {len(dataset)} graphs"
        )

    actual_num_nodes = int(dataset.num_nodes[graph_idx])
    adjacency = dataset.adjacencies[graph_idx][:actual_num_nodes, :actual_num_nodes]
    return GeneratedGraph(adjacency=adjacency, num_nodes=actual_num_nodes)


def _resolve_sbm_partition_mode(
    *,
    graph_type: str,
    graph_config: dict[str, Any],
) -> str | None:
    """Resolve SBM partition-mode defaults from the graph config."""

    partition_mode = graph_config.get("partition_mode")
    if (
        graph_type == "sbm"
        and partition_mode is None
        and "num_train_partitions" in graph_config
    ):
        return "enumerated"
    return partition_mode


def _generate_partitioned_sbm_split(
    *,
    num_nodes: int,
    graph_config: dict[str, Any],
    seed: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate SBM train, val, and test splits from partition rules."""

    rng = np.random.default_rng(seed)
    p_intra: float = graph_config.get("p_intra", 0.7)
    p_inter: float = graph_config.get("p_inter", 0.1)
    partition_mode = graph_config.get("partition_mode", "equal")

    if partition_mode == "fixed":
        block_sizes = graph_config["block_sizes"]
        return (
            np.array([generate_sbm_adjacency(block_sizes, p_intra, p_inter, rng=rng)]),
            np.array([generate_sbm_adjacency(block_sizes, p_intra, p_inter, rng=rng)]),
            np.array([generate_sbm_adjacency(block_sizes, p_intra, p_inter, rng=rng)]),
        )

    all_partitions = generate_block_sizes(
        graph_config.get("num_nodes", num_nodes),
        min_blocks=graph_config.get("min_blocks", 2),
        max_blocks=graph_config.get("max_blocks", 4),
        min_size=graph_config.get("min_block_size", 2),
        max_size=graph_config.get("max_block_size", 15),
    )

    num_train = graph_config.get("num_train_partitions", 10)
    num_test = graph_config.get("num_test_partitions", 10)
    total_needed = num_train + num_test

    if len(all_partitions) < total_needed:
        raise ValueError(
            f"Not enough valid SBM partitions ({len(all_partitions)}) for "
            f"requested train ({num_train}) and test ({num_test}) partitions."
        )

    py_rng = random.Random(seed)
    train_partitions = py_rng.sample(all_partitions, num_train)
    remaining = [
        partition for partition in all_partitions if partition not in train_partitions
    ]
    num_val = min(5, max(1, len(remaining) // 2))
    val_partitions = py_rng.sample(remaining, num_val)
    test_remaining = [
        partition for partition in remaining if partition not in val_partitions
    ]
    test_partitions = py_rng.sample(test_remaining, num_test)

    return (
        _generate_from_partitions(
            train_partitions, p_intra=p_intra, p_inter=p_inter, rng=rng
        ),
        _generate_from_partitions(
            val_partitions, p_intra=p_intra, p_inter=p_inter, rng=rng
        ),
        _generate_from_partitions(
            test_partitions, p_intra=p_intra, p_inter=p_inter, rng=rng
        ),
    )


def _generate_from_partitions(
    partitions: list[list[int]],
    *,
    p_intra: float,
    p_inter: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    """Sample one adjacency per SBM partition."""

    return np.array(
        [
            generate_sbm_adjacency(partition, p_intra, p_inter, rng=rng)
            for partition in partitions
        ]
    )

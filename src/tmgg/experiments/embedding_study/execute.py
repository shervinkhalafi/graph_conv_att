"""Hydra-compatible execution entry point for embedding dimension study.

Dispatches to run or analyze phase based on ``config.phase``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf


def execute_embedding_study(config: DictConfig) -> dict[str, Any]:
    """Execute an embedding dimension study phase from Hydra config.

    Parameters
    ----------
    config
        Resolved OmegaConf configuration. Required keys:

        - ``phase``: one of ``"run"``, ``"analyze"``
        - ``paths.output_dir``: base output directory
        - ``seed``: random seed

    Returns
    -------
    dict
        Result dict with at least ``{"status": "completed", "phase": ...}``.
    """
    phase = str(config.phase)
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "run": _run_study,
        "analyze": _run_analyze,
    }

    if phase not in dispatch:
        raise ValueError(
            f"Unknown embedding_study phase: {phase!r}. "
            f"Expected one of {list(dispatch)}."
        )

    return dispatch[phase](config, output_dir)


def _run_study(config: DictConfig, output_dir: Path) -> dict[str, Any]:
    from tmgg.experiments.embedding_study.collector import EmbeddingCollector
    from tmgg.experiments.embedding_study.embeddings.dimension_search import (
        DimensionSearcher,
        EmbeddingType,
    )

    # Parse embedding types
    methods = str(config.get("methods", "all"))
    if methods == "all":
        embedding_types = list(EmbeddingType)
    elif methods == "symmetric":
        embedding_types = [
            EmbeddingType.LPCA_SYMMETRIC,
            EmbeddingType.DOT_PRODUCT_SYMMETRIC,
            EmbeddingType.DOT_THRESHOLD_SYMMETRIC,
            EmbeddingType.DISTANCE_THRESHOLD,
            EmbeddingType.ORTHOGONAL,
        ]
    else:
        embedding_types = [EmbeddingType(m.strip()) for m in methods.split(",")]

    # Search tolerances
    search_cfg = config.get("search", {})
    tol_fnorm: float = search_cfg.get("tol_fnorm", 0.01)
    tol_accuracy: float = search_cfg.get("tol_accuracy", 0.99)
    fitter = str(config.get("fitter", "both"))

    searcher = DimensionSearcher(
        tol_fnorm=tol_fnorm,
        tol_accuracy=tol_accuracy,
        fitter=fitter,  # type: ignore[arg-type]  # validated by DimensionSearcher
    )
    collector = EmbeddingCollector(
        searcher=searcher,
        output_dir=output_dir,
        embedding_types=embedding_types,
    )

    # Process datasets
    datasets_raw = config.get("datasets", ["sbm"])
    datasets: list[str] = (
        cast(list[str], OmegaConf.to_container(datasets_raw, resolve=True))
        if OmegaConf.is_list(datasets_raw)
        else [str(datasets_raw)]
    )

    num_graphs: int = config.get("num_graphs", 10)
    num_nodes: int = config.get("num_nodes", 50)

    total_processed = 0
    for dataset_name in datasets:
        graphs = _generate_graphs(dataset_name, num_graphs, num_nodes, config.seed)
        collector.process_dataset(graphs, dataset_name, max_graphs=num_graphs)
        total_processed += len(graphs)

    stats_path, embeddings_path = collector.save_results()

    return {
        "status": "completed",
        "phase": "run",
        "total_graphs": total_processed,
        "stats_path": str(stats_path),
        "embeddings_path": str(embeddings_path),
    }


def _run_analyze(config: DictConfig, output_dir: Path) -> dict[str, Any]:
    from tmgg.experiments.embedding_study.analysis import analyze_results, print_summary

    results_path = output_dir / "embedding_study.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No results file at {results_path}. Run the 'run' phase first."
        )

    summaries = analyze_results(results_path)
    print_summary(summaries)

    return {
        "status": "completed",
        "phase": "analyze",
        "num_embedding_types": len(summaries),
    }


def _generate_graphs(
    dataset_name: str,
    num_graphs: int,
    num_nodes: int,
    seed: int,
) -> list[torch.Tensor]:
    """Generate graphs for the embedding study.

    Parameters
    ----------
    dataset_name
        Name of the dataset type (currently only "sbm").
    num_graphs
        Number of graphs to generate.
    num_nodes
        Number of nodes per graph.
    seed
        Random seed.

    Returns
    -------
    list[torch.Tensor]
        Dense square adjacency tensors, one per generated graph.
    """
    torch.manual_seed(seed)

    if dataset_name == "sbm":
        from tmgg.data.datasets.sbm import generate_sbm_batch

        batch = generate_sbm_batch(
            num_graphs=num_graphs, num_nodes=num_nodes, seed=seed
        )
        return list(torch.from_numpy(batch).float())

    raise ValueError(f"Unknown dataset for embedding study: {dataset_name!r}")

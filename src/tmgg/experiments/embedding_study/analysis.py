"""Analysis utilities for embedding dimension studies.

Provides functions to summarize and compare embedding dimension
results across graphs and embedding types.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DimensionSummary:
    """Summary statistics for embedding dimensions.

    Attributes
    ----------
    embedding_type
        Name of the embedding type.
    num_graphs
        Number of graphs analyzed.
    num_converged
        Number that achieved target quality.
    mean_dimension
        Mean minimum dimension.
    std_dimension
        Standard deviation of dimensions.
    min_dimension
        Smallest dimension found.
    max_dimension
        Largest dimension found.
    mean_ratio_sqrt_n
        Mean ratio of dimension to sqrt(n).
    mean_accuracy
        Mean final edge accuracy.
    """

    embedding_type: str
    num_graphs: int
    num_converged: int
    mean_dimension: float
    std_dimension: float
    min_dimension: int
    max_dimension: int
    mean_ratio_sqrt_n: float
    mean_accuracy: float


def analyze_results(
    results_path: Path | str,
) -> dict[str, DimensionSummary]:
    """Analyze collected embedding study results.

    Parameters
    ----------
    results_path
        Path to JSON results file from EmbeddingCollector.

    Returns
    -------
    dict
        Mapping from embedding type to summary statistics.
    """
    with open(results_path) as f:
        data = json.load(f)

    # Group by embedding type
    by_type: dict[str, list[dict[str, Any]]] = {}

    for study in data:
        metadata = study["metadata"]
        num_nodes = metadata["num_nodes"]
        sqrt_n = math.sqrt(num_nodes)

        for etype, result in study["results"].items():
            if etype not in by_type:
                by_type[etype] = []

            by_type[etype].append(
                {
                    "dimension": result["min_dimension"],
                    "converged": result["converged"],
                    "accuracy": result["final_accuracy"],
                    "sqrt_n": sqrt_n,
                    "ratio": result["min_dimension"] / sqrt_n,
                }
            )

    # Compute summaries
    summaries = {}
    for etype, records in by_type.items():
        dimensions = [r["dimension"] for r in records]
        accuracies = [r["accuracy"] for r in records]
        ratios = [r["ratio"] for r in records]
        converged = sum(1 for r in records if r["converged"])

        n = len(dimensions)
        mean_dim = sum(dimensions) / n
        var_dim = sum((d - mean_dim) ** 2 for d in dimensions) / n
        std_dim = math.sqrt(var_dim)

        summaries[etype] = DimensionSummary(
            embedding_type=etype,
            num_graphs=n,
            num_converged=converged,
            mean_dimension=mean_dim,
            std_dimension=std_dim,
            min_dimension=min(dimensions),
            max_dimension=max(dimensions),
            mean_ratio_sqrt_n=sum(ratios) / n,
            mean_accuracy=sum(accuracies) / n,
        )

    return summaries


def print_summary(summaries: dict[str, DimensionSummary]) -> None:
    """Print formatted summary of embedding study results.

    Parameters
    ----------
    summaries
        Dictionary of summaries from analyze_results.
    """
    print("\n" + "=" * 80)
    print("EMBEDDING DIMENSION STUDY SUMMARY")
    print("=" * 80 + "\n")

    # Sort by mean dimension
    sorted_types = sorted(summaries.keys(), key=lambda t: summaries[t].mean_dimension)

    header = f"{'Embedding Type':<30} {'n':>5} {'Conv':>5} {'Mean':>8} {'Std':>8} {'Min':>5} {'Max':>5} {'d/√n':>8}"
    print(header)
    print("-" * len(header))

    for etype in sorted_types:
        s = summaries[etype]
        row = (
            f"{s.embedding_type:<30} "
            f"{s.num_graphs:>5} "
            f"{s.num_converged:>5} "
            f"{s.mean_dimension:>8.2f} "
            f"{s.std_dimension:>8.2f} "
            f"{s.min_dimension:>5} "
            f"{s.max_dimension:>5} "
            f"{s.mean_ratio_sqrt_n:>8.2f}"
        )
        print(row)

    print("\n" + "=" * 80)

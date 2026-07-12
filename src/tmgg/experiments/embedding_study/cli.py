"""Command-line interface for embedding dimension studies.

Usage:
    tmgg-embedding-study run --datasets sbm --output results/
    tmgg-embedding-study analyze --input results/embedding_study.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tmgg.experiments.embedding_study.analysis import (
    analyze_results,
    print_summary,
)
from tmgg.experiments.embedding_study.collector import EmbeddingCollector

from .embeddings.dimension_search import (
    DimensionSearcher,
    EmbeddingType,
)
from .embeddings.fitters.gauge_stabilized import GaugeStabilizedConfig


def run_study(args: argparse.Namespace) -> None:
    """Run embedding dimension study on specified datasets."""
    output_dir = Path(args.output)

    # Parse embedding types
    if args.methods == "all":
        embedding_types = list(EmbeddingType)
    elif args.methods == "symmetric":
        embedding_types = [
            EmbeddingType.LPCA_SYMMETRIC,
            EmbeddingType.DOT_PRODUCT_SYMMETRIC,
            EmbeddingType.DOT_THRESHOLD_SYMMETRIC,
            EmbeddingType.DISTANCE_THRESHOLD,
            EmbeddingType.ORTHOGONAL,
        ]
    else:
        # Parse comma-separated list
        embedding_types = [EmbeddingType(m.strip()) for m in args.methods.split(",")]

    # Create gauge config if using gauge-stabilized fitter
    gauge_config = None
    if args.fitter == "gauge-stabilized":
        gauge_config = GaugeStabilizedConfig(
            alpha=args.hadamard_alpha,
            lambda_had=args.hadamard_lambda,
            use_anchor=not args.no_hadamard_anchor,
        )

    # Create searcher and collector
    searcher = DimensionSearcher(
        tol_fnorm=args.tol_fnorm,
        tol_accuracy=args.tol_accuracy,
        fitter=args.fitter,
        gauge_config=gauge_config,
    )
    collector = EmbeddingCollector(
        searcher=searcher,
        output_dir=output_dir,
        embedding_types=embedding_types,
    )

    # Process datasets
    datasets = args.datasets.split(",")

    for dataset_name in datasets:
        dataset_name = dataset_name.strip()
        print(f"\nProcessing dataset: {dataset_name}")

        if dataset_name == "sbm":
            from tmgg.data.datasets.sbm import generate_sbm_batch

            batch = generate_sbm_batch(
                num_graphs=args.num_graphs, num_nodes=args.num_nodes
            )
            graphs = list(torch.from_numpy(batch).float())
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        for idx, graph in enumerate(graphs):
            print(f"  Graph {idx + 1}/{len(graphs)} (n={graph.shape[0]})...", end=" ")
            study = collector.process_graph(graph, dataset_name, idx)
            dims = [r["min_dimension"] for r in study.results.values()]
            print(f"dims: {min(dims)}-{max(dims)}")

    # Save results (stats JSON + embeddings safetensors)
    stats_path, embeddings_path = collector.save_results()
    print(f"\nStats saved to: {stats_path}")
    print(f"Embeddings saved to: {embeddings_path}")

    # Print summary
    summaries = analyze_results(stats_path)
    print_summary(summaries)


def analyze_study(args: argparse.Namespace) -> None:
    """Analyze existing embedding study results."""
    results_path = Path(args.input)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    summaries = analyze_results(results_path)
    print_summary(summaries)


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Embedding dimension study for graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run embedding dimension study",
    )
    run_parser.add_argument(
        "--datasets",
        type=str,
        default="sbm",
        help="Comma-separated list of datasets (default: sbm)",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default="results/embedding_study",
        help="Output directory (default: results/embedding_study)",
    )
    run_parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Embedding methods: 'all', 'symmetric', or comma-separated list",
    )
    run_parser.add_argument(
        "--fitter",
        type=str,
        choices=[
            "gradient",
            "spectral",
            "both",
            "gauge-stabilized",
            "gauge-stabilized-svd",
        ],
        default="both",
        help=(
            "Fitter type. 'gauge-stabilized' uses Θ-space interpolation, "
            "'gauge-stabilized-svd' uses SVD-space interpolation (both LPCA only)"
        ),
    )
    run_parser.add_argument(
        "--hadamard-alpha",
        type=float,
        default=0.1,
        help="Interpolation weight for gauge-stabilized init (default: 0.1 = 10%% Hadamard)",
    )
    run_parser.add_argument(
        "--hadamard-lambda",
        type=float,
        default=0.01,
        help="Hadamard anchor regularization weight (default: 0.01)",
    )
    run_parser.add_argument(
        "--no-hadamard-anchor",
        action="store_true",
        help="Disable Hadamard anchor regularization (init-only mode)",
    )
    run_parser.add_argument(
        "--tol-fnorm",
        type=float,
        default=0.01,
        help="Frobenius norm tolerance (default: 0.01)",
    )
    run_parser.add_argument(
        "--tol-accuracy",
        type=float,
        default=0.99,
        help="Edge accuracy tolerance (default: 0.99)",
    )
    run_parser.add_argument(
        "--num-graphs",
        type=int,
        default=10,
        help="Number of graphs to generate per synthetic dataset (default: 10)",
    )
    run_parser.add_argument(
        "--num-nodes",
        type=int,
        default=50,
        help="Number of nodes per graph (default: 50)",
    )
    run_parser.set_defaults(func=run_study)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze existing study results",
    )
    analyze_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to results JSON file",
    )
    analyze_parser.set_defaults(func=analyze_study)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

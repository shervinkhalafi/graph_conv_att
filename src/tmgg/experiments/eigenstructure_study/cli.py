"""CLI interface for eigenstructure study tools.

Provides three commands:
- collect: Compute and store eigendecompositions for a dataset
- analyze: Run spectral analysis on collected data
- noised: Collect eigendecompositions for noised graphs
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import click
from loguru import logger


def setup_logging(verbose: bool) -> None:
    """Configure loguru for CLI output."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<level>{level: <8}</level> | {message}",
    )


@click.group()
def main() -> None:
    """Graph eigenstructure study tools.

    Three-phase pipeline for analyzing graph spectral properties:

    \b
    1. collect - Compute eigendecompositions for a dataset
    2. analyze - Compute spectral statistics on collected data
    3. noised  - Collect decompositions for noised graphs
    """
    pass


@main.command()
@click.option(
    "--dataset",
    "-d",
    required=True,
    help="Dataset type (sbm, anu, classical, nx, regular, tree, lfr, er, ws, rg, cm, qm9, enzymes, proteins)",
)
@click.option(
    "--dataset-config",
    "-c",
    required=True,
    help="JSON config string or path to JSON file with dataset parameters",
)
@click.option(
    "--output-dir",
    "-o",
    default="results/eigenstructure_study",
    type=click.Path(path_type=Path),
    help="Output directory for safetensors files (default: results/eigenstructure_study)",
)
@click.option(
    "--batch-size",
    "-b",
    default=64,
    help="Number of graphs per batch",
)
@click.option(
    "--seed",
    "-s",
    default=42,
    help="Random seed for reproducibility",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable debug logging",
)
def collect(
    dataset: str,
    dataset_config: str,
    output_dir: Path,
    batch_size: int,
    seed: int,
    verbose: bool,
) -> None:
    """Phase 1: Collect eigendecompositions for a dataset.

    Computes both adjacency and Laplacian decompositions for all graphs
    in the specified dataset, storing results in safetensors format.

    \b
    Examples:
        # SBM dataset with fixed partition
        tmgg-eigenstructure collect -d sbm \\
            -c '{"num_nodes": 50, "p_intra": 0.8, "p_inter": 0.1, "num_partitions": 100}' \\
            -o ./eigen_data/sbm

        # Erdos-Renyi graphs
        tmgg-eigenstructure collect -d er \\
            -c '{"num_nodes": 30, "num_graphs": 200, "p": 0.15}' \\
            -o ./eigen_data/er

        # PyG benchmark
        tmgg-eigenstructure collect -d enzymes \\
            -c '{"max_graphs": 500}' \\
            -o ./eigen_data/enzymes
    """
    setup_logging(verbose)

    # Parse config: either JSON string or file path
    if dataset_config.startswith("{"):
        config = json.loads(dataset_config)
    else:
        config_path = Path(dataset_config)
        if not config_path.exists():
            raise click.ClickException(f"Config file not found: {config_path}")
        with open(config_path) as f:
            config = json.load(f)

    from .collector import EigenstructureCollector

    collector = EigenstructureCollector(
        dataset_name=dataset,
        dataset_config=config,
        output_dir=output_dir,
        batch_size=batch_size,
        seed=seed,
    )

    collector.collect()
    click.echo(f"Collection complete. Output: {output_dir}")


@main.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory with collected eigendecompositions",
)
@click.option(
    "--output-dir",
    "-o",
    default="results/eigenstructure_study",
    type=click.Path(path_type=Path),
    help="Output directory for analysis results (default: results/eigenstructure_study)",
)
@click.option(
    "--subspace-k",
    "-k",
    default=10,
    help="Number of eigenvectors for subspace analysis",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable debug logging",
)
def analyze(
    input_dir: Path,
    output_dir: Path,
    subspace_k: int,
    verbose: bool,
) -> None:
    """Phase 2: Analyze collected eigenstructure data.

    Computes spectral statistics including:
    - Spectral gap (adjacency)
    - Algebraic connectivity (Laplacian)
    - Eigenvalue entropy
    - Eigenvector coherence
    - Effective rank
    - Subspace distances (optional)

    \b
    Examples:
        tmgg-eigenstructure analyze -i ./eigen_data/sbm -o ./analysis/sbm
    """
    setup_logging(verbose)

    from .analyzer import SpectralAnalyzer

    analyzer = SpectralAnalyzer(input_dir)

    result = analyzer.analyze()

    # Keep the main analysis artifact on the analyzer-owned save path so the
    # CLI and Hydra execution route through one write boundary.
    output_path = analyzer.save_results(result, output_dir)

    # Compute and save subspace distances if requested
    if subspace_k > 0:
        subspace_results = analyzer.compute_subspace_distances(k=subspace_k)
        subspace_path = output_dir / "subspace_analysis.json"
        with open(subspace_path, "w") as f:
            json.dump(subspace_results, f, indent=2)
        click.echo(f"Subspace analysis saved to: {subspace_path}")

    click.echo(f"Analysis complete. Results: {output_path}")
    click.echo()
    click.echo("Summary:")
    click.echo(f"  Dataset: {result.dataset_name}")
    click.echo(f"  Graphs: {result.num_graphs}")
    click.echo(
        f"  Spectral gap: {result.spectral_gap_mean:.4f} +/- {result.spectral_gap_std:.4f}"
    )
    click.echo(
        f"  Algebraic connectivity: {result.algebraic_connectivity_mean:.4f} +/- {result.algebraic_connectivity_std:.4f}"
    )
    click.echo(
        f"  Coherence: {result.coherence_mean:.4f} +/- {result.coherence_std:.4f}"
    )
    click.echo(f"  Effective rank (adj): {result.effective_rank_adj_mean:.2f}")


@main.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory with original (Phase 1) decompositions",
)
@click.option(
    "--output-dir",
    "-o",
    default="results/eigenstructure_study/noised",
    type=click.Path(path_type=Path),
    help="Output directory for noised decompositions (default: results/eigenstructure_study/noised)",
)
@click.option(
    "--noise-type",
    "-t",
    required=True,
    type=click.Choice(["gaussian", "digress", "rotation"]),
    help="Type of noise to apply",
)
@click.option(
    "--noise-levels",
    "-n",
    required=True,
    help="Comma-separated noise levels (e.g., '0.01,0.05,0.1,0.2')",
)
@click.option(
    "--rotation-k",
    default=None,
    type=int,
    help="Dimension for rotation noise skew matrix (required for rotation noise)",
)
@click.option(
    "--seed",
    "-s",
    default=42,
    help="Random seed",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable debug logging",
)
def noised(
    input_dir: Path,
    output_dir: Path,
    noise_type: str,
    noise_levels: str,
    rotation_k: int | None,
    seed: int,
    verbose: bool,
) -> None:
    """Phase 3: Collect eigendecompositions for noised graphs.

    Reads original adjacency matrices from Phase 1 output, applies noise
    at the specified levels, and computes new decompositions.

    Creates subdirectories eps_X.XXXX/ for each noise level.

    \b
    Examples:
        # Gaussian noise at multiple levels
        tmgg-eigenstructure noised -i ./eigen_data/sbm -o ./noised_data/sbm \\
            -t gaussian -n 0.01,0.05,0.1,0.2,0.3

        # Digress (edge flip) noise
        tmgg-eigenstructure noised -i ./eigen_data/er -o ./noised_data/er \\
            -t digress -n 0.05,0.1,0.2

        # Rotation noise
        tmgg-eigenstructure noised -i ./eigen_data/sbm -o ./noised_data/sbm_rot \\
            -t rotation -n 0.1,0.3,0.5 --rotation-k 20
    """
    setup_logging(verbose)

    # Parse noise levels
    levels = [float(x.strip()) for x in noise_levels.split(",")]

    from .noised_collector import NoisedEigenstructureCollector

    collector = NoisedEigenstructureCollector(
        input_dir=input_dir,
        output_dir=output_dir,
        noise_type=noise_type,
        noise_levels=levels,
        rotation_k=rotation_k,
        seed=seed,
    )
    collector.collect()
    click.echo(f"Noised collection complete. Output: {output_dir}")
    click.echo(f"Noise levels: {levels}")


@main.command()
@click.option(
    "--original-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory with original (Phase 1) decompositions",
)
@click.option(
    "--noised-dir",
    "-n",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory with noised decompositions (contains eps_*/ subdirs)",
)
@click.option(
    "--output-dir",
    "-o",
    default="results/eigenstructure_study",
    type=click.Path(path_type=Path),
    help="Output directory for comparison results (default: results/eigenstructure_study)",
)
@click.option(
    "--subspace-k",
    "-k",
    default=10,
    help="Number of eigenvectors for subspace comparison",
)
@click.option(
    "--procrustes-k",
    "-p",
    default="1,2,4,8,16",
    help="Comma-separated k values for Procrustes rotation analysis (default: 1,2,4,8,16)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable debug logging",
)
def compare(
    original_dir: Path,
    noised_dir: Path,
    output_dir: Path,
    subspace_k: int,
    procrustes_k: str,
    verbose: bool,
) -> None:
    """Compare original and noised eigenstructure.

    Computes delta metrics between original and noised graphs:
    - Eigengap delta: Change in spectral gap (lambda_max - lambda_{max-1})
    - Algebraic connectivity delta: Change in Fiedler value (lambda_2 of Laplacian)
    - Eigenvalue drift: Relative L2 distance between spectra
    - Subspace distance: Frobenius norm of projection difference
    - Procrustes rotation: Optimal rotation angle and residual for eigenvector alignment

    \b
    Examples:
        tmgg-eigenstructure compare \\
            -i ./eigen_data/sbm \\
            -n ./noised_data/sbm \\
            -o ./comparison/sbm
    """
    setup_logging(verbose)

    # Parse Procrustes k values
    procrustes_k_values = [int(x.strip()) for x in procrustes_k.split(",")]

    from .noised_collector import NoisedAnalysisComparator

    comparator = NoisedAnalysisComparator(original_dir, noised_dir)

    results = comparator.compute_full_comparison(
        k=subspace_k, procrustes_k_values=procrustes_k_values
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison.json"
    with open(output_path, "w") as f:
        json.dump([result.to_json_dict() for result in results], f, indent=2)

    click.echo(f"Comparison complete. Results: {output_path}")
    click.echo()
    click.echo("Summary (delta metrics):")
    for r in results:
        eps = r.noise_level
        click.echo(f"  eps={eps:.4f}:")
        click.echo(
            f"    eigengap_delta={r.eigengap_delta_relative.mean:+.4f} +/- "
            f"{r.eigengap_delta_relative.std:.4f}"
        )
        click.echo(
            f"    alg_conn_delta={r.algebraic_connectivity_delta_relative.mean:+.4f} +/- "
            f"{r.algebraic_connectivity_delta_relative.std:.4f}"
        )
        click.echo(
            f"    eigenvalue_drift_adj={r.eigenvalue_drift_adjacency.mean:.4f} +/- "
            f"{r.eigenvalue_drift_adjacency.std:.4f}"
        )
        click.echo(
            f"    subspace_distance={r.subspace_distance.mean:.4f} +/- "
            f"{r.subspace_distance.std:.4f}"
        )
        # Output Procrustes rotation metrics
        click.echo("    Procrustes rotation:")
        for k in procrustes_k_values:
            stats = r.procrustes_by_k.get(k)
            if stats is not None:
                click.echo(
                    f"      k={k}: angle={stats.angle.mean:.4f}+/-{stats.angle.std:.4f} rad"
                )


@main.command()
@click.option(
    "--original-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory with original (Phase 1) decompositions",
)
@click.option(
    "--noised-dir",
    "-n",
    default=None,
    type=click.Path(path_type=Path),
    help="Directory with noised decompositions (optional, for evolution study)",
)
@click.option(
    "--output-dir",
    "-o",
    default="results/eigenstructure_study",
    type=click.Path(path_type=Path),
    help="Output directory for covariance results (default: results/eigenstructure_study)",
)
@click.option(
    "--matrix-type",
    "-m",
    default="adjacency",
    type=click.Choice(["adjacency", "laplacian"]),
    help="Matrix type for eigenvalue covariance (default: adjacency)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable debug logging",
)
def covariance(
    original_dir: Path,
    noised_dir: Path | None,
    output_dir: Path,
    matrix_type: str,
    verbose: bool,
) -> None:
    """Compute eigenvalue covariance matrix.

    Computes the k×k covariance matrix of eigenvalues across all graphs,
    showing how eigenvalue positions covary in the population. High off-diagonal
    values indicate eigenvalue positions that tend to move together.

    If --noised-dir is provided, also computes covariance evolution showing
    how the covariance structure changes at each noise level.

    \b
    Examples:
        # Covariance for original dataset only
        tmgg-eigenstructure covariance \\
            --original-dir ./eigen_data/sbm \\
            --output-dir ./results/covariance

        # Covariance evolution with noise
        tmgg-eigenstructure covariance \\
            --original-dir ./eigen_data/sbm \\
            --noised-dir ./noised_data/sbm \\
            --output-dir ./results/covariance \\
            --matrix-type laplacian
    """
    setup_logging(verbose)

    output_dir.mkdir(parents=True, exist_ok=True)

    if noised_dir is not None and Path(noised_dir).exists():
        # Compute covariance evolution across noise levels
        from .noised_collector import NoisedAnalysisComparator

        comparator = NoisedAnalysisComparator(original_dir, noised_dir)
        evolution = comparator.compute_covariance_evolution(matrix_type)

        output_path = output_dir / "covariance_evolution.json"
        with open(output_path, "w") as f:
            json.dump(evolution.to_json_dict(), f, indent=2)

        click.echo(f"Covariance evolution saved to: {output_path}")
        click.echo()
        click.echo("Original covariance summary:")
        orig = evolution.original
        click.echo(f"  Frobenius norm: {orig.frobenius_norm:.4f}")
        click.echo(f"  Trace (total variance): {orig.trace:.4f}")
        click.echo(f"  Off-diagonal ratio: {orig.off_diagonal_ratio:.4f}")
        click.echo()
        click.echo("Evolution with noise:")
        for item in evolution.per_noise_level:
            eps = item.noise_level
            click.echo(
                f"  eps={eps:.4f}: "
                f"frob_delta={item.frobenius_delta_relative:+.4f}, "
                f"off_diag_delta={item.off_diagonal_delta_relative:+.4f}"
            )

    else:
        # Compute covariance for original dataset only
        from .analyzer import SpectralAnalyzer

        analyzer = SpectralAnalyzer(original_dir)
        result = analyzer.compute_eigenvalue_covariance(matrix_type)

        output_path = output_dir / "covariance.json"
        with open(output_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        click.echo(f"Covariance analysis saved to: {output_path}")
        click.echo()
        click.echo("Summary:")
        click.echo(f"  Matrix type: {result.matrix_type}")
        click.echo(f"  Graphs: {result.num_graphs}")
        click.echo(f"  Eigenvalues: {result.num_eigenvalues}")
        click.echo(f"  Frobenius norm: {result.frobenius_norm:.4f}")
        click.echo(f"  Trace (total variance): {result.trace:.4f}")
        click.echo(f"  Off-diagonal sum: {result.off_diagonal_sum:.4f}")
        click.echo(f"  Off-diagonal ratio: {result.off_diagonal_ratio:.4f}")
        click.echo(f"  Condition number: {result.condition_number:.4f}")


@main.command("list-remote")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable debug logging",
)
def list_remote(verbose: bool) -> None:
    """List eigenstructure studies in Modal volume.

    Requires Modal app to be deployed and Modal credentials configured.

    \b
    Examples:
        tmgg-eigenstructure list-remote
    """
    setup_logging(verbose)

    try:
        import modal
    except ImportError:
        raise click.ClickException(
            "modal package is required: pip install modal"
        ) from None

    try:
        # Get reference to deployed function
        fn = modal.Function.from_name("graph-denoising", "eigenstructure_list")
        studies = fn.remote()

        if not studies:
            click.echo("No eigenstructure studies found in Modal volume.")
            return

        click.echo(f"Found {len(studies)} study directories:")
        click.echo()
        for study in studies:
            click.echo(f"  {study['name']}/")
            if study["has_manifest"]:
                click.echo(f"    Batches: {study['batch_count']}")
            if study["noised_levels"]:
                click.echo(f"    Noised levels: {', '.join(study['noised_levels'])}")

    except modal.exception.NotFoundError:
        raise click.ClickException(
            "Modal function not found. Deploy the Modal app first with: "
            "mise run modal-deploy"
        ) from None


@main.command()
@click.option(
    "--remote-path",
    "-r",
    required=True,
    help="Path to study within Modal volume (e.g., 'sbm_study')",
)
@click.option(
    "--local-path",
    "-l",
    required=True,
    type=click.Path(path_type=Path),
    help="Local destination directory",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable debug logging",
)
def download(
    remote_path: str,
    local_path: Path,
    verbose: bool,
) -> None:
    """Download eigenstructure study from Modal volume.

    Downloads safetensors, metadata, and analysis files from Modal
    eigenstructure volume to local filesystem.

    Requires Modal app to be deployed and Modal credentials configured.

    \b
    Examples:
        tmgg-eigenstructure download \\
            --remote-path sbm_study \\
            --local-path ./downloaded/sbm

        tmgg-eigenstructure download \\
            -r my_experiment/original \\
            -l ./local_data
    """
    setup_logging(verbose)

    try:
        import modal
    except ImportError:
        raise click.ClickException(
            "modal package is required: pip install modal"
        ) from None

    try:
        local_path.mkdir(parents=True, exist_ok=True)

        click.echo(f"Downloading from Modal: {remote_path}")
        click.echo(f"To local: {local_path}")

        # Get references to deployed Modal functions
        list_files_fn = modal.Function.from_name(
            "graph-denoising", "eigenstructure_list_files"
        )
        read_file_fn = modal.Function.from_name(
            "graph-denoising", "eigenstructure_read_file"
        )

        # List files in remote path
        files = list_files_fn.remote(remote_path)

        if not files:
            raise click.ClickException(f"No files found at: {remote_path}")

        click.echo(f"Found {len(files)} files to download")

        # Download each file
        from tqdm import tqdm

        for file_info in tqdm(files, desc="Downloading"):
            rel_path = file_info["rel_path"]
            remote_file_path = file_info["path"]

            local_file_path = local_path / rel_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file content
            content = read_file_fn.remote(remote_file_path)

            with open(local_file_path, "wb") as f:
                f.write(content)

        click.echo(f"Download complete: {len(files)} files")

    except modal.exception.NotFoundError:
        raise click.ClickException(
            "Modal functions not found. Deploy the Modal app first with: "
            "mise run modal-deploy"
        ) from None


if __name__ == "__main__":
    main()

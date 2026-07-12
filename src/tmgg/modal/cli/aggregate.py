"""Pull MMD evaluation results from the Modal volume into a Polars Parquet file.

Downloads ``mmd_evaluation_*.json`` files written by the ``evaluate``
command, parses them, and produces a flat table with one row per
(run_id, checkpoint, evaluation-label).
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import click
import polars as pl
from tqdm import tqdm

MODAL_VOLUME_NAME = "graph-denoising-outputs"


# ---------------------------------------------------------------------------
# Helpers (pure logic, no CLI concerns)
# ---------------------------------------------------------------------------


def parse_run_id(run_id: str) -> dict[str, str | int | None]:
    """Extract metadata fields from a run ID string.

    Expected pattern:
    ``stage2_{model}_{dataset}_{embedding}_lr{lr}_wd{wd}_k{k}_s{seed}``

    Parameters
    ----------
    run_id
        Full run identifier.

    Returns
    -------
    dict
        Extracted metadata. Keys are always present; values are ``None``
        when the pattern does not match.
    """
    pattern = r"stage2_(digress_transformer)_(\w+)_(\w+)_lr([\d\w-]+)_wd([\d\w-]+)_k(\d+)_s(\d+)"
    match = re.match(pattern, run_id)
    if match:
        return {
            "model_type": match.group(1),
            "dataset": match.group(2),
            "embedding": match.group(3),
            "lr": match.group(4),
            "wd": match.group(5),
            "k": int(match.group(6)),
            "seed_run": int(match.group(7)),
        }
    return {
        "model_type": None,
        "dataset": None,
        "embedding": None,
        "lr": None,
        "wd": None,
        "k": None,
        "seed_run": None,
    }


def list_evaluation_files(pattern: str | None = None) -> list[dict[str, str]]:
    """List ``mmd_evaluation_*.json`` files on the Modal volume.

    Parameters
    ----------
    pattern
        Optional glob to filter run IDs before scanning their contents.

    Returns
    -------
    list[dict]
        Each entry has ``run_id``, ``checkpoint``, and ``remote_path`` keys.
    """
    cmd = ["modal", "volume", "ls", MODAL_VOLUME_NAME, "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    try:
        entries: list[dict[str, str]] = json.loads(result.stdout)
    except json.JSONDecodeError:
        entries = [
            {"Filename": line.strip()}
            for line in result.stdout.strip().split("\n")
            if line
        ]

    run_dirs = [
        entry.get("Filename", entry.get("filename", "")).rstrip("/")
        for entry in entries
        if entry.get("Filename", entry.get("filename", "")).startswith("stage2_")
    ]

    evaluation_files: list[dict[str, str]] = []
    for run_id in tqdm(run_dirs, desc="Scanning run directories", unit="dir"):
        if pattern and not fnmatch.fnmatch(run_id, pattern):
            continue
        try:
            cmd = ["modal", "volume", "ls", MODAL_VOLUME_NAME, run_id, "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            try:
                run_entries: list[dict[str, str]] = json.loads(result.stdout)
            except json.JSONDecodeError:
                run_entries = [
                    {"Filename": line.strip()}
                    for line in result.stdout.strip().split("\n")
                    if line
                ]
            for entry in run_entries:
                filename = entry.get("Filename", entry.get("filename", ""))
                if filename.startswith("mmd_evaluation_") and filename.endswith(
                    ".json"
                ):
                    checkpoint = filename.removeprefix("mmd_evaluation_").removesuffix(
                        ".json"
                    )
                    evaluation_files.append(
                        {
                            "run_id": run_id,
                            "checkpoint": checkpoint,
                            "remote_path": f"{run_id}/{filename}",
                        }
                    )
        except subprocess.CalledProcessError:
            continue

    return evaluation_files


def download_file(remote_path: str, cache_dir: Path) -> Path:
    """Download a single file from the Modal volume, returning its local path."""
    local_path = cache_dir / remote_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        return local_path
    cmd = ["modal", "volume", "get", MODAL_VOLUME_NAME, remote_path, str(local_path)]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return local_path


def download_files(
    files: list[dict[str, str]], cache_dir: Path, *, use_cache: bool = True
) -> list[Path]:
    """Download evaluation JSON files from the Modal volume.

    Parameters
    ----------
    files
        Output of :func:`list_evaluation_files`.
    cache_dir
        Local directory for cached downloads.
    use_cache
        Skip files that already exist locally.
    """
    local_paths: list[Path] = []
    for file_info in tqdm(files, desc="Downloading", unit="file"):
        remote_path = file_info["remote_path"]
        local_path = cache_dir / remote_path
        if use_cache and local_path.exists():
            local_paths.append(local_path)
            continue
        try:
            local_paths.append(download_file(remote_path, cache_dir))
        except subprocess.CalledProcessError as exc:
            click.echo(f"Warning: failed to download {remote_path}: {exc}", err=True)
    return local_paths


def load_and_flatten(json_path: Path) -> list[dict[str, Any]]:
    """Load one evaluation JSON and flatten to one row per evaluation label."""
    with open(json_path) as f:
        data = json.load(f)

    metadata = parse_run_id(data["run_id"])
    rows: list[dict[str, Any]] = []
    for label, metrics in data.get("results", {}).items():
        rows.append(
            {
                "run_id": data["run_id"],
                "checkpoint_name": data.get("checkpoint_name", "unknown"),
                "split": label,
                "degree_mmd": metrics.get("degree_mmd"),
                "clustering_mmd": metrics.get("clustering_mmd"),
                "spectral_mmd": metrics.get("spectral_mmd"),
                "timestamp": data.get("timestamp"),
                "num_samples": data.get("params", {}).get("num_samples"),
                "num_steps": data.get("params", {}).get("num_steps"),
                "mmd_kernel": data.get("params", {}).get("mmd_kernel"),
                "mmd_sigma": data.get("params", {}).get("mmd_sigma"),
                "seed": data.get("params", {}).get("seed"),
                **metadata,
            }
        )
    return rows


_SCHEMA = {
    "run_id": pl.Utf8,
    "checkpoint_name": pl.Utf8,
    "split": pl.Utf8,
    "degree_mmd": pl.Float64,
    "clustering_mmd": pl.Float64,
    "spectral_mmd": pl.Float64,
    "timestamp": pl.Utf8,
    "num_samples": pl.Int64,
    "num_steps": pl.Int64,
    "mmd_kernel": pl.Utf8,
    "mmd_sigma": pl.Float64,
    "seed": pl.Int64,
    "model_type": pl.Utf8,
    "dataset": pl.Utf8,
    "embedding": pl.Utf8,
    "lr": pl.Utf8,
    "wd": pl.Utf8,
    "k": pl.Int64,
    "seed_run": pl.Int64,
}


def create_dataframe(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a typed Polars DataFrame from flattened row dicts."""
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows)
    if "timestamp" in df.columns:
        df = df.with_columns(
            pl.col("timestamp").str.to_datetime(strict=False).alias("timestamp")
        )
    return df


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Parquet output path. Required unless --dry-run.",
)
@click.option(
    "--pattern",
    "-p",
    default=None,
    help="Glob pattern to filter run IDs (e.g. 'stage2_*_sbm_*').",
)
@click.option(
    "--dataset",
    "-d",
    multiple=True,
    help="Keep only these dataset(s) in the final table. Repeatable.",
)
@click.option(
    "--checkpoints",
    "-c",
    multiple=True,
    help="Keep only these checkpoint name(s). Repeatable.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Persistent cache directory. Uses a temp dir if omitted.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Re-download even if files are cached.",
)
@click.option(
    "--dry-run", is_flag=True, default=False, help="List files without downloading."
)
def aggregate(
    output: Path | None,
    pattern: str | None,
    dataset: tuple[str, ...],
    checkpoints: tuple[str, ...],
    cache_dir: Path | None,
    no_cache: bool,
    dry_run: bool,
) -> None:
    """Pull MMD evaluation results from the Modal volume into Parquet.

    \b
    Examples:
      tmgg-modal aggregate -o results/mmd.parquet
      tmgg-modal aggregate -p 'stage2_*_sbm_*' -o results/sbm.parquet
      tmgg-modal aggregate --dry-run
      tmgg-modal aggregate -d sbm er --cache-dir results/cache -o out.parquet
    """
    if not dry_run and output is None:
        raise click.UsageError("--output is required unless --dry-run is set.")

    click.echo("Listing evaluation files on Modal volume...")
    try:
        files = list_evaluation_files(pattern=pattern)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"Could not list Modal volume. Is Modal authenticated?\n{exc}"
        ) from exc

    if not files:
        click.echo("No evaluation files found matching criteria.")
        return

    click.echo(f"Found {len(files)} evaluation file(s)")

    if checkpoints:
        files = [f for f in files if f["checkpoint"] in checkpoints]
        click.echo(f"After checkpoint filter: {len(files)} file(s)")

    if dry_run:
        click.echo("\nWould download:")
        for f in files:
            click.echo(f"  {f['remote_path']}")
        click.echo(f"\nTotal: {len(files)} files")
        return

    # Resolve cache directory
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"Cache directory: {cache_dir}")
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix="mmd_eval_"))
        click.echo(f"Temporary cache: {cache_dir}")

    local_paths = download_files(files, cache_dir, use_cache=not no_cache)
    if not local_paths:
        raise click.ClickException("No files downloaded successfully.")

    click.echo(f"Downloaded {len(local_paths)} file(s)")
    click.echo("Parsing results...")

    all_rows: list[dict[str, Any]] = []
    for path in tqdm(local_paths, desc="Processing", unit="file"):
        try:
            all_rows.extend(load_and_flatten(path))
        except (json.JSONDecodeError, KeyError) as exc:
            click.echo(f"Warning: could not parse {path}: {exc}", err=True)

    if not all_rows:
        raise click.ClickException("No valid data rows extracted.")

    click.echo(f"Extracted {len(all_rows)} rows")

    df = create_dataframe(all_rows)

    if dataset:
        df = df.filter(pl.col("dataset").is_in(list(dataset)))
        click.echo(f"After dataset filter: {len(df)} rows")

    assert output is not None  # guaranteed by earlier validation
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output)

    click.echo(f"\nSaved to: {output}")
    click.echo(f"Shape: {df.shape}")
    click.echo("\nSchema:")
    for col, dtype in df.schema.items():
        click.echo(f"  {col}: {dtype}")

    click.echo("\nSummary:")
    if "dataset" in df.columns:
        for row in df.group_by("dataset").len().sort("dataset").iter_rows(named=True):
            click.echo(f"  {row['dataset']}: {row['len']} rows")
    if "checkpoint_name" in df.columns:
        for row in (
            df.group_by("checkpoint_name")
            .len()
            .sort("checkpoint_name")
            .iter_rows(named=True)
        ):
            click.echo(f"  {row['checkpoint_name']}: {row['len']} rows")

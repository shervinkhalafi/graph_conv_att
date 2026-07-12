"""Storage utilities for eigenstructure data using safetensors.

Provides functions for saving and loading eigendecomposition batches
with associated metadata in an efficient binary format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

_RESERVED_TENSOR_KEYS = frozenset(
    {
        "eigenvalues_adj",
        "eigenvectors_adj",
        "eigenvalues_lap",
        "eigenvectors_lap",
        "adjacency",
    }
)


def save_decomposition_batch(
    output_dir: Path,
    batch_index: int,
    eigenvalues_adj: torch.Tensor,
    eigenvectors_adj: torch.Tensor,
    eigenvalues_lap: torch.Tensor,
    eigenvectors_lap: torch.Tensor,
    adjacency_matrices: torch.Tensor,
    metadata_list: list[dict[str, Any]],
    extra_tensors: dict[str, torch.Tensor] | None = None,
) -> Path:
    """
    Save a batch of eigendecompositions to safetensors format.

    Parameters
    ----------
    output_dir : Path
        Directory to write files to.
    batch_index : int
        Index of this batch (used in filename).
    eigenvalues_adj : torch.Tensor
        Adjacency eigenvalues, shape (batch_size, n).
    eigenvectors_adj : torch.Tensor
        Adjacency eigenvectors, shape (batch_size, n, n).
    eigenvalues_lap : torch.Tensor
        Laplacian eigenvalues, shape (batch_size, n).
    eigenvectors_lap : torch.Tensor
        Laplacian eigenvectors, shape (batch_size, n, n).
    adjacency_matrices : torch.Tensor
        Original adjacency matrices, shape (batch_size, n, n).
    metadata_list : list[dict]
        Per-graph metadata dictionaries.
    extra_tensors : dict[str, torch.Tensor], optional
        Additional named tensors to persist alongside the canonical
        decomposition (e.g. ``B_k4`` projection matrices used by the
        improvement-gap surrogate). Keys must not collide with the
        reserved names enumerated in ``_RESERVED_TENSOR_KEYS``.

    Returns
    -------
    Path
        Path to the saved safetensors file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors = {
        "eigenvalues_adj": eigenvalues_adj.contiguous(),
        "eigenvectors_adj": eigenvectors_adj.contiguous(),
        "eigenvalues_lap": eigenvalues_lap.contiguous(),
        "eigenvectors_lap": eigenvectors_lap.contiguous(),
        "adjacency": adjacency_matrices.contiguous(),
    }

    if extra_tensors is not None:
        for key, tensor in extra_tensors.items():
            if key in _RESERVED_TENSOR_KEYS:
                raise ValueError(
                    f"extra_tensors key '{key}' collides with a reserved "
                    f"decomposition tensor name"
                )
            tensors[key] = tensor.contiguous()

    safetensors_path = output_dir / f"batch_{batch_index:06d}.safetensors"
    metadata_path = output_dir / f"batch_{batch_index:06d}_metadata.json"

    save_file(tensors, str(safetensors_path))

    with open(metadata_path, "w") as f:
        json.dump(metadata_list, f, indent=2)

    return safetensors_path


def load_decomposition_batch(
    batch_path: Path,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """
    Load a batch of decompositions and their metadata.

    Parameters
    ----------
    batch_path : Path
        Path to the safetensors file.

    Returns
    -------
    tuple[dict[str, torch.Tensor], list[dict]]
        Tuple of (tensors_dict, metadata_list).
    """
    batch_path = Path(batch_path)
    tensors = load_file(str(batch_path))

    # Metadata file has same name but with _metadata.json suffix
    stem = batch_path.stem  # e.g., "batch_000000"
    metadata_path = batch_path.parent / f"{stem}_metadata.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    return tensors, metadata


def save_dataset_manifest(
    output_dir: Path,
    dataset_name: str,
    dataset_config: dict[str, Any],
    num_graphs: int,
    num_batches: int,
    seed: int,
) -> None:
    """
    Save a manifest describing the collected dataset.

    Parameters
    ----------
    output_dir : Path
        Directory containing the batch files.
    dataset_name : str
        Name of the source dataset.
    dataset_config : dict
        Configuration used to generate/load the dataset.
    num_graphs : int
        Total number of graphs collected.
    num_batches : int
        Number of batch files written.
    seed : int
        Random seed used for reproducibility.
    """
    manifest = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "num_graphs": num_graphs,
        "num_batches": num_batches,
        "seed": seed,
        "schema_version": "1.0",
    }

    output_dir = Path(output_dir)
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(input_dir: Path) -> dict[str, Any]:
    """
    Load the dataset manifest.

    Parameters
    ----------
    input_dir : Path
        Directory containing the manifest.json file.

    Returns
    -------
    dict
        Manifest data.
    """
    with open(Path(input_dir) / "manifest.json") as f:
        return json.load(f)


def iter_batches(input_dir: Path) -> list[Path]:
    """
    Get sorted list of batch file paths in a directory.

    Parameters
    ----------
    input_dir : Path
        Directory containing batch_*.safetensors files.

    Returns
    -------
    list[Path]
        Sorted list of batch file paths.
    """
    input_dir = Path(input_dir)
    return sorted(input_dir.glob("batch_*.safetensors"))

"""SPECTRE planar fixture (planar_64_200.pt) loader + split helper.

Mirrors :mod:`tmgg.data.datasets.spectre_sbm`. The planar fixture is
200 dense adjacency matrices, all at n = 64, used by DiGress's public
planar experiment (Table 1 in Vignac et al., ICLR 2023).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# Source of truth for the fixture. Pinned to a raw GitHub URL on the
# SPECTRE repository's ``main`` branch.
SPECTRE_PLANAR_URL = (
    "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt"
)

# Default on-disk cache path. Mirrors the SBM convention.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tmgg" / "spectre"
DEFAULT_FIXTURE_PATH = DEFAULT_CACHE_DIR / "planar_64_200.pt"

# Upstream split geometry: total 200 graphs at n = 64. 20 % test,
# 80 % of the remainder train. Matches the upstream DiGress rule.
SPECTRE_PLANAR_TOTAL = 200
SPECTRE_PLANAR_TEST_LEN = int(round(SPECTRE_PLANAR_TOTAL * 0.2))  # 40
SPECTRE_PLANAR_TRAIN_LEN = int(
    round((SPECTRE_PLANAR_TOTAL - SPECTRE_PLANAR_TEST_LEN) * 0.8)
)  # 128
SPECTRE_PLANAR_VAL_LEN = (
    SPECTRE_PLANAR_TOTAL - SPECTRE_PLANAR_TEST_LEN - SPECTRE_PLANAR_TRAIN_LEN
)  # 32
SPECTRE_PLANAR_SPLIT_SEED = 0

# Pinned SHA-256 of the upstream fixture (downloaded 2026-04-28).
# Used by tests; production code does not enforce.
SPECTRE_PLANAR_SHA256 = (
    "063dc3e675a5c63144e56aa974ca961abfcba02914368192a09b21a364df38fc"
)


def download_spectre_planar_fixture(
    dest: Path | None = None,
    *,
    url: str = SPECTRE_PLANAR_URL,
    force: bool = False,
) -> Path:
    """Download the SPECTRE planar fixture to a local path."""
    target = Path(dest) if dest is not None else DEFAULT_FIXTURE_PATH
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)  # noqa: S310 - pinned https URL
    return target


def load_spectre_planar_fixture(
    path: Path | None = None,
) -> tuple[list[torch.Tensor], list[int]]:
    """Load the SPECTRE planar fixture and return adjacencies + node counts.

    Returns the same shape as :func:`load_spectre_sbm_fixture`: a list
    of dense adjacency tensors and a parallel list of node counts (all
    64 for planar). Uses ``weights_only=False`` because the file
    predates the PyTorch default; the fixture is first-party-trusted
    via the SPECTRE upstream URL.
    """
    fixture_path = path if path is not None else download_spectre_planar_fixture()
    obj: list[Any] = torch.load(fixture_path, weights_only=False)
    if not isinstance(obj, list) or len(obj) < 4:
        raise RuntimeError(
            f"Unexpected SPECTRE planar fixture structure at {fixture_path}: "
            f"expected a list of length ≥ 4, got {type(obj).__name__} "
            f"of length {len(obj) if hasattr(obj, '__len__') else '?'}."
        )
    adjs: list[torch.Tensor] = obj[0]
    n_nodes: list[int] = [int(n) for n in obj[3]]
    if len(adjs) != len(n_nodes):
        raise RuntimeError(
            f"SPECTRE planar fixture inconsistent: {len(adjs)} adjacencies vs "
            f"{len(n_nodes)} node-counts."
        )
    return adjs, n_nodes


def split_spectre_planar(
    num_graphs: int = SPECTRE_PLANAR_TOTAL,
    *,
    seed: int = SPECTRE_PLANAR_SPLIT_SEED,
) -> dict[str, list[int]]:
    """Produce train/val/test indices matching upstream DiGress."""
    generator = torch.Generator()
    _ = generator.manual_seed(seed)
    permutation = torch.randperm(num_graphs, generator=generator).tolist()

    test_len = int(round(num_graphs * 0.2))
    train_len = int(round((num_graphs - test_len) * 0.8))

    train_indices = permutation[:train_len]
    val_indices = permutation[train_len : num_graphs - test_len]
    test_indices = permutation[num_graphs - test_len :]
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def adjacency_to_pyg_data(adj: torch.Tensor) -> Data:
    """Convert one dense adjacency tensor to a :class:`Data` instance."""
    adj_f = adj.float()
    edge_index, _ = dense_to_sparse(adj_f)
    return Data(edge_index=edge_index, num_nodes=adj_f.shape[0])


class SpectrePlanarDataset(Dataset[Data]):
    """Indexable view over a SPECTRE-planar split."""

    def __init__(self, adjacencies: list[torch.Tensor]) -> None:
        self._adjacencies = adjacencies

    def __len__(self) -> int:
        return len(self._adjacencies)

    def __getitem__(self, idx: int) -> Data:
        return adjacency_to_pyg_data(self._adjacencies[idx])

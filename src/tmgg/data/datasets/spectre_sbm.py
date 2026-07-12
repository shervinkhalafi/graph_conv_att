"""SPECTRE SBM fixture loader.

The upstream DiGress SBM experiment trains on the 200-graph SPECTRE SBM
fixture (Martinkus et al., NeurIPS 2022). Each graph is an undirected
stochastic block model sample with variable node count (~44-187),
2-5 communities, community-internal edge density ~0.3, and
community-external edge density ~0.005.

The fixture is distributed as a single ``.pt`` file on the SPECTRE
repository. Loading it via ``torch.load`` yields a
``list[list[Tensor] | list[int] | bool | int]`` of length 8:

| Index | Field         | Shape / type                            |
|-------|---------------|-----------------------------------------|
| 0     | adjs          | ``list[Tensor]`` of shape ``(n_i, n_i)``|
| 1     | eigvals       | ``list[Tensor]`` of shape ``(n_i,)``    |
| 2     | eigvecs       | ``list[Tensor]`` of shape ``(n_i, n_i)``|
| 3     | n_nodes       | ``list[int]``                           |
| 4     | max_eigval    | ``float``                               |
| 5     | min_eigval    | ``float``                               |
| 6     | same_sample   | ``bool``                                |
| 7     | n_max         | ``int``                                 |

Only ``adjs`` and ``n_nodes`` are load-bearing for this project; the
spectral fields and scalars are ignored (graph features are derived at
training time by the extra-features pipeline instead).

Train/val/test splits follow upstream DiGress's convention:
``torch.randperm`` with a generator seeded to 0, then the first 128
indices become train, the next 32 val, and the last 40 test. The seed
and split sizes match upstream so a re-run reproduces the same cut.

References
----------
.. [1] Martinkus, Loukas, Perraudin, Wattenhofer (NeurIPS 2022).
   SPECTRE. https://github.com/KarolisMart/SPECTRE
.. [2] Vignac et al. (ICLR 2023). DiGress. Upstream loader:
   https://github.com/cvignac/DiGress/blob/main/src/datasets/spectre_dataset.py
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
SPECTRE_SBM_URL = (
    "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt"
)

# Default on-disk cache path. Keeps the fixture outside the repo so a
# clean checkout does not carry a 20 MB binary.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tmgg" / "spectre"
DEFAULT_FIXTURE_PATH = DEFAULT_CACHE_DIR / "sbm_200.pt"

# Upstream split geometry: total 200 graphs, 20 % test, 80 % of the
# remainder train. Ordering and sizes match upstream DiGress.
SPECTRE_SBM_TOTAL = 200
SPECTRE_SBM_TEST_LEN = int(round(SPECTRE_SBM_TOTAL * 0.2))  # 40
SPECTRE_SBM_TRAIN_LEN = int(
    round((SPECTRE_SBM_TOTAL - SPECTRE_SBM_TEST_LEN) * 0.8)
)  # 128
SPECTRE_SBM_VAL_LEN = (
    SPECTRE_SBM_TOTAL - SPECTRE_SBM_TEST_LEN - SPECTRE_SBM_TRAIN_LEN
)  # 32
SPECTRE_SPLIT_SEED = 0


def download_spectre_sbm_fixture(
    dest: Path | None = None,
    *,
    url: str = SPECTRE_SBM_URL,
    force: bool = False,
) -> Path:
    """Download the SPECTRE SBM fixture to a local path.

    Parameters
    ----------
    dest
        Destination file path. When ``None``, writes to
        :data:`DEFAULT_FIXTURE_PATH`.
    url
        Source URL. Defaults to the raw-GitHub SPECTRE URL; override
        only to test with a mirror.
    force
        When True, re-download even if ``dest`` already exists.

    Returns
    -------
    Path
        Path to the downloaded fixture.
    """
    target = Path(dest) if dest is not None else DEFAULT_FIXTURE_PATH
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # Stream directly to disk — fixture is ~20 MB so the whole transfer
    # fits in a single call without chunking gymnastics.
    urllib.request.urlretrieve(url, target)  # noqa: S310 - pinned https URL
    return target


def load_spectre_sbm_fixture(
    path: Path | None = None,
) -> tuple[list[torch.Tensor], list[int]]:
    """Load the SPECTRE SBM fixture and return adjacencies + node counts.

    Parameters
    ----------
    path
        Path to a pre-downloaded ``sbm_200.pt``. When ``None``, downloads
        to :data:`DEFAULT_FIXTURE_PATH` (no-op if already present).

    Returns
    -------
    adjs
        200 undirected binary adjacency matrices of varying node count.
    n_nodes
        Matching node counts (``adjs[i].shape[0] == n_nodes[i]``).
    """
    fixture_path = path if path is not None else download_spectre_sbm_fixture()
    # `weights_only=False` mirrors upstream; the file contains Python
    # lists of tensors, not a state_dict, and the upstream format
    # predates weights_only becoming the PyTorch default.
    obj: list[Any] = torch.load(fixture_path, weights_only=False)
    if not isinstance(obj, list) or len(obj) < 4:
        raise RuntimeError(
            f"Unexpected SPECTRE fixture structure at {fixture_path}: "
            f"expected a length-8 list, got {type(obj).__name__} "
            f"of length {len(obj) if hasattr(obj, '__len__') else '?'}."
        )
    adjs: list[torch.Tensor] = obj[0]
    n_nodes: list[int] = [int(n) for n in obj[3]]
    if len(adjs) != len(n_nodes):
        raise RuntimeError(
            f"SPECTRE fixture inconsistent: {len(adjs)} adjacencies vs "
            f"{len(n_nodes)} node-counts."
        )
    return adjs, n_nodes


def split_spectre_sbm(
    num_graphs: int = SPECTRE_SBM_TOTAL,
    *,
    seed: int = SPECTRE_SPLIT_SEED,
) -> dict[str, list[int]]:
    """Produce train/val/test indices matching upstream DiGress's split.

    Upstream DiGress (``src/datasets/spectre_dataset.py``) calls
    ``torch.randperm(num_graphs, generator=g)`` with ``g.manual_seed(0)``
    once, then takes the first ``train_len`` entries for train, the
    next ``val_len`` for val, and the final ``test_len`` for test.

    Parameters
    ----------
    num_graphs
        Total graph count. The SPECTRE fixture is 200 graphs; overriding
        this is only for tests.
    seed
        Permutation seed. Upstream uses 0 for reproducibility.

    Returns
    -------
    dict[str, list[int]]
        Keys ``"train"``, ``"val"``, ``"test"`` with integer indices into
        the 200-element adjacency list.
    """
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
    """Convert one dense adjacency tensor to a ``torch_geometric.data.Data``.

    Mirrors upstream DiGress's conversion: build a sparse ``edge_index``
    via ``dense_to_sparse`` and carry ``num_nodes`` explicitly (otherwise
    PyG infers it from the maximum node id, which would miss isolated
    vertices).
    """
    adj_f = adj.float()
    edge_index, _ = dense_to_sparse(adj_f)
    return Data(edge_index=edge_index, num_nodes=adj_f.shape[0])


class SpectreSBMDataset(Dataset[Data]):
    """Indexable view over a SPECTRE-SBM split.

    Holds the list of adjacency tensors for a given split plus a
    per-index converter to PyG ``Data``. The conversion happens lazily on
    ``__getitem__`` so batch iteration is still fast (edge-index
    construction is sub-millisecond for ~150-node graphs) but a fresh
    PyG ``Data`` is built each access — matching the rest of the project.
    """

    def __init__(self, adjs: list[torch.Tensor]) -> None:
        if not adjs:
            raise ValueError("SpectreSBMDataset requires a non-empty adjacency list.")
        self._adjs: list[torch.Tensor] = adjs

    def __len__(self) -> int:
        return len(self._adjs)

    def __getitem__(self, idx: int) -> Data:
        return adjacency_to_pyg_data(self._adjs[idx])

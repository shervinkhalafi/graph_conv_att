"""Abstract :class:`MolecularGraphDataset` with on-disk shard cache.

Subclasses (:class:`QM9Dataset`, :class:`MOSESDataset`,
:class:`GuacaMolDataset`) override :meth:`download_smiles_split` and
:meth:`make_codec` only. The cache machinery here is dataset-agnostic.

Cache layout:
``<cache_root>/<dataset_name>/preprocessed/<codec_hash>/<split>/<shard_idx>.pt``

``<codec_hash>`` is :meth:`SMILESCodec.cache_key` — any change to
vocab/codec params invalidates the entire preprocessed tree.
"""

from __future__ import annotations

import abc
import logging
import os
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from tmgg.data.datasets.graph_types import DenseGraphState
from tmgg.data.datasets.molecular.codec import SMILESCodec

logger = logging.getLogger(__name__)

# Shard size cap for MOSES/GuacaMol; QM9 fits in a single shard.
DEFAULT_SHARD_SIZE = 50_000


def _local_cache_root() -> Path:
    """Resolve the local cache root.

    ``/data/datasets`` exists on Modal containers (mounted via the
    graph-denoising-datasets volume per ``tmgg.modal._lib.volumes.DATASETS_MOUNT``);
    otherwise fall back to ``~/.cache/tmgg/``. This means a Modal
    training container reads + writes the persisted volume directly,
    while a host-side dev run reads + writes the user's home cache.
    """
    modal_root = Path("/data/datasets")
    if modal_root.exists() and modal_root.is_dir():
        return modal_root
    return Path.home() / ".cache" / "tmgg"


class MolecularGraphDataset(Dataset[Data], abc.ABC):
    """ABC for the three molecular datasets.

    Subclasses provide:
    - ``DATASET_NAME``: subdirectory name under the cache root.
    - ``RAW_FILES``: mapping ``split → URL`` for the raw SMILES files.
    - ``make_codec``: classmethod returning the dataset's
      :class:`SMILESCodec`.
    - ``download_smiles_split``: returns a list[str] of SMILES for
      a given split (uses ``RAW_FILES`` by default).

    Lifecycle:

    1. ``prepare_data()`` is called by the matching DataModule and
       downloads raw SMILES. Idempotent.
    2. ``setup(split)`` populates ``self._graphs`` (a list of
       single-graph :class:`DenseGraphState`) from the cached shards,
       preprocessing on first call.
    3. ``__getitem__`` returns a sparse PyG :class:`~torch_geometric.data.Data`
       carrying ``edge_index``, ``num_nodes`` and ``x`` (integer
       atom-class indices). The :class:`DenseGraphState` shards are
       kept in memory and converted on demand so the ``DataLoader``
       worker feeds :class:`GraphDataCollator` (which expects PyG
       ``Data``) directly. The collator then assembles the sparse
       :class:`GraphState` batch on the hot path. No RDKit calls.
    """

    DATASET_NAME: str = ""  # subclass override
    RAW_FILES: dict[str, str] = {}  # split → URL
    DEFAULT_MAX_ATOMS: int = 30  # subclass override

    def __init__(
        self,
        *,
        split: str,
        cache_root: Path | None = None,
        shard_size: int = DEFAULT_SHARD_SIZE,
    ) -> None:
        if not self.DATASET_NAME:
            raise NotImplementedError(f"{type(self).__name__} must set DATASET_NAME.")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split!r}.")
        self.split = split
        self.cache_root = cache_root or _local_cache_root()
        self.shard_size = shard_size
        self._codec: SMILESCodec | None = None
        self._graphs: list[DenseGraphState] | None = None

    # ------------------------------------------------------------------
    # subclass hooks
    # ------------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def make_codec(cls) -> SMILESCodec:
        """Return the canonical codec for this dataset."""

    @abc.abstractmethod
    def download_smiles_split(self, split: str) -> list[str]:
        """Return the raw SMILES list for a split.

        Implementations should download into
        ``self.cache_root / self.DATASET_NAME / "raw"`` and parse the
        SMILES out of whatever format the upstream uses (CSV, txt,
        etc.).
        """

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------

    def _raw_dir(self) -> Path:
        return self.cache_root / self.DATASET_NAME / "raw"

    def _preprocessed_dir(self) -> Path:
        codec = self._codec or self.make_codec()
        return self.cache_root / self.DATASET_NAME / "preprocessed" / codec.cache_key()

    def _shard_dir(self) -> Path:
        return self._preprocessed_dir() / self.split

    # ------------------------------------------------------------------
    # default URL-based downloader
    # ------------------------------------------------------------------

    def _default_download(self, split: str) -> Path:
        if split not in self.RAW_FILES:
            raise KeyError(
                f"{self.DATASET_NAME}: no RAW_FILES URL for split {split!r}."
            )
        url = self.RAW_FILES[split]
        target = self._raw_dir() / Path(url).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return target
        logger.info("Downloading %s → %s", url, target)
        # Retry 3x with exponential backoff for transient network errors.
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, target)  # noqa: S310
                return target
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < 2:
                    import time

                    time.sleep(2**attempt)
        raise RuntimeError(f"Failed to download {url} after 3 attempts: {last_exc!r}")

    # ------------------------------------------------------------------
    # preprocessing + caching
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Download all raw splits. Safe to call multiple times."""
        for split in ("train", "val", "test"):
            try:
                _ = self.download_smiles_split(split)
            except KeyError:
                # Dataset may not declare every split (e.g. GuacaMol
                # 'test' is the held-out reference set; declare what
                # exists in the subclass).
                logger.info(
                    "Skipping %s/%s: not declared.",
                    self.DATASET_NAME,
                    split,
                )

    def setup(self) -> None:
        """Populate ``self._graphs`` for the configured split."""
        codec = self.make_codec()
        self._codec = codec
        shard_dir = self._shard_dir()
        if shard_dir.exists() and any(shard_dir.iterdir()):
            self._graphs = self._load_shards()
            return

        logger.info(
            "%s/%s: preprocessing (cache miss at %s)",
            self.DATASET_NAME,
            self.split,
            shard_dir,
        )
        smiles = self.download_smiles_split(self.split)
        graphs, counters = codec.encode_dataset_with_stats(smiles)
        logger.info(
            "%s/%s: preprocessed; counters=%s",
            self.DATASET_NAME,
            self.split,
            counters,
        )
        self._write_shards(graphs)
        self._graphs = graphs

    # ------------------------------------------------------------------
    # shard I/O
    # ------------------------------------------------------------------

    def _write_shards(self, graphs: list[DenseGraphState]) -> None:
        # Atomic write: ``torch.save`` to a sibling ``<idx>.pt.tmp`` then
        # ``os.replace`` to the final ``<idx>.pt``. ``os.replace`` is
        # atomic on POSIX when src and dst live on the same filesystem
        # (Modal volumes satisfy this — both are under the same mount).
        # Without this, a SIGTERM / OOM mid-``torch.save`` leaves a
        # half-written ``.pt`` whose internal zip directory is missing
        # later tensors; the next training run then dies at
        # ``PytorchStreamReader failed locating file data/N`` and Modal
        # crash-loops the run on retry. We hit this exact failure mode
        # in the Phase 8 smoke runs (543 retries from corrupted shards).
        shard_dir = self._shard_dir()
        shard_dir.mkdir(parents=True, exist_ok=True)
        for shard_idx, start in enumerate(range(0, len(graphs), self.shard_size)):
            shard = graphs[start : start + self.shard_size]
            shard_path = shard_dir / f"{shard_idx:04d}.pt"
            tmp_path = shard_dir / f"{shard_idx:04d}.pt.tmp"
            # ``weights_only=False`` on load is required because
            # ``DenseGraphState`` is a dataclass; these are first-party
            # files we wrote ourselves so the security memory's
            # "no third-party pickles" rule does not apply.
            torch.save(shard, tmp_path)
            os.replace(tmp_path, shard_path)

    def _load_shards(self) -> list[DenseGraphState]:
        shard_dir = self._shard_dir()
        graphs: list[DenseGraphState] = []
        for shard_path in sorted(shard_dir.glob("*.pt")):
            shard: list[DenseGraphState] = torch.load(shard_path, weights_only=False)
            graphs.extend(shard)
        return graphs

    # ------------------------------------------------------------------
    # Dataset[Data] surface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._graphs is None:
            raise RuntimeError(f"{type(self).__name__} not set up. Call setup() first.")
        return len(self._graphs)

    def __getitem__(self, idx: int) -> Data:
        if self._graphs is None:
            raise RuntimeError(f"{type(self).__name__} not set up. Call setup() first.")
        return self._graphdata_to_pyg(self._graphs[idx])

    @staticmethod
    def _graphdata_to_pyg_one_hot(gd: DenseGraphState) -> Data:
        """Convert codec :class:`DenseGraphState` to PyG :class:`Data` with one-hot ``x``/``edge_attr``.

        Variant of :meth:`_graphdata_to_pyg` for the *raw* PyG path
        consumed by :func:`count_node_classes_sparse` and
        :func:`count_edge_classes_sparse` (and ultimately
        :meth:`CategoricalNoiseProcess.initialize_from_data`). Those
        helpers expect:

        - ``x``: ``(N, num_atom_types)`` float one-hot, summable along
          ``dim=0`` for per-class node histograms.
        - ``edge_attr``: ``(E, num_bond_types)`` float one-hot, with
          row-zero reserved for the implicit "no-edge" class (the sparse
          counter discards it because no-edge entries do not appear in
          ``edge_index``).

        SBM/Planar satisfy the same contract by *omitting* ``x`` and
        ``edge_attr`` entirely; the counters fall back to a fixed-class
        branch in that case. Molecular datasets need real per-class
        histograms, so we populate them explicitly.

        The training path uses :meth:`_graphdata_to_pyg` (the
        index variant) which is what :meth:`GraphState.from_pyg_batch`
        densifies into the per-edge ``edge_class`` / per-node
        ``x_class`` collator output. Don't merge the two: the
        index path requires integer indices for its ``F.one_hot``
        densification step, and the sparse counter path requires
        float one-hot for its ``x.sum(dim=0)`` aggregation step.
        """
        if gd.X_class is None or gd.E_class is None:
            raise ValueError(
                "MolecularGraphDataset._graphdata_to_pyg_one_hot requires "
                "X_class and E_class to be populated by the codec."
            )
        n = int(gd.num_nodes_per_graph[0].item())
        num_bond_types = int(gd.E_class.shape[-1])
        e_argmax = gd.E_class[0, :n, :n].argmax(dim=-1)
        adj = (e_argmax != 0).to(torch.float32)
        edge_index, _ = dense_to_sparse(adj)
        src, dst = edge_index[0], edge_index[1]
        bond_idx = e_argmax[src, dst].long()
        edge_attr = torch.nn.functional.one_hot(
            bond_idx, num_classes=num_bond_types
        ).to(torch.float32)
        # ``X_class`` is already one-hot, just slice + cast.
        x = gd.X_class[0, :n].to(torch.float32)
        return Data(
            edge_index=edge_index,
            num_nodes=n,
            x=x,
            edge_attr=edge_attr,
        )

    @staticmethod
    def _graphdata_to_pyg(gd: DenseGraphState) -> Data:
        """Convert a single-graph codec :class:`DenseGraphState` to PyG :class:`Data`.

        The codec emits a leading batch dim of 1 plus the dense
        categorical fields ``X_class`` (atom one-hot,
        ``(1, n, num_atom_types)``) and ``E_class`` (bond one-hot,
        ``(1, n, n, num_bond_types)`` with channel 0 = NONE). This
        helper extracts:

        - ``edge_index``: COO indices for every edge with
          ``argmax_E_class != 0`` (i.e. any non-NONE bond), via
          :func:`torch_geometric.utils.dense_to_sparse`.
        - ``edge_attr``: per-edge bond-class indices aligned with
          ``edge_index``, integer dtype. Mirrors the ``x`` plumbing
          for atom classes — :meth:`GraphState.from_pyg_batch`
          densifies these into a multi-class ``edge_class`` of width
          ``num_bond_types_e`` so that bond multiplicity (NONE /
          SINGLE / DOUBLE / TRIPLE / AROMATIC) survives the
          dataset → collator boundary instead of collapsing to
          binary edge presence. See the diagnosis in
          ``internal notes (not included)``
          item #3.3.
        - ``num_nodes``: real atom count, taken from
          ``num_nodes_per_graph[0]``.
        - ``x``: integer atom-class indices, shape ``(n_real,)``.
          :meth:`GraphState.from_pyg_batch` densifies this back into
          ``x_class`` on the collator side.

        Parameters
        ----------
        gd
            Single-graph :class:`DenseGraphState` produced by
            :meth:`SMILESCodec.encode`. ``X_class`` and ``E_class``
            MUST be populated; ``num_nodes_per_graph`` shape ``(1,)``.

        Returns
        -------
        torch_geometric.data.Data
            Sparse representation suitable for PyG batching.
        """
        if gd.X_class is None or gd.E_class is None:
            raise ValueError(
                "MolecularGraphDataset._graphdata_to_pyg requires X_class "
                "and E_class to be populated by the codec."
            )
        n = int(gd.num_nodes_per_graph[0].item())
        # Per-position bond-class argmax over the codec's one-hot E_class.
        # 0 = NONE; 1+ = present bond types.
        e_argmax = gd.E_class[0, :n, :n].argmax(dim=-1)
        # Adjacency: any non-NONE bond counts as an edge for edge_index
        # extraction. dense_to_sparse builds the COO index from this.
        adj = (e_argmax != 0).to(torch.float32)
        edge_index, _ = dense_to_sparse(adj)
        # Pull the bond class out of e_argmax for each retained edge so it
        # rides alongside edge_index as edge_attr. Integer dtype because
        # GraphState.from_pyg_batch's edge_attr branch refuses floats.
        src, dst = edge_index[0], edge_index[1]
        edge_attr = e_argmax[src, dst].long()
        x = gd.X_class[0, :n].argmax(dim=-1).long()
        return Data(
            edge_index=edge_index,
            num_nodes=n,
            x=x,
            edge_attr=edge_attr,
        )

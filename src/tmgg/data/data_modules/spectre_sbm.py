"""Lightning DataModule for the SPECTRE SBM fixture.

Unlike :class:`SyntheticCategoricalDataModule`, which generates SBM
graphs at runtime, this datamodule loads the upstream SPECTRE fixture
(``sbm_200.pt``) used by DiGress's public SBM experiment. The fixture is
fixed in both graph count (200) and per-graph node count (variable in
[44, 187]); splits match upstream DiGress exactly.

Downstream wiring is identical to the synthetic path: every dataloader
yields sparse :class:`~tmgg.data.datasets.graph_types.GraphState` batches
via :class:`~tmgg.data.data_modules.multigraph_data_module.GraphDataCollator`,
so ``DiffusionModule`` picks this datamodule up unchanged.

Static-pad default: ``pad_to_static_n_max=True`` makes the collator
pad every batch to ``num_nodes_max_static`` (200 by default for the
SPECTRE fixture). Unlocks ``torch.compile`` and ``cuda.graph`` capture
downstream by removing the variable per-batch ``n_max``. Flipped from
the historical ``False`` default in commit ``0d07fb63``; see
``docs/performance-toggles.md`` and
``internal notes (not included)``

**Upstream-parity note:** the static pad is mathematically equivalent
to upstream cvignac/DiGress's per-batch dynamic padding for both the
loss and the extra-features (cycles, eigenvalues, eigenvectors).
``ExtraFeatures`` masks ``A`` to zero on padded positions before
computing the Laplacian and adds a ``2*N`` sentinel on the padded
diagonal (``extra_features.py:518-525``); the eigenvalues of the real
sub-block are independent of the surrounding zero/sentinel block,
which gets discarded by ``get_eigenvalues_features``'s "first k
non-zero" filter. Eigenvectors are masked back to zero on padded
positions before being returned. So whether ``N=200`` (our static
pad) or ``N=150`` (some batch's dynamic pad), features at real
nodes are byte-equivalent — only padded compute differs (a wash
under ``node_mask`` zeroing downstream).

See :mod:`tmgg.data.datasets.spectre_sbm` for the load/split helpers
and ``internal notes (not included)`` for
the numerical parity rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from torch.utils.data import DataLoader
from torch_geometric.data import Data

from tmgg.data.data_modules.base_data_module import BaseGraphDataModule
from tmgg.data.data_modules.multigraph_data_module import (
    GraphDataCollator,
    RawPyGCollator,
    _ListDataset,
)
from tmgg.data.datasets.graph_types import GraphState
from tmgg.data.datasets.spectre_sbm import (
    SpectreSBMDataset,
    load_spectre_sbm_fixture,
    split_spectre_sbm,
)
from tmgg.utils.noising.size_distribution import SizeDistribution


class SpectreSBMDataModule(BaseGraphDataModule):
    """Load the SPECTRE SBM fixture and serve it via the TMGG collator.

    Parameters
    ----------
    batch_size, num_workers, pin_memory, seed
        Forwarded to :class:`BaseGraphDataModule`. ``seed`` only affects
        the training-split dataloader shuffle; the SPECTRE split itself
        is seeded separately (upstream-matching, see
        :func:`split_spectre_sbm`).
    cache_dir
        Directory under which ``sbm_200.pt`` is cached when the fixture
        has to be downloaded. When ``None``, uses the default cache
        (``~/.cache/tmgg/spectre/``).
    fixture_path
        Path to a pre-downloaded fixture. When set, no network call is
        attempted. Useful when running inside a Modal container whose
        image already contains the file, or for tests.
    """

    # The fixture contains variable-size graphs; ``num_nodes`` is set to
    # the maximum observed node count for size-distribution-aware code
    # paths that need a padding ceiling. Individual graphs retain their
    # real node counts via the per-graph PyG ``num_nodes`` attribute.
    num_nodes: int

    def __init__(
        self,
        *,
        batch_size: int = 12,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        seed: int = 42,
        cache_dir: str | None = None,
        fixture_path: str | None = None,
        num_nodes_max_static: int = 200,
        pad_to_static_n_max: bool = True,
        **_metadata: object,
    ) -> None:
        # ``**_metadata`` swallows informational keys (notably
        # ``eval_meta``) that some upstream config blocks attach to the
        # data namespace for downstream Hydra interpolation. They have
        # no runtime effect on the datamodule itself; rejecting them
        # would force every parity-fix config to special-case the
        # ``+data=spectre_sbm`` overlay.
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            seed=seed,
        )
        # Parity #42 / D-11: safe upper bound on node count, exposed
        # for Hydra interpolation by model configs that need
        # ``max_n_nodes``. Default 200 covers the SPECTRE SBM fixture's
        # observed maximum of 187.
        self.num_nodes_max_static = num_nodes_max_static
        # Off by default: when ``True``, the collator pads every batch
        # to ``num_nodes_max_static`` so downstream tensors emerge at a
        # single fixed shape, unlocking ``torch.compile`` and
        # ``cuda.graph`` capture. Costs extra padded compute on
        # average-sized batches; only flip on once the Tier 3 capture
        # work is ready to use the shape stability. See
        # ``internal notes (not included)``
        self.pad_to_static_n_max = pad_to_static_n_max
        self._cache_dir: Path | None = Path(cache_dir) if cache_dir else None
        self._fixture_path: Path | None = Path(fixture_path) if fixture_path else None

        # Populated by setup().
        self._train_data: list[Data] | None = None
        self._val_data: list[Data] | None = None
        self._test_data: list[Data] | None = None
        self._train_node_counts: list[int] | None = None

        # Informational metadata; mirrors ``MultiGraphDataModule`` so
        # helper code that reads ``num_nodes`` keeps working.
        self.graph_type = "sbm"
        self.num_nodes = 0  # updated by setup() from the fixture

        # Ignore keys leaked from the base config's inline SBM ``data:``
        # block (notably ``num_nodes``, ``graph_type``, ``num_graphs``,
        # ``train_ratio``, ``val_ratio``, ``graph_config``, ``eval_meta``).
        # Without this, ``save_hyperparameters()`` would capture every
        # ``**_metadata`` key and trigger the LightningModule/DataModule
        # hparams-merge collision when an experiment yaml overrides
        # ``model.num_nodes`` to a value other than ``data.num_nodes=20``.
        self.save_hyperparameters(ignore=list(_metadata.keys()))

    @override
    def setup(self, stage: str | None = None) -> None:
        """Load the fixture and materialise PyG Data lists for each split."""
        if self._train_data is not None:
            return

        fixture = self._fixture_path
        if fixture is None and self._cache_dir is not None:
            fixture = self._cache_dir / "sbm_200.pt"

        adjs, n_nodes = load_spectre_sbm_fixture(path=fixture)
        splits = split_spectre_sbm()

        # Track the training-split node counts for size-distribution use,
        # and remember the max for ``num_nodes``. Other splits iterate
        # their adjacencies lazily via the dataset indexer.
        train_adjs = [adjs[i] for i in splits["train"]]
        val_adjs = [adjs[i] for i in splits["val"]]
        test_adjs = [adjs[i] for i in splits["test"]]

        self._train_data = [
            SpectreSBMDataset(train_adjs)[i] for i in range(len(train_adjs))
        ]
        self._val_data = [SpectreSBMDataset(val_adjs)[i] for i in range(len(val_adjs))]
        self._test_data = [
            SpectreSBMDataset(test_adjs)[i] for i in range(len(test_adjs))
        ]

        self._train_node_counts = [n_nodes[i] for i in splits["train"]]
        # Derive num_nodes from train+val only, never test. Mirrors upstream
        # DiGress AbstractDatasetInfos.complete_infos (abstract_dataset.py:95-100).
        # Including test would leak test-set graph sizes into model construction.
        train_val_node_counts = [n_nodes[i] for i in splits["train"]] + [
            n_nodes[i] for i in splits["val"]
        ]
        self.num_nodes = int(max(train_val_node_counts))

    def _dense_collator(self) -> GraphDataCollator:
        """Build the GraphState collator with the configured pad mode.

        ``n_max_static`` is retained as an inert collator field while
        downstream Phase-6 work migrates static-pad to ``to_dense()``
        call sites; it has no effect on the sparse output here.
        """
        return GraphDataCollator(
            n_max_static=self.num_nodes_max_static if self.pad_to_static_n_max else None
        )

    @override
    def train_dataloader(self) -> DataLoader[GraphState]:
        if self._train_data is None:
            raise RuntimeError("SpectreSBMDataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._train_data),
            shuffle=True,
            collate_fn=self._dense_collator(),
        )

    @override
    def val_dataloader(self) -> DataLoader[GraphState]:
        if self._val_data is None:
            raise RuntimeError("SpectreSBMDataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._val_data),
            shuffle=False,
            collate_fn=self._dense_collator(),
        )

    @override
    def test_dataloader(self) -> DataLoader[GraphState]:
        if self._test_data is None:
            raise RuntimeError("SpectreSBMDataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._test_data),
            shuffle=False,
            collate_fn=self._dense_collator(),
        )

    @override
    def train_dataloader_raw_pyg(self) -> DataLoader[object]:
        """Raw PyG ``Batch`` training loader for the parity-port π estimator."""
        if self._train_data is None:
            raise RuntimeError("SpectreSBMDataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._train_data),
            shuffle=False,
            collate_fn=RawPyGCollator(),
        )

    @override
    def get_size_distribution(self, split: str | None = None) -> SizeDistribution:
        """Return the empirical size distribution for a split.

        Unlike the synthetic SBM path, SPECTRE graphs have strongly
        variable node counts (44-187) and the size distribution is a
        first-class training signal — ``log p(N)`` enters the VLB.
        """
        split_map: dict[str | None, list[Data] | None] = {
            "train": self._train_data,
            "val": self._val_data,
            "test": self._test_data,
        }

        if split is None:
            data_list = (
                (self._train_data or [])
                + (self._val_data or [])
                + (self._test_data or [])
            )
        elif split in split_map:
            data_list = split_map[split] or []
        else:
            raise ValueError(
                f"Unknown split {split!r}. Expected 'train', 'val', 'test', or None."
            )

        if not data_list:
            # Pre-setup call with no node count information yet.
            return SizeDistribution.fixed(self.num_nodes or 1)

        node_counts = [int(d.num_nodes) for d in data_list]  # type: ignore[arg-type]
        return SizeDistribution.from_node_counts(node_counts)

"""Lightning DataModule for the SPECTRE planar fixture.

Mirrors :class:`tmgg.data.data_modules.spectre_sbm.SpectreSBMDataModule`
but loads the SPECTRE planar fixture (n=64, 200 graphs) used in
DiGress Table 1.
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
from tmgg.data.datasets.spectre_planar import (
    SpectrePlanarDataset,
    load_spectre_planar_fixture,
    split_spectre_planar,
)
from tmgg.utils.noising.size_distribution import SizeDistribution


class SpectrePlanarDataModule(BaseGraphDataModule):
    """Load the SPECTRE planar fixture and serve via the TMGG collator."""

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
        num_nodes_max_static: int = 64,
        pad_to_static_n_max: bool = True,
        **_metadata: object,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            seed=seed,
        )
        self.num_nodes_max_static = num_nodes_max_static
        self.pad_to_static_n_max = pad_to_static_n_max
        self._cache_dir: Path | None = Path(cache_dir) if cache_dir else None
        self._fixture_path: Path | None = Path(fixture_path) if fixture_path else None

        self._train_data: list[Data] | None = None
        self._val_data: list[Data] | None = None
        self._test_data: list[Data] | None = None
        self._train_node_counts: list[int] | None = None

        self.graph_type = "planar"
        self.num_nodes = 0  # updated by setup()

        # ``**_metadata`` swallows keys leaked from the base config's
        # inline SBM ``data:`` block (notably ``num_nodes``,
        # ``graph_type``, ``num_graphs``, ``train_ratio``, ...). Lightning's
        # ``save_hyperparameters()`` would otherwise capture every
        # unpacked key into ``self.hparams`` and trigger the
        # LightningModule/LightningDataModule hparams-merge collision
        # check (e.g. DM ``num_nodes=20`` from the leaked SBM block vs
        # Module ``num_nodes=64`` from the planar yaml override).
        self.save_hyperparameters(ignore=list(_metadata.keys()))

    @override
    def setup(self, stage: str | None = None) -> None:
        if self._train_data is not None:
            return

        fixture = self._fixture_path
        if fixture is None and self._cache_dir is not None:
            fixture = self._cache_dir / "planar_64_200.pt"

        adjs, n_nodes = load_spectre_planar_fixture(path=fixture)
        splits = split_spectre_planar()

        train_adjs = [adjs[i] for i in splits["train"]]
        val_adjs = [adjs[i] for i in splits["val"]]
        test_adjs = [adjs[i] for i in splits["test"]]

        self._train_data = [
            SpectrePlanarDataset(train_adjs)[i] for i in range(len(train_adjs))
        ]
        self._val_data = [
            SpectrePlanarDataset(val_adjs)[i] for i in range(len(val_adjs))
        ]
        self._test_data = [
            SpectrePlanarDataset(test_adjs)[i] for i in range(len(test_adjs))
        ]

        self._train_node_counts = [n_nodes[i] for i in splits["train"]]
        train_val_node_counts = [n_nodes[i] for i in splits["train"]] + [
            n_nodes[i] for i in splits["val"]
        ]
        self.num_nodes = int(max(train_val_node_counts))

    def _dense_collator(self) -> GraphDataCollator:
        return GraphDataCollator(
            n_max_static=self.num_nodes_max_static if self.pad_to_static_n_max else None
        )

    @override
    def train_dataloader(self) -> DataLoader[GraphState]:
        if self._train_data is None:
            raise RuntimeError("SpectrePlanarDataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._train_data),
            shuffle=True,
            collate_fn=self._dense_collator(),
        )

    @override
    def val_dataloader(self) -> DataLoader[GraphState]:
        if self._val_data is None:
            raise RuntimeError("SpectrePlanarDataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._val_data),
            shuffle=False,
            collate_fn=self._dense_collator(),
        )

    @override
    def test_dataloader(self) -> DataLoader[GraphState]:
        if self._test_data is None:
            raise RuntimeError("SpectrePlanarDataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._test_data),
            shuffle=False,
            collate_fn=self._dense_collator(),
        )

    @override
    def train_dataloader_raw_pyg(self) -> DataLoader[object]:
        if self._train_data is None:
            raise RuntimeError("SpectrePlanarDataModule not setup. Call setup() first.")
        return self._make_dataloader(
            _ListDataset(self._train_data),
            shuffle=False,
            collate_fn=RawPyGCollator(),
        )

    @override
    def get_size_distribution(self, split: str | None = None) -> SizeDistribution:
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
            return SizeDistribution.fixed(self.num_nodes or 1)
        node_counts = [int(d.num_nodes) for d in data_list]  # type: ignore[arg-type]
        return SizeDistribution.from_node_counts(node_counts)

"""GuacaMol DataModule: pins GuacaMolDataset."""

from __future__ import annotations

from tmgg.data.data_modules.molecular.base import MolecularDataModule
from tmgg.data.datasets.molecular.guacamol import GuacaMolDataset


class GuacaMolDataModule(MolecularDataModule):
    """GuacaMol DataModule."""

    dataset_cls = GuacaMolDataset

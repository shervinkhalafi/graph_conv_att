"""MOSES DataModule: pins MOSESDataset."""

from __future__ import annotations

from tmgg.data.data_modules.molecular.base import MolecularDataModule
from tmgg.data.datasets.molecular.moses import MOSESDataset


class MOSESDataModule(MolecularDataModule):
    """MOSES DataModule."""

    dataset_cls = MOSESDataset

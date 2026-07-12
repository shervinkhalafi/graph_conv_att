"""QM9 DataModule: pins QM9Dataset."""

from __future__ import annotations

from tmgg.data.data_modules.molecular.base import MolecularDataModule
from tmgg.data.datasets.molecular.qm9 import QM9Dataset


class QM9DataModule(MolecularDataModule):
    """QM9 (no-H) DataModule. Mirrors SpectreSBMDataModule shape."""

    dataset_cls = QM9Dataset

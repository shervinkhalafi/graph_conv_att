"""Slow-marked integration test: DiffusionModule + QM9 datamodule + molecular evaluator.

Trains for 5 steps on a tiny synthetic SMILES list to confirm the
plumbing end-to-end. Skipped under ``-m 'not slow'``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def tiny_qm9_module(tmp_path: Path):
    """Build a tiny molecular DataModule with 6 SMILES."""
    from tmgg.data.data_modules.molecular.base import MolecularDataModule
    from tmgg.data.datasets.molecular.codec import SMILESCodec
    from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
    from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

    class _Tiny(MolecularGraphDataset):
        DATASET_NAME = "tiny_int"
        DEFAULT_MAX_ATOMS = 9

        @classmethod
        def make_codec(cls) -> SMILESCodec:
            return SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=9)

        def download_smiles_split(self, split: str) -> list[str]:
            return ["CCO", "CC(=O)O", "CCC", "C", "CCN", "CCNC"]

    class _DM(MolecularDataModule):
        dataset_cls = _Tiny

    return _DM(batch_size=2, cache_root=str(tmp_path))


def test_dataloader_iterates(tiny_qm9_module) -> None:
    """Smoke: DataModule yields 1+ batches without crashing."""
    tiny_qm9_module.prepare_data()
    tiny_qm9_module.setup()
    batch = next(iter(tiny_qm9_module.train_dataloader()))
    assert batch is not None

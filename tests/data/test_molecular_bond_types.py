"""Tests that bond multiplicity (single/double/triple/aromatic)
survives the dataset → collator → GraphData round-trip.

Regression test for the architectural shortcut documented in
internal notes (not included) item #1.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tmgg.data.data_modules.molecular.base import MolecularDataModule
from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary


class _Tiny(MolecularGraphDataset):
    DATASET_NAME = "tiny_bondtest"
    DEFAULT_MAX_ATOMS = 10

    @classmethod
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.qm9(), max_atoms=cls.DEFAULT_MAX_ATOMS
        )

    def download_smiles_split(self, split: str) -> list[str]:
        # CC=O: SINGLE + DOUBLE
        # CC#N: SINGLE + TRIPLE
        # CCO:  SINGLE only
        return ["CC=O", "CC#N", "CCO"]


class _DM(MolecularDataModule):
    dataset_cls = _Tiny


def test_e_class_carries_bond_types(tmp_path: Path) -> None:
    """Batch's edge_class should be 5-wide (NONE/SINGLE/DOUBLE/TRIPLE/AROMATIC)
    AND should carry classes beyond just NONE/SINGLE for the test fixtures.

    Datamodules now emit a sparse :class:`GraphState`; ``edge_class`` is
    the per-edge categorical one-hot, shape ``(sum_E, num_bond_types)``.
    The sparse encoding only stores active edges, so ``argmax`` directly
    yields one of the active bond classes (no NO_BOND padding to scan).
    """
    dm = _DM(batch_size=3, cache_root=str(tmp_path))
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch.edge_class is not None
    assert batch.edge_class.shape[-1] == 5, (
        f"edge_class width {batch.edge_class.shape[-1]} != 5 "
        f"(NONE+SINGLE+DOUBLE+TRIPLE+AROMATIC)"
    )
    ec_argmax = batch.edge_class.argmax(dim=-1)
    distinct = set(int(c) for c in torch.unique(ec_argmax).tolist())
    # CC=O contributes a DOUBLE (class 2); CC#N contributes a TRIPLE (class 3).
    assert 2 in distinct or 3 in distinct, (
        f"distinct bond classes {distinct} contains neither DOUBLE (2) nor "
        f"TRIPLE (3); bond types are still being collapsed"
    )

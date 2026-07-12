"""MOSES dataset — DiGress repro Table 5.

SMILES source: the canonical molecularsets/moses GitHub-LFS CSVs,
the same ones upstream DiGress downloads. Each CSV has a ``SMILES``
column (uppercase). Atom decoder: (C, N, S, O, F, Cl, Br, H).

The Python ``moses`` package on PyPI (versions ≤ 0.10) is a
namespace stub with no actual code; the real package
``molsets`` exposes ``moses.get_dataset`` but adds heavy deps for no
benefit here. We follow upstream's lead and just download the CSVs.

DiGress evaluates against MOSES's scaffold split (held-out scaffolds)
for the FCD/SNN metrics. Our DataModule serves
``train`` + ``val`` (= ``test_scaffolds``) + ``test`` (= ``test``).
"""

from __future__ import annotations

import csv

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

# Upstream MOSES splits, hosted via molecularsets/moses GitHub LFS.
# These match upstream DiGress's ``train_url`` / ``val_url`` /
# ``test_url`` (note upstream's somewhat confusing naming: it maps
# the MOSES ``test`` CSV onto its own ``val`` slot and the
# ``test_scaffolds`` CSV onto its own ``test`` slot — we keep the
# name MOSES uses on disk).
_MOSES_URLS = {
    "train": (
        "https://media.githubusercontent.com/media/molecularsets/moses/"
        "master/data/train.csv"
    ),
    "val": (
        "https://media.githubusercontent.com/media/molecularsets/moses/"
        "master/data/test_scaffolds.csv"
    ),
    "test": (
        "https://media.githubusercontent.com/media/molecularsets/moses/"
        "master/data/test.csv"
    ),
}


class MOSESDataset(MolecularGraphDataset):
    """MOSES split. Atom decoder: (C, N, S, O, F, Cl, Br, H)."""

    DATASET_NAME = "moses"
    DEFAULT_MAX_ATOMS = 30
    RAW_FILES = _MOSES_URLS

    @classmethod
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.moses(),
            remove_h=True,
            kekulize=True,
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    def download_smiles_split(self, split: str) -> list[str]:
        """Download the MOSES CSV for a split and return its SMILES column."""
        path = self._default_download(split)
        smiles_out: list[str] = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Upstream uses ``pd.read_csv(path)['SMILES']`` — column
                # is uppercase in MOSES CSVs.
                smi = row.get("SMILES") or row.get("smiles")
                if smi:
                    smiles_out.append(smi.strip())
        if not smiles_out:
            raise RuntimeError(
                f"MOSES split {split!r}: 0 SMILES read from {path}; "
                "column 'SMILES' may be missing or file is empty."
            )
        return smiles_out

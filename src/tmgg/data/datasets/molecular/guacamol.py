"""GuacaMol dataset — DiGress repro Table 6.

SMILES source: GuacaMol's published train/val/test ``.smiles`` files
(one canonical SMILES per line, no header), the same ones upstream
DiGress fetches.

URL host: ``ndownloader.figshare.com`` rather than
``figshare.com/ndownloader``. The latter is fronted by an AWS WAF that
returns ``HTTP 202 + x-amzn-waf-action: challenge`` for non-browser
clients, yielding zero bytes; the former 302-redirects to a signed S3
URL that any client can follow. The numeric file IDs are unchanged.

Atom decoder: (C, N, O, F, B, Br, Cl, I, P, S, Se, Si).
"""

from __future__ import annotations

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

# Upstream GuacaMol splits — file IDs match upstream DiGress; we just
# use the WAF-bypassing ndownloader subdomain so curl/urllib succeed.
_GUACAMOL_URLS = {
    "train": "https://ndownloader.figshare.com/files/13612760",
    "val": "https://ndownloader.figshare.com/files/13612766",
    "test": "https://ndownloader.figshare.com/files/13612757",
}


class GuacaMolDataset(MolecularGraphDataset):
    """GuacaMol split. Atom decoder: 12-element vocabulary."""

    DATASET_NAME = "guacamol"
    DEFAULT_MAX_ATOMS = 88
    RAW_FILES = _GUACAMOL_URLS

    @classmethod
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.guacamol(),
            remove_h=True,
            kekulize=True,
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    def download_smiles_split(self, split: str) -> list[str]:
        path = self._default_download(split)
        smiles_out = [line.strip() for line in path.open("r") if line.strip()]
        if not smiles_out:
            raise RuntimeError(
                f"GuacaMol split {split!r}: 0 SMILES read from {path}; "
                "the figshare URL may have been replaced by an HTML challenge page."
            )
        return smiles_out

"""QM9 dataset (no-H by default) — DiGress repro Table 4.

SMILES source: DeepChem's QM9 mirror (``molnet_publish/qm9.zip``),
which is the same archive upstream DiGress reads. The zip contains
``gdb9.sdf`` (3D conformers for ~134k molecules), ``gdb9.sdf.csv``
(physical properties only — *no* SMILES column), and
``uncharacterized.txt`` (indices of molecules to skip).

The CSV in the upstream archive has no SMILES column; SMILES live in
the SDF. We extract them once with RDKit's ``SDMolSupplier`` and cache
the result as ``qm9.smi`` so subsequent splits reuse the parsed list.

Splits: 80 % train / 10 % val / 10 % test, mirroring upstream's
``np.split`` recipe (seeded shuffle).
"""

from __future__ import annotations

import random
from pathlib import Path

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

# DiGress's upstream URL — molnet_publish bundle (zip), not gdb9.tar.gz.
# The tar.gz mirror is the *raw* QM9 archive without the same layout
# and was the source of the silent zero-row bug we fixed here.
_QM9_ZIP_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
)
# Indices of QM9 molecules flagged uncharacterized by Ramakrishnan
# et al. (2014). Mirrors upstream DiGress's ``raw_url2`` retrieval.
_QM9_UNCHAR_URL = "https://ndownloader.figshare.com/files/3195404"

# Upstream split rule: random 80/10/10 with seed 0.
_QM9_SPLIT_SEED = 0


class QM9Dataset(MolecularGraphDataset):
    """QM9 (no-H) split. Atom decoder: (C, N, O, F)."""

    DATASET_NAME = "qm9"
    DEFAULT_MAX_ATOMS = 9
    # No per-split URLs — single archive; we split locally.
    RAW_FILES: dict[str, str] = {}

    @classmethod
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.qm9(remove_h=True),
            remove_h=True,
            kekulize=True,
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    def download_smiles_split(self, split: str) -> list[str]:
        """Read all SMILES from the QM9 SDF and return the requested split."""
        smi_path = self._raw_dir() / "qm9.smi"
        smi_path.parent.mkdir(parents=True, exist_ok=True)
        if not smi_path.exists():
            self._materialise_qm9_smi(smi_path)

        with smi_path.open("r") as f:
            all_smiles = [line.strip() for line in f if line.strip()]

        rng = random.Random(_QM9_SPLIT_SEED)
        indices = list(range(len(all_smiles)))
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        if split == "train":
            picked = indices[:n_train]
        elif split == "val":
            picked = indices[n_train : n_train + n_val]
        elif split == "test":
            picked = indices[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split {split!r}.")
        return [all_smiles[i] for i in picked]

    def _materialise_qm9_smi(self, target: Path) -> None:
        """Download the QM9 zip and extract SMILES from gdb9.sdf.

        The zip ships the SDF + the physical-property CSV; SMILES are
        not present in the CSV column set. We use RDKit's
        :class:`~rdkit.Chem.SDMolSupplier` to walk the SDF and emit
        canonical SMILES, dropping molecules flagged uncharacterized
        in the upstream filter list.
        """
        import urllib.request
        import zipfile

        from rdkit import (
            Chem,
            RDLogger,  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
        )

        raw_dir = target.parent
        archive = raw_dir / "qm9.zip"
        if not archive.exists():
            urllib.request.urlretrieve(_QM9_ZIP_URL, archive)  # noqa: S310

        sdf_path = raw_dir / "gdb9.sdf"
        csv_path = raw_dir / "gdb9.sdf.csv"
        if not sdf_path.exists() or not csv_path.exists():
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(raw_dir)
        if not sdf_path.exists():
            raise RuntimeError(
                f"Expected gdb9.sdf inside {archive}; archive layout has changed."
            )

        # Uncharacterized list: 1-based indices of molecules to drop.
        # Upstream parses this with ``[int(x.split()[0]) - 1 for x in
        # f.read().split('\n')[9:-2]]``. We mirror that exactly.
        unchar_path = raw_dir / "uncharacterized.txt"
        if not unchar_path.exists():
            urllib.request.urlretrieve(_QM9_UNCHAR_URL, unchar_path)  # noqa: S310
        with unchar_path.open("r") as f:
            skip = {
                int(line.split()[0]) - 1
                for line in f.read().split("\n")[9:-2]
                if line.strip()
            }

        RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
        smiles_out: list[str] = []
        for i, mol in enumerate(suppl):
            if i in skip or mol is None:
                continue
            # Drop molecules with non-zero formal charges. The codec's
            # categorical atom decoder records element only, so charged
            # species silently lose their ``+``/``-`` on decode and
            # round-trip mismatch. Roughly 0.6 % of QM9 carries a
            # formal charge; dropping them keeps the dataset honest.
            if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
                continue
            try:
                # ``isomericSmiles=False`` drops stereo (atom chirality
                # and bond E/Z). The codec's categorical bond decoder
                # has no stereo channel, so keeping isomeric SMILES
                # here would guarantee a round-trip mismatch on every
                # stereo-bearing molecule. Upstream DiGress also drops
                # stereo at the graph-encoding boundary, so this only
                # matches what training would see.
                smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            except Exception:  # noqa: BLE001
                continue
            if smi:
                smiles_out.append(smi)

        if not smiles_out:
            raise RuntimeError(
                f"Extracted 0 SMILES from {sdf_path}; SDF appears empty or unreadable."
            )

        with target.open("w") as f:
            for smi in smiles_out:
                f.write(smi + "\n")

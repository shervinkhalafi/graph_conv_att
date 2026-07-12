"""Tests for MolecularGraphDataset shard caching.

Uses a tiny tmp-fixture subclass to avoid hitting the real QM9/MOSES/
GuacaMol downloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary


class _TinyDataset(MolecularGraphDataset):
    DATASET_NAME = "tiny"
    DEFAULT_MAX_ATOMS = 5
    SAMPLE_SMILES = ["CCO", "CC(=O)O", "CCC"]

    @classmethod
    @override
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.qm9(),
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    @override
    def download_smiles_split(self, split: str) -> list[str]:
        return list(self.SAMPLE_SMILES)


def test_first_setup_preprocesses_and_caches(tmp_path: Path) -> None:
    ds = _TinyDataset(split="train", cache_root=tmp_path)
    ds.setup()
    assert len(ds) == 3
    # Shard file should exist.
    shard_dir = tmp_path / "tiny" / "preprocessed" / ds._codec.cache_key() / "train"  # type: ignore[union-attr]
    assert shard_dir.exists()
    assert any(shard_dir.iterdir())


def test_second_setup_hits_cache(tmp_path: Path) -> None:
    ds1 = _TinyDataset(split="train", cache_root=tmp_path)
    ds1.setup()

    # Mutate the SAMPLE_SMILES on the second instance — if cache hit
    # is broken, ds2 would re-encode and pick up the new SMILES.
    class _MutantDataset(_TinyDataset):
        SAMPLE_SMILES = ["NEVER_CALLED"]

    ds2 = _MutantDataset(split="train", cache_root=tmp_path)
    ds2.setup()
    assert len(ds2) == 3  # cached from ds1, not regenerated


def test_codec_change_invalidates_cache(tmp_path: Path) -> None:
    ds1 = _TinyDataset(split="train", cache_root=tmp_path)
    ds1.setup()

    class _OtherCodecDataset(_TinyDataset):
        @classmethod
        @override
        def make_codec(cls) -> SMILESCodec:
            return SMILESCodec(
                vocab=AtomBondVocabulary.moses(),  # different vocab
                max_atoms=cls.DEFAULT_MAX_ATOMS,
            )

    ds2 = _OtherCodecDataset(split="train", cache_root=tmp_path)
    ds2.setup()
    # Different cache key directory.
    k1 = ds1._codec.cache_key()  # type: ignore[union-attr]
    k2 = ds2._codec.cache_key()  # type: ignore[union-attr]
    assert k1 != k2
    assert (tmp_path / "tiny" / "preprocessed" / k1).exists()
    assert (tmp_path / "tiny" / "preprocessed" / k2).exists()


def test_write_shards_is_atomic_no_tmp_files_after_success(tmp_path: Path) -> None:
    """``_write_shards`` must leave no ``.pt.tmp`` files behind on success.

    Starting state: a fresh ``_TinyDataset`` writes its shards via
    ``setup()`` (cache miss path).

    Invariant: after ``setup()`` returns cleanly, only the final
    ``<idx>.pt`` files exist in the shard directory — no
    ``<idx>.pt.tmp`` siblings should remain. The atomic-rename pattern
    is what protects against the Phase-8 ``PytorchStreamReader failed
    locating file data/N`` corruption that bricked an entire
    smoke-run session; the regression test exists so a future revert
    that drops the rename gets caught immediately.
    """
    ds = _TinyDataset(split="train", cache_root=tmp_path)
    ds.setup()
    shard_dir = tmp_path / "tiny" / "preprocessed" / ds._codec.cache_key() / "train"  # type: ignore[union-attr]
    pt_files = sorted(shard_dir.glob("*.pt"))
    tmp_files = sorted(shard_dir.glob("*.pt.tmp"))
    assert pt_files, "expected at least one final shard file"
    assert not tmp_files, f"unexpected .pt.tmp leftovers: {tmp_files}"

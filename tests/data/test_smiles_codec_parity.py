"""Slow-marked: per-dataset train-set round-trip ≥ 99% canonical match.

Uses the dataset's own preprocessing — therefore depends on
``rdkit`` + the dataset packages being installed and the raw files
being downloadable.
"""

from __future__ import annotations

import random

import pytest

pytestmark = pytest.mark.slow

_SAMPLE_N = 1000


def _round_trip_match_rate(smiles: list[str], codec) -> float:
    matched = 0
    n_attempted = 0
    for s in smiles:
        encoded = codec.encode(s)
        if encoded is None:
            continue
        decoded = codec.decode(encoded)
        if decoded is None:
            continue
        # Compare canonical forms.
        from rdkit import Chem

        # rdkit-stubs types ``MolFromSmiles`` as ``Mol | None`` but
        # ``MolToSmiles`` as taking ``Mol``. The real RDKit returns ``None``
        # for invalid SMILES, so a runtime guard would be safer; in this
        # test we rely on the SMILES list being valid. Drop this marker
        # once rdkit-stubs offers an overload that accepts ``Mol | None``.
        original_canon = Chem.MolToSmiles(Chem.MolFromSmiles(s))  # type: ignore[reportArgumentType]
        n_attempted += 1
        if decoded == original_canon:
            matched += 1
    if n_attempted == 0:
        return 0.0
    return matched / n_attempted


@pytest.mark.parametrize(
    "dataset_name",
    ["qm9", "moses", "guacamol"],
)
def test_round_trip_match_rate_above_99pct(dataset_name: str, tmp_path) -> None:
    if dataset_name == "qm9":
        from tmgg.data.datasets.molecular.qm9 import QM9Dataset

        ds = QM9Dataset(split="train", cache_root=tmp_path)
    elif dataset_name == "moses":
        from tmgg.data.datasets.molecular.moses import MOSESDataset

        ds = MOSESDataset(split="train", cache_root=tmp_path)
    else:
        from tmgg.data.datasets.molecular.guacamol import GuacaMolDataset

        ds = GuacaMolDataset(split="train", cache_root=tmp_path)

    smiles = ds.download_smiles_split("train")
    rng = random.Random(0)
    sample = rng.sample(smiles, min(_SAMPLE_N, len(smiles)))
    rate = _round_trip_match_rate(sample, ds.make_codec())
    assert rate >= 0.99, f"{dataset_name} round-trip match rate {rate:.4f} < 0.99"

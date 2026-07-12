"""Tests for AtomBondVocabulary presets and properties."""

from __future__ import annotations

import pytest

from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary


def test_qm9_preset_atom_decoder() -> None:
    v = AtomBondVocabulary.qm9(remove_h=True)
    assert v.atom_decoder == ("C", "N", "O", "F")
    assert v.num_atom_types == 4


def test_qm9_preset_with_h() -> None:
    v = AtomBondVocabulary.qm9(remove_h=False)
    assert v.atom_decoder == ("H", "C", "N", "O", "F")
    assert v.num_atom_types == 5


def test_moses_preset_atom_decoder() -> None:
    v = AtomBondVocabulary.moses()
    assert v.atom_decoder == ("C", "N", "S", "O", "F", "Cl", "Br", "H")


def test_guacamol_preset_atom_decoder() -> None:
    v = AtomBondVocabulary.guacamol()
    assert v.atom_decoder == (
        "C",
        "N",
        "O",
        "F",
        "B",
        "Br",
        "Cl",
        "I",
        "P",
        "S",
        "Se",
        "Si",
    )


def test_vocabulary_is_hashable() -> None:
    v = AtomBondVocabulary.qm9()
    h1 = hash(v)
    h2 = hash(AtomBondVocabulary.qm9())
    assert h1 == h2


def test_encode_atom_roundtrip() -> None:
    v = AtomBondVocabulary.qm9()
    assert v.decode_atom(v.encode_atom("C")) == "C"


def test_encode_atom_unknown_raises() -> None:
    v = AtomBondVocabulary.qm9()
    with pytest.raises(ValueError):
        v.encode_atom("Xx")


def test_encode_bond_roundtrip() -> None:
    v = AtomBondVocabulary.qm9()
    assert v.decode_bond(v.encode_bond("AROMATIC")) == "AROMATIC"


def test_max_valence_lookup() -> None:
    v = AtomBondVocabulary.qm9()
    assert v.max_valence("C") == 4
    assert v.max_valence("N") == 3
    assert v.max_valence("O") == 2


def test_repr_is_stable_across_calls() -> None:
    """Stable __repr__ matters for cache-key hashing."""
    v1 = AtomBondVocabulary.qm9()
    v2 = AtomBondVocabulary.qm9()
    assert repr(v1) == repr(v2)

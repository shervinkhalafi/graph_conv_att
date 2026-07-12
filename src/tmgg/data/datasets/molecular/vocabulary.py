"""Frozen atom + bond vocabulary for molecular categorical diffusion.

Constants mirror upstream DiGress
(``digress-upstream-readonly/src/datasets/{qm9,moses,guacamol}_dataset.py``).
The class is hashable so its ``repr`` doubles as a stable cache key
component for :class:`SMILESCodec`'s preprocessed-shard directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

# Bond decoder is shared across all three molecular datasets in DiGress.
# Order matters: the index into this tuple is the categorical edge class.
# ``"NONE"`` is the no-edge class (class 0). Order matches upstream's
# ``bonds = {Chem.BondType.SINGLE: 1, ...}`` mapping.
BOND_DECODER: tuple[str, ...] = ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

# Per-atom max valence used for valency-mask construction in extra
# features. Mirrors upstream constants in the per-dataset modules.
_DEFAULT_MAX_VALENCES: Mapping[str, int] = {
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Br": 1,
    "Cl": 1,
    "I": 1,
    "P": 5,
    "S": 6,
    "Se": 6,
    "Si": 4,
    "H": 1,
}


@dataclass(frozen=True)
class AtomBondVocabulary:
    """Mapping atom symbol ↔ class idx + bond type ↔ class idx.

    Frozen + hashable: instances can serve as cache keys (their
    ``__repr__`` is stable across runs by virtue of the frozen
    dataclass machinery). Equality is structural over the decoder
    tuples.

    Parameters
    ----------
    atom_decoder
        Tuple of atom symbols in class-index order. ``atom_decoder[0]``
        is class 0, etc. Upstream DiGress reserves no "no-atom" slot
        for the molecular datasets — every node is a real atom.
    bond_decoder
        Tuple of bond-type names in class-index order. ``bond_decoder[0]``
        is the no-edge class.
    max_valences
        Per-atom maximum valence. Used by valency-aware extra features.

    Notes
    -----
    Use the classmethod presets (:meth:`qm9`, :meth:`moses`,
    :meth:`guacamol`) rather than instantiating directly; they pin
    the upstream constants.
    """

    atom_decoder: tuple[str, ...]
    bond_decoder: tuple[str, ...] = BOND_DECODER
    max_valences: tuple[tuple[str, int], ...] = field(
        default_factory=lambda: tuple(sorted(_DEFAULT_MAX_VALENCES.items()))
    )

    @classmethod
    def qm9(cls, *, remove_h: bool = True) -> AtomBondVocabulary:
        """QM9 (no-H) atom decoder. Matches upstream
        ``qm9_dataset.py:atom_decoder`` for ``remove_h=True``."""
        atom_decoder = ("C", "N", "O", "F") if remove_h else ("H", "C", "N", "O", "F")
        return cls(atom_decoder=atom_decoder)

    @classmethod
    def moses(cls) -> AtomBondVocabulary:
        """MOSES atom decoder. Matches upstream
        ``moses_dataset.py:atom_decoder``."""
        return cls(atom_decoder=("C", "N", "S", "O", "F", "Cl", "Br", "H"))

    @classmethod
    def guacamol(cls) -> AtomBondVocabulary:
        """GuacaMol atom decoder. Matches upstream
        ``guacamol_dataset.py:atom_decoder``."""
        return cls(
            atom_decoder=(
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
        )

    @property
    def num_atom_types(self) -> int:
        return len(self.atom_decoder)

    @property
    def num_bond_types(self) -> int:
        return len(self.bond_decoder)

    @property
    def atom_encoder(self) -> Mapping[str, int]:
        return {symbol: idx for idx, symbol in enumerate(self.atom_decoder)}

    @property
    def bond_encoder(self) -> Mapping[str, int]:
        return {symbol: idx for idx, symbol in enumerate(self.bond_decoder)}

    def encode_atom(self, symbol: str) -> int:
        try:
            return self.atom_encoder[symbol]
        except KeyError as exc:
            raise ValueError(
                f"Atom {symbol!r} not in vocabulary {self.atom_decoder}."
            ) from exc

    def decode_atom(self, idx: int) -> str:
        if not 0 <= idx < self.num_atom_types:
            raise ValueError(
                f"Atom index {idx} out of range [0, {self.num_atom_types})."
            )
        return self.atom_decoder[idx]

    def encode_bond(self, bond_type_name: str) -> int:
        try:
            return self.bond_encoder[bond_type_name]
        except KeyError as exc:
            raise ValueError(
                f"Bond type {bond_type_name!r} not in {self.bond_decoder}."
            ) from exc

    def decode_bond(self, idx: int) -> str:
        if not 0 <= idx < self.num_bond_types:
            raise ValueError(
                f"Bond index {idx} out of range [0, {self.num_bond_types})."
            )
        return self.bond_decoder[idx]

    def max_valence(self, symbol: str) -> int:
        for sym, valence in self.max_valences:
            if sym == symbol:
                return valence
        raise ValueError(f"No max valence registered for atom {symbol!r}.")

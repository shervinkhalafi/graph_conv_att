"""RDKit-based generation-quality metrics: Validity / Uniqueness / Novelty.

Mirrors upstream DiGress's ``rdkit_functions.py`` formulas:

- Validity = |valid molecules| / |attempted|
- Uniqueness = |distinct canonical SMILES among valid| / |valid|
- Novelty = |canonical SMILES not in train set| / |valid|
"""

from __future__ import annotations

from collections.abc import Sequence

from tmgg.evaluation.molecular.metric import MolecularMetric


def _to_canonical(smiles: str) -> str | None:
    """Canonicalise a SMILES via RDKit; return None if invalid."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


class ValidityMetric(MolecularMetric):
    """Fraction of generated SMILES that pass RDKit sanitisation."""

    name = "validity"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        del reference
        if not generated:
            return 0.0
        n_valid = sum(1 for s in generated if _to_canonical(s) is not None)
        return n_valid / len(generated)


class UniquenessMetric(MolecularMetric):
    """Fraction of valid molecules that are unique by canonical SMILES."""

    name = "uniqueness"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        del reference
        canonical = [_to_canonical(s) for s in generated]
        valid = [c for c in canonical if c is not None]
        if not valid:
            return 0.0
        return len(set(valid)) / len(valid)


class NoveltyMetric(MolecularMetric):
    """Fraction of valid molecules NOT in the reference (training) set."""

    name = "novelty"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError(
                "NoveltyMetric requires a reference set of training SMILES."
            )
        ref_canonical = {
            c for c in (_to_canonical(s) for s in reference) if c is not None
        }
        canonical = [_to_canonical(s) for s in generated]
        valid = [c for c in canonical if c is not None]
        if not valid:
            return 0.0
        novel = [c for c in valid if c not in ref_canonical]
        return len(novel) / len(valid)

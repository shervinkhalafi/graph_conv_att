"""Tests for the RDKit family of molecular metrics."""

from __future__ import annotations

import math

from tmgg.evaluation.molecular.rdkit_metrics import (
    NoveltyMetric,
    UniquenessMetric,
    ValidityMetric,
)


def test_validity_basic() -> None:
    gen = ["CCO", "not_a_smiles", "CC(=O)O"]
    v = ValidityMetric().compute(gen)
    assert math.isclose(v, 2 / 3, rel_tol=1e-6)


def test_validity_empty() -> None:
    assert ValidityMetric().compute([]) == 0.0


def test_uniqueness_with_duplicates() -> None:
    gen = ["CCO", "CCO", "CC(=O)O", "not_a_smiles"]
    # 3 valid, 2 distinct
    u = UniquenessMetric().compute(gen)
    assert math.isclose(u, 2 / 3, rel_tol=1e-6)


def test_novelty_against_train_set() -> None:
    gen = ["CCO", "CC(=O)O", "CCC", "not_a_smiles"]
    train = ["CCO"]
    n = NoveltyMetric().compute(gen, train)
    # 3 valid; novel = {CC(=O)O, CCC} = 2/3
    assert math.isclose(n, 2 / 3, rel_tol=1e-6)

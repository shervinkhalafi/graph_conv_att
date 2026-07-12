"""Tests for MolecularEvaluator + classmethod presets."""

from __future__ import annotations

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary
from tmgg.evaluation.molecular import MolecularEvaluator
from tmgg.evaluation.molecular.rdkit_metrics import (
    UniquenessMetric,
    ValidityMetric,
)


def _qm9_codec() -> SMILESCodec:
    return SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)


def test_for_qm9_metric_keys() -> None:
    ev = MolecularEvaluator.for_qm9()
    names = [m.name for m in ev.metrics]
    assert names == ["validity", "uniqueness", "novelty"]


def test_for_moses_metric_keys() -> None:
    ev = MolecularEvaluator.for_moses()
    names = [m.name for m in ev.metrics]
    assert names == [
        "validity",
        "uniqueness",
        "novelty",
        "fcd",
        "snn",
        "int_div",
        "filters",
        "scaffold_novelty",
    ]


def test_for_guacamol_metric_keys() -> None:
    ev = MolecularEvaluator.for_guacamol()
    names = [m.name for m in ev.metrics]
    assert names == [
        "validity",
        "uniqueness",
        "novelty",
        "kl_div_property",
        "fcd_chembl",
    ]


def test_classmethod_presets_swallow_leaked_kwargs() -> None:
    """Hydra-leaked evaluator fields must not crash the classmethod call.

    Starting state: each molecular experiment yaml inherits the
    ``evaluator:`` block from ``discrete_sbm_official`` (because the
    yaml does ``override /models/discrete@model: discrete_sbm_official``)
    and only overrides ``_target_``. Hydra deep-merges, so
    ``eval_num_samples``, ``p_intra``, ``p_inter``, and
    ``clustering_sigma`` get passed through to the classmethod call.

    Invariant: the three classmethod presets MUST accept these
    Hydra-leaked kwargs without raising. Was caught by the Phase 8
    QM9 Modal smoke run (TypeError at ``MolecularEvaluator.for_qm9()``).
    """
    leaked = {
        "eval_num_samples": 40,
        "p_intra": 0.7,
        "p_inter": 0.1,
        "clustering_sigma": 0.1,
    }
    ev_qm9 = MolecularEvaluator.for_qm9(**leaked)
    ev_moses = MolecularEvaluator.for_moses(**leaked)
    ev_guacamol = MolecularEvaluator.for_guacamol(**leaked)
    assert ev_qm9.metrics
    assert ev_moses.metrics
    assert ev_guacamol.metrics


def test_evaluate_on_decoded_graphs() -> None:
    """End-to-end: evaluator decodes GraphData and runs metrics."""
    codec = _qm9_codec()
    g_ccol = codec.encode("CCO")
    g_acet = codec.encode("CC(=O)O")
    assert g_ccol is not None and g_acet is not None
    ev = MolecularEvaluator(
        metrics=[ValidityMetric(), UniquenessMetric()],
        codec=codec,
    )
    refs = [g_ccol]
    gen = [g_ccol, g_acet]
    results = ev.evaluate(refs, gen)
    assert "validity" in results.values
    assert "uniqueness" in results.values
    assert results.values["validity"] == 1.0

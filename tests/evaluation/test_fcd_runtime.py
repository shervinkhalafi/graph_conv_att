"""Runtime smoke test for FCDMetric.

Verifies that ChemNet weights are reachable and the FCD computation
returns a finite scalar. Slow-marked because the first run downloads
ChemNet (~50 MB).
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.slow


_GEN = [
    "CCO",
    "CC(=O)O",
    "CCN",
    "CCCO",
    "CCCC",
    "c1ccccc1",
    "Cc1ccccc1",
    "CCOCC",
]
_REF = [
    "CCO",
    "CC(=O)O",
    "CN",
    "CCO",
    "CCCN",
    "CCC",
    "OCCO",
    "CCN",
]


def test_fcd_metric_finite_on_distinct_sets() -> None:
    """FCD between two non-trivial SMILES sets is a finite, positive number."""
    from tmgg.evaluation.molecular.moses_metrics import FCDMetric

    v = FCDMetric().compute(_GEN, _REF)
    assert math.isfinite(v), f"FCD returned non-finite: {v}"
    assert v >= 0.0, f"FCD must be non-negative, got {v}"


def test_fcd_metric_near_zero_on_identical_sets() -> None:
    """FCD between identical sets is near zero."""
    from tmgg.evaluation.molecular.moses_metrics import FCDMetric

    m = FCDMetric()
    v = m.compute(_GEN, _GEN)
    assert math.isfinite(v)
    # Identical sets should give FCD close to 0; allow numerical slack.
    assert v < 1.0, f"Identical-set FCD too large: {v} (expected < 1.0)"


def test_fcd_chembl_metric_finite() -> None:
    """FCDChEMBLMetric is a thin wrapper around the maintained ``fcd>=1.2.2`` — same shape."""
    from tmgg.evaluation.molecular.guacamol_metrics import FCDChEMBLMetric

    v = FCDChEMBLMetric().compute(_GEN, _REF)
    assert math.isfinite(v)
    assert v >= 0.0

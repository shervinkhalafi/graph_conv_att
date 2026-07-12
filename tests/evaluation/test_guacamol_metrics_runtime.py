"""Runtime smoke tests for vendored GuacaMol metrics.

Regression test for shims-doc item 4.1/4.2 (KLDivPropertyMetric ImportError
and pyright suppressions). After this fix, both metrics return finite
values without depending on the unmaintained `guacamol` package.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.slow  # FCDChEMBLMetric loads ChemNet weights


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


def test_kldiv_property_metric_in_unit_range() -> None:
    """KLDiv score is exp(-mean(KL)), so always in [0, 1]."""
    from tmgg.evaluation.molecular.guacamol_metrics import KLDivPropertyMetric

    v = KLDivPropertyMetric().compute(_GEN, _REF)
    assert math.isfinite(v)
    assert 0.0 <= v <= 1.0, f"KLDivProperty score {v} not in [0, 1]"


def test_kldiv_property_metric_near_one_on_identical_sets() -> None:
    """Identical distributions => KL = 0 => exp(0) = 1."""
    from tmgg.evaluation.molecular.guacamol_metrics import KLDivPropertyMetric

    v = KLDivPropertyMetric().compute(_GEN, _GEN)
    assert v > 0.95, f"Identical-set KLDiv score {v} not near 1.0"


def test_fcd_chembl_metric_finite() -> None:
    """FCDChEMBLMetric still works after dropping guacamol dep."""
    from tmgg.evaluation.molecular.guacamol_metrics import FCDChEMBLMetric

    v = FCDChEMBLMetric().compute(_GEN, _REF)
    assert math.isfinite(v) and v >= 0.0

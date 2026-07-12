"""Runtime smoke tests for MOSES eval metrics.

Regression test for the broken-package + ``rdkit.six`` issue documented
in ``internal notes (not included)`` items
4.1/4.2/4.3. The four ``MolecularMetric`` subclasses below all used to
crash because the project depended on the ``moses`` PyPI stub and even
the real ``molsets`` package fails to import on the current toolchain
(``rdkit.six`` removal, ``pandas.DataFrame.append`` removal, ``np.NaN``
removal, deprecated Morgan API segfaulting). The fix vendors the four
metric formulas plus the canonical MOSES filter SMARTS catalogues
directly into ``tmgg.evaluation.molecular.moses_metrics``. Each
``compute()`` call must therefore return a finite float in ``[0, 1]``.

The tests are marked ``slow`` because they pull in RDKit's full Morgan
fingerprint generator and Bemis-Murcko scaffold extraction, both of
which add a couple of seconds of import overhead - too long for the
default fast-test loop, fine for the nightly run.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.slow


_GEN = ["CCO", "CC(=O)O", "CCN", "CCCO", "CCCC", "c1ccccc1"]
_REF = ["CCO", "CC(=O)O", "CN", "CCO", "CCCN"]


def test_snn_metric_runs() -> None:
    from tmgg.evaluation.molecular.moses_metrics import SNNMetric

    v = SNNMetric().compute(_GEN, _REF)
    assert math.isfinite(v) and 0.0 <= v <= 1.0


def test_intdiv_metric_runs() -> None:
    from tmgg.evaluation.molecular.moses_metrics import IntDivMetric

    v = IntDivMetric().compute(_GEN)
    assert math.isfinite(v) and 0.0 <= v <= 1.0


def test_filters_metric_runs() -> None:
    from tmgg.evaluation.molecular.moses_metrics import FiltersMetric

    v = FiltersMetric().compute(_GEN)
    assert math.isfinite(v) and 0.0 <= v <= 1.0


def test_scaffold_split_metric_runs() -> None:
    from tmgg.evaluation.molecular.moses_metrics import ScaffoldSplitMetric

    v = ScaffoldSplitMetric().compute(_GEN, _REF)
    assert math.isfinite(v) and 0.0 <= v <= 1.0

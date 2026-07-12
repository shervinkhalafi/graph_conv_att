"""Modal-marked: confirm molecular deps are present in the image.

Skipped on host pytest unless ``-m modal`` is passed. Runs as part
of Modal-side CI / deploy verification.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.modal]


def test_rdkit_imports() -> None:
    from rdkit import Chem  # noqa: F401


def test_fcd_imports() -> None:
    """Verify the maintained ``fcd>=1.2.2`` (bioinf-jku) is reachable.

    Replaces the previous ``test_fcd_torch_imports`` which checked the
    frozen ``fcd_torch==1.0.7`` package; the project now depends on
    ``fcd>=1.2.2``, the maintained PyTorch refresh.
    """
    from fcd import canonical_smiles, get_fcd, load_ref_model  # noqa: F401


def test_moses_imports() -> None:
    import moses  # noqa: F401


def test_guacamol_imports() -> None:
    import guacamol  # noqa: F401

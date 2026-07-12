"""Tests for ``ExtraMolecularFeatures`` + ``CompositeExtraFeatures``.

Rationale
---------
- Confirms the port of upstream cvignac/DiGress's
  ``ExtraMolecularFeatures`` matches our
  :class:`tmgg.models.digress.extra_features.ExtraFeaturesProvider`
  protocol shape contract — (extra_X, extra_E, extra_y) widths
  (2, 0, 1) per provider call, and ``adjust_dims`` adding the same.
- Verifies known-input → known-output for each sub-feature against
  hand-computed ground truth on tiny toy graphs (so the upstream
  port is checked, not just that it runs).
- Exercises ``CompositeExtraFeatures`` as the bridge between
  structural ``ExtraFeatures(all)`` and ``ExtraMolecularFeatures``,
  which is the actual configuration the digress-repro yamls install.

Starting state assumed: noisy ``(X, E)`` are one-hot tensors after
the noise process; ``X`` width matches ``noise_process.x_classes``,
``E`` width = 5 (bond classes). Tests use plain torch tensors with
no noise applied so values stay deterministic.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.models.digress.extra_features import ExtraFeatures
from tmgg.models.digress.extra_molecular_features import (
    ChargeFeature,
    CompositeExtraFeatures,
    ExtraMolecularFeatures,
    ValencyFeature,
    WeightFeature,
)

# ------------------------------------------------------------------
# Toy graph fixture: 3-node QM9-style methanol (CH3-OH skeleton)
# ------------------------------------------------------------------
#
# atom_decoder = ['C', 'N', 'O', 'F'] → 4-class X
# Atoms: [C (idx 0), O (idx 2), F (idx 3)]
# Bonds: C--O single, O--F single (toy molecule, valencies will not all match)
# This is a tiny synthetic case; the goal is to verify shapes + math.


def _toy_qm9_graph(
    bs: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X, E, y, node_mask) for a 3-node QM9-style toy graph."""
    # 4 atom classes, 5 bond classes, 3 nodes.
    X = torch.zeros(bs, 3, 4)
    X[:, 0, 0] = 1.0  # C
    X[:, 1, 2] = 1.0  # O
    X[:, 2, 3] = 1.0  # F
    E = torch.zeros(bs, 3, 3, 5)
    # No-bond default; insert single bonds C-O and O-F (symmetric).
    E[..., 0] = 1.0
    E[:, 0, 1] = 0.0
    E[:, 0, 1, 1] = 1.0  # C-O single
    E[:, 1, 0] = 0.0
    E[:, 1, 0, 1] = 1.0
    E[:, 1, 2] = 0.0
    E[:, 1, 2, 1] = 1.0  # O-F single
    E[:, 2, 1] = 0.0
    E[:, 2, 1, 1] = 1.0
    # Self-loops: stay at no-bond.
    y = torch.zeros(bs, 0)
    node_mask = torch.ones(bs, 3, dtype=torch.bool)
    return X, E, y, node_mask


# ------------------------------------------------------------------
# Sub-feature unit tests
# ------------------------------------------------------------------


def test_valency_feature_shape() -> None:
    """ValencyFeature returns (bs, n) with sum-of-bond-orders per node."""
    _, E, _, _ = _toy_qm9_graph()
    out = ValencyFeature()(E)
    assert out.shape == (1, 3)
    # C has one single bond → 1; O has two single bonds → 2; F has one single → 1.
    expected = torch.tensor([[1.0, 2.0, 1.0]])
    assert torch.allclose(out, expected)


def test_charge_feature_shape_and_value() -> None:
    """Match upstream charge formula literally.

    Upstream's ``ChargeFeature`` is *not* the chemical formal-charge
    formula despite the name. It returns ``argmax(X * valencies) -
    sum(argmax(weighted_E))`` per node. Since ``argmax`` of a one-hot
    times-positive-constants is just ``argmax(X)``, the first term
    reduces to the **atom class index**, not the valency value. We
    port the upstream behaviour bit-for-bit and assert against that;
    the semantic question of "is this what upstream meant?" is out of
    scope for the parity port.
    """
    X, E, _, _ = _toy_qm9_graph()
    # valencies = [4, 3, 2, 1] (C, N, O, F)
    # Per-node: C(class 0)→argmax=0, current=1, charge = 0-1 = -1
    #           O(class 2)→argmax=2, current=2, charge = 2-2 =  0
    #           F(class 3)→argmax=3, current=1, charge = 3-1 =  2
    out = ChargeFeature(valencies=[4, 3, 2, 1])(X, E)
    assert out.shape == (1, 3)
    expected = torch.tensor([[-1.0, 0.0, 2.0]])
    assert torch.allclose(out, expected)


def test_weight_feature_shape_and_normalisation() -> None:
    """WeightFeature returns (bs, 1) normalised by max_weight."""
    X, _, _, _ = _toy_qm9_graph()
    # Per-atom weights: C=12, N=14, O=16, F=19; max_weight=150.
    out = WeightFeature(atom_weights=[12, 14, 16, 19], max_weight=150)(X)
    assert out.shape == (1, 1)
    # Total = 12 + 16 + 19 = 47 → 47 / 150 ≈ 0.3133.
    expected = torch.tensor([[47.0 / 150.0]])
    assert torch.allclose(out, expected)


# ------------------------------------------------------------------
# Provider-level tests
# ------------------------------------------------------------------


def test_extra_molecular_features_adjust_dims() -> None:
    """``adjust_dims`` adds exactly (+2, 0, +1)."""
    feats = ExtraMolecularFeatures(
        valencies=[4, 3, 2, 1],
        atom_weights=[12, 14, 16, 19],
        max_weight=150,
    )
    out = feats.adjust_dims({"X": 4, "E": 5, "y": 0})
    assert out == {"X": 6, "E": 5, "y": 1}


def test_extra_molecular_features_call_widths() -> None:
    """``__call__`` returns tensors with widths (2, 0, 1)."""
    X, E, y, mask = _toy_qm9_graph()
    feats = ExtraMolecularFeatures(
        valencies=[4, 3, 2, 1],
        atom_weights=[12, 14, 16, 19],
        max_weight=150,
    )
    extra_X, extra_E, extra_y = feats(X, E, y, mask)
    assert extra_X.shape == (1, 3, 2)
    assert extra_E.shape == (1, 3, 3, 0)
    assert extra_y.shape == (1, 1)


def test_extra_molecular_features_validates_dim_mismatch() -> None:
    """valencies and atom_weights must have the same length."""
    with pytest.raises(ValueError, match="must have the same length"):
        ExtraMolecularFeatures(
            valencies=[4, 3, 2, 1], atom_weights=[12, 14, 16], max_weight=150
        )


# ------------------------------------------------------------------
# Composite provider tests
# ------------------------------------------------------------------


def test_composite_extra_features_chains_adjust_dims() -> None:
    """``adjust_dims`` should sum each child's delta."""
    structural = ExtraFeatures(extra_features_type="all", max_n_nodes=9)
    domain = ExtraMolecularFeatures(
        valencies=[4, 3, 2, 1],
        atom_weights=[12, 14, 16, 19],
        max_weight=150,
    )
    comp = CompositeExtraFeatures(providers=[structural, domain])
    # ExtraFeatures(all) adds (+6, 0, +11); ExtraMolecularFeatures adds (+2, 0, +1).
    # Total: (+8, 0, +12).
    out = comp.adjust_dims({"X": 4, "E": 5, "y": 0})
    assert out == {"X": 12, "E": 5, "y": 12}


def test_composite_extra_features_call_concatenates_outputs() -> None:
    """Output widths should equal sum of child widths."""
    X, E, y, mask = _toy_qm9_graph()
    structural = ExtraFeatures(extra_features_type="all", max_n_nodes=9)
    domain = ExtraMolecularFeatures(
        valencies=[4, 3, 2, 1],
        atom_weights=[12, 14, 16, 19],
        max_weight=150,
    )
    comp = CompositeExtraFeatures(providers=[structural, domain])

    extra_X, extra_E, extra_y = comp(X, E, y, mask)
    # Structural "all": x=6, e=0, y=11; domain: x=2, e=0, y=1. Totals: x=8, e=0, y=12.
    assert extra_X.shape == (1, 3, 8)
    assert extra_E.shape == (1, 3, 3, 0)
    assert extra_y.shape == (1, 12)


def test_composite_rejects_empty_providers() -> None:
    """Composite with no providers is a config error."""
    with pytest.raises(ValueError, match="non-empty"):
        CompositeExtraFeatures(providers=[])

"""Domain-specific (molecular) extra features for discrete diffusion.

Ports cvignac/DiGress's ``ExtraMolecularFeatures``
(``src/diffusion/extra_features_molecular.py``) into the
:class:`tmgg.models.digress.extra_features.ExtraFeaturesProvider`
protocol so it composes cleanly with our existing structural
``ExtraFeatures`` (cycles + Laplacian eigenfeatures) at the
``GraphTransformer.extra_features`` slot.

Adds three per-graph chemistry features computed from the noisy
``(X, E)`` state:

- **Charge** (per node, +1 dim on X): the formal-charge proxy
  ``normal_valency(atom) - current_valency_from_edges``. Captures
  whether a node's currently-incident bond order is consistent
  with its atom type's typical valency.
- **Valency** (per node, +1 dim on X): the raw current valency
  derived from the edge tensor (with aromatic bonds counted as 1.5).
- **Weight** (per graph, +1 dim on y): the sum of per-atom weights
  divided by ``max_weight`` so the value lives in roughly ``[0, 1]``.

Together: ``adjust_dims`` adds **(+2, 0, +1)** to ``(X, E, y)``.

This module also exports a generic :class:`CompositeExtraFeatures`
that stacks multiple providers — needed because our
``GraphTransformer`` accepts a single ``extra_features`` provider but
upstream DiGress passes both structural *and* domain features.

References
----------
- ``/tmp/digress_upstream/src/diffusion/extra_features_molecular.py``
  (upstream implementation we mirror).
- Per-dataset ``valencies`` / ``atom_weights`` / ``max_weight``
  constants live in upstream's
  ``src/datasets/{qm9,moses,guacamol}_dataset.py`` and are passed into
  our yamls via the ``extra_features`` block (one-source-of-truth).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from tmgg.models.digress.extra_features import ExtraFeaturesProvider

# Bond order weights for the standard 5-class molecular edge encoding:
# ``[no-bond, single, double, triple, aromatic]``. Aromatic is 1.5 so
# the per-node valency from the edge tensor matches RDKit's convention.
# Identical to upstream ``extra_features_molecular.py:27,43``.
_BOND_ORDERS = (0.0, 1.0, 2.0, 3.0, 1.5)


# ------------------------------------------------------------------
# Per-feature helpers
# ------------------------------------------------------------------


class ChargeFeature:
    """Per-node formal-charge proxy from atom type and incident edges.

    Computes ``normal_valency(atom) - current_valency_from_edges``,
    where the normal valency is read from the atom's one-hot type
    via the ``valencies`` table. A negative value means the node is
    over-bonded relative to its atom type; positive means
    under-bonded.

    Parameters
    ----------
    valencies
        Per-atom-type expected valency, indexed by the same column
        as the one-hot atom feature. For QM9 (no-H): ``[4, 3, 2, 1]``
        (C, N, O, F).
    """

    def __init__(self, valencies: Sequence[int]) -> None:
        self._valencies = tuple(valencies)

    def __call__(self, X: Tensor, E: Tensor) -> Tensor:
        """Return the per-node charge proxy of shape ``(bs, n)``."""
        bond_orders = torch.tensor(_BOND_ORDERS, device=E.device).reshape(1, 1, 1, -1)
        weighted_E = E * bond_orders  # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)  # (bs, n)

        valencies = torch.tensor(self._valencies, device=X.device).reshape(1, 1, -1)
        normal_valencies = (X * valencies).argmax(dim=-1)  # (bs, n)

        return (normal_valencies - current_valencies).type_as(X)


class ValencyFeature:
    """Per-node current valency from the noisy edge tensor.

    Returns the sum (per node) of bond orders over incident edges,
    using the standard 5-class encoding with aromatic = 1.5. Shape
    ``(bs, n)``.
    """

    def __call__(self, E: Tensor) -> Tensor:
        bond_orders = torch.tensor(_BOND_ORDERS, device=E.device).reshape(1, 1, 1, -1)
        weighted_E = E * bond_orders  # (bs, n, n, de)
        valencies = weighted_E.argmax(dim=-1).sum(dim=-1)  # (bs, n)
        return valencies.type_as(E)


class WeightFeature:
    """Per-graph (normalised) molecular weight.

    Sum of per-atom weights, divided by ``max_weight`` so the output
    lives in ``[0, ~1]``. Shape ``(bs, 1)``.

    Notes
    -----
    Matches upstream's behaviour exactly, including the absence of a
    node mask in the sum: padded positions of ``X`` *can* argmax to a
    real atom slot under noise, so this slightly over-counts mass on
    padded rows. Upstream has the same property, so we match for
    parity. If desired, mask in a future revision.
    """

    def __init__(self, atom_weights: Sequence[float], max_weight: float) -> None:
        self._atom_weights = tuple(float(w) for w in atom_weights)
        self._max_weight = float(max_weight)

    def __call__(self, X: Tensor) -> Tensor:
        atom_weights = torch.tensor(self._atom_weights, device=X.device)
        atom_idx = X.argmax(dim=-1)  # (bs, n)
        per_node_weight = atom_weights[atom_idx]  # (bs, n)
        return (per_node_weight.sum(dim=-1) / self._max_weight).unsqueeze(-1).type_as(X)


# ------------------------------------------------------------------
# Public provider
# ------------------------------------------------------------------


class ExtraMolecularFeatures:
    """Charge + valency + weight as an ``ExtraFeaturesProvider``.

    Adds **(+2, 0, +1)** to ``(X, E, y)``. Charge and valency ride
    on X; mol weight on y. Edge dim is unchanged.

    Parameters
    ----------
    valencies
        Per-atom-type expected valency. List indexed by the one-hot
        atom column; e.g. QM9 (no-H) is ``[4, 3, 2, 1]``.
    atom_weights
        Per-atom-type molecular weight. Same indexing as
        ``valencies``.
    max_weight
        Normaliser for the total per-graph weight (so the output is
        in ``[0, ~1]``). Set to a per-dataset upper bound; e.g. 150
        for QM9, 350 for MOSES, 1000 for GuacaMol.

    Notes
    -----
    All three constants are dataset-specific. Upstream hardcodes them
    in ``src/datasets/{qm9,moses,guacamol}_dataset.py``; we pass them
    through Hydra so the yaml is the single source of truth.
    """

    def __init__(
        self,
        valencies: Sequence[int],
        atom_weights: Sequence[float],
        max_weight: float,
    ) -> None:
        if len(valencies) != len(atom_weights):
            raise ValueError(
                "valencies and atom_weights must have the same length "
                f"(got {len(valencies)} vs {len(atom_weights)})."
            )
        self._charge = ChargeFeature(valencies=valencies)
        self._valency = ValencyFeature()
        self._weight = WeightFeature(atom_weights=atom_weights, max_weight=max_weight)

    def __call__(
        self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute (extra_X, extra_E, extra_y) of widths (2, 0, 1).

        Parameters
        ----------
        X
            One-hot atom features ``(bs, n, num_atom_types)``.
        E
            One-hot edge features ``(bs, n, n, 5)`` with the canonical
            bond-order encoding (no-bond, single, double, triple,
            aromatic).
        y
            Global features ``(bs, dy)`` — unused; reproduced for
            protocol conformance.
        node_mask
            ``(bs, n)`` bool. Unused for parity with upstream; see
            :class:`WeightFeature` notes.
        """
        del y, node_mask  # unused; kept for protocol conformance.
        charge = self._charge(X, E).unsqueeze(-1)  # (bs, n, 1)
        valency = self._valency(E).unsqueeze(-1)  # (bs, n, 1)
        weight = self._weight(X)  # (bs, 1)

        extra_X = torch.cat([charge, valency], dim=-1)  # (bs, n, 2)
        extra_E = torch.zeros((*E.shape[:-1], 0), device=E.device, dtype=E.dtype)
        extra_y = weight  # (bs, 1)
        return extra_X, extra_E, extra_y

    def adjust_dims(self, input_dims: dict[str, int]) -> dict[str, int]:
        """Add (+2, 0, +1) to (X, E, y)."""
        return {
            "X": input_dims["X"] + 2,
            "E": input_dims["E"] + 0,
            "y": input_dims["y"] + 1,
        }


# ------------------------------------------------------------------
# Composite provider
# ------------------------------------------------------------------


class CompositeExtraFeatures:
    """Stack multiple ``ExtraFeaturesProvider`` instances side-by-side.

    Concatenates each provider's output along the last axis of its
    respective tensor (X, E, y). ``adjust_dims`` chains the providers'
    own deltas. Use to combine structural features
    (:class:`ExtraFeatures`) with domain features
    (:class:`ExtraMolecularFeatures`) under our single-provider
    transformer interface.

    Parameters
    ----------
    providers
        Ordered list. Concatenation order in the output follows the
        list order, so e.g. cycle-x features come before
        charge/valency in the X dimension.

    Examples
    --------
    Hydra yaml::

        extra_features:
          _target_: tmgg.models.digress.extra_molecular_features.CompositeExtraFeatures
          providers:
            - _target_: tmgg.models.digress.extra_features.ExtraFeatures
              extra_features_type: all
              max_n_nodes: 9
            - _target_: tmgg.models.digress.extra_molecular_features.ExtraMolecularFeatures
              valencies: [4, 3, 2, 1]
              atom_weights: [12, 14, 16, 19]
              max_weight: 150
    """

    def __init__(self, providers: Sequence[ExtraFeaturesProvider]) -> None:
        if not providers:
            raise ValueError(
                "CompositeExtraFeatures: providers list must be non-empty."
            )
        self._providers: list[ExtraFeaturesProvider] = list(providers)

    def __call__(
        self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        x_outs: list[Tensor] = []
        e_outs: list[Tensor] = []
        y_outs: list[Tensor] = []
        for p in self._providers:
            xs, es, ys = p(X, E, y, node_mask)
            x_outs.append(xs)
            e_outs.append(es)
            y_outs.append(ys)
        return (
            torch.cat(x_outs, dim=-1),
            torch.cat(e_outs, dim=-1),
            torch.cat(y_outs, dim=-1),
        )

    def adjust_dims(self, input_dims: dict[str, int]) -> dict[str, int]:
        out = dict(input_dims)
        for p in self._providers:
            out = p.adjust_dims(out)
        return out

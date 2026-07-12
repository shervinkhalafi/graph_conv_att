"""Regression tests for the ``use_cycles`` flag on ``PEARLExtraFeatures``.

Three contracts:

1. ``use_cycles=True`` (default) preserves the historic shape contract:
   ``extra_X = (bs, n, 3 + pearl_dim)``, ``extra_y = (bs, 5)``. This is
   the pre-flag behaviour shipped before 2026-05-21 and must keep
   working byte-identically for backward compatibility with prior runs
   under the ``_pearl_repro`` configs.

2. ``use_cycles=False`` drops the 3 per-node cycle channels and the 4
   per-graph cycle-global channels: ``extra_X = (bs, n, pearl_dim)``,
   ``extra_y = (bs, 1)`` (only the normalised node count survives).
   ``ncycles`` is not allocated at construction time.

3. ``pearl_extra_features_dims`` and ``adjust_dims`` agree with the
   ``__call__`` output shapes for both flag values, so the model's
   input projection sizes correctly regardless of the flag.

The padded shapes from a dense ``GraphState`` are exercised: ``E[..., 0]``
is the "no edge" channel and ``E[..., 1]`` carries the binary "edge present"
signal, matching the SBM datamodule output post sparse-default refactor.
"""

from __future__ import annotations

import torch

from tmgg.models.digress.pearl_extra_features import (
    PEARLExtraFeatures,
    pearl_extra_features_dims,
)


def _make_dense_inputs(
    bs: int = 2, n: int = 8, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a small symmetric two-class edge tensor for shape checks."""
    g = torch.Generator().manual_seed(seed)
    X = torch.zeros(bs, n, 1)
    E = torch.zeros(bs, n, n, 2)
    edge_mask = (torch.rand(bs, n, n, generator=g) > 0.5).float()
    edge_mask = (edge_mask + edge_mask.transpose(-1, -2)).clamp_(max=1.0)
    # Zero the diagonal so cycle counts are non-degenerate.
    diag = torch.eye(n).bool().unsqueeze(0).expand(bs, -1, -1)
    edge_mask[diag] = 0.0
    E[..., 1] = edge_mask
    E[..., 0] = 1.0 - edge_mask
    y = torch.zeros(bs, 0)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    return X, E, y, node_mask


def test_default_use_cycles_true_preserves_historic_shapes() -> None:
    """Default constructor keeps the pre-flag 3-cycle + PEARL contract."""
    pearl_dim = 16
    extra = PEARLExtraFeatures(
        max_n_nodes=20,
        pearl_output_dim=pearl_dim,
        pearl_num_layers=2,
        pearl_mode="random",
        pearl_hidden_dim=16,
        pearl_input_samples=4,
    )
    X, E, y, node_mask = _make_dense_inputs(bs=2, n=8)
    extra_x, extra_e, extra_y = extra(X, E, y, node_mask)

    assert extra_x.shape == (2, 8, 3 + pearl_dim), (
        "default contract: 3 cycle channels prepended to k PEARL channels"
    )
    assert extra_e.shape == (2, 8, 8, 0)
    assert extra_y.shape == (2, 5), "default contract: 1 n + 4 cycle globals"


def test_use_cycles_false_drops_cycle_channels() -> None:
    """``use_cycles=False`` returns PEARL-only extras: no cycle channels."""
    pearl_dim = 16
    extra = PEARLExtraFeatures(
        max_n_nodes=20,
        pearl_output_dim=pearl_dim,
        pearl_num_layers=2,
        pearl_mode="random",
        pearl_hidden_dim=16,
        pearl_input_samples=4,
        use_cycles=False,
    )
    X, E, y, node_mask = _make_dense_inputs(bs=2, n=8)
    extra_x, extra_e, extra_y = extra(X, E, y, node_mask)

    assert extra_x.shape == (2, 8, pearl_dim), (
        "PEARL-only contract: pearl_dim node channels, no 3-cycle prefix"
    )
    assert extra_e.shape == (2, 8, 8, 0)
    assert extra_y.shape == (2, 1), (
        "PEARL-only contract: only normalised node count, no 4 cycle globals"
    )


def test_use_cycles_false_skips_ncycles_allocation() -> None:
    """``ncycles`` is None when cycles are disabled — prevents accidental use."""
    extra = PEARLExtraFeatures(
        max_n_nodes=20,
        pearl_output_dim=8,
        pearl_num_layers=2,
        pearl_mode="random",
        pearl_hidden_dim=16,
        pearl_input_samples=4,
        use_cycles=False,
    )
    assert extra.ncycles is None
    assert extra.use_cycles is False


def test_use_cycles_true_allocates_ncycles() -> None:
    """Default keeps the ``NodeCycleFeatures`` instance allocated."""
    extra = PEARLExtraFeatures(
        max_n_nodes=20,
        pearl_output_dim=8,
        pearl_num_layers=2,
        pearl_mode="random",
        pearl_hidden_dim=16,
        pearl_input_samples=4,
    )
    assert extra.ncycles is not None
    assert extra.use_cycles is True


def test_pearl_extra_features_dims_with_and_without_cycles() -> None:
    """Module-level dim helper agrees with ``__call__`` output widths."""
    # With cycles: 3 + k node channels, 0 edge, 1 + 4 global channels.
    assert pearl_extra_features_dims(16) == (3 + 16, 0, 1 + 4)
    assert pearl_extra_features_dims(16, use_cycles=True) == (3 + 16, 0, 5)
    # Without cycles: k node channels, 0 edge, 1 global channel.
    assert pearl_extra_features_dims(16, use_cycles=False) == (16, 0, 1)
    assert pearl_extra_features_dims(0, use_cycles=False) == (0, 0, 1)


def test_adjust_dims_reflects_use_cycles_flag() -> None:
    """``adjust_dims`` returns input widths bumped by the same widths."""
    input_dims = {"X": 1, "E": 2, "y": 0}

    with_cycles = PEARLExtraFeatures(
        max_n_nodes=20, pearl_output_dim=16, pearl_num_layers=2,
        pearl_mode="random", pearl_hidden_dim=16, pearl_input_samples=4,
    ).adjust_dims(input_dims)
    assert with_cycles == {"X": 1 + 3 + 16, "E": 2, "y": 0 + 5}

    no_cycles = PEARLExtraFeatures(
        max_n_nodes=20, pearl_output_dim=16, pearl_num_layers=2,
        pearl_mode="random", pearl_hidden_dim=16, pearl_input_samples=4,
        use_cycles=False,
    ).adjust_dims(input_dims)
    assert no_cycles == {"X": 1 + 16, "E": 2, "y": 0 + 1}

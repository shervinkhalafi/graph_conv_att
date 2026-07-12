"""Tests for DenseGraphState.mask() symmetry invariants.

Testing Strategy
----------------
``DenseGraphState.mask()`` must enforce edge-feature symmetry for
undirected graphs. The original DiGress baseline asserts this; TMGG
previously discarded the check result (P2.7 bug). These tests confirm
the restored assertion catches asymmetric inputs and passes symmetric
ones.

See also tests/models/test_graph_types.py for comprehensive coverage.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.data.datasets.graph_types import DenseGraphState, collapse_to_indices


def _state(
    bs: int,
    n: int,
    *,
    X: torch.Tensor,
    E: torch.Tensor,
    num_nodes_per_graph: torch.Tensor | None = None,
) -> DenseGraphState:
    if num_nodes_per_graph is None:
        num_nodes_per_graph = torch.full((bs,), n, dtype=torch.long)
    y = torch.zeros(bs, 0)
    return DenseGraphState(
        num_nodes_per_graph=num_nodes_per_graph,
        y=y,
        X_class=X,
        E_class=E,
    )


class TestDenseGraphStateMask:
    """Verify the symmetry assertion in DenseGraphState.mask().

    Starting state: DenseGraphState with batch of (X_class, E_class, y)
    tensors and node_mask derived from num_nodes_per_graph. The
    ``mask()`` method zeros out masked positions, asserts E_class
    symmetry, and returns a new frozen instance.
    """

    def test_symmetric_edges_pass(self) -> None:
        """Symmetric E after masking raises no error.

        Starting state: E is explicitly symmetric, every position kept.
        Invariant: mask() returns without assertion failure; result is a
        new instance (frozen dataclass).
        """
        bs, n, dx, de = 2, 4, 3, 2
        X = torch.randn(bs, n, dx)
        E_upper = torch.randn(bs, n, n, de)
        E = (E_upper + E_upper.transpose(1, 2)) / 2

        data = _state(bs, n, X=X, E=E)
        result = data.mask()
        assert result is not data

    def test_symmetric_edges_with_partial_mask(self) -> None:
        """Masking some nodes preserves symmetry when E starts symmetric.

        Starting state: symmetric E, last node padded out per batch.
        Invariant: element-wise masking (E * e_mask1 * e_mask2) preserves
        symmetry because the mask itself is symmetric in (i, j).
        """
        bs, n, dx, de = 2, 5, 3, 2
        X = torch.randn(bs, n, dx)
        E_upper = torch.randn(bs, n, n, de)
        E = (E_upper + E_upper.transpose(1, 2)) / 2
        # Drop the last position per batch by setting num_nodes = n - 1.
        num_nodes = torch.full((bs,), n - 1, dtype=torch.long)

        data = _state(bs, n, X=X, E=E, num_nodes_per_graph=num_nodes)
        result = data.mask()
        assert result is not data

    def test_asymmetric_edges_raise(self) -> None:
        """Asymmetric E triggers AssertionError after masking.

        Starting state: E is deliberately asymmetric.
        Invariant: the assertion catches the asymmetry and reports
        the magnitude of the discrepancy.
        """
        bs, n, dx, de = 1, 4, 3, 2
        X = torch.randn(bs, n, dx)
        E = torch.randn(bs, n, n, de)
        E[0, 0, 1, 0] = 10.0
        E[0, 1, 0, 0] = -10.0

        data = _state(bs, n, X=X, E=E)
        with pytest.raises(AssertionError, match="symmetric"):
            data.mask()

    def test_collapse_to_indices_skips_symmetry_check(self) -> None:
        """collapse_to_indices converts to class indices without symmetry assertion.

        Starting state: asymmetric E (would fail mask()).
        Invariant: collapse_to_indices produces integer indices via argmax
        without checking E symmetry.
        """
        bs, n, dx, de = 1, 4, 3, 2
        X = torch.randn(bs, n, dx)
        E = torch.randn(bs, n, n, de)
        E[0, 0, 1, 0] = 10.0
        E[0, 1, 0, 0] = -10.0

        data = _state(bs, n, X=X, E=E)
        E_idx, X_idx = collapse_to_indices(data)
        assert X_idx is not None
        assert X_idx.shape == (bs, n)
        assert E_idx.shape == (bs, n, n)

"""Parity tests: unconditional-pad vs conditional-pad eigenfeature gather.

Test rationale: ``get_eigenvalues_features`` and ``get_eigenvectors_features``
in ``tmgg.models.digress.extra_features`` were ported verbatim from upstream
DiGress, including a conditional ``if to_extend > 0: pad`` branch that reads
``max(n_connected_components)`` to host RAM via ``.item()`` — triggering a
CUDA host-device sync at every training step. The ports were rewritten to
unconditionally pad by ``k`` slots, which is always sufficient because
``max(n_connected_components) + k - n <= k`` (n_connected_components is
bounded above by n). These tests pin the bitwise equivalence of the two
formulations on inputs that exercise both branches of the conditional code.

Assumed starting state:
* ``get_eigenvalues_features(eigenvalues, k)`` returns
  ``(n_connected_components, first_k_ev)`` from sorted ascending eigenvalues.
* ``get_eigenvectors_features(vectors, node_mask, n_connected, k)`` returns
  ``(not_lcc_indicator, first_k_ev)`` from columns-as-eigenvectors and the
  per-graph component count.

Invariants pinned:
* For every batch where the conditional code path either does or does not
  pad (``to_extend > 0`` vs ``to_extend <= 0``), the unconditional pad
  produces bitwise-identical ``first_k_ev`` values.
* The connected-component count itself is unaffected (it's computed from
  raw eigenvalues before any padding).
"""

from __future__ import annotations

import pytest
import torch

from tmgg.models.digress.extra_features import (
    get_eigenvalues_features,
    get_eigenvectors_features,
)

# ---------------------------------------------------------------------------
# Reference implementations: literal upstream-DiGress conditional pad.
# These mirror the pre-refactor code so we can assert the new helpers'
# outputs are bitwise-identical.
# ---------------------------------------------------------------------------


def _ref_get_eigenvalues_features(
    eigenvalues: torch.Tensor, k: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conditional-pad reference. Identical to upstream DiGress's pre-refactor code."""
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)

    to_extend = int(n_connected_components.max().item()) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack(
            (eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues))
        )
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(
        0
    ) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def _ref_get_eigenvectors_features(
    vectors: torch.Tensor,
    node_mask: torch.Tensor,
    n_connected: torch.Tensor,
    k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conditional-pad reference for eigenvector gather only.

    The not-LCC-indicator computation contains a non-deterministic
    ``torch.randn`` seed for tie-breaking on padding positions; we pin
    the seed before each call to compare outputs deterministically.
    """
    bs, n = vectors.size(0), vectors.size(1)

    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values
    mask = ~(first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    to_extend = int(n_connected.max().item()) + k - n
    if to_extend > 0:
        vectors = torch.cat(
            (vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2
        )
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(
        0
    ) + n_connected.unsqueeze(2)
    indices = indices.expand(-1, n, -1)
    first_k_ev = torch.gather(vectors, dim=2, index=indices)
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


# ---------------------------------------------------------------------------
# Helper: synthesize a sorted-ascending eigenvalue batch with a controlled
# number of leading zeros per row, exercising both code paths.
# ---------------------------------------------------------------------------


def _make_eigenvalue_batch(
    bs: int,
    n: int,
    n_zeros_per_row: list[int],
    seed: int = 0,
) -> torch.Tensor:
    """Build a sorted ascending eigenvalue tensor (bs, n) with n_zeros_per_row[i] zeros at the start of row i."""
    assert len(n_zeros_per_row) == bs
    g = torch.Generator().manual_seed(seed)
    rows: list[torch.Tensor] = []
    for nz in n_zeros_per_row:
        zeros = torch.zeros(nz)
        # Non-zero eigenvalues, sorted ascending and bounded in (0, 2).
        # The threshold for "is zero" is 1e-5; lower bound 1e-3 keeps all
        # non-zero eigenvalues firmly above it.
        positives = torch.sort(1e-3 + torch.rand(n - nz, generator=g) * 1.5).values
        rows.append(torch.cat([zeros, positives], dim=0))
    return torch.stack(rows, dim=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bs,n,n_zeros_per_row,k",
    [
        # Case A — to_extend <= 0: graphs large enough that gather doesn't
        # cross the n boundary. Conditional code skips the pad; unconditional
        # code pads anyway. Should match.
        (4, 20, [1, 2, 3, 1], 5),
        (8, 50, [1, 1, 4, 2, 3, 1, 5, 2], 5),
        # Case B — to_extend > 0: at least one row has so many components
        # that gather indices push past n. Conditional code pads by exactly
        # the deficit; unconditional code over-pads by k. Outputs match.
        (4, 8, [4, 5, 6, 7], 5),  # row 3 needs n_components+k = 12 slots
        (4, 10, [8, 9, 10, 7], 5),  # extreme: row 2 has n=10 zeros
        # Case C — boundary: to_extend == 0 exactly.
        (3, 10, [5, 5, 5], 5),  # max_components(5) + k(5) - n(10) = 0
    ],
)
def test_get_eigenvalues_features_matches_conditional_reference(
    bs: int, n: int, n_zeros_per_row: list[int], k: int
) -> None:
    """Unconditional-pad output equals upstream conditional-pad output bitwise."""
    eigenvalues = _make_eigenvalue_batch(bs, n, n_zeros_per_row, seed=42)

    n_components_ref, first_k_ref = _ref_get_eigenvalues_features(eigenvalues, k=k)
    n_components_new, first_k_new = get_eigenvalues_features(eigenvalues, k=k)

    torch.testing.assert_close(n_components_ref, n_components_new, rtol=0, atol=0)
    torch.testing.assert_close(first_k_ref, first_k_new, rtol=0, atol=0)


@pytest.mark.parametrize(
    "bs,n,k",
    [
        (4, 20, 2),
        (4, 8, 2),
        (3, 10, 2),
    ],
)
def test_get_eigenvectors_features_matches_conditional_reference(
    bs: int, n: int, k: int
) -> None:
    """Unconditional-pad eigenvector gather equals conditional reference bitwise."""
    # Deterministic eigenvector matrices: column j is meant to be the
    # j-th eigenvector. We don't need them to be true eigenvectors of any
    # matrix — gather just reads columns, so the values are arbitrary as
    # long as both functions see identical inputs.
    torch.manual_seed(7)
    vectors = torch.randn(bs, n, n)

    # node_mask: roughly half the nodes valid in some rows
    node_mask = torch.zeros(bs, n, dtype=torch.bool)
    valid_counts = [n, n - 3, n // 2 + 1, max(k + 2, n // 3)][:bs]
    for i, vc in enumerate(valid_counts):
        node_mask[i, :vc] = True

    # Make n_connected exercise both code paths: include rows where
    # n_components + k > n (forces conditional pad) and rows where it
    # doesn't.
    n_connected = torch.tensor(
        [[min(2, n - k - 1)], [min(3, n - k)], [min(n - 1, n - k + 1)], [n - 1]][:bs],
        dtype=torch.long,
    )

    # Pin the torch RNG immediately before each call: the not-LCC indicator
    # uses torch.randn for tie-breaking on padding positions.
    torch.manual_seed(123)
    not_lcc_ref, first_k_ref = _ref_get_eigenvectors_features(
        vectors, node_mask, n_connected, k=k
    )

    torch.manual_seed(123)
    not_lcc_new, first_k_new = get_eigenvectors_features(
        vectors, node_mask, n_connected, k=k
    )

    torch.testing.assert_close(not_lcc_ref, not_lcc_new, rtol=0, atol=0)
    torch.testing.assert_close(first_k_ref, first_k_new, rtol=0, atol=0)


def test_get_eigenvalues_features_no_sync_in_implementation() -> None:
    """Canary: the new implementation must not call ``.item()`` or ``.tolist()``.

    Test rationale: the entire point of the unconditional-pad refactor was
    to remove the ``.item()``-driven host-device sync. AST-based check
    (rather than substring search) so docstring and comment mentions of
    ``.item()`` don't trigger false positives.
    """
    import ast
    import inspect

    from tmgg.models.digress import extra_features as ef

    SYNC_METHODS = {"item", "tolist", "cpu", "numpy"}

    def _find_sync_calls(func) -> list[str]:
        tree = ast.parse(inspect.getsource(func))
        offenders: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in SYNC_METHODS
            ):
                offenders.append(f"line {node.lineno}: .{node.func.attr}() call")
        return offenders

    eigvals_offenders = _find_sync_calls(ef.get_eigenvalues_features)
    assert not eigvals_offenders, (
        "get_eigenvalues_features contains host-device sync call(s): "
        f"{eigvals_offenders}. The unconditional-pad refactor must keep "
        "this function fully GPU-resident."
    )

    eigvec_offenders = _find_sync_calls(ef.get_eigenvectors_features)
    assert not eigvec_offenders, (
        "get_eigenvectors_features contains host-device sync call(s): "
        f"{eigvec_offenders}. The unconditional-pad refactor must keep "
        "this function fully GPU-resident."
    )

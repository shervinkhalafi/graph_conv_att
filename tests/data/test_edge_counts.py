"""Tests for the upstream-parity sparse edge / node count helpers.

Test rationale
--------------
:mod:`tmgg.data.utils.edge_counts` ports upstream DiGress's
``AbstractDatasetInfos.edge_counts`` (sparse PyG iteration) verbatim so
audits comparing the two codebases find the same algorithm in both.
The numerical claim, separately, is that the new sparse path produces
the same ``(K,)`` PMF as the old dense upper-triangle counter on
``GraphData.E_class`` it replaced. These tests pin both invariants on a
small SBM-shaped fixture so any regression shows up loudly.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from torch_geometric.utils import dense_to_sparse

from tmgg.data.utils.edge_counts import (
    count_edge_classes_sparse,
    count_node_classes_sparse,
)


def _dense_upper_tri_edge_counts(
    e_class: torch.Tensor,
    node_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Reference implementation: previous dense upper-triangle counter.

    Mirrors the ``CategoricalNoiseProcess.initialize_from_data``
    body before the parity-port: count one-hot E_class mass over the
    strict upper triangle restricted to real-node positions. Used here
    to check that the sparse port matches the dense path numerically.
    """
    if node_mask.dim() == 1:
        node_mask = node_mask.unsqueeze(0)
        e_class = e_class.unsqueeze(0)
    _, n = node_mask.shape
    upper_triangle = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=node_mask.device),
        diagonal=1,
    ).unsqueeze(0)
    valid_edges = node_mask.unsqueeze(1) & node_mask.unsqueeze(2) & upper_triangle
    counts = torch.zeros(num_classes, dtype=torch.float64)
    if valid_edges.any():
        counts += e_class[valid_edges].sum(dim=0).to(torch.float64)
    return counts


def test_count_edge_classes_sparse_two_node_one_edge() -> None:
    """2-node, 1-undirected-edge case mirrors upstream's directed-pair counting.

    PyG enumerates both directions, so num_edges=2 and all_pairs=2*1=2;
    no-edge count is 0. Counts are intentionally un-normalised so they
    can be summed across batches.
    """
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = Batch.from_data_list(
        cast(list[BaseData], [Data(edge_index=edge_index, num_nodes=2)])
    )
    counts = count_edge_classes_sparse(batch, num_edge_classes=2)
    torch.testing.assert_close(counts, torch.tensor([0.0, 2.0], dtype=torch.float64))


def test_count_edge_classes_sparse_no_edges_returns_only_non_edges() -> None:
    """3-node graph with no edges: 6 ordered non-edge pairs and zero edges."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = Batch.from_data_list(
        cast(list[BaseData], [Data(edge_index=edge_index, num_nodes=3)])
    )
    counts = count_edge_classes_sparse(batch, num_edge_classes=2)
    torch.testing.assert_close(counts, torch.tensor([6.0, 0.0], dtype=torch.float64))


def test_count_edge_classes_sparse_strips_self_loops() -> None:
    """Self-loops on the sparse edge_index must not contribute to counts."""
    # 2-node graph: one self-loop on node 0, no real edge between 0 and 1.
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    batch = Batch.from_data_list(
        cast(list[BaseData], [Data(edge_index=edge_index, num_nodes=2)])
    )
    counts = count_edge_classes_sparse(batch, num_edge_classes=2)
    # 2 ordered pairs, 0 real edges (self-loop stripped) → 2 non-edges.
    torch.testing.assert_close(counts, torch.tensor([2.0, 0.0], dtype=torch.float64))


def test_count_edge_classes_sparse_matches_dense_upper_tri_on_sbm_fixture() -> None:
    """Sparse counter agrees with the previous dense upper-triangle counter.

    Generates a tiny SBM-shaped batch, computes counts both ways, and
    checks they produce the same un-normalised ``(K,)`` vector. This is
    the parity invariant the D-3 port has to preserve.
    """
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    # Build three small graphs of varying density.
    adjs: list[torch.Tensor] = []
    for n, p in [(5, 0.5), (6, 0.3), (4, 0.7)]:
        upper = (torch.from_numpy(rng.random((n, n))) < p).float()
        upper = torch.triu(upper, diagonal=1)
        adj = upper + upper.T
        adjs.append(adj)

    pyg_data: list[Data] = []
    n_max = max(int(a.shape[0]) for a in adjs)
    e_class_list: list[torch.Tensor] = []
    node_mask_list: list[torch.Tensor] = []
    for adj in adjs:
        n = int(adj.shape[0])
        edge_index, _ = dense_to_sparse(adj)
        pyg_data.append(Data(edge_index=edge_index, num_nodes=n))

        # Dense E_class for the reference path: pad to n_max, build
        # one-hot [no-edge, edge] with diagonal zeroed (parity #4).
        padded = torch.zeros(n_max, n_max)
        padded[:n, :n] = adj
        e = torch.stack([1.0 - padded, padded], dim=-1)
        diag = torch.arange(n_max)
        e[diag, diag, :] = 0.0
        e_class_list.append(e)

        nm = torch.zeros(n_max, dtype=torch.bool)
        nm[:n] = True
        node_mask_list.append(nm)

    batch = Batch.from_data_list(cast(list[BaseData], pyg_data))
    sparse_counts = count_edge_classes_sparse(batch, num_edge_classes=2)

    dense_counts = torch.zeros(2, dtype=torch.float64)
    for e_class, nm in zip(e_class_list, node_mask_list, strict=True):
        dense_counts += _dense_upper_tri_edge_counts(
            e_class.unsqueeze(0), nm.unsqueeze(0), num_classes=2
        )
    # Dense counter sums one-hot mass over the strict upper triangle, so
    # its "edge" total equals the number of undirected edges. The sparse
    # counter sums over directed edges (PyG enumerates both directions),
    # so its edge count is doubled. The "no-edge" entry is doubled too
    # (ordered pairs = 2 * unordered pairs). Compare the normalised PMFs
    # — both should yield identical π_E.
    sparse_pmf = sparse_counts / sparse_counts.sum()
    dense_pmf = dense_counts / dense_counts.sum()
    torch.testing.assert_close(sparse_pmf, dense_pmf, atol=1e-12, rtol=0.0)


def test_count_node_classes_sparse_structure_only_fallback() -> None:
    """Without ``x`` features, all nodes count toward class 1."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = Batch.from_data_list(
        cast(
            list[BaseData],
            [
                Data(edge_index=edge_index, num_nodes=3),
                Data(edge_index=edge_index, num_nodes=2),
            ],
        )
    )
    counts = count_node_classes_sparse(batch, num_node_classes=2)
    torch.testing.assert_close(counts, torch.tensor([0.0, 5.0], dtype=torch.float64))


def test_count_node_classes_sparse_uses_x_when_present() -> None:
    """When ``x`` is present, counts come from summing the one-hot rows."""
    x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float,
    )
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = Batch.from_data_list([Data(x=x, edge_index=edge_index, num_nodes=3)])
    counts = count_node_classes_sparse(batch, num_node_classes=3)
    torch.testing.assert_close(
        counts, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    )


def test_count_node_classes_sparse_single_class_structure_only() -> None:
    """Abstract-graph datasets (SBM / Planar / Comm20 / Ego / Protein) have a
    single node class: every node is simply a node. The "class 0 = no node"
    convention from upstream DiGress's molecular datasets doesn't apply, so
    all structure-only node mass lands on class 0 (the only class).

    Regression test for the refusal that previously forced
    :class:`CategoricalNoiseProcess.initialize_from_data` callers to
    hand-roll the single-class stationary distribution. See
    ``an internal validation harness (not included)``'s
    pre-fix ``init_noise_process`` helper for the dead bypass.
    """
    edge_index = torch.empty((2, 0), dtype=torch.long)
    batch = Batch.from_data_list(
        cast(
            list[BaseData],
            [
                Data(edge_index=edge_index, num_nodes=3),
                Data(edge_index=edge_index, num_nodes=2),
            ],
        )
    )
    counts = count_node_classes_sparse(batch, num_node_classes=1)
    torch.testing.assert_close(counts, torch.tensor([5.0], dtype=torch.float64))


def test_count_edge_classes_sparse_single_class_no_edge_attr() -> None:
    """Single-class edge vocabularies: every node pair lands on class 0.

    A 1-class edge space has no "present vs absent" distinction. The only
    sensible count is ``all_pairs`` on the single class. Kept for symmetry
    with the single-class node path; unlikely to appear in production
    configs but well-defined if it does.
    """
    # Two nodes, one directed pair in each direction: all_pairs = 2.
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = Batch.from_data_list(
        cast(list[BaseData], [Data(edge_index=edge_index, num_nodes=2)])
    )
    counts = count_edge_classes_sparse(batch, num_edge_classes=1)
    torch.testing.assert_close(counts, torch.tensor([2.0], dtype=torch.float64))

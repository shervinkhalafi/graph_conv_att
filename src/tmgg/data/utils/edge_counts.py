"""Sparse edge / node class counters for empirical π estimation.

The two counters here are direct ports of upstream DiGress's
``AbstractDatasetInfos.edge_counts`` and ``AbstractDatasetInfos.node_types``
(``digress-upstream-readonly/src/datasets/abstract_dataset.py:34-72``).
The point of porting them rather than computing the same statistics on
our ``GraphState`` collator output is grep-friendly parity: auditors
comparing the two codebases see the same algorithm in both.

The hot training loop now consumes sparse ``GraphState`` directly.
These helpers are only used during datamodule setup, on the *raw* PyG
batches that sit one layer below the ``GraphDataCollator``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import remove_self_loops


def count_edge_classes_sparse(
    pyg_batch: Batch,
    num_edge_classes: int,
) -> Tensor:
    """Count one-hot edge-class occurrences across a PyG batch.

    Mirrors upstream DiGress's
    :meth:`AbstractDatasetInfos.edge_counts`
    (``digress-upstream-readonly/src/datasets/abstract_dataset.py:48-72``).
    For each per-graph node count ``c`` in the batch, the total number of
    ordered node pairs is ``c * (c - 1)``. The implicit "no-edge" count
    is the difference between the sum of those pair counts and the
    actual number of directed edges in ``edge_index``. Per-class edge
    counts come from summing ``edge_attr`` (one-hot) over the edge
    dimension and discarding the no-edge slot at index 0.

    Self-loops are stripped before counting, mirroring our collator
    path which removes them in :meth:`GraphState.from_pyg_batch` (parity
    fix #2 / D-1).

    Parameters
    ----------
    pyg_batch
        A PyG batch yielded by ``Batch.from_data_list``. May expose
        ``edge_attr`` of shape ``(num_edges, num_edge_classes)`` (one-hot
        per edge, with index 0 reserved for "no edge"). When absent, we
        assume the binary-edge convention used throughout the
        SBM/synthetic pipeline: every edge is class 1.
    num_edge_classes
        Number of edge classes including the implicit "no-edge" class at
        index 0.

    Returns
    -------
    Tensor
        Un-normalised counts, shape ``(num_edge_classes,)``, ``float64``.
    """
    edge_index_in: Tensor = getattr(pyg_batch, "edge_index")  # noqa: B009
    edge_index, _ = remove_self_loops(edge_index_in)
    num_edges = int(edge_index.shape[1])

    batch_vec: Tensor = getattr(pyg_batch, "batch")  # noqa: B009
    _, counts = torch.unique(batch_vec, return_counts=True)
    all_pairs = int((counts * (counts - 1)).sum().item())
    num_non_edges = all_pairs - num_edges
    if num_non_edges < 0:
        raise AssertionError(
            "count_edge_classes_sparse: implicit no-edge count went "
            f"negative (all_pairs={all_pairs}, num_edges={num_edges}). "
            "Either edge_index contains parallel edges or graphs in the "
            "batch have inconsistent node accounting."
        )

    edge_attr = getattr(pyg_batch, "edge_attr", None)
    if edge_attr is None:
        # Binary-edge convention for K>=2: class 0 = no edge, class 1 = edge.
        # Degenerate K==1 collapses both onto the single class, so the whole
        # all_pairs count lands there (no present-vs-absent distinction).
        if num_edge_classes < 1:
            raise ValueError(
                "count_edge_classes_sparse with no edge_attr requires "
                f"num_edge_classes>=1; got {num_edge_classes}."
            )
        per_class = torch.zeros(num_edge_classes, dtype=torch.float64)
        if num_edge_classes == 1:
            per_class[0] = float(all_pairs)
        else:
            per_class[1] = float(num_edges)
    else:
        # Shape tripwire: ``edge_attr`` must be a 2-D one-hot
        # ``(E, K)`` tensor. A 1-D ``(E,)`` of class indices would
        # crash ``edge_attr.sum(dim=0)`` to a scalar and silently
        # corrupt the per-class histogram. Surface clearly here.
        if edge_attr.dim() != 2:
            raise AssertionError(
                f"count_edge_classes_sparse expects ``edge_attr`` to be a "
                f"2-D one-hot ``(E, num_edge_classes)`` tensor, got shape "
                f"{tuple(edge_attr.shape)}. If your dataset emits 1-D "
                f"class indices, either omit ``edge_attr`` from the PyG "
                f"``Data`` (fallback branch handles binary-edge batches) "
                f"or convert to one-hot before batching."
            )
        if edge_attr.shape[0] != edge_index.shape[1]:
            # Self-loop edges were removed above; if the dataset attaches
            # edge_attr we need the dataset to provide compatible counts.
            # A future molecular wiring should re-emit edge_attr aligned
            # with the post-self-loop-removal edge_index.
            raise AssertionError(
                "count_edge_classes_sparse: edge_attr length "
                f"{edge_attr.shape[0]} disagrees with post-self-loop "
                f"edge_index length {edge_index.shape[1]}."
            )
        per_class = edge_attr.sum(dim=0).to(torch.float64)

    counts_out = torch.zeros(num_edge_classes, dtype=torch.float64)
    if num_edge_classes == 1:
        # K==1 has no "no-edge" slot; the per_class above already carries
        # the full all_pairs mass on class 0.
        counts_out[0] = per_class[0]
    else:
        counts_out[0] = float(num_non_edges)
        counts_out[1:] = per_class[1:]
    return counts_out


def count_node_classes_sparse(
    pyg_batch: Batch,
    num_node_classes: int,
) -> Tensor:
    """Count one-hot node-class occurrences across a PyG batch.

    Mirrors upstream DiGress's :meth:`AbstractDatasetInfos.node_types`
    (``digress-upstream-readonly/src/datasets/abstract_dataset.py:34-46``):
    sum the node-feature one-hots over the node axis. Datasets in the
    structure-only path do not carry per-node class features, so the
    helper falls back to the degenerate "every padded slot is a node"
    convention by counting nodes from the batch vector and placing the
    mass on class index 1 (matches ``_read_categorical_x`` in
    :mod:`tmgg.diffusion.noise_process`).

    Parameters
    ----------
    pyg_batch
        A PyG batch. May expose ``x`` of shape ``(num_nodes, K)`` with
        one-hot rows (with class 0 reserved for "no node"); when absent,
        the structure-only fallback above kicks in.
    num_node_classes
        Number of node classes including any implicit "no-node" class.

    Returns
    -------
    Tensor
        Un-normalised counts, shape ``(num_node_classes,)``, ``float64``.
    """
    x = getattr(pyg_batch, "x", None)
    counts_out = torch.zeros(num_node_classes, dtype=torch.float64)
    if x is None:
        if num_node_classes < 1:
            raise ValueError(
                "count_node_classes_sparse without per-node features requires "
                f"num_node_classes>=1; got {num_node_classes}."
            )
        batch_vec: Tensor = getattr(pyg_batch, "batch")  # noqa: B009
        num_nodes = int(batch_vec.numel())
        # For K>=2: upstream-parity convention puts every node on class 1
        # (class 0 is reserved for "no node" in molecular datasets).
        # For K==1: abstract graphs where every node is the sole class,
        # so all mass lands on class 0.
        target_class = 0 if num_node_classes == 1 else 1
        counts_out[target_class] = float(num_nodes)
        return counts_out
    # Shape tripwire: ``x`` must be a 2-D one-hot ``(N, K)``. A 1-D
    # ``(N,)`` would crash ``x.shape[1]`` with ``IndexError: tuple index
    # out of range`` — surface a clear actionable error instead. The
    # molecular ``MolecularGraphDataset`` raw-PyG path emits one-hot
    # explicitly via ``_graphdata_to_pyg_one_hot``; SBM/Planar omit
    # ``x`` entirely (handled by the fallback above). 1-D ``x`` would
    # mean a caller fed indices through this path by mistake.
    if x.dim() != 2:
        raise AssertionError(
            f"count_node_classes_sparse expects ``x`` to be a 2-D one-hot "
            f"``(N, num_node_classes)`` tensor, got shape {tuple(x.shape)}. "
            f"If your dataset emits 1-D class indices, either omit ``x`` "
            f"from the PyG ``Data`` (fallback branch handles structure-"
            f"only batches) or convert to one-hot before batching."
        )
    if x.shape[1] != num_node_classes:
        raise AssertionError(
            f"count_node_classes_sparse: per-node feature dimension "
            f"{x.shape[1]} disagrees with num_node_classes={num_node_classes}."
        )
    counts_out += x.sum(dim=0).to(torch.float64)
    return counts_out

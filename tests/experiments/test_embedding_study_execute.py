"""Tests for embedding-study execution helpers.

Test rationale
--------------
The embedding study is intentionally lightweight and does not go through the
training stack. These tests lock two contracts that are easy to drift during
cleanup: graph generation returns adjacency tensors, and the helper keeps the
output surface narrow and explicit.
"""

from __future__ import annotations

import torch

from tmgg.experiments.embedding_study.execute import _generate_graphs


def test_generate_graphs_returns_adjacency_tensors() -> None:
    """SBM generation should return a list of square float adjacency tensors."""
    graphs = _generate_graphs(
        dataset_name="sbm",
        num_graphs=3,
        num_nodes=12,
        seed=123,
    )

    assert isinstance(graphs, list)
    assert len(graphs) == 3

    for graph in graphs:
        assert isinstance(graph, torch.Tensor)
        assert graph.dtype == torch.float32
        assert graph.shape == (12, 12)

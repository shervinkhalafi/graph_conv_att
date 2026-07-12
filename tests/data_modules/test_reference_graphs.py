"""Tests for BaseGraphDataModule.get_reference_graphs().

Test Rationale
--------------
``get_reference_graphs()`` is the public API for extracting reference
graphs from dataset splits. Post the sparse-default refactor
(``internal notes (not included)``) it
returns ``list[DenseGraphState]`` — each entry a per-graph dense state
with a leading batch dim of 1 — so existing downstream consumers
indexing ``X_class[0]`` / ``E_class[0]`` continue to work. The nx view
is recovered at the consumption site via
:meth:`DenseGraphState.to_networkx`.

Tests verify the type contract, truncation to ``max_graphs``, error
handling for invalid stages, and the nx-conversion round-trip
preserves node count + undirected topology.

The fixture uses ``MultiGraphDataModule`` with SBM and fixed-size graphs
(``num_nodes=8``, ``num_graphs=20``) so node counts are deterministic
and the test dataset is small enough to be fast.
"""

from __future__ import annotations

import networkx as nx
import pytest

from tmgg.data.data_modules.multigraph_data_module import MultiGraphDataModule
from tmgg.data.datasets.graph_types import DenseGraphState


@pytest.fixture
def data_module() -> MultiGraphDataModule:
    """Small SBM data module for testing."""
    dm = MultiGraphDataModule(
        graph_type="sbm",
        num_graphs=20,
        num_nodes=8,
        batch_size=4,
        num_workers=0,
    )
    dm.setup("fit")
    return dm


class TestGetReferenceGraphs:
    """Verify get_reference_graphs() returns per-graph DenseGraphState."""

    def test_val_returns_graph_data(self, data_module: MultiGraphDataModule) -> None:
        graphs = data_module.get_reference_graphs("val", max_graphs=5)
        assert len(graphs) > 0
        assert all(isinstance(g, DenseGraphState) for g in graphs)

    def test_test_returns_graph_data(self, data_module: MultiGraphDataModule) -> None:
        graphs = data_module.get_reference_graphs("test", max_graphs=5)
        assert len(graphs) > 0
        assert all(isinstance(g, DenseGraphState) for g in graphs)

    def test_max_graphs_caps_output(self, data_module: MultiGraphDataModule) -> None:
        graphs = data_module.get_reference_graphs("val", max_graphs=3)
        assert len(graphs) <= 3

    def test_max_graphs_larger_than_dataset(
        self, data_module: MultiGraphDataModule
    ) -> None:
        graphs = data_module.get_reference_graphs("val", max_graphs=10000)
        # Should return all available, not crash
        assert len(graphs) > 0

    def test_invalid_stage_raises(self, data_module: MultiGraphDataModule) -> None:
        with pytest.raises(ValueError, match="stage must be"):
            data_module.get_reference_graphs("train", max_graphs=5)

    def test_graphs_have_correct_node_count(
        self, data_module: MultiGraphDataModule
    ) -> None:
        graphs = data_module.get_reference_graphs("val", max_graphs=5)
        for gd in graphs:
            # Each per-graph slice carries a leading bs dim of 1; the
            # derived ``node_mask`` is shape (1, n_max). SBM with
            # num_nodes=8 yields 8 valid nodes per graph (fixed-size).
            assert int(gd.node_mask.sum().item()) == 8
            assert gd.to_networkx(batch_index=0).number_of_nodes() == 8

    def test_to_networkx_round_trip_preserves_undirected(
        self, data_module: MultiGraphDataModule
    ) -> None:
        graphs = data_module.get_reference_graphs("val", max_graphs=5)
        for gd in graphs:
            g_nx = gd.to_networkx(batch_index=0)
            assert isinstance(g_nx, nx.Graph)
            assert not g_nx.is_directed()

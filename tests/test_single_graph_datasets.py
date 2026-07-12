"""Unit tests for SingleGraphDataModule with various graph types.

Tests verify that each supported graph type:
1. Generates valid symmetric binary adjacency matrices
2. Properly implements same_graph_all_splits protocol
3. Creates functional dataloaders with correct batch shapes

Test Rationale
--------------
The single-graph training protocol is central to stage 1 and stage 1.5 experiments.
Each graph type has different generation logic and potential edge cases:
- Synthetic graphs (ER, regular, tree): Use numpy-based generators
- NetworkX graphs (ring_of_cliques, LFR): External library with specific constraints
- PyG graphs: External datasets requiring download and format conversion

These tests ensure the SingleGraphDataModule abstraction works uniformly across
all graph types, catching integration issues early.
"""

from typing import Any, cast

import networkx as nx
import numpy as np
import pytest
import torch

from tmgg.data.data_modules.single_graph_data_module import (
    SingleGraphDataModule,
)
from tmgg.data.datasets.graph_types import GraphData, GraphState


def _graphdata_to_numpy(batch: GraphData, index: int = 0) -> np.ndarray:
    """Extract one graph from a sparse-default batch.

    Rationale
    ---------
    The cleaned datamodule contract exposes split access through
    dataloaders and ``get_reference_graphs()``. Tests reconstruct dense
    adjacency matrices from those boundaries instead of relying on the
    deleted graph accessors. Post sparse-default refactor the datamodule
    yields ``GraphState``; ``dense_adjacency()`` is defined on every
    concrete carrier and reads ``num_nodes_per_graph`` for per-graph
    sizing without going through ``node_mask`` (which sparse types do
    not carry).
    """
    num_nodes = int(batch.num_nodes_per_graph[index].item())
    adj = batch.dense_adjacency()[index, :num_nodes, :num_nodes]
    return adj.cpu().numpy()


def _first_train_graph(dm: SingleGraphDataModule) -> np.ndarray:
    """Return the first training graph from the train dataloader."""
    return _graphdata_to_numpy(next(iter(dm.train_dataloader())))


def _first_reference_graph(dm: SingleGraphDataModule, stage: str) -> np.ndarray:
    """Return one validation or test graph through the public reference API.

    Per the 2026-05-01 universal-transport refactor get_reference_graphs
    returns list[GraphData]; convert at the leaf via to_networkx() to
    keep the downstream nx.to_numpy_array assertion intact. Each list
    entry is itself a batched dense carrier with B==1, so we explicitly
    select ``batch_index=0`` for the conversion.
    """
    gd = dm.get_reference_graphs(stage, max_graphs=1)[0]
    return np.asarray(nx.to_numpy_array(gd.to_networkx(0)), dtype=np.float32)


class TestSyntheticGraphs:
    """Test synthetic graph types that don't require external downloads."""

    @pytest.mark.parametrize(
        "graph_type,graph_config",
        [
            ("sbm", {"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3}),
            ("erdos_renyi", {"p": 0.1}),
            ("regular", {"d": 3}),
            ("tree", {}),
        ],
    )
    def test_synthetic_graph_generation(self, graph_type: str, graph_config: dict):
        """Test that synthetic graphs generate valid adjacency matrices."""
        dm = SingleGraphDataModule(
            graph_type=graph_type,
            num_nodes=50,
            graph_config=graph_config,
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
        )
        dm.setup()

        A = _first_train_graph(dm)

        # Shape check
        assert A.shape == (50, 50), f"Expected (50, 50), got {A.shape}"

        # Symmetry check
        assert np.allclose(A, A.T), "Adjacency matrix should be symmetric"

        # Binary check
        unique_vals = set(np.unique(A))
        assert unique_vals.issubset({0.0, 1.0}), f"Expected binary, got {unique_vals}"

        # No self-loops
        assert np.allclose(np.diag(A), 0), "Diagonal should be zero (no self-loops)"

    @pytest.mark.parametrize(
        "graph_type,graph_config",
        [
            ("sbm", {"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3}),
            ("erdos_renyi", {"p": 0.1}),
            ("regular", {"d": 3}),
            ("tree", {}),
        ],
    )
    def test_same_graph_all_splits(self, graph_type: str, graph_config: dict):
        """Test that same_graph_all_splits=True uses identical graphs."""
        dm = SingleGraphDataModule(
            graph_type=graph_type,
            num_nodes=50,
            graph_config=graph_config,
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
        )
        dm.setup()

        train_graph = _first_train_graph(dm)
        val_graph = _first_reference_graph(dm, "val")
        test_graph = _first_reference_graph(dm, "test")

        assert np.array_equal(
            train_graph, val_graph
        ), "Train and val should be identical"
        assert np.array_equal(
            train_graph, test_graph
        ), "Train and test should be identical"

    @pytest.mark.parametrize(
        "graph_type,graph_config",
        [
            ("sbm", {"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3}),
            ("erdos_renyi", {"p": 0.1}),
        ],
    )
    def test_different_graphs_when_disabled(self, graph_type: str, graph_config: dict):
        """Test that same_graph_all_splits=False uses different graphs."""
        dm = SingleGraphDataModule(
            graph_type=graph_type,
            num_nodes=50,
            graph_config=graph_config,
            same_graph_all_splits=False,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
        )
        dm.setup()

        train_graph = _first_train_graph(dm)
        val_graph = _first_reference_graph(dm, "val")
        test_graph = _first_reference_graph(dm, "test")

        # Graphs should differ (with high probability for random graphs)
        assert not np.array_equal(train_graph, val_graph), "Train and val should differ"
        assert not np.array_equal(
            train_graph, test_graph
        ), "Train and test should differ"

    @pytest.mark.parametrize(
        "graph_type,graph_config",
        [
            ("sbm", {"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3}),
            ("erdos_renyi", {"p": 0.1}),
            ("regular", {"d": 3}),
            ("tree", {}),
        ],
    )
    def test_dataloader_batch_shape(self, graph_type: str, graph_config: dict):
        """Test that dataloaders return correctly shaped batches."""
        batch_size = 4
        dm = SingleGraphDataModule(
            graph_type=graph_type,
            num_nodes=50,
            graph_config=graph_config,
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=batch_size,
        )
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        assert isinstance(batch, GraphData)
        adj = batch.dense_adjacency()
        assert adj.shape == (
            batch_size,
            50,
            50,
        ), f"Expected ({batch_size}, 50, 50), got {adj.shape}"
        assert adj.dtype == torch.float32, f"Expected float32, got {adj.dtype}"


class TestNetworkXGraphs:
    """Test NetworkX-based graph types."""

    def test_ring_of_cliques_generation(self):
        """Test ring of cliques graph generation."""
        dm = SingleGraphDataModule(
            graph_type="ring_of_cliques",
            num_nodes=20,  # Ignored for this type
            graph_config={"num_cliques": 4, "clique_size": 5},
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
        )
        dm.setup()

        A = _first_train_graph(dm)

        # Total nodes = num_cliques * clique_size = 20
        assert A.shape == (20, 20), f"Expected (20, 20), got {A.shape}"
        assert np.allclose(A, A.T), "Should be symmetric"
        assert set(np.unique(A)).issubset({0.0, 1.0}), "Should be binary"

    def test_lfr_generation(self):
        """Test LFR benchmark graph generation."""
        dm = SingleGraphDataModule(
            graph_type="lfr",
            num_nodes=200,  # LFR needs large n for community constraints
            graph_config={
                "tau1": 2.0,
                "tau2": 1.1,
                "mu": 0.1,
                "average_degree": 8,
                "min_community": 15,
            },
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
        )
        dm.setup()

        A = _first_train_graph(dm)

        assert A.shape == (200, 200), f"Expected (200, 200), got {A.shape}"
        assert np.allclose(A, A.T), "Should be symmetric"
        assert set(np.unique(A)).issubset({0.0, 1.0}), "Should be binary"

    def test_ring_of_cliques_same_graph_protocol(self):
        """Test same graph protocol for ring of cliques."""
        dm = SingleGraphDataModule(
            graph_type="ring_of_cliques",
            num_nodes=20,
            graph_config={"num_cliques": 4, "clique_size": 5},
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
        )
        dm.setup()

        train_graph = _first_train_graph(dm)
        assert np.array_equal(train_graph, _first_reference_graph(dm, "val"))
        assert np.array_equal(train_graph, _first_reference_graph(dm, "test"))


class TestPyGGraphs:
    """Test PyTorch Geometric dataset loading.

    These tests require torch-geometric to be installed and may download
    datasets on first run. Marked as slow to allow skipping in fast test runs.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "graph_type",
        [
            "pyg_enzymes",
            "pyg_proteins",
            # "pyg_qm9",  # QM9 is very large, skip by default
        ],
    )
    def test_pyg_graph_loading(self, graph_type: str):
        """Test that PyG datasets load and convert to valid adjacency matrices."""
        pytest.importorskip("torch_geometric", reason="torch-geometric not installed")

        dm = SingleGraphDataModule(
            graph_type=graph_type,
            num_nodes=0,  # Ignored for PyG
            graph_config={"graph_idx": 0},
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
        )
        dm.setup()

        A = _first_train_graph(dm)

        # Shape should be square
        assert A.shape[0] == A.shape[1], f"Expected square matrix, got {A.shape}"

        # Symmetry
        assert np.allclose(A, A.T), "Should be symmetric"

        # Binary
        assert set(np.unique(A)).issubset({0.0, 1.0}), "Should be binary"

    @pytest.mark.slow
    def test_pyg_same_graph_protocol(self):
        """Test same graph protocol for PyG datasets."""
        pytest.importorskip("torch_geometric", reason="torch-geometric not installed")

        dm = SingleGraphDataModule(
            graph_type="pyg_enzymes",
            num_nodes=0,
            graph_config={"graph_idx": 0},
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
        )
        dm.setup()

        train_graph = _first_train_graph(dm)
        assert np.array_equal(train_graph, _first_reference_graph(dm, "val"))
        assert np.array_equal(train_graph, _first_reference_graph(dm, "test"))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_rejects_legacy_noise_levels_argument(self) -> None:
        """Legacy ``noise_levels`` should fail at construction."""
        legacy_kwargs = cast(Any, {"noise_levels": [0.1]})
        with pytest.raises(TypeError, match="noise_levels"):
            SingleGraphDataModule(
                graph_type="sbm",
                num_nodes=50,
                graph_config={"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3},
                **legacy_kwargs,
            )

    def test_rejects_legacy_noise_type_argument(self) -> None:
        """Legacy ``noise_type`` should fail at construction."""
        legacy_kwargs = cast(Any, {"noise_type": "digress"})
        with pytest.raises(TypeError, match="noise_type"):
            SingleGraphDataModule(
                graph_type="sbm",
                num_nodes=50,
                graph_config={"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3},
                **legacy_kwargs,
            )

    def test_unknown_graph_type_raises(self):
        """Test that unknown graph types raise ValueError."""
        dm = SingleGraphDataModule(
            graph_type="unknown_type",
            num_nodes=50,
            same_graph_all_splits=True,
        )
        with pytest.raises(ValueError, match="unknown_type"):
            dm.setup()

    def test_setup_required_before_access(self):
        """Dataloaders and reference-graph extraction require setup()."""
        dm = SingleGraphDataModule(
            graph_type="sbm",
            num_nodes=50,
            same_graph_all_splits=True,
        )
        with pytest.raises(RuntimeError, match="Call setup"):
            dm.train_dataloader()
        with pytest.raises(RuntimeError, match="Call setup"):
            dm.get_reference_graphs("val", max_graphs=1)

    def test_reproducibility_with_seed(self):
        """Test that the same seed produces identical graphs."""
        graph_config = {"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3}

        dm1 = SingleGraphDataModule(
            graph_type="sbm",
            num_nodes=50,
            graph_config=graph_config,
            train_seed=42,
            same_graph_all_splits=True,
        )
        dm1.setup()

        dm2 = SingleGraphDataModule(
            graph_type="sbm",
            num_nodes=50,
            graph_config=graph_config,
            train_seed=42,
            same_graph_all_splits=True,
        )
        dm2.setup()

        assert np.array_equal(_first_train_graph(dm1), _first_train_graph(dm2))

    def test_different_seeds_produce_different_graphs(self):
        """Test that different seeds produce different graphs."""
        graph_config = {"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3}

        dm1 = SingleGraphDataModule(
            graph_type="sbm",
            num_nodes=50,
            graph_config=graph_config,
            train_seed=42,
            same_graph_all_splits=True,
        )
        dm1.setup()

        dm2 = SingleGraphDataModule(
            graph_type="sbm",
            num_nodes=50,
            graph_config=graph_config,
            train_seed=123,
            same_graph_all_splits=True,
        )
        dm2.setup()

        assert not np.array_equal(_first_train_graph(dm1), _first_train_graph(dm2))

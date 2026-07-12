"""Tests for synthetic graph generators."""

import numpy as np
import pytest
from torch.utils.data import Dataset

from tmgg.data.datasets.synthetic_graphs import (
    SyntheticGraphDataset,
    generate_circular_ladder_graphs,
    generate_configuration_model_graphs,
    generate_erdos_renyi_graphs,
    generate_lollipop_graphs,
    generate_random_geometric_graphs,
    generate_regular_graphs,
    generate_square_grid_graphs,
    generate_star_graphs,
    generate_tree_graphs,
    generate_triangle_grid_graphs,
    generate_watts_strogatz_graphs,
)


class TestWattsStrogatzGraphs:
    """Tests for Watts-Strogatz small-world graph generator."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        n, num_graphs = 20, 5
        graphs = generate_watts_strogatz_graphs(n, num_graphs)
        assert graphs.shape == (num_graphs, n, n)

    def test_symmetry(self):
        """Test that graphs are symmetric (undirected)."""
        graphs = generate_watts_strogatz_graphs(15, 3)
        for A in graphs:
            assert np.allclose(A, A.T)

    def test_binary_edges(self):
        """Test that adjacency matrices are binary."""
        graphs = generate_watts_strogatz_graphs(15, 3)
        assert np.all((graphs == 0) | (graphs == 1))

    def test_k_parameter(self):
        """Test that k parameter affects connectivity."""
        n = 20
        graphs_k2 = generate_watts_strogatz_graphs(n, 5, k=2, p=0.0)
        graphs_k4 = generate_watts_strogatz_graphs(n, 5, k=4, p=0.0)

        # Higher k means more edges (at p=0, each node has exactly k neighbors)
        edges_k2 = graphs_k2.sum() / 2 / 5  # avg edges per graph
        edges_k4 = graphs_k4.sum() / 2 / 5
        assert edges_k4 > edges_k2

    def test_p_parameter_extremes(self):
        """Test rewiring probability at extremes."""
        n = 20
        # p=0: ring lattice (highly structured)
        graphs_p0 = generate_watts_strogatz_graphs(n, 3, k=4, p=0.0, seed=42)
        # p=1: random-like (less structured)
        graphs_p1 = generate_watts_strogatz_graphs(n, 3, k=4, p=1.0, seed=42)

        # Both should have same number of edges but different structure
        assert graphs_p0.shape == graphs_p1.shape

    def test_reproducibility(self):
        """Test that seed produces reproducible results."""
        g1 = generate_watts_strogatz_graphs(15, 3, seed=123)
        g2 = generate_watts_strogatz_graphs(15, 3, seed=123)
        assert np.allclose(g1, g2)

    def test_invalid_k_raises(self):
        """Test that invalid k values raise errors."""
        with pytest.raises(ValueError, match="k=.*must be less than"):
            generate_watts_strogatz_graphs(10, 3, k=10)

        with pytest.raises(ValueError, match="k must be even"):
            generate_watts_strogatz_graphs(10, 3, k=3)


class TestRandomGeometricGraphs:
    """Tests for random geometric graph generator."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        n, num_graphs = 20, 5
        graphs = generate_random_geometric_graphs(n, num_graphs)
        assert graphs.shape == (num_graphs, n, n)

    def test_symmetry(self):
        """Test that graphs are symmetric (undirected)."""
        graphs = generate_random_geometric_graphs(15, 3)
        for A in graphs:
            assert np.allclose(A, A.T)

    def test_binary_edges(self):
        """Test that adjacency matrices are binary."""
        graphs = generate_random_geometric_graphs(15, 3)
        assert np.all((graphs == 0) | (graphs == 1))

    def test_radius_parameter(self):
        """Test that radius affects edge density."""
        n = 30
        graphs_small_r = generate_random_geometric_graphs(n, 5, radius=0.2, seed=42)
        graphs_large_r = generate_random_geometric_graphs(n, 5, radius=0.5, seed=42)

        # Larger radius means more edges
        edges_small = graphs_small_r.sum()
        edges_large = graphs_large_r.sum()
        assert edges_large > edges_small

    def test_reproducibility(self):
        """Test that seed produces reproducible results."""
        g1 = generate_random_geometric_graphs(15, 3, seed=123)
        g2 = generate_random_geometric_graphs(15, 3, seed=123)
        assert np.allclose(g1, g2)


class TestConfigurationModelGraphs:
    """Tests for configuration model graph generator."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        n, num_graphs = 20, 5
        graphs = generate_configuration_model_graphs(n, num_graphs)
        assert graphs.shape == (num_graphs, n, n)

    def test_symmetry(self):
        """Test that graphs are symmetric (undirected)."""
        graphs = generate_configuration_model_graphs(15, 3)
        for A in graphs:
            assert np.allclose(A, A.T)

    def test_no_self_loops(self):
        """Test that graphs have no self-loops."""
        graphs = generate_configuration_model_graphs(15, 3)
        for A in graphs:
            assert np.allclose(np.diag(A), 0)

    def test_custom_degree_sequence(self):
        """Test with custom degree sequence."""
        n = 10
        # Regular degree sequence (all nodes degree 3)
        deg_seq = [3] * n
        graphs = generate_configuration_model_graphs(n, 3, degree_sequence=deg_seq)

        assert graphs.shape == (3, n, n)
        # Degrees should be close to 3 (may be slightly lower due to simplification)
        for A in graphs:
            degrees = A.sum(axis=1)
            assert np.all(degrees <= 3)

    def test_reproducibility(self):
        """Test that seed produces reproducible results."""
        g1 = generate_configuration_model_graphs(15, 3, seed=123)
        g2 = generate_configuration_model_graphs(15, 3, seed=123)
        assert np.allclose(g1, g2)


class TestSyntheticGraphDataset:
    """Tests for SyntheticGraphDataset class."""

    def test_dataset_inherits_torch_dataset_protocol(self) -> None:
        """SyntheticGraphDataset should be a real torch Dataset.

        Rationale
        ---------
        The denoising data wrappers expose ``__len__`` and ``__getitem__`` and
        are passed through generic torch data tooling. Inheriting the Dataset
        base class makes that contract explicit instead of relying on
        duck-typing.
        """
        dataset = SyntheticGraphDataset("ws", num_nodes=10, num_graphs=3)

        assert issubclass(SyntheticGraphDataset, Dataset)
        assert isinstance(dataset, Dataset)

    def test_watts_strogatz_type(self):
        """Test Watts-Strogatz graph generation via dataset."""
        dataset = SyntheticGraphDataset("watts_strogatz", num_nodes=15, num_graphs=5)
        assert len(dataset) == 5
        assert dataset[0].shape == (15, 15)
        assert dataset.graph_type == "watts_strogatz"

    def test_ws_alias(self):
        """Test ws alias for Watts-Strogatz."""
        dataset = SyntheticGraphDataset("ws", num_nodes=15, num_graphs=5, k=4, p=0.2)
        assert dataset.graph_type == "watts_strogatz"
        assert len(dataset) == 5

    def test_random_geometric_type(self):
        """Test random geometric graph generation via dataset."""
        dataset = SyntheticGraphDataset("random_geometric", num_nodes=20, num_graphs=5)
        assert len(dataset) == 5
        assert dataset[0].shape == (20, 20)
        assert dataset.graph_type == "random_geometric"

    def test_rg_alias(self):
        """Test rg alias for random geometric."""
        dataset = SyntheticGraphDataset("rg", num_nodes=15, num_graphs=5, radius=0.4)
        assert dataset.graph_type == "random_geometric"

    def test_configuration_model_type(self):
        """Test configuration model graph generation via dataset."""
        dataset = SyntheticGraphDataset(
            "configuration_model", num_nodes=15, num_graphs=5
        )
        assert len(dataset) == 5
        assert dataset[0].shape == (15, 15)
        assert dataset.graph_type == "configuration_model"

    def test_cm_alias(self):
        """Test cm alias for configuration model."""
        dataset = SyntheticGraphDataset("cm", num_nodes=15, num_graphs=5)
        assert dataset.graph_type == "configuration_model"

    def test_er_alias(self):
        """Test er alias for Erdős-Rényi."""
        dataset = SyntheticGraphDataset("er", num_nodes=15, num_graphs=5, p=0.3)
        assert dataset.graph_type == "erdos_renyi"

    def test_invalid_type_raises(self):
        """Test that invalid graph type raises error."""
        with pytest.raises(ValueError, match="graph_type must be one of"):
            SyntheticGraphDataset("invalid_type", num_nodes=10, num_graphs=5)

    def test_to_torch(self):
        """Test conversion to PyTorch tensor."""
        import torch

        dataset = SyntheticGraphDataset("ws", num_nodes=10, num_graphs=3)
        tensor = dataset.to_torch()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 10, 10)
        assert tensor.dtype == torch.float32

    def test_split_indices_on_adjacencies(self):
        """Test splitting dataset adjacencies via split_indices.

        Rationale
        ---------
        The split helper now lives with the datamodule generation helpers that
        own its only in-repo callers. This test keeps the direct helper
        contract covered at the new import path.
        """
        from tmgg.data.data_modules._split import split_indices

        dataset = SyntheticGraphDataset("rg", num_nodes=10, num_graphs=100)
        train_idx, val_idx, test_idx = split_indices(
            dataset.num_graphs, train_ratio=0.7, val_ratio=0.15, seed=42
        )

        train = dataset.adjacencies[train_idx]
        val = dataset.adjacencies[val_idx]
        test = dataset.adjacencies[test_idx]

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert train.shape[1:] == (10, 10)


class TestPyGDatasetWrapper:
    """Tests for the PyTorch Geometric dataset wrapper surface."""

    def test_wrapper_inherits_torch_dataset_protocol(self) -> None:
        """PyGDatasetWrapper should advertise the Dataset protocol explicitly.

        Rationale
        ---------
        This check is intentionally structural only. It avoids dataset
        downloads while still locking the wrapper onto the same Dataset
        surface as the synthetic graph wrapper.
        """
        pytest.importorskip("torch_geometric")

        from tmgg.data.datasets.pyg_datasets import PyGDatasetWrapper

        assert issubclass(PyGDatasetWrapper, Dataset)


class TestExistingGenerators:
    """Regression tests for existing generators to ensure they still work."""

    def test_regular_graphs(self):
        """Test regular graph generator."""
        graphs = generate_regular_graphs(n=20, d=3, num_graphs=5, seed=42)
        assert graphs.shape == (5, 20, 20)
        # Check all nodes have degree 3
        for A in graphs:
            degrees = A.sum(axis=1)
            assert np.allclose(degrees, 3)

    def test_erdos_renyi_graphs(self):
        """Test Erdős-Rényi graph generator."""
        graphs = generate_erdos_renyi_graphs(n=20, p=0.3, num_graphs=5, seed=42)
        assert graphs.shape == (5, 20, 20)

    def test_tree_graphs(self):
        """Test tree graph generator."""
        graphs = generate_tree_graphs(n=15, num_graphs=5, seed=42)
        assert graphs.shape == (5, 15, 15)
        # Check each graph is a tree (n-1 edges)
        for A in graphs:
            edges = A.sum() / 2
            assert edges == 14  # n-1 edges


class TestLollipopGraphs:
    """Tests for lollipop graph generator.

    A lollipop graph is a complete graph (cluster) connected to a path.
    Total nodes = cluster_size + path_length.
    """

    def test_output_shape(self):
        """Lollipop(5, 3) has 8 nodes; all copies are identical."""
        graphs = generate_lollipop_graphs(cluster_size=5, path_length=3, num_graphs=4)
        assert graphs.shape == (4, 8, 8)

    def test_symmetry(self):
        graphs = generate_lollipop_graphs(4, 2, 3)
        for A in graphs:
            assert np.allclose(A, A.T)

    def test_all_copies_identical(self):
        """Deterministic topology means every graph in the batch is the same."""
        graphs = generate_lollipop_graphs(4, 3, 5)
        for i in range(1, 5):
            assert np.allclose(graphs[0], graphs[i])


class TestCircularLadderGraphs:
    """Tests for circular ladder graph generator.

    circular_ladder_graph(n) produces 2n nodes.
    """

    def test_output_shape(self):
        graphs = generate_circular_ladder_graphs(n=6, num_graphs=3)
        assert graphs.shape == (3, 12, 12)  # 2*6 = 12 nodes

    def test_symmetry(self):
        graphs = generate_circular_ladder_graphs(5, 2)
        for A in graphs:
            assert np.allclose(A, A.T)

    def test_degree_three(self):
        """Every node in a circular ladder has degree 3."""
        graphs = generate_circular_ladder_graphs(8, 2)
        for A in graphs:
            degrees = A.sum(axis=1)
            assert np.allclose(degrees, 3)


class TestStarGraphs:
    """Tests for star graph generator.

    star_graph(n) produces n+1 nodes (1 hub + n leaves).
    """

    def test_output_shape(self):
        graphs = generate_star_graphs(n=7, num_graphs=3)
        assert graphs.shape == (3, 8, 8)  # 7+1 = 8 nodes

    def test_symmetry(self):
        graphs = generate_star_graphs(5, 2)
        for A in graphs:
            assert np.allclose(A, A.T)

    def test_hub_degree(self):
        """Hub node has degree n, leaves have degree 1."""
        n = 10
        graphs = generate_star_graphs(n, 2)
        for A in graphs:
            degrees = A.sum(axis=1)
            assert np.max(degrees) == n
            assert np.sum(degrees == 1) == n


class TestSquareGridGraphs:
    """Tests for square grid graph generator.

    Grid dimensions are chosen so h*w >= n. Actual node count may
    differ from requested n.
    """

    def test_node_count_at_least_n(self):
        n = 20
        graphs = generate_square_grid_graphs(n, 3)
        actual_n = graphs.shape[-1]
        assert actual_n >= n

    def test_symmetry(self):
        graphs = generate_square_grid_graphs(16, 2)
        for A in graphs:
            assert np.allclose(A, A.T)

    def test_binary_edges(self):
        graphs = generate_square_grid_graphs(25, 2)
        assert np.all((graphs == 0) | (graphs == 1))


class TestTriangleGridGraphs:
    """Tests for triangular lattice graph generator.

    Lattice dimension is chosen so the graph has at least n nodes.
    """

    def test_node_count_at_least_n(self):
        n = 20
        graphs = generate_triangle_grid_graphs(n, 3)
        actual_n = graphs.shape[-1]
        assert actual_n >= n

    def test_symmetry(self):
        graphs = generate_triangle_grid_graphs(10, 2)
        for A in graphs:
            assert np.allclose(A, A.T)


class TestNewTypesViaSyntheticGraphDataset:
    """Test the 5 new types through SyntheticGraphDataset dispatch."""

    def test_lollipop_type(self):
        ds = SyntheticGraphDataset(
            "lollipop", num_nodes=10, num_graphs=3, cluster_size=5, path_length=5
        )
        assert ds.graph_type == "lollipop"
        assert len(ds) == 3
        assert ds[0].shape == (10, 10)

    def test_circular_ladder_type(self):
        ds = SyntheticGraphDataset("circular_ladder", num_nodes=6, num_graphs=3)
        assert ds.graph_type == "circular_ladder"
        # circular_ladder_graph(6) has 12 nodes
        assert ds[0].shape == (12, 12)

    def test_star_type(self):
        ds = SyntheticGraphDataset("star", num_nodes=7, num_graphs=3)
        assert ds.graph_type == "star"
        # star_graph(7) has 8 nodes
        assert ds[0].shape == (8, 8)

    def test_square_grid_type(self):
        ds = SyntheticGraphDataset("square_grid", num_nodes=16, num_graphs=3)
        assert ds.graph_type == "square_grid"
        assert ds[0].shape[0] >= 16

    def test_triangle_grid_type(self):
        ds = SyntheticGraphDataset("triangle_grid", num_nodes=10, num_graphs=3)
        assert ds.graph_type == "triangle_grid"
        assert ds[0].shape[0] >= 10


class TestVariableSizeSupport:
    """Test node_counts, max_n, get_node_counts(), and get_masks()."""

    def test_fixed_size_node_counts(self):
        """Fixed-size types report uniform node counts equal to n."""
        ds = SyntheticGraphDataset("er", num_nodes=15, num_graphs=5, p=0.3)
        counts = ds.get_node_counts()
        assert counts.shape == (5,)
        assert np.all(counts == 15)
        assert ds.max_n == 15

    def test_masks_all_true_for_fixed_size(self):
        """Fixed-size types have all-True masks (no padding)."""
        ds = SyntheticGraphDataset("regular", num_nodes=10, num_graphs=3, d=3)
        masks = ds.get_masks()
        assert masks.shape == (3, 10)
        assert masks.all()

    def test_deterministic_type_node_counts(self):
        """Deterministic types derive actual_n from the generated graph."""
        ds = SyntheticGraphDataset("star", num_nodes=7, num_graphs=3)
        # star_graph(7) → 8 nodes
        assert ds.max_n == 8
        assert np.all(ds.get_node_counts() == 8)

    def test_grid_type_node_counts(self):
        """Grid types may have actual_n > requested n."""
        ds = SyntheticGraphDataset("square_grid", num_nodes=20, num_graphs=2)
        assert ds.max_n >= 20
        counts = ds.get_node_counts()
        assert np.all(counts == ds.max_n)

    def test_masks_shape_and_dtype(self):
        ds = SyntheticGraphDataset("circular_ladder", num_nodes=5, num_graphs=4)
        masks = ds.get_masks()
        assert masks.shape == (4, ds.max_n)
        assert masks.dtype == bool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

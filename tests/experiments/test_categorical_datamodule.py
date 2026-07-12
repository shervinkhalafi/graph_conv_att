"""Tests for SyntheticCategoricalDataModule.

Verifies that synthetic graph generation, splitting, collation, marginals
computation, round-trip fidelity, and size distribution interface all
behave correctly.

Internal storage is ``list[Data]`` (PyG sparse COO format). DataLoaders
now collate into sparse ``GraphState`` batches per the 2026-05-07
sparse-default refactor; per-graph reference extraction returns
``DenseGraphState`` (universal evaluation transport).
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from tmgg.data.data_modules.synthetic_categorical import (
    SyntheticCategoricalDataModule,
)
from tmgg.data.datasets.graph_types import DenseGraphState, GraphState
from tmgg.utils.noising.size_distribution import SizeDistribution

# -- Shared fixtures -------------------------------------------------------


@pytest.fixture()
def small_dm() -> SyntheticCategoricalDataModule:
    """A small SBM data module with 100 graphs of 20 nodes.

    Uses default SBM parameters (2 blocks, p_in=0.7, p_out=0.1) and an
    80/10/10 split, producing 80 train, 10 val, 10 test graphs.
    """
    dm = SyntheticCategoricalDataModule(
        graph_type="sbm",
        num_nodes=20,
        num_graphs=100,
        train_ratio=0.8,
        val_ratio=0.1,
        batch_size=8,
        seed=123,
    )
    dm.setup()
    return dm


@pytest.fixture()
def er_dm() -> SyntheticCategoricalDataModule:
    """An Erdos-Renyi data module to test non-SBM graph types."""
    dm = SyntheticCategoricalDataModule(
        graph_type="er",
        num_nodes=15,
        num_graphs=50,
        train_ratio=0.8,
        val_ratio=0.1,
        batch_size=4,
        seed=99,
        graph_config={"p": 0.3},
    )
    dm.setup()
    return dm


# -- TestSetup --------------------------------------------------------------


class TestSetup:
    """Verify dataset generation produces correct split sizes and shapes."""

    def test_split_sizes(self, small_dm: SyntheticCategoricalDataModule) -> None:
        """The 80/10/10 split of 100 graphs should yield 80, 10, 10."""
        assert small_dm._train_data is not None
        assert small_dm._val_data is not None
        assert small_dm._test_data is not None
        assert len(small_dm._train_data) == 80
        assert len(small_dm._val_data) == 10
        assert len(small_dm._test_data) == 10

    def test_pyg_data_structure(self, small_dm: SyntheticCategoricalDataModule) -> None:
        """Each stored element must be a PyG Data with edge_index and num_nodes.

        Internal storage uses sparse COO format; dense GraphData is only
        constructed at collation time by the DataLoader.
        """
        assert small_dm._train_data is not None
        d = small_dm._train_data[0]
        n = small_dm.num_nodes

        assert isinstance(d, Data)
        assert d.num_nodes == n
        assert d.edge_index is not None
        assert d.edge_index.shape[0] == 2  # COO format: (2, num_edges)

    def test_idempotent_setup(self, small_dm: SyntheticCategoricalDataModule) -> None:
        """Calling setup() a second time should be a no-op (same data objects)."""
        train_before = small_dm._train_data
        small_dm.setup()
        assert small_dm._train_data is train_before

    def test_no_overlap_between_splits(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Train, val, and test splits should contain no duplicate graphs.

        We compare by hashing the sorted edge_index tensor for each graph.
        """
        assert small_dm._train_data is not None
        assert small_dm._val_data is not None
        assert small_dm._test_data is not None

        def _hash_graph(d: Data) -> int:
            assert d.edge_index is not None
            return hash(d.edge_index.numpy().tobytes())

        train_hashes = {_hash_graph(d) for d in small_dm._train_data}
        val_hashes = {_hash_graph(d) for d in small_dm._val_data}
        test_hashes = {_hash_graph(d) for d in small_dm._test_data}

        assert train_hashes.isdisjoint(val_hashes)
        assert train_hashes.isdisjoint(test_hashes)
        assert val_hashes.isdisjoint(test_hashes)

    def test_er_generation(self, er_dm: SyntheticCategoricalDataModule) -> None:
        """Non-SBM types (Erdos-Renyi) should also produce valid data."""
        assert er_dm._train_data is not None
        assert len(er_dm._train_data) == 40  # 0.8 * 50


# -- TestDataloaders --------------------------------------------------------


class TestDataloaders:
    """Verify that DataLoaders yield properly collated sparse ``GraphState`` batches."""

    def test_train_batch_shape(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """A training batch should be a structure-only ``GraphState`` with correct shapes.

        Sparse-default refactor: ``x_class`` is ``None`` (structure-only);
        ``edge_class`` has shape ``(sum_E, 2)`` and ``num_nodes_per_graph``
        carries the per-graph node count.
        """
        batch: GraphState = next(iter(small_dm.train_dataloader()))
        assert batch.x_class is None
        assert batch.edge_class is not None
        bs = int(batch.num_nodes_per_graph.shape[0])

        assert isinstance(batch, GraphState)
        # Every graph in this fixture has the same fixed size.
        assert int(batch.num_nodes_per_graph[0].item()) == small_dm.num_nodes
        assert batch.edge_class.shape[-1] == 2
        assert batch.y.shape == (bs, 0)

    def test_val_batch_no_error(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Validation dataloader should produce at least one batch."""
        batch = next(iter(small_dm.val_dataloader()))
        assert isinstance(batch, GraphState)

    def test_test_batch_no_error(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Test dataloader should produce at least one batch."""
        batch = next(iter(small_dm.test_dataloader()))
        assert isinstance(batch, GraphState)

    def test_collated_feature_dims(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Collated output should have de=2 edge classes (x_class=None for structure-only)."""
        batch: GraphState = next(iter(small_dm.train_dataloader()))
        assert batch.x_class is None
        assert batch.edge_class is not None
        assert batch.edge_class.shape[-1] == 2  # de


# -- TestCategoricalBatches -------------------------------------------------


class TestCategoricalBatches:
    """Verify the categorical batch contract exposed by the datamodule."""

    def test_edge_features_have_expected_class_dim(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Edge features should expose the binary categorical basis."""
        batch: GraphState = next(iter(small_dm.train_dataloader()))
        assert batch.edge_class is not None
        assert batch.edge_class.shape[-1] == 2

    def test_node_features_are_absent_for_structure_only_datasets(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Structure-only synthetic datasets carry ``x_class=None``.

        The spec forbids datasets emitting a degenerate node-present /
        node-absent one-hot since it just re-encodes node presence; any
        architecture needing a per-node feature synthesises it
        internally.
        """
        batch: GraphState = next(iter(small_dm.train_dataloader()))
        assert batch.x_class is None

    def test_edge_features_are_one_hot_on_real_edges(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Per-edge categorical features should sum to one along the class axis.

        ``GraphState.edge_class`` carries one row per directed edge in
        ``edge_index``; every such row must be a one-hot. Sparse layouts
        do not store diagonal / inactive-pair entries, so the diagonal-
        is-zero invariant is enforced by construction (no self-loops in
        ``edge_index``).
        """
        batch: GraphState = next(iter(small_dm.train_dataloader()))
        assert batch.edge_class is not None
        edge_sums = batch.edge_class.sum(dim=-1)
        assert torch.allclose(edge_sums, torch.ones_like(edge_sums))

        # Self-loop guard: ``GraphState.__post_init__`` rejects self-loops
        # outright (see ``_check_sparse_invariants``). Asserting on
        # ``edge_index`` here is belt-and-braces.
        assert (batch.edge_index[0] != batch.edge_index[1]).all()

    def test_node_presence_tracked_by_num_nodes_per_graph(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """``num_nodes_per_graph`` is the single source of truth for node presence.

        Sparse-default refactor: ``x_class`` is ``None`` for
        structure-only datasets and the per-graph node count lives on
        ``num_nodes_per_graph``. Every synthesised graph has at least
        one real node.
        """
        batch: GraphState = next(iter(small_dm.train_dataloader()))
        assert batch.x_class is None
        assert (batch.num_nodes_per_graph > 0).all()


# -- TestReferenceGraphs ----------------------------------------------------


class TestReferenceGraphs:
    """Verify the val/test graph extraction boundary."""

    def test_val_returns_networkx_graphs(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Reference extraction returns per-graph ``DenseGraphState`` carriers.

        Per ``internal notes (not included)``
        the universal evaluation transport format is dense per-graph
        ``DenseGraphState``; the base datamodule densifies the sparse
        loader output before slicing into single-graph batches.
        """
        graphs = small_dm.get_reference_graphs("val", max_graphs=3)
        assert len(graphs) == 3
        assert all(isinstance(gd, DenseGraphState) for gd in graphs)
        assert all(int(gd.node_mask.sum().item()) == 20 for gd in graphs)


# -- TestRoundTrip ----------------------------------------------------------


class TestRoundTrip:
    """Verify adjacency -> PyG Data -> dense GraphData -> adjacency roundtrip.

    The invariant: a binary adjacency matrix stored as sparse COO in a
    PyG ``Data`` object, then collated into a dense ``GraphData`` batch,
    should recover the original adjacency via ``to_binary_adjacency()``.
    """

    def test_single_graph_roundtrip(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Convert a single stored Data to dense and verify the adjacency
        round-trip through ``dense_adjacency()``.

        Uses ``DenseGraphState.from_pyg_batch`` for the dense fast path
        (the sparse-default ``GraphData`` abstract base no longer carries
        a ``from_pyg_batch`` factory; the dense and sparse concrete types
        each have their own).
        """
        from typing import cast

        from torch_geometric.data import Batch
        from torch_geometric.data.data import BaseData

        assert small_dm._train_data is not None
        d = small_dm._train_data[0]

        # Convert single Data to dense via batch of size 1
        batch = Batch.from_data_list(cast(list[BaseData], [d]))
        g = DenseGraphState.from_pyg_batch(batch)

        recovered_adj = g.dense_adjacency()

        # Build original adjacency from the one-hot E
        assert g.E_class is not None
        original_adj = g.E_class[..., 1]  # class 1 = edge

        assert torch.allclose(recovered_adj, original_adj)

    def test_batch_roundtrip(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Densified train batch should roundtrip ``E_class[...,1]`` through
        ``dense_adjacency()`` at every real (b, i, j) position.

        The dataloader emits sparse ``GraphState`` batches now; we
        densify locally for the dense-adjacency assertion.
        """
        sparse: GraphState = next(iter(small_dm.train_dataloader()))
        from tmgg.data.datasets.graph_types import state_to_dense_sample

        batch = state_to_dense_sample(sparse)
        recovered = batch.dense_adjacency()

        assert batch.E_class is not None
        expected = batch.E_class[..., 1]
        mask_2d = batch.node_mask.unsqueeze(-1) & batch.node_mask.unsqueeze(-2)
        assert torch.allclose(recovered * mask_2d.float(), expected * mask_2d.float())


# -- TestSizeDistribution ---------------------------------------------------


class TestSizeDistribution:
    """Verify the size distribution interface on SyntheticCategoricalDataModule.

    For fixed-size synthetic data, ``get_size_distribution`` returns a
    degenerate distribution and sampling through that distribution
    returns a constant tensor. These tests confirm both the datamodule
    integration and the contract that the generative pipeline relies on.
    """

    def test_default_is_degenerate(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Fixed-size data should produce a degenerate distribution."""
        dist = small_dm.get_size_distribution()
        assert dist.is_degenerate
        assert dist.sizes == (20,)
        assert dist.max_size == 20

    def test_train_split(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Training split should also be degenerate for fixed-size data."""
        dist = small_dm.get_size_distribution("train")
        assert dist.is_degenerate
        assert dist.sizes == (20,)
        # All 80 training graphs have 20 nodes
        assert dist.counts == (80,)

    def test_val_split(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        dist = small_dm.get_size_distribution("val")
        assert dist.is_degenerate
        assert dist.counts == (10,)

    def test_test_split(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        dist = small_dm.get_size_distribution("test")
        assert dist.is_degenerate
        assert dist.counts == (10,)

    def test_invalid_split_raises(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            small_dm.get_size_distribution("predict")

    def test_before_setup_returns_fixed(self) -> None:
        """Before setup, should fall back to SizeDistribution.fixed(num_nodes)."""
        dm = SyntheticCategoricalDataModule(num_graphs=10, num_nodes=15)
        dist = dm.get_size_distribution("train")
        assert dist == SizeDistribution.fixed(15)

    def test_distribution_sample_shape(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Sampling via SizeDistribution should return a 1-D tensor."""
        result = small_dm.get_size_distribution("train").sample(4)
        assert result.shape == (4,)
        assert result.dtype == torch.long

    def test_distribution_sample_all_equal(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """For fixed-size data, sampled sizes should all equal num_nodes."""
        result = small_dm.get_size_distribution("train").sample(16)
        assert (result == 20).all()

    def test_serialization_roundtrip(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """The distribution should survive a to_dict/from_dict round-trip."""
        dist = small_dm.get_size_distribution("train")
        restored = SizeDistribution.from_dict(dist.to_dict())
        assert restored == dist

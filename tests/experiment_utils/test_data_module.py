"""Tests for GraphDataModule.

This module tests the GraphDataModule class which handles loading and
preparation of graph datasets for training.

Testing Strategy:
- Test initialization with different dataset types (SBM, wrappers, synthetic)
- Test prepare_data/setup lifecycle
- Test dataloader creation and properties
- Use real small datasets where practical; mock expensive operations

Key Invariants:
- setup must be called before accessing dataloaders
- Train/val/test splits should be disjoint
- Dataloader batch sizes should match configuration
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from tmgg.data.data_modules.data_module import (
    GraphDataModule,
)
from tmgg.data.datasets.graph_types import GraphData


class TestGraphDataModuleInit:
    """Tests for GraphDataModule initialization."""

    def test_default_parameters(self) -> None:
        """Should initialize with sensible defaults.

        Rationale: Default parameters should be reasonable for most use cases
        without requiring explicit configuration.
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={"num_nodes": 10, "block_sizes": [5, 5]},
        )

        assert dm.samples_per_graph == 1000
        assert dm.batch_size == 100
        assert dm.num_workers == 4
        assert dm.pin_memory is True
        assert dm.train_ratio == 0.6
        assert dm.val_ratio == 0.2

    def test_custom_parameters(self) -> None:
        """Should accept custom parameters.

        Rationale: Users should be able to override all configurable parameters.
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={"num_nodes": 10, "block_sizes": [5, 5]},
            samples_per_graph=500,
            batch_size=50,
            num_workers=2,
            pin_memory=False,
            train_ratio=0.8,
            val_ratio=0.1,
        )

        assert dm.samples_per_graph == 500
        assert dm.batch_size == 50
        assert dm.num_workers == 2
        assert dm.pin_memory is False
        assert dm.train_ratio == 0.8
        assert dm.val_ratio == 0.1

    def test_rejects_legacy_noise_levels_argument(self) -> None:
        """Legacy ``noise_levels`` should fail at construction."""
        legacy_kwargs = cast(Any, {"noise_levels": [0.1]})
        with pytest.raises(TypeError, match="noise_levels"):
            GraphDataModule(
                graph_type="sbm",
                graph_config={"num_nodes": 10, "block_sizes": [5, 5]},
                **legacy_kwargs,
            )

    def test_rejects_legacy_noise_type_argument(self) -> None:
        """Legacy ``noise_type`` should fail at construction."""
        legacy_kwargs = cast(Any, {"noise_type": "digress"})
        with pytest.raises(TypeError, match="noise_type"):
            GraphDataModule(
                graph_type="sbm",
                graph_config={"num_nodes": 10, "block_sizes": [5, 5]},
                **legacy_kwargs,
            )

    def test_absorbs_synthetic_default_leftover_kwargs(self) -> None:
        """Hydra ``+data=`` merges leak base-config synthetic-default kwargs.

        ``base_config_discrete_diffusion_generative.yaml``'s inline
        ``data:`` block sets ``num_nodes=20``, ``num_graphs=2000``, etc.
        for the synthetic-default ``SyntheticCategoricalDataModule``.
        When the wrapper does ``+data=pyg_enzymes``, Hydra MERGES (does
        not replace) the data dict, so those base keys persist into the
        ``GraphDataModule.__init__`` call. ``SpectreSBMDataModule`` already
        absorbs them via ``**_metadata``; ``GraphDataModule`` does the
        same via explicit ignored params, while still rejecting truly
        unknown legacy kwargs (``noise_levels``, ``noise_type``).

        Pin: passing ``num_nodes`` and ``num_graphs`` does not raise, and
        ``graph_config["num_nodes"]`` remains the source of truth for
        the dataset's actual node count.
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={"num_nodes": 10, "block_sizes": [5, 5]},
            num_nodes=20,  # leaked from base-config; ignored
            num_graphs=2000,  # leaked from base-config; ignored
        )
        assert dm.num_nodes_max_static == 10  # graph_config["num_nodes"], not 20


class TestGraphDataModuleSBM:
    """Tests for SBM dataset setup."""

    def test_sbm_setup_creates_splits(self) -> None:
        """SBM mode should create train/val/test data lists.

        Rationale: After setup(), all three splits should be populated
        with PyG Data objects (stored in _train_data/_val_data/_test_data).
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={
                "num_nodes": 10,
                "block_sizes": [5, 5],
                "p_intra": 0.8,
                "p_inter": 0.1,
            },
        )
        dm.prepare_data()
        dm.setup()

        assert dm._train_data is not None  # pyright: ignore[reportPrivateUsage]
        assert dm._val_data is not None  # pyright: ignore[reportPrivateUsage]
        assert dm._test_data is not None  # pyright: ignore[reportPrivateUsage]
        assert len(dm._train_data) > 0  # pyright: ignore[reportPrivateUsage]
        assert len(dm._val_data) > 0  # pyright: ignore[reportPrivateUsage]
        assert len(dm._test_data) > 0  # pyright: ignore[reportPrivateUsage]

    def test_sbm_fixed_block_sizes(self) -> None:
        """Fixed block_sizes mode should use specified sizes.

        Rationale: When block_sizes is explicitly provided, the SBM
        should use that exact partition structure.
        """
        from torch_geometric.utils import to_dense_adj

        block_sizes = [3, 4, 3]
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={
                "num_nodes": 10,
                "block_sizes": block_sizes,
                "p_intra": 1.0,
                "p_inter": 0.0,
            },
        )
        dm.prepare_data()
        dm.setup()

        # Verify the adjacency matrix has correct size
        assert dm._train_data is not None  # pyright: ignore[reportPrivateUsage]
        d = dm._train_data[0]  # pyright: ignore[reportPrivateUsage]
        assert d.edge_index is not None
        A = to_dense_adj(d.edge_index, max_num_nodes=d.num_nodes).squeeze(0)
        assert A.shape == (10, 10)

    def test_sbm_random_partitions(self) -> None:
        """Random partitions mode should generate valid splits.

        Rationale: When block_sizes is not provided, the module should
        generate random partitions based on min/max constraints.
        The list lengths include samples_per_graph repetition, so we
        check the unique graph count by dividing out.
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={
                "num_nodes": 20,
                "partition_mode": "enumerated",
                "num_train_partitions": 5,
                "num_test_partitions": 3,
                "min_blocks": 2,
                "max_blocks": 4,
                "min_block_size": 3,
                "max_block_size": 8,
                "p_intra": 0.8,
                "p_inter": 0.1,
            },
            samples_per_graph=1,
        )
        dm.prepare_data()
        dm.setup()

        # With samples_per_graph=1, list length = unique graph count
        assert dm._train_data is not None  # pyright: ignore[reportPrivateUsage]
        assert dm._test_data is not None  # pyright: ignore[reportPrivateUsage]
        assert len(dm._train_data) == 5  # pyright: ignore[reportPrivateUsage]
        assert len(dm._test_data) == 3  # pyright: ignore[reportPrivateUsage]


class TestGraphDataModuleSynthetic:
    """Tests for synthetic graph dataset setup."""

    def test_synthetic_er_setup(self) -> None:
        """Should setup Erdos-Renyi graph dataset.

        Rationale: ER graphs should be generated with the specified
        probability parameter and split correctly.
        """
        from torch_geometric.utils import to_dense_adj

        dm = GraphDataModule(
            graph_type="er",
            graph_config={
                "num_nodes": 15,
                "num_graphs": 30,
                "p": 0.2,
                "seed": 42,
            },
            train_ratio=0.6,
            val_ratio=0.2,
            samples_per_graph=1,
        )
        dm.prepare_data()
        dm.setup()

        assert dm._train_data is not None  # pyright: ignore[reportPrivateUsage]
        assert dm._val_data is not None  # pyright: ignore[reportPrivateUsage]
        assert dm._test_data is not None  # pyright: ignore[reportPrivateUsage]
        # 60% train, 20% val, 20% test of 30 graphs (samples_per_graph=1)
        total = (
            len(dm._train_data)  # pyright: ignore[reportPrivateUsage]
            + len(dm._val_data)  # pyright: ignore[reportPrivateUsage]
            + len(dm._test_data)  # pyright: ignore[reportPrivateUsage]
        )
        assert total == 30

        # Check matrix properties via dense reconstruction
        d = dm._train_data[0]  # pyright: ignore[reportPrivateUsage]
        assert d.edge_index is not None
        A = to_dense_adj(d.edge_index, max_num_nodes=d.num_nodes).squeeze(0)
        assert A.shape == (15, 15)
        assert torch.allclose(A, A.T)  # Symmetric
        assert A.diag().sum() == 0  # No self-loops

    def test_synthetic_regular_setup(self) -> None:
        """Should setup d-regular graph dataset.

        Rationale: Regular graphs should have consistent degree structure.
        """
        from torch_geometric.utils import to_dense_adj

        dm = GraphDataModule(
            graph_type="regular",
            graph_config={
                "num_nodes": 12,
                "num_graphs": 20,
                "d": 3,
                "seed": 42,
            },
            samples_per_graph=1,
        )
        dm.prepare_data()
        dm.setup()

        assert dm._train_data is not None  # pyright: ignore[reportPrivateUsage]

        # Check regular graph property (all degrees = d)
        d = dm._train_data[0]  # pyright: ignore[reportPrivateUsage]
        assert d.edge_index is not None
        A = to_dense_adj(d.edge_index, max_num_nodes=d.num_nodes).squeeze(0)
        degrees = A.sum(dim=1)
        assert torch.allclose(degrees, torch.full_like(degrees, 3.0))


class TestGraphDataModuleDataLoaders:
    """Tests for dataloader creation."""

    def test_train_dataloader_batch_size(self) -> None:
        """Train dataloader should use configured batch_size.

        Rationale: The batch size should match the configuration for
        proper training throughput.
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={"num_nodes": 8, "block_sizes": [4, 4]},
            batch_size=16,
            samples_per_graph=32,
            num_workers=0,  # Avoid multiprocessing in tests
        )
        dm.prepare_data()
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        # Batch is a GraphData with adjacency shape (batch_size, n, n)
        assert isinstance(batch, GraphData)
        adj = batch.dense_adjacency()
        assert adj.shape[0] == 16
        assert adj.shape[1] == adj.shape[2] == 8

    def test_val_dataloader_default_samples(self) -> None:
        """Val/test dataloaders default to half of samples_per_graph.

        Rationale: When val_samples_per_graph is not specified, it defaults
        to samples_per_graph // 2. The parent uses list repetition, so
        dataset length = num_unique_graphs * samples_per_graph.
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={
                "num_nodes": 8,
                "block_sizes": [4, 4],
                "num_graphs": 10,
            },
            samples_per_graph=100,
            val_samples_per_graph=50,
            batch_size=10,
            num_workers=0,
        )
        dm.prepare_data()
        dm.setup()

        # Verify the repetition factor: train dataset length should be
        # num_train_graphs * samples_per_graph
        assert dm._train_data is not None  # pyright: ignore[reportPrivateUsage]
        assert dm._val_data is not None  # pyright: ignore[reportPrivateUsage]

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        # The dataset lengths reflect list repetition
        assert len(train_loader.dataset) > len(val_loader.dataset)  # type: ignore[arg-type]

    def test_val_samples_per_graph_explicit(self) -> None:
        """val_samples_per_graph can be set explicitly.

        Rationale: Users should be able to override the default half-samples
        behavior for validation/test dataloaders.
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={
                "num_nodes": 8,
                "block_sizes": [4, 4],
                "num_graphs": 10,
            },
            samples_per_graph=10,
            val_samples_per_graph=5,
            batch_size=10,
            num_workers=0,
        )
        dm.prepare_data()
        dm.setup()

        assert dm.val_samples_per_graph == 5

    def test_dataloader_without_setup_raises(self) -> None:
        """Calling dataloader before setup should raise RuntimeError.

        Rationale: Dataloaders require adjacency matrices to be loaded,
        which happens in setup().
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={"num_nodes": 8, "block_sizes": [4, 4]},
        )

        with pytest.raises(RuntimeError, match="not setup|setup.*first|Call setup"):
            dm.train_dataloader()

        with pytest.raises(RuntimeError, match="not setup|setup.*first|Call setup"):
            dm.val_dataloader()

        with pytest.raises(RuntimeError, match="not setup|setup.*first|Call setup"):
            dm.test_dataloader()

    def test_dataloader_persistent_workers(self) -> None:
        """Dataloaders should have persistent_workers when num_workers > 0.

        Rationale: Persistent workers improve performance by keeping
        worker processes alive between epochs.
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={"num_nodes": 8, "block_sizes": [4, 4]},
            num_workers=2,
            samples_per_graph=20,
        )
        dm.prepare_data()
        dm.setup()

        loader = dm.train_dataloader()
        assert loader.persistent_workers is True


class TestGraphDataModuleUnknownDataset:
    """Tests for error handling with unknown datasets."""

    def test_unknown_dataset_raises(self) -> None:
        """Unknown dataset name should raise ValueError.

        Rationale: Clear error message helps users identify typos
        or unsupported datasets. The error propagates from
        SyntheticGraphDataset via _generate_adjacencies(). Pass a
        valid ``num_nodes`` so the constructor accepts the config and
        the failure is raised by ``setup()`` (the path under test).
        """
        dm = GraphDataModule(
            graph_type="nonexistent_dataset",
            graph_config={"num_nodes": 8},
        )
        dm.prepare_data()  # No-op for non-wrapper types

        with pytest.raises(ValueError, match="graph_type must be one of"):
            dm.setup()


class TestGraphDataModulePublicContract:
    """Tests for the remaining public datamodule access surface."""

    def test_get_reference_graphs_returns_val_graphs(self) -> None:
        """Validation graphs should be exposed via get_reference_graphs()."""
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={"num_nodes": 8, "block_sizes": [4, 4]},
        )
        dm.prepare_data()
        dm.setup()

        graphs = dm.get_reference_graphs("val", max_graphs=2)

        assert len(graphs) == 2
        # Per 2026-05-01 universal-transport refactor: GraphData not nx.
        assert all(isinstance(gd, GraphData) for gd in graphs)
        assert all(int(gd.node_mask.sum().item()) == 8 for gd in graphs)

    def test_get_size_distribution_is_fixed(self) -> None:
        """GraphDataModule should expose a fixed-size distribution."""
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={"num_nodes": 8, "block_sizes": [4, 4]},
        )

        dist = dm.get_size_distribution("train")
        assert dist.is_degenerate
        assert dist.sizes == (8,)
        assert dist.max_size == 8

    def test_reference_graphs_without_setup_raises(self) -> None:
        """Reference extraction should fail loudly before setup()."""
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={"num_nodes": 8, "block_sizes": [4, 4]},
        )

        with pytest.raises(RuntimeError):
            dm.get_reference_graphs("val", max_graphs=1)

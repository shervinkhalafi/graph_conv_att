"""Regression tests for all 4 datamodule classes.

These tests verify the full lifecycle (init → setup → dataloader → batch) for
each datamodule, focusing on the external contracts that consumers rely on. They
serve as a safety net for the BaseGraphDataModule unification refactoring: any
change that breaks the batch format, split sizes, or metadata contracts should
cause a failure here.

The existing test files (test_data_module.py, test_single_graph_datasets.py,
test_categorical_datamodule.py, test_generative_integration.py) cover each
module in depth. This file focuses on cross-cutting contract verification.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from tmgg.data.data_modules.data_module import GraphDataModule
from tmgg.data.data_modules.graph_generation import (
    generate_adjacency_batch,
    generate_multigraph_split,
)
from tmgg.data.data_modules.multigraph_data_module import (
    MultiGraphDataModule,
)
from tmgg.data.data_modules.single_graph_data_module import (
    SingleGraphDataModule,
)
from tmgg.data.data_modules.synthetic_categorical import (
    SyntheticCategoricalDataModule,
)
from tmgg.data.datasets.graph_types import GraphData


def _graphdata_to_numpy(batch: GraphData, index: int = 0) -> np.ndarray:
    """Extract one graph from a (sparse-default) GraphData batch.

    Post the sparse-default refactor the datamodule emits ``GraphState``
    (sparse), which has no ``node_mask`` attribute. ``num_nodes_per_graph``
    is the universal source of truth on every concrete carrier.
    """
    num_nodes = int(batch.num_nodes_per_graph[index].item())
    adj = batch.dense_adjacency()[index, :num_nodes, :num_nodes]
    return adj.cpu().numpy()


def _first_train_graph(dm: SingleGraphDataModule) -> np.ndarray:
    """Read the training graph through the public dataloader boundary."""
    return _graphdata_to_numpy(next(iter(dm.train_dataloader())))


def _first_reference_graph(dm: SingleGraphDataModule, stage: str) -> np.ndarray:
    """Read one val/test graph through get_reference_graphs().

    Post the 2026-05-01 universal-transport refactor get_reference_graphs
    yields list[GraphData]; convert at the leaf via to_networkx() to keep
    downstream nx.to_numpy_array assertions intact. Each entry is itself
    a batched dense carrier with B==1, so we explicitly pass
    ``batch_index=0``.
    """
    gd = dm.get_reference_graphs(stage, max_graphs=1)[0]
    return np.asarray(nx.to_numpy_array(gd.to_networkx(0)), dtype=np.float32)


# ---------------------------------------------------------------------------
# GraphDataModule lifecycle
# ---------------------------------------------------------------------------


class TestGraphDataModuleContract:
    """Full lifecycle for GraphDataModule (denoising, multi-source)."""

    def test_sbm_fixed_lifecycle(self) -> None:
        """SBM with fixed block_sizes: init → prepare → setup → batch.

        Rationale: Fixed-partition SBM is the most common denoising config.
        The batch is a plain Tensor of shape (bs, n, n).
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={
                "num_nodes": 12,
                "block_sizes": [6, 6],
                "p_intra": 0.8,
                "p_inter": 0.1,
            },
            samples_per_graph=16,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.prepare_data()
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, GraphData)
        assert batch.dense_adjacency().shape == (4, 12, 12)
        assert batch.dense_adjacency().dtype == torch.float32

        val_batch = next(iter(dm.val_dataloader()))
        assert isinstance(val_batch, GraphData)
        assert val_batch.dense_adjacency().shape[1:] == (12, 12)

        test_batch = next(iter(dm.test_dataloader()))
        assert isinstance(test_batch, GraphData)
        assert test_batch.dense_adjacency().shape[1:] == (12, 12)

    def test_sbm_enumerated_lifecycle(self) -> None:
        """SBM with enumerated partitions: different partitions per split.

        Rationale: Enumerated partitions are the denoising SBM variant that
        tests generalization to unseen community structures. Train and test
        should receive different adjacency matrices.
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={
                "num_nodes": 20,
                "partition_mode": "enumerated",
                "num_train_partitions": 3,
                "num_test_partitions": 2,
                "min_blocks": 2,
                "max_blocks": 4,
                "min_block_size": 3,
                "max_block_size": 8,
                "p_intra": 0.8,
                "p_inter": 0.1,
            },
            samples_per_graph=8,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.prepare_data()
        dm.setup()

        # samples_per_graph=8, val_samples_per_graph defaults to 8//2=4
        # train: 3 unique graphs * 8 repetitions = 24
        # test:  2 unique graphs * 4 repetitions = 8
        assert dm._train_data is not None  # pyright: ignore[reportPrivateUsage]
        assert dm._test_data is not None  # pyright: ignore[reportPrivateUsage]
        assert len(dm._train_data) == 3 * 8  # pyright: ignore[reportPrivateUsage]
        assert len(dm._test_data) == 2 * 4  # pyright: ignore[reportPrivateUsage]

        batch = next(iter(dm.train_dataloader()))
        assert batch.dense_adjacency().shape == (4, 20, 20)

    def test_er_lifecycle(self) -> None:
        """Erdos-Renyi: synthetic graph generation through SyntheticGraphDataset."""
        dm = GraphDataModule(
            graph_type="er",
            graph_config={"num_nodes": 10, "num_graphs": 20, "p": 0.3, "seed": 42},
            batch_size=4,
            num_workers=0,
            samples_per_graph=8,
        )
        dm.prepare_data()
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, GraphData)
        assert batch.dense_adjacency().shape[1:] == (10, 10)


# ---------------------------------------------------------------------------
# SingleGraphDataModule lifecycle
# ---------------------------------------------------------------------------


class TestSingleGraphDataModuleContract:
    """Full lifecycle for SingleGraphDataModule (single-graph denoising)."""

    def test_sbm_same_graph_lifecycle(self) -> None:
        """SBM with same_graph_all_splits=True: all splits use identical graph.

        Rationale: Stage 1 protocol — model sees only noise variation.
        """
        dm = SingleGraphDataModule(
            graph_type="sbm",
            num_nodes=20,
            graph_config={"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 2},
            same_graph_all_splits=True,
            num_train_samples=16,
            num_val_samples=8,
            num_test_samples=8,
            batch_size=4,
            num_workers=0,
        )
        dm.setup()

        train_graph = _first_train_graph(dm)
        assert np.array_equal(train_graph, _first_reference_graph(dm, "val"))
        assert np.array_equal(train_graph, _first_reference_graph(dm, "test"))

        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, GraphData)
        assert batch.dense_adjacency().shape == (4, 20, 20)
        assert batch.dense_adjacency().dtype == torch.float32

    def test_er_different_graphs_lifecycle(self) -> None:
        """ER with same_graph_all_splits=False: different graphs per split.

        Rationale: Stage 2+ protocol — test generalization to new structures.
        """
        dm = SingleGraphDataModule(
            graph_type="erdos_renyi",
            num_nodes=15,
            graph_config={"p": 0.2},
            same_graph_all_splits=False,
            num_train_samples=8,
            num_val_samples=4,
            num_test_samples=4,
            batch_size=2,
            num_workers=0,
        )
        dm.setup()

        train_graph = _first_train_graph(dm)
        assert not np.array_equal(train_graph, _first_reference_graph(dm, "val"))

        batch = next(iter(dm.train_dataloader()))
        assert batch.dense_adjacency().shape == (2, 15, 15)

    def test_graph_properties(self) -> None:
        """Generated graphs should be symmetric, binary, with zero diagonal."""
        dm = SingleGraphDataModule(
            graph_type="sbm",
            num_nodes=20,
            graph_config={"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3},
            same_graph_all_splits=True,
        )
        dm.setup()

        A = _first_train_graph(dm)
        assert A.shape == (20, 20)
        assert np.allclose(A, A.T), "Should be symmetric"
        assert set(np.unique(A)).issubset({0.0, 1.0}), "Should be binary"
        assert np.allclose(np.diag(A), 0), "No self-loops"


# ---------------------------------------------------------------------------
# MultiGraphDataModule lifecycle
# ---------------------------------------------------------------------------


class TestMultiGraphDataModuleContract:
    """Full lifecycle for MultiGraphDataModule (gaussian generative)."""

    def test_sbm_lifecycle(self) -> None:
        """SBM: init → setup → all 3 dataloaders → batch shape.

        Rationale: The gaussian diffusion generative module serves plain
        Tensor batches of shape (bs, n, n), same as denoising.
        """
        dm = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=30,
            train_ratio=0.8,
            val_ratio=0.1,
            batch_size=4,
            num_workers=0,
            seed=42,
            graph_config={"num_blocks": 2, "p_intra": 0.7, "p_inter": 0.1},
        )
        dm.setup()

        # Train dataloader
        train_batch = next(iter(dm.train_dataloader()))
        assert isinstance(train_batch, GraphData)
        assert train_batch.dense_adjacency().shape == (4, 16, 16)
        assert train_batch.dense_adjacency().dtype == torch.float32

        # Val dataloader
        val_batch = next(iter(dm.val_dataloader()))
        assert isinstance(val_batch, GraphData)
        assert val_batch.dense_adjacency().shape[1:] == (16, 16)

        # Test dataloader
        test_batch = next(iter(dm.test_dataloader()))
        assert isinstance(test_batch, GraphData)
        assert test_batch.dense_adjacency().shape[1:] == (16, 16)

    def test_er_lifecycle(self) -> None:
        """ER graphs through the generative pipeline."""
        dm = MultiGraphDataModule(
            graph_type="er",
            num_nodes=12,
            num_graphs=20,
            batch_size=4,
            num_workers=0,
            seed=42,
            graph_config={"p": 0.3},
        )
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        assert batch.dense_adjacency().shape == (4, 12, 12)

    def test_split_sizes(self) -> None:
        """Train/val/test split sizes should match ratios."""
        dm = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=10,
            num_graphs=100,
            train_ratio=0.8,
            val_ratio=0.1,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.setup()

        assert dm._train_data is not None  # pyright: ignore[reportPrivateUsage]
        assert dm._val_data is not None  # pyright: ignore[reportPrivateUsage]
        assert dm._test_data is not None  # pyright: ignore[reportPrivateUsage]
        assert len(dm._train_data) == 80  # pyright: ignore[reportPrivateUsage]
        assert len(dm._val_data) == 10  # pyright: ignore[reportPrivateUsage]
        assert len(dm._test_data) == 10  # pyright: ignore[reportPrivateUsage]

    def test_graph_validity(self) -> None:
        """Generated graphs should be binary, symmetric, zero-diagonal.

        After the PyG Data storage refactor, _train_data is a list[Data]
        with COO edge_index. We reconstruct dense adjacency to verify the
        same invariants.
        """
        from torch_geometric.utils import to_dense_adj

        dm = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=10,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.setup()

        assert dm._train_data is not None  # pyright: ignore[reportPrivateUsage]
        graphs = dm._train_data  # pyright: ignore[reportPrivateUsage]
        for g in graphs:
            assert g.edge_index is not None
            adj = to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes).squeeze(0)
            assert torch.all((adj == 0) | (adj == 1))
            assert torch.allclose(adj, adj.T)
            assert torch.all(adj.diagonal() == 0)

    def test_size_distribution_contract(self) -> None:
        """Fixed-size multigraph modules expose a degenerate size distribution."""
        dm = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=50,
        )
        dist = dm.get_size_distribution("train")
        assert dist.is_degenerate
        assert dist.sizes == (16,)
        assert dist.max_size == 16

    def test_idempotent_setup(self) -> None:
        """Calling setup() twice should not regenerate data."""
        dm = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=10,
            num_graphs=20,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.setup()
        train_before = dm._train_data  # pyright: ignore[reportPrivateUsage]
        dm.setup()
        assert dm._train_data is train_before  # pyright: ignore[reportPrivateUsage]

    def test_setup_required_for_dataloaders(self) -> None:
        """Accessing dataloaders before setup() should raise RuntimeError."""
        dm = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=10,
            num_graphs=20,
        )
        with pytest.raises(RuntimeError, match="not setup|not set up"):
            dm.train_dataloader()
        with pytest.raises(RuntimeError, match="not setup|not set up"):
            dm.val_dataloader()
        with pytest.raises(RuntimeError, match="not setup|not set up"):
            dm.test_dataloader()


# ---------------------------------------------------------------------------
# SyntheticCategoricalDataModule lifecycle
# ---------------------------------------------------------------------------


class TestSyntheticCategoricalDataModuleContract:
    """Full lifecycle for SyntheticCategoricalDataModule (discrete generative)."""

    def test_sbm_lifecycle(self) -> None:
        """SBM: init → setup → dataloader → (X, E, y, node_mask) batch.

        Rationale: Discrete diffusion requires categorical tuple batches,
        NOT plain tensors. This is the fundamental format difference.
        """
        dm = SyntheticCategoricalDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=30,
            train_ratio=0.8,
            val_ratio=0.1,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.setup()

        batch = next(iter(dm.train_dataloader()))

        # Sparse-default refactor: dataloader emits GraphState. Convert to
        # dense for the structural assertions; the no-edge fill keeps the
        # categorical [no-edge, edge] one-hot semantics that this test
        # codifies.
        no_edge_fill = torch.tensor([1.0, 0.0])
        dense = batch.to_dense(edge_class_fill=no_edge_fill)
        # Structure-only SBM data emits X_class=None. E_class holds the
        # two-channel [no-edge, edge] one-hot; node_mask marks real vs
        # padded positions.
        assert dense.X_class is None
        assert dense.E_class is not None
        assert dense.E_class.shape == (4, 16, 16, 2)  # (bs, n, n, de)
        assert dense.y.shape == (4, 0)  # (bs, 0) — no global features
        assert dense.node_mask.shape == (4, 16)  # (bs, n)
        assert dense.node_mask.dtype == torch.bool

    def test_train_batch_contract(self) -> None:
        """Train batches expose valid one-hot categorical node and edge features.

        Rationale: the diffusion pipeline now learns empirical stationary PMFs
        from the train loader directly, so the datamodule contract is the batch
        shape and masking behavior rather than cached marginal vectors.
        """
        dm = SyntheticCategoricalDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=50,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        # Sparse-default refactor: dataloader emits GraphState; convert to
        # dense for the categorical structural assertions. Structure-only
        # datasets emit X_class=None; node_mask carries which positions
        # are real.
        no_edge_fill = torch.tensor([1.0, 0.0])
        dense = batch.to_dense(edge_class_fill=no_edge_fill)
        assert dense.X_class is None
        assert dense.E_class is not None
        # Off-diagonal positions must be valid one-hot; the diagonal is
        # emitted as all-zero (upstream encode_no_edge parity — see
        # ``GraphData.from_pyg_batch``).
        n = dense.node_mask.size(-1)
        off_diag = ~torch.eye(n, dtype=torch.bool, device=dense.node_mask.device)
        edge_mask = dense.node_mask.unsqueeze(1) & dense.node_mask.unsqueeze(2)
        edge_mask = edge_mask & off_diag.unsqueeze(0)
        edge_sums = dense.E_class[edge_mask].sum(dim=-1)

        assert dense.E_class.shape[-1] == 2
        assert torch.allclose(edge_sums, torch.ones_like(edge_sums))

    def test_reference_graphs_contract(self) -> None:
        """Categorical modules expose val graphs through get_reference_graphs()."""
        dm = SyntheticCategoricalDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=50,
        )
        dm.setup()

        graphs = dm.get_reference_graphs("val", max_graphs=3)
        assert len(graphs) == 3
        # Per 2026-05-01 universal-transport refactor get_reference_graphs
        # returns GraphData; per-graph node count lives on
        # num_nodes_per_graph (universal across the type grid).
        assert all(int(gd.num_nodes_per_graph.sum().item()) == 16 for gd in graphs)

    def test_er_lifecycle(self) -> None:
        """Non-SBM graph types should also produce valid categorical data."""
        dm = SyntheticCategoricalDataModule(
            graph_type="er",
            num_nodes=12,
            num_graphs=20,
            batch_size=4,
            num_workers=0,
            seed=42,
            graph_config={"p": 0.3},
        )
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        # Sparse-default refactor: dataloader emits GraphState; convert
        # to dense for the categorical edge-shape assertion.
        no_edge_fill = torch.tensor([1.0, 0.0])
        dense = batch.to_dense(edge_class_fill=no_edge_fill)
        assert dense.X_class is None
        assert dense.E_class is not None
        assert dense.E_class.shape[1:] == (12, 12, 2)


# ---------------------------------------------------------------------------
# Cross-cutting: reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Verify that all datamodules produce identical output given identical seeds."""

    def test_graph_distribution_reproducible(self) -> None:
        """Two MultiGraphDataModule instances with the same seed
        should produce identical training data."""

        def _make_gdm() -> MultiGraphDataModule:
            return MultiGraphDataModule(
                graph_type="sbm",
                num_nodes=10,
                num_graphs=20,
                batch_size=4,
                num_workers=0,
                seed=42,
            )

        dm1 = _make_gdm()
        dm1.setup()
        dm2 = _make_gdm()
        dm2.setup()

        assert dm1._train_data is not None and dm2._train_data is not None  # pyright: ignore[reportPrivateUsage]
        # Compare edge_index tensors of corresponding Data objects
        for d1, d2 in zip(dm1._train_data, dm2._train_data, strict=False):  # pyright: ignore[reportPrivateUsage]
            assert d1.edge_index is not None and d2.edge_index is not None
            assert torch.equal(d1.edge_index, d2.edge_index)

    def test_categorical_reproducible(self) -> None:
        """Two SyntheticCategoricalDataModule instances with the same seed
        should produce identical training graphs."""

        def _make_cat_dm() -> SyntheticCategoricalDataModule:
            return SyntheticCategoricalDataModule(
                graph_type="sbm",
                num_nodes=10,
                num_graphs=20,
                batch_size=4,
                num_workers=0,
                seed=42,
            )

        dm1 = _make_cat_dm()
        dm1.setup()
        dm2 = _make_cat_dm()
        dm2.setup()

        assert dm1._train_data is not None and dm2._train_data is not None  # pyright: ignore[reportPrivateUsage]
        for d1, d2 in zip(dm1._train_data, dm2._train_data, strict=False):  # pyright: ignore[reportPrivateUsage]
            assert d1.edge_index is not None and d2.edge_index is not None
            assert torch.equal(d1.edge_index, d2.edge_index)


# ---------------------------------------------------------------------------
# SBM block_size_alpha (paper α) dispatch contract  — REPRODUCE.md flag #2
# ---------------------------------------------------------------------------


class TestSBMBlockSizeAlphaDispatch:
    """Regression: ``block_size_alpha`` must flow through the SBM dispatch.

    Rationale (REPRODUCE.md flag #2; paper Appendix B.2.3, Eq. 28):
        The paper's synthetic-SBM diversity sweep is the symmetric-Dirichlet
        block-size concentration ``graph_config.block_size_alpha``
        (``s ~ n·Dirichlet(α,…,α)``). Before this fix the datamodule dispatch
        dropped the key — ``generate_adjacency_batch`` forwarded only
        ``num_blocks``/``p_intra``/``p_inter``/``seed`` — *and* the paper's
        ``sbm_n100``/``sbm_n200`` presets route through the enumerated-partition
        path that never calls ``generate_sbm_batch``. So the knob was a silent
        no-op through config. These tests are the regression guard.

    Invariants:
        With ``p_intra=1.0`` and ``p_inter=0.0`` each graph is a disjoint union
        of cliques, so a node's degree equals ``(its block size − 1)``. The
        within-graph spread of node degrees is therefore a direct proxy for
        block-size heterogeneity: near-zero for equal blocks, large for a
        low-α (unbalanced) Dirichlet partition.
    """

    _BASE: dict[str, float] = {
        "num_nodes": 100,
        "min_blocks": 2,
        "max_blocks": 4,
        "p_intra": 1.0,
        "p_inter": 0.0,
    }

    @staticmethod
    def _mean_within_graph_degree_std(adjacencies: np.ndarray) -> float:
        """Mean over graphs of the within-graph node-degree standard deviation."""
        degrees = adjacencies.sum(axis=-1)  # (num_graphs, n)
        return float(np.mean(degrees.std(axis=-1)))

    def _batch(self, *, alpha: float | None, num_graphs: int = 64) -> np.ndarray:
        graph_config = dict(self._BASE)
        if alpha is not None:
            graph_config["block_size_alpha"] = alpha
        return generate_adjacency_batch(
            graph_type="sbm",
            num_nodes=100,
            num_graphs=num_graphs,
            graph_config=graph_config,
            seed=0,
        )

    def test_alpha_controls_heterogeneity_direction(self) -> None:
        """Small α → more heterogeneous blocks than large α (paper's claim)."""
        het = self._mean_within_graph_degree_std(self._batch(alpha=0.1))
        hom = self._mean_within_graph_degree_std(self._batch(alpha=10.0))
        assert het > hom, (
            f"expected α=0.1 more heterogeneous than α=10.0, got "
            f"het={het:.3f} hom={hom:.3f}"
        )

    def test_alpha_is_not_a_noop_vs_equal_blocks(self) -> None:
        """α=0.1 (Dirichlet) must differ from the equal-block baseline (no α)."""
        equal = self._mean_within_graph_degree_std(self._batch(alpha=None))
        dirichlet = self._mean_within_graph_degree_std(self._batch(alpha=0.1))
        assert dirichlet > equal + 1.0, (
            f"block_size_alpha appears ignored: dirichlet={dirichlet:.3f} "
            f"not clearly above equal-block baseline={equal:.3f}"
        )

    def test_alpha_forces_batch_path_over_enumerated(self) -> None:
        """block_size_alpha overrides enumerated mode → per-graph Dirichlet batch.

        With ``num_train_partitions`` present the SBM path would normally route
        to ``_generate_partitioned_sbm_split`` (enumerated: ~25 graphs total).
        Setting ``block_size_alpha`` must instead take the batch path, which
        emits exactly ``num_graphs`` graphs split by the ratios.
        """
        graph_config = dict(self._BASE)
        graph_config.update(
            num_train_partitions=10,
            num_test_partitions=10,
            block_size_alpha=0.1,
        )
        train, val, test = generate_multigraph_split(
            graph_type="sbm",
            num_nodes=100,
            num_graphs=50,
            graph_config=graph_config,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=0,
        )
        assert len(train) + len(val) + len(test) == 50, (
            "block_size_alpha did not force the batch path: total graphs "
            f"{len(train) + len(val) + len(test)} != num_graphs=50 "
            "(enumerated path would emit ~25)"
        )
        # And the batch is genuinely heterogeneous (Dirichlet, not equal blocks).
        assert self._mean_within_graph_degree_std(train) > 1.0

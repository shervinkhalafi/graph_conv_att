"""Field-occupancy invariants and helper round-trips for DenseGraphState.

Covers the unified GraphData schema
(see ``internal notes (not included)``):

- Valid combinations of optional split fields construct cleanly.
- Invalid combinations raise ``ValueError``.
- ``DenseGraphState.replace`` behaves like :func:`dataclasses.replace`.
- The Wave 1.2 split-field helpers (``from_structure_only`` /
  ``from_edge_scalar`` / ``to_edge_scalar``) round-trip as expected.
- Every concrete datamodule emits a sparse ``GraphState`` whose
  ``edge_class`` / ``x_class`` (lower-case sparse fields) match the
  no-X_class / E_class invariants documented for the new sparse-default
  pipeline. The dense view is recovered via
  :func:`state_to_dense_sample` at the consumption site.
"""

from __future__ import annotations

import pytest
import torch

from tests._helpers.graph_builders import binary_graphdata
from tmgg.data.datasets.graph_types import (
    DenseGraphState,
    GraphState,
    state_to_dense_sample,
)

# ---------------------------------------------------------------------------
# Constructor invariants
# ---------------------------------------------------------------------------


def _num_nodes(node_mask: torch.Tensor) -> torch.Tensor:
    """Helper: derive ``num_nodes_per_graph`` from a (bs, n) boolean mask."""
    return node_mask.long().sum(dim=-1)


def test_xclass_plus_eclass_construction_ok() -> None:
    """X_class + E_class populated side-by-side construct cleanly."""
    bs, n = 2, 3
    y = torch.zeros(bs, 0)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    X_class = torch.zeros(bs, n, 4)
    E_class = torch.zeros(bs, n, n, 2)
    E_class[..., 0] = 1.0
    data = DenseGraphState(
        num_nodes_per_graph=_num_nodes(node_mask),
        y=y,
        X_class=X_class,
        E_class=E_class,
    )
    assert data.X_class is X_class
    assert data.E_class is E_class


def test_eclass_only_construction_ok() -> None:
    """E_class only, no X fields."""
    bs, n = 2, 3
    y = torch.zeros(bs, 0)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    E_class = torch.zeros(bs, n, n, 2)
    E_class[..., 0] = 1.0
    data = DenseGraphState(
        num_nodes_per_graph=_num_nodes(node_mask), y=y, E_class=E_class
    )
    assert data.X_class is None
    assert data.X_feat is None
    assert data.E_class is E_class


def test_efeat_only_construction_ok() -> None:
    """E_feat only."""
    bs, n = 2, 3
    y = torch.zeros(bs, 0)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    E_feat = torch.zeros(bs, n, n, 1)
    data = DenseGraphState(
        num_nodes_per_graph=_num_nodes(node_mask), y=y, E_feat=E_feat
    )
    assert data.E_feat is E_feat
    assert data.E_class is None


def test_eclass_plus_efeat_construction_ok() -> None:
    """Both E_class and E_feat populated."""
    bs, n = 2, 3
    y = torch.zeros(bs, 0)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    E_class = torch.zeros(bs, n, n, 2)
    E_class[..., 0] = 1.0
    E_feat = torch.zeros(bs, n, n, 1)
    data = DenseGraphState(
        num_nodes_per_graph=_num_nodes(node_mask),
        y=y,
        E_class=E_class,
        E_feat=E_feat,
    )
    assert data.E_class is E_class
    assert data.E_feat is E_feat


def test_xfeat_plus_efeat_construction_ok() -> None:
    """X_feat + E_feat populated."""
    bs, n = 2, 3
    y = torch.zeros(bs, 0)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    X_feat = torch.zeros(bs, n, 5)
    E_feat = torch.zeros(bs, n, n, 1)
    data = DenseGraphState(
        num_nodes_per_graph=_num_nodes(node_mask),
        y=y,
        X_feat=X_feat,
        E_feat=E_feat,
    )
    assert data.X_feat is X_feat
    assert data.E_feat is E_feat


def test_missing_all_edges_raises_value_error() -> None:
    """No split edge fields → ValueError."""
    bs, n = 2, 3
    y = torch.zeros(bs, 0)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one of E_class"):
        DenseGraphState(num_nodes_per_graph=_num_nodes(node_mask), y=y)


def test_xclass_wrong_leading_dim_raises() -> None:
    """A split-feature tensor with mismatched ``n`` leading dim → ValueError.

    ``DenseGraphState.__post_init__`` infers ``n_max`` from the first
    populated split-feature tensor (``X_class`` here) and then validates
    every other field. So a (bs, n+1, 4) ``X_class`` paired with a
    (bs, n, n, 2) ``E_class`` surfaces as an ``E_class leading dims``
    mismatch.
    """
    bs, n = 2, 3
    y = torch.zeros(bs, 0)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    bad_X_class = torch.zeros(bs, n + 1, 4)  # wrong n dim
    with pytest.raises(ValueError, match="leading dims"):
        DenseGraphState(
            num_nodes_per_graph=_num_nodes(node_mask),
            y=y,
            X_class=bad_X_class,
            E_class=torch.zeros(bs, n, n, 2),
        )


def test_replace_preserves_other_fields() -> None:
    """DenseGraphState.replace swaps one field and preserves the rest."""
    bs, n = 2, 3
    y = torch.zeros(bs, 0)
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    E_class = torch.zeros(bs, n, n, 2)
    E_class[..., 0] = 1.0
    data = DenseGraphState(
        num_nodes_per_graph=_num_nodes(node_mask), y=y, E_class=E_class
    )

    new_E_class = torch.zeros(bs, n, n, 2)
    new_E_class[..., 1] = 1.0
    updated = data.replace(E_class=new_E_class)

    assert updated.E_class is new_E_class
    assert updated.y is data.y
    # ``node_mask`` is now a cached_property derived from
    # ``num_nodes_per_graph``; assert tensor-equality rather than identity.
    assert updated.num_nodes_per_graph is data.num_nodes_per_graph
    assert torch.equal(updated.node_mask, data.node_mask)


# ---------------------------------------------------------------------------
# from_structure_only / from_edge_scalar / to_edge_scalar
# ---------------------------------------------------------------------------


def test_from_structure_only_to_edge_scalar_feat_roundtrip() -> None:
    """(n, n) scalar adjacency round-trips via from_structure_only + to_edge_scalar('feat')."""
    n = 4
    adj = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    node_mask = torch.ones(n, dtype=torch.bool)
    data = DenseGraphState.from_structure_only(node_mask, adj)

    assert data.E_feat is not None
    # ``from_structure_only`` always batches; a single graph yields shape
    # ``(1, n, n, 1)``.
    assert data.E_feat.shape == (1, n, n, 1)
    assert data.E_class is None
    assert data.X_class is None
    assert data.X_feat is None

    recovered = data.to_edge_scalar(source="feat")
    assert recovered.shape == (1, n, n)
    assert torch.allclose(recovered.squeeze(0), adj)


def test_from_edge_scalar_class_to_edge_scalar_class_roundtrip() -> None:
    """from_edge_scalar(target='E_class') + to_edge_scalar('class') recovers 0/1 adjacency."""
    bs, n = 2, 3
    adj = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        ]
    )
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    data = DenseGraphState.from_edge_scalar(adj, node_mask=node_mask, target="E_class")

    assert data.E_class is not None
    assert data.E_class.shape == (bs, n, n, 2)
    assert data.E_feat is None
    assert data.X_class is None

    recovered = data.to_edge_scalar(source="class")
    assert recovered.shape == (bs, n, n)
    # 1 − P(no_edge) round-trips exactly for 0/1 inputs.
    assert torch.allclose(recovered, adj)


def test_to_edge_scalar_feat_without_efeat_raises() -> None:
    """Calling to_edge_scalar(source='feat') when only E_class is set raises."""
    bs, n = 2, 3
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    data = DenseGraphState.from_edge_scalar(
        torch.zeros(bs, n, n), node_mask=node_mask, target="E_class"
    )
    # Sanity: constructed via from_edge_scalar, E_feat should be None.
    assert data.E_feat is None
    with pytest.raises(ValueError, match="E_feat to be populated"):
        _ = data.to_edge_scalar(source="feat")


def test_from_edge_scalar_feat_equals_from_structure_only() -> None:
    """target='E_feat' dispatches to from_structure_only for identical shape."""
    bs, n = 2, 3
    adj = torch.rand(bs, n, n)
    adj = 0.5 * (adj + adj.transpose(1, 2))  # symmetric
    node_mask = torch.ones(bs, n, dtype=torch.bool)

    via_feat = DenseGraphState.from_edge_scalar(
        adj, node_mask=node_mask, target="E_feat"
    )
    via_struct = DenseGraphState.from_structure_only(node_mask, adj)

    assert via_feat.E_feat is not None
    assert via_struct.E_feat is not None
    assert torch.allclose(via_feat.E_feat, via_struct.E_feat)


# ---------------------------------------------------------------------------
# Categorical datasets populate E_class (and optionally X_class)
#
# After the Wave 9 removal wave the legacy ``X`` / ``E`` fields are gone;
# every test below verifies the invariant that datamodules and
# ``from_*`` helpers write directly into the split categorical fields.
# ---------------------------------------------------------------------------


def test_from_binary_adjacency_populates_class_fields() -> None:
    """from_binary_adjacency writes X_class and E_class only."""
    bs, n = 2, 4
    adj = torch.zeros(bs, n, n)
    adj[0, 0, 1] = adj[0, 1, 0] = 1.0
    adj[1, 2, 3] = adj[1, 3, 2] = 1.0
    data = binary_graphdata(adj)
    assert data.X_class is not None
    assert data.E_class is not None
    assert data.X_class.shape == (bs, n, 2)
    assert data.E_class.shape == (bs, n, n, 2)
    assert data.X_feat is None
    assert data.E_feat is None


def test_from_binary_adjacency_single_graph_populates_class_fields() -> None:
    """Single-graph branch also writes the split fields.

    The ``binary_graphdata`` helper always produces a batched output
    (leading batch dim of 1) — the test legacy-shape assertions
    (``(3, 2)`` / ``(3, 3, 2)``) reflected the removed unbatched path.
    Updated to the post-refactor batched-of-one shape.
    """
    adj = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    data = binary_graphdata(adj)
    assert data.X_class is not None
    assert data.E_class is not None
    assert data.X_class.shape == (1, 3, 2)
    assert data.E_class.shape == (1, 3, 3, 2)


def test_collate_populates_class_fields() -> None:
    """DenseGraphState.collate pads the split categorical fields uniformly."""
    n1, n2 = 3, 5
    adj1 = torch.zeros(n1, n1)
    adj1[0, 1] = adj1[1, 0] = 1.0
    adj2 = torch.zeros(n2, n2)
    adj2[2, 3] = adj2[3, 2] = 1.0
    g1 = binary_graphdata(adj1)
    g2 = binary_graphdata(adj2)
    batch = DenseGraphState.collate([g1, g2])
    assert batch.X_class is not None
    assert batch.E_class is not None
    assert batch.X_class.shape == (2, n2, 2)
    assert batch.E_class.shape == (2, n2, n2, 2)


def test_from_pyg_batch_populates_class_fields() -> None:
    """GraphState.from_pyg_batch writes edge_class; x_class is None (Wave 9.3).

    The sparse path replaces the legacy dense-eager
    ``GraphData.from_pyg_batch``; structure-only datasets emit
    ``x_class=None`` and the spec forbids re-encoding ``node_mask`` as a
    degenerate ``X_class``.
    """
    from typing import cast

    from torch_geometric.data import Batch, Data
    from torch_geometric.data.data import BaseData

    data_list = [
        Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), num_nodes=3),
        Data(
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long),
            num_nodes=4,
        ),
    ]
    batch = Batch.from_data_list(cast(list[BaseData], data_list))
    state = GraphState.from_pyg_batch(batch)
    # Structure-only datasets emit x_class=None.
    assert state.x_class is None
    assert state.edge_class is not None


def test_from_pyg_batch_emits_symmetric_E_class() -> None:
    """Parity #4: dense view of ``GraphState.from_pyg_batch`` is symmetric.

    Upstream DiGress establishes ``E`` symmetry implicitly via
    ``to_dense_adj`` on a symmetric ``edge_index``. Our densification
    path is now the sparse-to-dense path on ``GraphState``: this
    regression test densifies the sparse output and pins that the
    resulting ``E_class`` is symmetric for both single- and multi-graph
    batches with mixed node counts. See
    ``internal notes (not included)``
    (#4).
    """
    from typing import cast

    from torch_geometric.data import Batch, Data
    from torch_geometric.data.data import BaseData

    data_list = [
        Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), num_nodes=3),
        Data(
            edge_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
            ),
            num_nodes=5,
        ),
    ]
    batch = Batch.from_data_list(cast(list[BaseData], data_list))
    state = GraphState.from_pyg_batch(batch)
    dense = state_to_dense_sample(state)
    assert dense.E_class is not None
    e = dense.E_class
    assert torch.allclose(e, e.transpose(1, 2))


def test_multigraph_datamodule_batch_populates_class_fields() -> None:
    """End-to-end: a MultiGraphDataModule batch carries edge_class; x_class stays None.

    Datamodules now emit a sparse ``GraphState`` (lower-case fields).
    Structure-only synthetic graphs leave ``x_class`` empty.
    """
    from tmgg.data.data_modules.multigraph_data_module import MultiGraphDataModule

    dm = MultiGraphDataModule(
        graph_type="er",
        num_nodes=10,
        num_graphs=8,
        train_ratio=0.5,
        val_ratio=0.25,
        graph_config={"p": 0.3},
        batch_size=4,
        num_workers=0,
        seed=0,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, GraphState)
    assert batch.x_class is None
    assert batch.edge_class is not None


def test_synthetic_categorical_datamodule_batch_is_structure_only() -> None:
    """SyntheticCategoricalDataModule batches are structure-only (x_class=None).

    Wave 9.3: structure-only datasets stop emitting a degenerate X_class
    that merely re-encodes ``node_mask``; architectures synthesise a
    per-node feature internally when needed.
    """
    from tmgg.data.data_modules.synthetic_categorical import (
        SyntheticCategoricalDataModule,
    )

    dm = SyntheticCategoricalDataModule(
        graph_type="er",
        num_nodes=8,
        num_graphs=8,
        train_ratio=0.5,
        val_ratio=0.25,
        graph_config={"p": 0.3},
        batch_size=4,
        num_workers=0,
        seed=0,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, GraphState)
    assert batch.x_class is None
    assert batch.edge_class is not None


def test_single_graph_datamodule_batch_is_structure_only() -> None:
    """SingleGraphDataModule batches are structure-only (x_class=None)."""
    from tmgg.data.data_modules.single_graph_data_module import SingleGraphDataModule

    dm = SingleGraphDataModule(
        graph_type="er",
        num_nodes=10,
        graph_config={"p": 0.3},
        num_train_samples=4,
        num_val_samples=2,
        num_test_samples=2,
        batch_size=2,
        num_workers=0,
        train_seed=0,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, GraphState)
    assert batch.x_class is None
    assert batch.edge_class is not None


def test_spectre_sbm_datamodule_batch_is_structure_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """SpectreSBMDataModule batches are structure-only (x_class=None)."""
    pytest.importorskip("torch_geometric")

    from tmgg.data.data_modules.spectre_sbm import SpectreSBMDataModule
    from tmgg.data.datasets.spectre_sbm import SPECTRE_SBM_TOTAL

    n = 10
    adjs: list[torch.Tensor] = []
    for i in range(SPECTRE_SBM_TOTAL):
        g = torch.Generator().manual_seed(i)
        triu = torch.bernoulli(torch.full((n, n), 0.3), generator=g).to(torch.float32)
        triu = torch.triu(triu, diagonal=1)
        adjs.append(triu + triu.T)

    fixture_path = tmp_path / "sbm_200_mock.pt"
    fixture_obj = [
        adjs,
        [torch.zeros(n) for _ in adjs],  # eigvals
        [torch.zeros(n, n) for _ in adjs],  # eigvecs
        [n] * len(adjs),
        2.0,  # max_eigval
        0.0,  # min_eigval
        False,  # same_sample
        n,  # n_max
    ]
    torch.save(fixture_obj, fixture_path)

    dm = SpectreSBMDataModule(
        batch_size=4,
        num_workers=0,
        fixture_path=str(fixture_path),
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))

    assert isinstance(batch, GraphState)
    assert batch.x_class is None
    assert batch.edge_class is not None

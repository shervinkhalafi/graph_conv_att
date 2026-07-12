"""Tests for the molecular DataModule batch shape + collator integration."""

from __future__ import annotations

from pathlib import Path

from tmgg.data.data_modules.molecular.base import MolecularDataModule
from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary


class _TinyDataset(MolecularGraphDataset):
    DATASET_NAME = "tiny_dm"
    DEFAULT_MAX_ATOMS = 5

    @classmethod
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.qm9(),
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    def download_smiles_split(self, split: str) -> list[str]:
        # 3 each: train/val/test
        return ["CCO", "CC(=O)O", "CCC"]


class _TinyDataModule(MolecularDataModule):
    dataset_cls = _TinyDataset


def test_dataloader_yields_graphdata(tmp_path: Path) -> None:
    """Datamodule emits a sparse ``GraphState``; check sparse shapes.

    Two molecules concatenated (PyG-flat layout): ``x_class`` has shape
    ``(sum_n, d_xc)`` and ``edge_class`` has shape ``(sum_E, d_ec)``.
    ``num_nodes_per_graph`` carries the per-graph node count and so its
    leading dim is the batch size.
    """
    from tmgg.data.datasets.graph_types import GraphState

    dm = _TinyDataModule(batch_size=2, cache_root=str(tmp_path))
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, GraphState)
    assert batch.x_class is not None
    assert batch.edge_class is not None
    assert batch.num_nodes_per_graph.shape == (2,)
    # Sparse layout: every node row in batch.x_class corresponds to one
    # real node across the two graphs, and its sum matches the
    # per-graph node count.
    assert batch.x_class.shape[0] == int(batch.num_nodes_per_graph.sum().item())


def test_size_distribution_populated(tmp_path: Path) -> None:
    dm = _TinyDataModule(batch_size=1, cache_root=str(tmp_path))
    dm.prepare_data()
    dm.setup()
    sd = dm.get_size_distribution("train")
    # Three molecules, sizes ∈ {3, 4, 3}.
    sample = sd.sample(100)
    assert sample.min() >= 3
    assert sample.max() <= 4


def test_train_dataloader_raw_pyg_yields_pyg_batch(tmp_path: Path) -> None:
    """``train_dataloader_raw_pyg`` must produce a PyG ``Batch``.

    Starting state: a tiny molecular DataModule with three SMILES.
    Invariant: the raw loader yields a ``torch_geometric.data.Batch``
    (the noise-process initialiser walks this once before training to
    estimate empirical (atom, bond)-class marginals). Without an
    override on ``MolecularDataModule``, this would have raised
    ``NotImplementedError`` at first iter — and previously,
    ``BaseGraphDataModule`` even allowed instantiation of subclasses
    without the override (caught only at preflight). This test pins
    both fixes (override + abstract enforcement upstream).
    """
    from torch_geometric.data import Batch

    dm = _TinyDataModule(batch_size=2, cache_root=str(tmp_path))
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader_raw_pyg()))
    assert isinstance(batch, Batch)
    assert hasattr(batch, "x")
    assert hasattr(batch, "edge_index")
    # Two molecules collated; node count should be sum of per-graph
    # node counts (sizes ∈ {3, 4, 3}, so first 2 → 3 + 4 = 7).
    assert batch.num_graphs == 2
    assert batch.num_nodes >= 6  # type: ignore[attr-defined]  # PyG runtime attr


def test_train_dataloader_raw_pyg_emits_one_hot_x_and_edge_attr(
    tmp_path: Path,
) -> None:
    """Raw-PyG batch must carry one-hot ``x`` + ``edge_attr``.

    Starting state: tiny molecular DataModule, three SMILES.

    Invariants (the contract every consumer of
    ``train_dataloader_raw_pyg`` already assumes — SBM/Planar satisfy
    it by *omitting* ``x``/``edge_attr`` and falling back to the
    fixed-class branch in :func:`count_node_classes_sparse` /
    :func:`count_edge_classes_sparse`; molecular populates them
    explicitly so per-class atom + bond histograms can be summed):

    1. ``batch.x`` has shape ``(sum_N, num_atom_types)`` and is
       float-valued one-hot (every row sums to 1.0). The previous
       implementation emitted 1-D ``argmax`` indices, which crashed
       :func:`count_node_classes_sparse` with ``IndexError: tuple
       index out of range`` on ``x.shape[1]``.
    2. ``batch.edge_attr`` has shape ``(E, num_bond_types)`` and is
       one-hot. ``count_edge_classes_sparse`` reads ``edge_attr.sum(
       dim=0)``; without the second axis it would crash the same way.
    3. Each row sums to exactly 1.0 (single-class one-hot) so the
       per-class counts coming out of the sparse counters add up to
       the total node / edge count.
    """
    import torch

    from tmgg.data.utils.edge_counts import (
        count_edge_classes_sparse,
        count_node_classes_sparse,
    )

    dm = _TinyDataModule(batch_size=2, cache_root=str(tmp_path))
    dm.prepare_data()
    dm.setup()
    codec = _TinyDataset.make_codec()
    num_atom_types = codec.vocab.num_atom_types
    num_bond_types = codec.vocab.num_bond_types

    batch = next(iter(dm.train_dataloader_raw_pyg()))
    # 1. x: (sum_N, num_atom_types) one-hot.
    assert (
        batch.x.dim() == 2
    ), f"x must be 2-D one-hot, got shape {tuple(batch.x.shape)}"
    assert batch.x.shape[1] == num_atom_types, (
        f"x second axis must equal vocab.num_atom_types={num_atom_types}, "
        f"got {batch.x.shape[1]}"
    )
    assert torch.allclose(
        batch.x.sum(dim=-1), torch.ones(batch.x.shape[0])
    ), "every row of x must sum to 1.0 (one-hot)"
    # 2. edge_attr: (E, num_bond_types) one-hot.
    assert (
        batch.edge_attr.dim() == 2
    ), f"edge_attr must be 2-D one-hot, got shape {tuple(batch.edge_attr.shape)}"
    assert batch.edge_attr.shape[1] == num_bond_types, (
        f"edge_attr second axis must equal vocab.num_bond_types={num_bond_types}, "
        f"got {batch.edge_attr.shape[1]}"
    )
    assert torch.allclose(
        batch.edge_attr.sum(dim=-1), torch.ones(batch.edge_attr.shape[0])
    ), "every row of edge_attr must sum to 1.0 (one-hot)"
    # 3. The sparse counters that crashed before now return histograms
    # whose totals match the underlying node / edge counts.
    node_counts = count_node_classes_sparse(batch, num_atom_types)
    assert node_counts.sum().item() == batch.x.shape[0], (
        f"per-class node counts ({node_counts.tolist()}) must sum to total "
        f"node count ({batch.x.shape[0]})"
    )
    edge_counts = count_edge_classes_sparse(batch, num_bond_types)
    # Edge counts include the implicit no-edge slot at index 0; classes 1+
    # should sum to the actual number of present edges in edge_index.
    present_edges = int(batch.edge_attr.shape[0])
    assert int(edge_counts[1:].sum().item()) == present_edges, (
        f"per-class present-edge counts ({edge_counts[1:].tolist()}) must "
        f"sum to len(edge_attr)={present_edges}"
    )


def test_basegraph_subclass_without_raw_pyg_override_cannot_instantiate() -> None:
    """Abstract-method enforcement: instantiation refused at object creation.

    Starting state: a synthetic subclass of ``BaseGraphDataModule`` that
    overrides ``setup``/``train_dataloader``/``val_dataloader``/
    ``test_dataloader`` but NOT ``train_dataloader_raw_pyg``.
    Invariant: ``Subclass()`` must raise ``TypeError`` from Python's
    abstract-class machinery, mentioning the missing method by name —
    so a future regression that drops the override surfaces at unit-
    test time, not during a Modal preflight on a remote A100.
    """
    import pytest

    from tmgg.data.data_modules.base_data_module import BaseGraphDataModule

    class _Incomplete(BaseGraphDataModule):
        def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
            pass

        def train_dataloader(self):  # type: ignore[no-untyped-def]
            pass

        def val_dataloader(self):  # type: ignore[no-untyped-def]
            pass

        def test_dataloader(self):  # type: ignore[no-untyped-def]
            pass

    with pytest.raises(TypeError, match="train_dataloader_raw_pyg"):
        # ``_Incomplete`` is abstract by design here; the test asserts
        # that Python's runtime refuses the instantiation.
        _Incomplete()  # pyright: ignore[reportAbstractUsage]

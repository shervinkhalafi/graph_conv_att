"""Tests for SpectrePlanarDataModule, mirroring the SBM tests' structure."""

from __future__ import annotations

import pytest

from tmgg.data.data_modules.spectre_planar import SpectrePlanarDataModule
from tmgg.data.datasets.spectre_planar import (
    SPECTRE_PLANAR_TEST_LEN,
    SPECTRE_PLANAR_TRAIN_LEN,
    SPECTRE_PLANAR_VAL_LEN,
    split_spectre_planar,
)


def test_split_sizes_match_spec() -> None:
    splits = split_spectre_planar()
    assert len(splits["train"]) == SPECTRE_PLANAR_TRAIN_LEN
    assert len(splits["val"]) == SPECTRE_PLANAR_VAL_LEN
    assert len(splits["test"]) == SPECTRE_PLANAR_TEST_LEN
    assert SPECTRE_PLANAR_TRAIN_LEN == 128
    assert SPECTRE_PLANAR_VAL_LEN == 32
    assert SPECTRE_PLANAR_TEST_LEN == 40


def test_setup_loads_fixture() -> None:
    """Requires fixture at ~/.cache/tmgg/spectre/planar_64_200.pt."""
    pytest.importorskip("torch_geometric")
    dm = SpectrePlanarDataModule()
    dm.setup("fit")
    assert dm.num_nodes == 64
    assert dm._train_data is not None
    assert len(dm._train_data) == SPECTRE_PLANAR_TRAIN_LEN


def test_dataloader_shapes() -> None:
    """Sanity-check that the collator produces a sparse ``GraphState``.

    Post the sparse-default refactor the planar collator emits a
    PyG-flat ``GraphState`` with a 4-graph batch and per-graph node
    counts of 64 (the planar fixture is fixed-size). Densifying to
    ``DenseGraphState`` recovers the legacy ``(bs, n_max, n_max, d_ec)``
    ``E_class`` shape — useful for the shape-pin invariant.
    """
    from tmgg.data.datasets.graph_types import GraphState, state_to_dense_sample

    pytest.importorskip("torch_geometric")
    dm = SpectrePlanarDataModule(batch_size=4)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, GraphState)
    assert batch.num_nodes_per_graph.shape == (4,)
    dense = state_to_dense_sample(batch)
    assert dense.E_class is not None
    assert dense.E_class.shape[0] == 4  # batch dim
    assert dense.E_class.shape[1] == 64  # n_max


def test_metadata_swallow_does_not_pollute_hparams() -> None:
    """Leaked Hydra keys must not enter ``self.hparams``.

    Starting state: instantiate the planar DM the way Hydra does when
    the planar experiment yaml inherits from the base config's inline
    SBM ``data:`` block — the unused SBM keys (``num_nodes``,
    ``graph_type``, ``num_graphs``, ``train_ratio``, ``val_ratio``,
    ``graph_config``, ``eval_meta``) get unpacked into ``**_metadata``
    on the planar DM constructor.

    Invariant: those keys MUST NOT appear in ``hparams_initial``.
    Otherwise Lightning's ``Trainer._log_hyperparams`` collision check
    raises when an experiment yaml overrides ``model.num_nodes`` to a
    value other than the leaked ``data.num_nodes=20`` (was caught by
    the Phase 8 planar Modal smoke run, run id ``2rvpiku0``).
    """
    dm = SpectrePlanarDataModule(
        batch_size=12,
        num_nodes=20,
        graph_type="sbm",
        num_graphs=2000,
        train_ratio=0.8,
        val_ratio=0.1,
        graph_config={"num_blocks": 2, "p_intra": 0.7, "p_inter": 0.1},
        eval_meta={"p_intra": 0.7, "p_inter": 0.1},
    )
    leaked_keys = {
        "num_nodes",
        "graph_type",
        "num_graphs",
        "train_ratio",
        "val_ratio",
        "graph_config",
        "eval_meta",
    }
    assert leaked_keys.isdisjoint(dm.hparams.keys()), (
        f"leaked keys appeared in hparams: " f"{leaked_keys & set(dm.hparams.keys())}"
    )

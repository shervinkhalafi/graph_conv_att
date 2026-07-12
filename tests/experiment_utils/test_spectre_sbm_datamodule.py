# pyright: reportAttributeAccessIssue=false
"""Tests for the SPECTRE SBM datamodule.

Covers:

- Split math matches upstream DiGress: 128/32/40 with the same
  ``torch.randperm`` seed (0).
- Fixture loader handles the real on-disk tuple structure and rejects
  malformed files.
- Datamodule ``setup`` is idempotent and populates per-split lists.
- ``get_reference_graphs("val", N)`` returns a list of NetworkX graphs
  with node counts drawn from the fixture's variable-n distribution.
- ``get_size_distribution("train")`` is non-degenerate (at least two
  distinct sizes, since SPECTRE graphs are 44-187 nodes).

The tests point at a tiny mock fixture (4 graphs) so they do not rely
on the real 20 MB file being downloaded. One optional test loads the
real fixture from a fixed local path when present (the session
downloaded it to ``/tmp/spectre-download/sbm_200.pt`` during setup) so
we exercise the real-data path at least once in local development,
skipping when the file is absent (e.g. in CI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from tmgg.data.data_modules.spectre_sbm import SpectreSBMDataModule
from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.datasets.spectre_sbm import (
    SPECTRE_SBM_TEST_LEN,
    SPECTRE_SBM_TOTAL,
    SPECTRE_SBM_TRAIN_LEN,
    SPECTRE_SBM_VAL_LEN,
    SpectreSBMDataset,
    adjacency_to_pyg_data,
    load_spectre_sbm_fixture,
    split_spectre_sbm,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _make_random_symmetric_adj(n: int, p: float = 0.3, seed: int = 0) -> torch.Tensor:
    """Generate an undirected binary adjacency for test fixtures.

    Rationale: tests use graphs with known size + rough density so the
    downstream checks (``num_edges > 0``, ``symmetric``, ``no
    self-loops``) are deterministic.
    """
    g = torch.Generator().manual_seed(seed)
    triu = torch.bernoulli(torch.full((n, n), p), generator=g).to(torch.float32)
    triu = torch.triu(triu, diagonal=1)
    return triu + triu.T


def _write_mock_fixture(path: Path, sizes: list[int]) -> None:
    """Serialise a minimal upstream-shaped fixture to *path*.

    The real fixture is a length-8 list
    ``[adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval,
    same_sample, n_max]``. Only indices 0 and 3 are load-bearing for
    the datamodule; other slots carry placeholder scalars so the
    structural check in ``load_spectre_sbm_fixture`` passes.
    """
    adjs = [_make_random_symmetric_adj(n, seed=i) for i, n in enumerate(sizes)]
    n_nodes = list(sizes)
    placeholder_tensor_lists: list[Any] = [
        [torch.zeros(n) for n in sizes] for _ in range(2)
    ]
    fixture = [
        adjs,
        placeholder_tensor_lists[0],  # eigvals
        placeholder_tensor_lists[1],  # eigvecs
        n_nodes,
        2.0,  # max_eigval
        0.0,  # min_eigval
        False,  # same_sample
        max(sizes),  # n_max
    ]
    torch.save(fixture, path)


# ---------------------------------------------------------------------
# Split math
# ---------------------------------------------------------------------


class TestSplitMath:
    def test_split_sizes_match_upstream(self) -> None:
        splits = split_spectre_sbm()
        assert len(splits["train"]) == SPECTRE_SBM_TRAIN_LEN == 128
        assert len(splits["val"]) == SPECTRE_SBM_VAL_LEN == 32
        assert len(splits["test"]) == SPECTRE_SBM_TEST_LEN == 40
        assert (
            SPECTRE_SBM_TRAIN_LEN + SPECTRE_SBM_VAL_LEN + SPECTRE_SBM_TEST_LEN
            == SPECTRE_SBM_TOTAL
        )

    def test_splits_disjoint_and_cover_all_indices(self) -> None:
        splits = split_spectre_sbm()
        all_indices = splits["train"] + splits["val"] + splits["test"]
        assert len(all_indices) == SPECTRE_SBM_TOTAL
        assert len(set(all_indices)) == SPECTRE_SBM_TOTAL
        assert set(all_indices) == set(range(SPECTRE_SBM_TOTAL))

    def test_split_is_deterministic_under_same_seed(self) -> None:
        """Upstream DiGress uses seed 0; regenerating with the same seed
        must reproduce the split byte-for-byte, otherwise parity claims
        are invalid.
        """
        a = split_spectre_sbm()
        b = split_spectre_sbm()
        assert a == b

    def test_split_first_train_indices_are_upstream_fixed(self) -> None:
        """Locks in the first five training indices so an accidental
        change to ``split_spectre_sbm`` is caught as a regression.
        The values come from ``torch.randperm(200, seed=0)`` and were
        observed during the 2026-04-15 integration.
        """
        splits = split_spectre_sbm()
        assert splits["train"][:5] == [44, 57, 157, 64, 19]


# ---------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------


class TestFixtureLoader:
    def test_loads_mock_fixture(self, tmp_path: Path) -> None:
        fixture = tmp_path / "sbm_mock.pt"
        _write_mock_fixture(fixture, sizes=[5, 6, 7, 8])
        adjs, n_nodes = load_spectre_sbm_fixture(path=fixture)
        assert len(adjs) == 4
        assert n_nodes == [5, 6, 7, 8]
        # Shape agreement — n_nodes[i] must equal adjs[i].shape[0].
        for adj, n in zip(adjs, n_nodes, strict=True):
            assert adj.shape == (n, n)

    def test_rejects_malformed_fixture(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pt"
        torch.save({"not": "a list"}, path)
        with pytest.raises(RuntimeError, match="Unexpected SPECTRE fixture"):
            _ = load_spectre_sbm_fixture(path=path)

    def test_rejects_fixture_with_length_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "mismatch.pt"
        fixture = [
            [torch.zeros(3, 3), torch.zeros(4, 4)],  # 2 adjs
            [],  # eigvals
            [],  # eigvecs
            [3, 4, 5],  # n_nodes — length 3, inconsistent
            0.0,
            0.0,
            False,
            5,
        ]
        torch.save(fixture, path)
        with pytest.raises(RuntimeError, match="inconsistent"):
            _ = load_spectre_sbm_fixture(path=path)


# ---------------------------------------------------------------------
# Dataset and datamodule
# ---------------------------------------------------------------------


class TestSpectreSBMDataset:
    def test_indexable_and_converts_to_pyg(self, tmp_path: Path) -> None:
        adj = _make_random_symmetric_adj(10, p=0.3, seed=42)
        ds = SpectreSBMDataset([adj, adj.clone()])
        assert len(ds) == 2
        data = ds[0]
        assert data.num_nodes == 10
        edge_index = data.edge_index
        assert edge_index is not None
        assert edge_index.shape[0] == 2
        # Symmetry of ``edge_index``: every edge (u, v) appears twice.
        assert edge_index.shape[1] % 2 == 0

    def test_rejects_empty_list(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _ = SpectreSBMDataset([])


class TestSpectreSBMDataModule:
    def _make_dm(self, tmp_path: Path) -> SpectreSBMDataModule:
        # A minimal 200-graph fixture with all identical 10-node cliques
        # — size distribution is degenerate, but every downstream path
        # still exercises correctly. The real fixture has variable n;
        # see ``test_real_fixture_via_local_cache`` below.
        fixture = tmp_path / "sbm_200_mock.pt"
        _write_mock_fixture(fixture, sizes=[10] * SPECTRE_SBM_TOTAL)
        return SpectreSBMDataModule(
            batch_size=4,
            num_workers=0,
            fixture_path=str(fixture),
        )

    def test_setup_is_idempotent(self, tmp_path: Path) -> None:
        dm = self._make_dm(tmp_path)
        dm.setup("fit")
        first_train = dm._train_data  # type: ignore[attr-defined]
        dm.setup("fit")
        # Second call must not re-run the loader; list identity stable.
        assert dm._train_data is first_train  # type: ignore[attr-defined]

    def test_split_lengths_match_upstream(self, tmp_path: Path) -> None:
        dm = self._make_dm(tmp_path)
        dm.setup("fit")
        assert len(dm._train_data) == SPECTRE_SBM_TRAIN_LEN  # type: ignore[arg-type]
        assert len(dm._val_data) == SPECTRE_SBM_VAL_LEN  # type: ignore[arg-type]
        assert len(dm._test_data) == SPECTRE_SBM_TEST_LEN  # type: ignore[arg-type]

    def test_dataloaders_yield_graphdata_batches(self, tmp_path: Path) -> None:
        dm = self._make_dm(tmp_path)
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))
        # Wave 9.3 + 2026-05-07 sparse-default refactor: structure-only
        # datasets emit ``GraphState`` (sparse) batches, with ``x_class``
        # absent and ``edge_class`` populated. Batch size comes from the
        # length of ``num_nodes_per_graph``.
        assert batch.x_class is None
        assert batch.edge_class is not None
        assert batch.num_nodes_per_graph.shape[0] == 4  # batch size
        assert batch.edge_class.shape[-1] == 2  # edge categorical (present/absent)

    def test_get_reference_graphs_returns_networkx(self, tmp_path: Path) -> None:
        dm = self._make_dm(tmp_path)
        dm.setup("fit")
        refs = dm.get_reference_graphs("val", max_graphs=3)
        assert len(refs) == 3
        for gd in refs:
            # Per 2026-05-01 universal-transport refactor: GraphData not nx.
            assert isinstance(gd, GraphData)
            assert int(gd.node_mask.sum().item()) == 10  # mock fixture node count

    def test_size_distribution_exposes_sizes(self, tmp_path: Path) -> None:
        dm = self._make_dm(tmp_path)
        dm.setup("fit")
        dist = dm.get_size_distribution("train")
        # Mock fixture has all 10-node graphs -> a single size.
        assert tuple(dist.sizes) == (10,)


_REAL_FIXTURE = Path("/tmp/spectre-download/sbm_200.pt")


@pytest.mark.skipif(
    not _REAL_FIXTURE.exists(),
    reason=f"Real SPECTRE fixture not available at {_REAL_FIXTURE}",
)
class TestRealFixture:
    """Smoke test against the actual SPECTRE fixture when present.

    Protects against the mock fixture diverging from the real file
    format. Skipped automatically on CI where the file is not provided.
    """

    def test_real_fixture_has_variable_sizes_and_produces_nontrivial_split(
        self,
    ) -> None:
        adjs, n_nodes = load_spectre_sbm_fixture(path=_REAL_FIXTURE)
        assert len(adjs) == SPECTRE_SBM_TOTAL
        assert len(set(n_nodes)) > 10  # variable — not a single size
        assert min(n_nodes) >= 40
        assert max(n_nodes) <= 200

    def test_real_fixture_drives_datamodule(self) -> None:
        dm = SpectreSBMDataModule(
            batch_size=4,
            num_workers=0,
            fixture_path=str(_REAL_FIXTURE),
        )
        dm.setup("fit")
        dist = dm.get_size_distribution("train")
        # Real SPECTRE has dozens of distinct sizes.
        assert len(dist.sizes) > 20


class TestAdjacencyConversion:
    def test_single_adjacency_to_pyg_preserves_structure(self) -> None:
        adj = _make_random_symmetric_adj(6, p=0.5, seed=7)
        data = adjacency_to_pyg_data(adj)
        assert data.num_nodes == 6
        # Edge count should equal twice the number of upper-triangular
        # ones (undirected graphs emit both directions in ``edge_index``).
        expected = int(torch.triu(adj, diagonal=1).sum().item()) * 2
        edge_index = data.edge_index
        assert edge_index is not None
        assert edge_index.shape[1] == expected

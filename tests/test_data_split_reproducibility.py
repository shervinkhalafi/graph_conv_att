"""Data splits must be reproducible given the same seed.

Rationale
---------
``data_module.py`` used Python's global ``random.shuffle``/``random.sample``
without passing a seed.  Two runs with the same config therefore produced
different train/val/test splits.  The fix introduces ``self._rng``, a
``random.Random(seed)`` instance, so that all partition sampling and matrix
shuffling become deterministic for a given seed.

Starting state: two freshly constructed ``GraphDataModule`` instances with
identical parameters (including ``seed``).

Invariants tested:
    - Identical seeds produce identical train/val/test Data objects.
    - The SBM adjacency generation is also seeded, so the *content* of the
      graphs (not just the partition selection) matches.
"""

from __future__ import annotations

from typing import Any

import torch


def test_sbm_splits_reproducible():
    """Two GraphDataModule instances with the same seed produce identical splits."""
    from tmgg.data.data_modules.data_module import GraphDataModule

    common: dict[str, Any] = dict(
        graph_type="sbm",
        graph_config={
            "num_nodes": 12,
            "p_intra": 0.8,
            "p_inter": 0.1,
            "min_blocks": 2,
            "max_blocks": 3,
            "min_block_size": 2,
            "max_block_size": 8,
            "num_train_partitions": 5,
            "num_test_partitions": 5,
        },
        batch_size=4,
        num_workers=0,
        seed=42,
        samples_per_graph=1,  # No repetition for cleaner comparison
    )

    dm1 = GraphDataModule(**common)
    dm1.prepare_data()
    dm1.setup()

    dm2 = GraphDataModule(**common)
    dm2.prepare_data()
    dm2.setup()

    # --- train ---
    assert dm1._train_data is not None  # pyright: ignore[reportPrivateUsage]
    assert dm2._train_data is not None  # pyright: ignore[reportPrivateUsage]
    assert len(dm1._train_data) == len(dm2._train_data)  # pyright: ignore[reportPrivateUsage]
    for i, (a, b) in enumerate(
        zip(dm1._train_data, dm2._train_data, strict=False)  # pyright: ignore[reportPrivateUsage]
    ):
        assert a.edge_index is not None and b.edge_index is not None
        assert torch.equal(
            a.edge_index, b.edge_index
        ), f"Train graph {i} differs between runs with same seed"

    # --- val ---
    assert dm1._val_data is not None  # pyright: ignore[reportPrivateUsage]
    assert dm2._val_data is not None  # pyright: ignore[reportPrivateUsage]
    assert len(dm1._val_data) == len(dm2._val_data)  # pyright: ignore[reportPrivateUsage]
    for i, (a, b) in enumerate(
        zip(dm1._val_data, dm2._val_data, strict=False)  # pyright: ignore[reportPrivateUsage]
    ):
        assert a.edge_index is not None and b.edge_index is not None
        assert torch.equal(
            a.edge_index, b.edge_index
        ), f"Val graph {i} differs between runs with same seed"

    # --- test ---
    assert dm1._test_data is not None  # pyright: ignore[reportPrivateUsage]
    assert dm2._test_data is not None  # pyright: ignore[reportPrivateUsage]
    assert len(dm1._test_data) == len(dm2._test_data)  # pyright: ignore[reportPrivateUsage]
    for i, (a, b) in enumerate(
        zip(dm1._test_data, dm2._test_data, strict=False)  # pyright: ignore[reportPrivateUsage]
    ):
        assert a.edge_index is not None and b.edge_index is not None
        assert torch.equal(
            a.edge_index, b.edge_index
        ), f"Test graph {i} differs between runs with same seed"


def test_different_seeds_produce_different_splits():
    """Different seeds should (with overwhelming probability) produce different splits."""
    from tmgg.data.data_modules.data_module import GraphDataModule

    base: dict[str, Any] = dict(
        graph_type="sbm",
        graph_config={
            "num_nodes": 12,
            "p_intra": 0.8,
            "p_inter": 0.1,
            "min_blocks": 2,
            "max_blocks": 3,
            "min_block_size": 2,
            "max_block_size": 8,
            "num_train_partitions": 5,
            "num_test_partitions": 5,
        },
        batch_size=4,
        num_workers=0,
        samples_per_graph=1,
    )

    dm1 = GraphDataModule(**base, seed=42)
    dm1.prepare_data()
    dm1.setup()

    dm2 = GraphDataModule(**base, seed=99)
    dm2.prepare_data()
    dm2.setup()

    assert dm1._train_data is not None  # pyright: ignore[reportPrivateUsage]
    assert dm2._train_data is not None  # pyright: ignore[reportPrivateUsage]

    # At least one graph should differ (the SBM adjacency generation is random)
    def _edges_equal(a: object, b: object) -> bool:
        ea = getattr(a, "edge_index", None)
        eb = getattr(b, "edge_index", None)
        assert ea is not None, "PyG Data.edge_index must be populated"
        assert eb is not None, "PyG Data.edge_index must be populated"
        return torch.equal(ea, eb)

    any_differ = any(
        not _edges_equal(a, b)
        for a, b in zip(
            dm1._train_data,
            dm2._train_data,
            strict=False,  # pyright: ignore[reportPrivateUsage]
        )
    )
    assert (
        any_differ
    ), "Different seeds produced identical splits -- seeding is likely broken"

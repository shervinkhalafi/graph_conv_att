"""End-to-end smoke for the sparse-default refactor.

Loads the SPECTRE SBM datamodule and confirms that a batch is a
:class:`GraphState` (the canonical sparse carrier). Exists to pin the
post-refactor datamodule emit contract; downstream training and
evaluation pipelines depend on this carrier choice.
"""

from __future__ import annotations

import pytest

from tmgg.data.data_modules.spectre_sbm import SpectreSBMDataModule
from tmgg.data.datasets.graph_types import GraphState


@pytest.mark.smoke
def test_e2e_sbm_batch_is_graphstate() -> None:
    """SBM datamodule emits ``GraphState`` batches with sane shapes."""
    dm = SpectreSBMDataModule(batch_size=3, num_workers=0, pin_memory=False)
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert isinstance(batch, GraphState), (
        f"Expected GraphState, got {type(batch).__name__}"
    )

    # Spot-check shapes: ``num_nodes_per_graph`` indexes per-graph
    # node counts; ``edge_index`` is PyG-flat (2, sum_E).
    bs = int(batch.num_nodes_per_graph.shape[0])
    assert bs == 3 or bs == loader.batch_size
    assert batch.edge_index.shape[0] == 2

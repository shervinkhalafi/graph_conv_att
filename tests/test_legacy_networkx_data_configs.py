"""Regression tests for the legacy `nx*` data presets.

Test rationale
--------------
These configs used to target an older NetworkX dispatch layer that accepted
``graph_type: nx`` plus ad hoc ``clsname`` and ``repeat`` keys. The current
synthetic graph surface exposes the concrete topology names directly
(``ring_of_cliques``, ``square_grid``, ``star``) and rejects the old
dispatcher. This test keeps the legacy preset filenames working by asserting
they still compose into a valid GraphDataModule and complete a tiny setup.
"""

from collections.abc import Generator
from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import tmgg  # noqa: F401 - registers OmegaConf resolvers used by config tests


@pytest.fixture(autouse=True)
def clear_hydra() -> Generator[None, None, None]:
    """Clear Hydra global state before and after each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def exp_config_path() -> Path:
    """Return the experiment config root."""
    return Path(__file__).parent.parent / "src" / "tmgg" / "experiments" / "exp_configs"


def _minimal_overrides(tmp_path: Path) -> list[str]:
    """Return a tiny, logger-free override set for config setup tests."""
    return [
        f"paths.output_dir={tmp_path}",
        f"paths.results_dir={tmp_path}/results",
        f"hydra.run.dir={tmp_path}",
        "trainer.max_steps=1",
        "trainer.val_check_interval=1",
        "trainer.accelerator=cpu",
        "~logger",
        "data.batch_size=1",
        "data.num_workers=0",
        "++data.samples_per_graph=1",
        "++data.val_samples_per_graph=1",
    ]


@pytest.mark.integration
@pytest.mark.parametrize("data_config", ["nx", "nx_square", "nx_star"])
def test_legacy_networkx_presets_setup_successfully(
    data_config: str,
    exp_config_path: Path,
    tmp_path: Path,
) -> None:
    """Each legacy preset should map cleanly onto the current graph generator.

    The invariant here is stronger than mere YAML parsing: Hydra composition,
    datamodule instantiation, and ``setup('fit')`` must all succeed for the
    tiny override set. That catches stale graph-type names and stale graph
    parameter keys in one place.
    """
    overrides = _minimal_overrides(tmp_path)
    overrides.append(f"data={data_config}")

    with initialize_config_dir(version_base=None, config_dir=str(exp_config_path)):
        cfg = compose(config_name="base_config_gnn", overrides=overrides)

    data_module = hydra.utils.instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup("fit")

    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    # Sparse-default refactor: the dataloader emits ``GraphState`` (sparse).
    # Structure-only presets carry ``x_class is None``; per-graph node count
    # is read directly off ``num_nodes_per_graph``.
    assert batch.x_class is None
    assert batch.num_nodes_per_graph.shape[0] == 1

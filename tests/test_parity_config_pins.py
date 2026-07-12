"""Pin the parity-flip config defaults per the 2026-04-23 review.

Eight commits in the fix-parity wave changed defaults to match upstream
DiGress; none shipped with a regression test pinning the new value. A
future YAML edit could silently restore the upstream-divergent default.
This file pins each new default. On bisect, the parametrize id names
the offending pin.

The pins cover:
- parity #35 — ``GraphEvaluator.sbm_refinement_steps`` constructor default 100
- parity #39 — ``trainer/default.yaml::gradient_clip_val = null``
- parity #40 — ``base/callbacks/default.yaml`` early stopping patience 1000 / save_top_k -1
- parity #43 — ``models/discrete/discrete_sbm_official.yaml`` timesteps 1000
- parity D-9  — ``_base_infra.yaml`` scheduler default ``none``
- parity D-11 — per-data-config ``num_nodes_max_static``

Parity #4 (``from_pyg_batch`` self-loop strip) and #5 (train+val node
count) are structural, with regression tests in
``tests/data/test_graph_data_invariants.py``. Parity #8 (K=5 default)
is pinned in ``tests/test_beta_schedule_edge_classes.py``. Those are
intentionally not duplicated here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("config_path", "key", "expected"),
    [
        # parity #39 — gradient_clip_val null (configs/train/train_default.yaml:5)
        pytest.param(
            "src/tmgg/experiments/exp_configs/base/trainer/default.yaml",
            "gradient_clip_val",
            None,
            id="parity_39_trainer_gradient_clip_val_null",
        ),
        # parity #40 — early-stopping patience 1000 (no upstream ES; act as safety net)
        pytest.param(
            "src/tmgg/experiments/exp_configs/base/callbacks/default.yaml",
            "callbacks.early_stopping.patience",
            1000,
            id="parity_40_callbacks_early_stopping_patience_1000",
        ),
        # parity #40 — save_top_k -1 to support D-16c evaluate_all_checkpoints
        pytest.param(
            "src/tmgg/experiments/exp_configs/base/callbacks/default.yaml",
            "callbacks.checkpoint.save_top_k",
            -1,
            id="parity_40_callbacks_checkpoint_save_top_k_minus_one",
        ),
        # parity #43 — timesteps 1000 (configs/experiment/sbm.yaml:21)
        pytest.param(
            "src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_official.yaml",
            "noise_schedule.timesteps",
            1000,
            id="parity_43_discrete_sbm_official_timesteps_1000",
        ),
        # parity D-9 — scheduler default 'none' (no LR scheduler upstream)
        pytest.param(
            "src/tmgg/experiments/exp_configs/_base_infra.yaml",
            "scheduler_config.type",
            "none",
            id="parity_d9_base_infra_scheduler_none",
        ),
        # parity D-11 — per-data-config num_nodes_max_static
        pytest.param(
            "src/tmgg/experiments/exp_configs/data/sbm_default.yaml",
            "num_nodes_max_static",
            20,
            id="parity_d11_sbm_default_num_nodes_max_static_20",
        ),
        pytest.param(
            "src/tmgg/experiments/exp_configs/data/sbm_n100.yaml",
            "num_nodes_max_static",
            100,
            id="parity_d11_sbm_n100_num_nodes_max_static_100",
        ),
        pytest.param(
            "src/tmgg/experiments/exp_configs/data/sbm_n200.yaml",
            "num_nodes_max_static",
            200,
            id="parity_d11_sbm_n200_num_nodes_max_static_200",
        ),
        pytest.param(
            "src/tmgg/experiments/exp_configs/data/spectre_sbm.yaml",
            "num_nodes_max_static",
            200,
            id="parity_d11_spectre_sbm_num_nodes_max_static_200",
        ),
    ],
)
def test_parity_config_pin(config_path: str, key: str, expected: Any) -> None:
    """A YAML key pins to the upstream-parity default.

    A failing pin means a future commit silently broke parity. The
    parametrize id identifies the affected (file, key) under bisect.
    """
    full_path = REPO_ROOT / config_path
    assert full_path.exists(), f"config path missing: {full_path}"
    cfg = OmegaConf.load(full_path)
    actual = OmegaConf.select(cfg, key)
    assert actual == expected, (
        f"parity pin broken: {config_path}:{key} "
        f"expected={expected!r} got={actual!r}"
    )


def test_parity_35_graph_evaluator_sbm_refinement_steps_default() -> None:
    """``GraphEvaluator.sbm_refinement_steps`` defaults to upstream's effective 100.

    Upstream DiGress's ``compute_sbm_accuracy`` declares a 1000-step
    default but every live caller (``SpectreSamplingMetrics.forward``
    in ``spectre_utils.py:830``) overrides to 100. The constructor
    default in :class:`GraphEvaluator` matches the live override per
    parity #35; this test pins that constructor default so a future
    refactor cannot silently revert it.
    """
    from tmgg.evaluation.graph_evaluator import GraphEvaluator

    evaluator = GraphEvaluator(eval_num_samples=1)
    assert evaluator.sbm_refinement_steps == 100

"""Cross-yaml callback-wiring check for the discrete-diffusion repro panel.

For every ``configs/experiment/discrete_*_repro.yaml`` we resolve the
config through Hydra and assert that the resolved ``cfg.callbacks``
dict carries:

1. ``early_stopping`` and ``checkpoint`` entries that monitor
   ``val/epoch_NLL`` (the discrete-diffusion convention — the proper
   variational ELBO, not the per-step weighted CE in ``val/loss``).
2. An ``async_eval_spawn`` entry resolvable through
   ``hydra.utils.instantiate``. Without it, the trainer emits no
   ``gen-val/*`` metrics regardless of ``eval_every_n_steps`` / val
   schedule, because the spawn callback is the only consumer of the
   schedule.

Why this test exists
--------------------
On 2026-04-30 a panel run lost ~12 GPU-hours of would-be gen-val/*
data because ``base_config_discrete_diffusion_generative`` overrode
``base/callbacks`` to ``discrete_nll`` (which has no spawn callback)
instead of the new ``discrete_nll_with_async_eval`` (which composes
``discrete_nll + async_eval_spawn``). Training itself stayed healthy
— val/epoch_NLL improved monotonically and best ckpts were persisted
— but no async-eval call ever fired, so wandb summaryMetrics had zero
``gen-val/*`` keys ~80 min past the first scheduled trigger.

This test fails fast at unit-test time so the next person who tweaks
the discrete-diffusion callback group cannot ship that bug class
again. It also pins the val/epoch_NLL monitor convention so a future
"clean up" that swaps to ``default_with_async_eval`` (which monitors
``val/loss``) immediately surfaces as a behavioural change.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPRO_NAMES = (
    "discrete_sbm_vignac_repro",
    "discrete_planar_digress_repro",
    "discrete_qm9_digress_repro",
    "discrete_moses_digress_repro",
    "discrete_guacamol_digress_repro",
)


@pytest.fixture(autouse=True)
def _hydra_global_clear():
    """Hydra global state survives across tests; clear before + after."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def _config_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tmgg"
        / "experiments"
        / "exp_configs"
    )


@pytest.mark.parametrize("name", REPRO_NAMES)
def test_repro_yaml_callbacks_have_async_eval_and_nll_monitor(
    name: str,
) -> None:
    """Every panel yaml must compose discrete-NLL + async-eval callbacks."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(_config_dir()),
    ):
        cfg = compose(
            config_name="base_config_discrete_diffusion_generative",
            overrides=[f"+experiment={name}"],
        )

    cb = cfg.callbacks

    # 1. Discrete-NLL monitor on early_stopping + checkpoint.
    assert "early_stopping" in cb, (
        f"[{name}] cfg.callbacks.early_stopping missing — "
        "discrete_nll callback group not composed in"
    )
    assert cb.early_stopping.monitor == "val/epoch_NLL", (
        f"[{name}] cfg.callbacks.early_stopping.monitor="
        f"{cb.early_stopping.monitor!r} disagrees with discrete-diffusion "
        "convention (val/epoch_NLL). If this is intentional you are "
        "deviating from the variational-ELBO ckpt-selection criterion."
    )
    assert "checkpoint" in cb, f"[{name}] cfg.callbacks.checkpoint missing"
    assert cb.checkpoint.monitor == "val/epoch_NLL", (
        f"[{name}] cfg.callbacks.checkpoint.monitor="
        f"{cb.checkpoint.monitor!r} must match discrete-NLL convention"
    )

    # 2. Async-eval spawn callback present (gen-val/* emission gate).
    assert "async_eval_spawn" in cb, (
        f"[{name}] cfg.callbacks.async_eval_spawn missing — the "
        "trainer will emit zero gen-val/* metrics regardless of "
        "eval_every_n_steps. Bug class fingerprint: 'training healthy "
        "but no gen-val/* on wandb'. Compose discrete_nll_with_async_eval."
    )
    aes = cb.async_eval_spawn
    # The spawn callback can be enabled/disabled per-run; we only check
    # it is wired in. The launcher is responsible for setting enabled=true
    # plus a non-empty schedule on actual eval runs.
    assert "_target_" in aes, (
        f"[{name}] async_eval_spawn block has no _target_; "
        "hydra.utils.instantiate would fail at runtime"
    )
    assert (
        aes._target_
        == "tmgg.training.callbacks.async_eval_spawn.AsyncEvalSpawnCallback"
    ), (
        f"[{name}] async_eval_spawn._target_={aes._target_!r} "
        "diverges from canonical AsyncEvalSpawnCallback path"
    )

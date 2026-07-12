"""Hydra compose tests for the async-eval callback group.

Test rationale
--------------
Step 4 of the async-eval plan
(``internal async-eval design notes``) introduces
two new Hydra configs:

* ``base/callbacks/async_eval_spawn.yaml`` — the
  ``AsyncEvalSpawnCallback`` payload, instantiable via
  ``hydra.utils.instantiate(cfg.callbacks.async_eval_spawn)``.
* ``base/callbacks/default_with_async_eval.yaml`` — composes the
  existing ``default`` callback group with the new
  ``async_eval_spawn`` config so a launcher can enable async eval with
  a single override:
  ``base/callbacks=default_with_async_eval
  callbacks.async_eval_spawn.enabled=true ...``. The override path
  matches the defaults entry in ``_base_infra.yaml``
  (``base/callbacks: default``); a plain ``callbacks=...`` would not
  resolve to this nested group.

These tests verify the compose path resolves cleanly and that the
resulting config carries both the new ``_target_`` for the async-eval
callback and the preserved keys (``early_stopping``, ``checkpoint``)
that ``create_callbacks()`` reads.

Assumed starting state
----------------------
* ``base_config_spectral_arch`` is a known-good entry-point config
  (mirrors ``tests/test_config_composition.py``).
* ``run_id`` is set explicitly via override since
  ``async_eval_spawn.run_uid: ${run_id}`` would otherwise rely on
  runtime resolution.

Invariants
----------
* Composing with ``base/callbacks=default_with_async_eval`` succeeds.
* ``cfg.callbacks.async_eval_spawn._target_`` points at the callback
  class.
* ``cfg.callbacks.early_stopping`` and ``cfg.callbacks.checkpoint``
  remain present (default callbacks not lost).
* Override of ``callbacks.async_eval_spawn.schedule=[1000,2000]``
  populates the list; ``enabled=true`` flips the gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback
from tmgg.training.orchestration.run_experiment import create_callbacks

EXP_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "src" / "tmgg" / "experiments" / "exp_configs"
)


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _minimal_overrides(tmp_path: Path) -> list[str]:
    """Match ``tests/test_config_composition.get_minimal_overrides``."""
    return [
        f"paths.output_dir={tmp_path}",
        f"paths.results_dir={tmp_path}/results",
        "trainer.max_steps=2",
        "trainer.val_check_interval=1",
        "trainer.accelerator=cpu",
        "~logger",
        "data.batch_size=2",
        "data.num_workers=0",
        f"hydra.run.dir={tmp_path}",
        # ``run_id`` is auto-generated at runtime, but the
        # async_eval_spawn YAML interpolates ``${run_id}`` for ``run_uid``.
        # Pin it for the compose-time resolution.
        "+run_id=test-run-id",
    ]


def test_default_with_async_eval_composes(tmp_path: Path) -> None:
    """The ``default_with_async_eval`` group resolves and carries the new target."""
    overrides = _minimal_overrides(tmp_path) + [
        "base/callbacks=default_with_async_eval",
        "callbacks.async_eval_spawn.enabled=true",
        "callbacks.async_eval_spawn.schedule=[1000, 2000]",
        "callbacks.async_eval_spawn.run_uid=test-uid",
        "callbacks.async_eval_spawn.manifest_path=/tmp/test_manifest.jsonl",
    ]

    with initialize_config_dir(version_base=None, config_dir=str(EXP_CONFIG_PATH)):
        cfg = compose(
            config_name="base_config_spectral_arch",
            overrides=overrides,
        )

    # New callback registered with the expected target.
    assert "async_eval_spawn" in cfg.callbacks, (
        "default_with_async_eval should add an async_eval_spawn entry; "
        f"got keys: {list(cfg.callbacks.keys())}"
    )
    assert (
        cfg.callbacks.async_eval_spawn._target_
        == "tmgg.training.callbacks.async_eval_spawn.AsyncEvalSpawnCallback"
    )
    assert cfg.callbacks.async_eval_spawn.enabled is True
    assert list(cfg.callbacks.async_eval_spawn.schedule) == [1000, 2000]
    assert cfg.callbacks.async_eval_spawn.run_uid == "test-uid"
    assert cfg.callbacks.async_eval_spawn.manifest_path == "/tmp/test_manifest.jsonl"

    # Existing default callback keys must survive composition.
    assert "early_stopping" in cfg.callbacks, (
        "default callbacks lost after composing default_with_async_eval; "
        f"got keys: {list(cfg.callbacks.keys())}"
    )
    assert "checkpoint" in cfg.callbacks
    # Sanity-check default values still flow through.
    assert cfg.callbacks.early_stopping.monitor == "val/loss"
    assert cfg.callbacks.checkpoint.monitor == "val/loss"


def test_async_eval_spawn_defaults(tmp_path: Path) -> None:
    """Defaults in async_eval_spawn.yaml match the callback's __init__ defaults."""
    overrides = _minimal_overrides(tmp_path) + [
        "base/callbacks=default_with_async_eval",
    ]

    with initialize_config_dir(version_base=None, config_dir=str(EXP_CONFIG_PATH)):
        cfg = compose(
            config_name="base_config_spectral_arch",
            overrides=overrides,
        )

    aes = cfg.callbacks.async_eval_spawn
    assert aes.enabled is False, "async_eval_spawn must default to disabled"
    assert list(aes.schedule) == [], "schedule defaults to empty list"
    assert aes.gpu_tier == "standard"
    assert aes.eval_drain_idle_timeout_s == 900
    assert aes.eval_drain_poll_s == 30
    assert aes.keep_step_checkpoints is True
    assert aes.num_samples == 40
    assert aes.num_steps == 1000
    assert aes.modal_app_name == "graph-denoising"


def test_async_eval_spawn_callback_wired_when_enabled(tmp_path: Path) -> None:
    """``create_callbacks`` should append an ``AsyncEvalSpawnCallback`` when the
    ``callbacks.async_eval_spawn`` block is composed and ``enabled=true``.

    Regression rationale
    --------------------
    Before the fix, ``create_callbacks`` only read a hardcoded set of
    keys (``checkpoint``, ``early_stopping``, ``ema_decay``,
    ``final_sample_dump``, ``chain_saving``); the ``async_eval_spawn``
    block was silently dropped even when present in the resolved config
    and gated open. This regressed the smallest-config smoke run: the
    trainer ran 1000 steps cleanly, but no eval manifest was written and
    no step-stamped checkpoints were saved. The fix wires the block
    through ``hydra.utils.instantiate`` after stripping the ``enabled``
    gate (which is not an ``__init__`` kwarg of the callback).
    """
    overrides = _minimal_overrides(tmp_path) + [
        "base/callbacks=default_with_async_eval",
        "callbacks.async_eval_spawn.enabled=true",
        "callbacks.async_eval_spawn.schedule=[1000, 2000]",
        "callbacks.async_eval_spawn.run_uid=test-uid",
        "callbacks.async_eval_spawn.manifest_path=/tmp/test_manifest.jsonl",
        # ``~logger`` is already in _minimal_overrides; no need to mock W&B.
    ]

    with initialize_config_dir(version_base=None, config_dir=str(EXP_CONFIG_PATH)):
        cfg = compose(
            config_name="base_config_spectral_arch",
            overrides=overrides,
        )

    callbacks = create_callbacks(cfg)
    aes_callbacks = [cb for cb in callbacks if isinstance(cb, AsyncEvalSpawnCallback)]
    assert len(aes_callbacks) == 1, (
        f"expected exactly one AsyncEvalSpawnCallback in the returned list; "
        f"got {len(aes_callbacks)}. Full callback types: "
        f"{[type(cb).__name__ for cb in callbacks]}"
    )
    aes = aes_callbacks[0]
    assert aes.schedule == [1000, 2000]
    assert aes.run_uid == "test-uid"
    assert aes.manifest_path == "/tmp/test_manifest.jsonl"


def test_async_eval_spawn_callback_skipped_when_disabled(tmp_path: Path) -> None:
    """``create_callbacks`` must NOT append an ``AsyncEvalSpawnCallback``
    when the block is composed but ``enabled=false`` (the default).

    Regression rationale
    --------------------
    The gate must remain a hard off-switch. A launcher that composes
    ``base/callbacks=default_with_async_eval`` without flipping
    ``enabled=true`` should get the default callbacks plus nothing else;
    otherwise we'd be auto-spawning Modal jobs every run.
    """
    overrides = _minimal_overrides(tmp_path) + [
        "base/callbacks=default_with_async_eval",
        # enabled defaults to false in async_eval_spawn.yaml.
    ]

    with initialize_config_dir(version_base=None, config_dir=str(EXP_CONFIG_PATH)):
        cfg = compose(
            config_name="base_config_spectral_arch",
            overrides=overrides,
        )

    callbacks = create_callbacks(cfg)
    aes_callbacks = [cb for cb in callbacks if isinstance(cb, AsyncEvalSpawnCallback)]
    assert len(aes_callbacks) == 0, (
        f"expected zero AsyncEvalSpawnCallback when enabled=false; "
        f"got {len(aes_callbacks)}."
    )


def test_async_eval_callback_volume_commit_fn_wired_to_modal_helper(
    tmp_path: Path,
) -> None:
    """Bug #2 regression: ``create_callbacks`` must bind the callback's
    ``_volume_commit_fn`` to ``tmgg.modal._functions._commit_outputs_volume``
    so the trainer-side commits actually fire on Modal.

    Without this wiring the callback's ``volume_commit_fn`` stays
    ``None`` (its constructor default), the trainer never commits the
    persistent volume between writing the spawned manifest row and
    spawning the eval worker, and the worker's volume snapshot misses
    the row entirely. Symptom in the 2026-04-29 smoke: the manifest
    contained ``completed`` rows from the workers but no ``spawned``
    rows from the trainer.
    """
    from tmgg.modal._functions import _commit_outputs_volume

    overrides = _minimal_overrides(tmp_path) + [
        "base/callbacks=default_with_async_eval",
        "callbacks.async_eval_spawn.enabled=true",
        "callbacks.async_eval_spawn.schedule=[100]",
        "callbacks.async_eval_spawn.run_uid=test-uid",
        "callbacks.async_eval_spawn.manifest_path=/tmp/test_manifest.jsonl",
    ]

    with initialize_config_dir(version_base=None, config_dir=str(EXP_CONFIG_PATH)):
        cfg = compose(
            config_name="base_config_spectral_arch",
            overrides=overrides,
        )

    callbacks = create_callbacks(cfg)
    aes_callbacks = [cb for cb in callbacks if isinstance(cb, AsyncEvalSpawnCallback)]
    assert len(aes_callbacks) == 1
    aes = aes_callbacks[0]
    assert aes._volume_commit_fn is _commit_outputs_volume, (
        "create_callbacks must wire _volume_commit_fn to _commit_outputs_volume; "
        f"got {aes._volume_commit_fn!r}"
    )

"""Tests for ``AsyncEvalSpawnCallback`` (Step 3 of the async-eval plan).

Test rationale
--------------
The callback drives the async-eval loop from inside Lightning training:

* On ``on_train_start``, it must declare the W&B custom-step metric
  routing for ``gen-val/*`` (the eval worker logs gen-val metrics under
  ``trainer/global_step`` rather than the default sample-step axis).
  It must also capture the W&B run id so spawned eval workers can
  ``resume="must"`` back into the same run.
* On ``on_train_batch_end``, the callback compares the schedule head
  against ``trainer.global_step``. When one or more scheduled steps have
  been passed (drain semantics for ``accumulate_grad_batches > 1``), the
  callback saves a step-stamped checkpoint, commits the Modal output
  volume, spawns the appropriate ``modal_evaluate_mmd_async{,_debug,_fast}``
  Function, and appends a ``status="spawned"`` row to the manifest. The
  manifest row records both ``scheduled_step`` (target) and
  ``global_step`` (actual fired step).
* On ``on_fit_end``, the callback drains in-flight evals with a 15-min
  progress-reset idle timeout. Each newly-observed terminal status row
  (``completed`` or ``failed``) resets the idle timer; if the timer
  expires without progress, the callback exits and lets the post-hoc
  reconciler clean up.

The fail-loud rule from CLAUDE.md applies: a missing or non-WandbLogger
``trainer.logger`` is a configuration error, not a recoverable state.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


def _make_trainer(
    *,
    tmp_path: Path,
    global_step: int = 0,
    wandb_run_id: str = "abc1234",
    has_logger: bool = True,
) -> MagicMock:
    """Build a stub trainer with an attached WandbLogger-like mock."""
    trainer = MagicMock()
    trainer.global_step = global_step
    trainer.default_root_dir = str(tmp_path)
    trainer.save_checkpoint = MagicMock()
    if has_logger:
        experiment = MagicMock()
        experiment.id = wandb_run_id
        experiment.define_metric = MagicMock()
        trainer.logger = MagicMock()
        trainer.logger.experiment = experiment
    else:
        trainer.logger = None
    return trainer


def _make_resolver(spawn_call_id: str = "fc-123") -> MagicMock:
    """Build a Modal-Function resolver that returns a fake function with .spawn()."""
    fn = MagicMock()
    fake_call = MagicMock()
    fake_call.object_id = spawn_call_id
    fn.spawn = MagicMock(return_value=fake_call)
    resolver = MagicMock(return_value=fn)
    return resolver


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    """Read the manifest via the canonical helper, which auto-detects
    directory-of-rows vs legacy-JSONL layout."""
    import sys as _sys

    _repo = Path(__file__).resolve().parents[3]
    if str(_repo) not in _sys.path:
        _sys.path.insert(0, str(_repo))
    from tmgg.sweep._eval_manifest import (
        read_manifest as _read,
    )

    return _read(path)


# ---------------------------------------------------------------------------
# on_train_start
# ---------------------------------------------------------------------------


class TestOnTrainStart:
    def test_callback_calls_define_metric_on_train_start(self, tmp_path: Path) -> None:
        """``define_metric`` must declare gen-val/* under trainer/global_step.

        The eval worker logs metrics without a ``step=`` kwarg; W&B routes
        them via this metric definition. Without it, W&B uses the default
        sample-step axis and the metrics land on the wrong steps.
        """
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        trainer = _make_trainer(tmp_path=tmp_path)
        pl_module = MagicMock()
        cb = AsyncEvalSpawnCallback(
            schedule=[10, 20],
            run_uid="smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(tmp_path / "eval_manifest.jsonl"),
        )
        cb.on_train_start(trainer, pl_module)
        trainer.logger.experiment.define_metric.assert_called_once_with(
            "gen-val/*", step_metric="trainer/global_step"
        )

    def test_callback_captures_wandb_run_id_on_train_start(
        self, tmp_path: Path
    ) -> None:
        """The captured run id is forwarded to spawn() so the worker resumes the right run."""
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        trainer = _make_trainer(tmp_path=tmp_path, wandb_run_id="run-xyz")
        pl_module = MagicMock()
        cb = AsyncEvalSpawnCallback(
            schedule=[10],
            run_uid="uid",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(tmp_path / "eval_manifest.jsonl"),
        )
        cb.on_train_start(trainer, pl_module)
        assert cb._wandb_run_id == "run-xyz"

    def test_callback_raises_loudly_when_logger_missing(self, tmp_path: Path) -> None:
        """No silent fallback: missing logger is a config error."""
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        trainer = _make_trainer(tmp_path=tmp_path, has_logger=False)
        pl_module = MagicMock()
        cb = AsyncEvalSpawnCallback(
            schedule=[10],
            run_uid="uid",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(tmp_path / "eval_manifest.jsonl"),
        )
        with pytest.raises(RuntimeError, match="WandbLogger"):
            cb.on_train_start(trainer, pl_module)


# ---------------------------------------------------------------------------
# on_train_batch_end — schedule-driven spawn
# ---------------------------------------------------------------------------


class TestOnTrainBatchEnd:
    def test_callback_spawns_at_scheduled_step(self, tmp_path: Path) -> None:
        """Two scheduled steps in the [1..15] range => exactly two spawn calls."""
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        manifest = tmp_path / "eval_manifest.jsonl"
        trainer = _make_trainer(tmp_path=tmp_path)
        pl_module = MagicMock()
        resolver = _make_resolver()
        volume_commit = MagicMock()
        cb = AsyncEvalSpawnCallback(
            schedule=[5, 10],
            run_uid="uid",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(manifest),
            modal_function_resolver=resolver,
            volume_commit_fn=volume_commit,
        )
        cb.on_train_start(trainer, pl_module)
        for step in range(1, 16):
            trainer.global_step = step
            cb.on_train_batch_end(
                trainer, pl_module, outputs=None, batch=None, batch_idx=step
            )

        fn = resolver.return_value
        assert fn.spawn.call_count == 2
        rows = _read_manifest(manifest)
        spawned_rows = [r for r in rows if r["status"] == "spawned"]
        assert len(spawned_rows) == 2
        assert {r["scheduled_step"] for r in spawned_rows} == {5, 10}

    def test_callback_drains_overshoot_steps(self, tmp_path: Path) -> None:
        """If global_step jumps from 4 to 12 (e.g. accumulate_grad_batches), drain both 5 and 10."""
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        manifest = tmp_path / "eval_manifest.jsonl"
        trainer = _make_trainer(tmp_path=tmp_path)
        pl_module = MagicMock()
        resolver = _make_resolver()
        cb = AsyncEvalSpawnCallback(
            schedule=[5, 10],
            run_uid="uid",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(manifest),
            modal_function_resolver=resolver,
            volume_commit_fn=MagicMock(),
        )
        cb.on_train_start(trainer, pl_module)
        # Step 4: nothing fires.
        trainer.global_step = 4
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
        # Step jumps to 12: both 5 and 10 should drain.
        trainer.global_step = 12
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=1)

        fn = resolver.return_value
        assert fn.spawn.call_count == 2
        rows = _read_manifest(manifest)
        spawned = [r for r in rows if r["status"] == "spawned"]
        assert len(spawned) == 2
        # Both fire at global_step=12, but record their scheduled step.
        for r in spawned:
            assert r["global_step"] == 12
            assert r["scheduled_step"] in {5, 10}

    def test_callback_writes_step_stamped_checkpoint(self, tmp_path: Path) -> None:
        """The saved checkpoint path lives under {default_root_dir}/checkpoints/step_{N}.ckpt."""
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        manifest = tmp_path / "eval_manifest.jsonl"
        trainer = _make_trainer(tmp_path=tmp_path)
        pl_module = MagicMock()
        cb = AsyncEvalSpawnCallback(
            schedule=[5],
            run_uid="uid",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(manifest),
            modal_function_resolver=_make_resolver(),
            volume_commit_fn=MagicMock(),
        )
        cb.on_train_start(trainer, pl_module)
        trainer.global_step = 7
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
        assert trainer.save_checkpoint.call_count == 1
        ckpt_path = Path(trainer.save_checkpoint.call_args.args[0])
        assert ckpt_path.name == "step_7.ckpt"
        assert ckpt_path.parent.name == "checkpoints"

    def test_callback_calls_volume_commit_before_spawn(self, tmp_path: Path) -> None:
        """Ordering contract:

        1. ``save_checkpoint`` writes the step-stamped ckpt to the volume.
        2. ``volume_commit`` flushes the ckpt so the spawned worker sees it.
        3. ``spawn`` enqueues the Modal call.
        4. ``volume_commit`` (again) flushes the ``spawned`` manifest row so the
           worker can correlate its ``completed`` row against it (bug #2 fix).
        """
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        manifest = tmp_path / "eval_manifest.jsonl"
        trainer = _make_trainer(tmp_path=tmp_path)
        pl_module = MagicMock()

        events: list[str] = []
        save_ckpt = MagicMock(
            side_effect=lambda *a, **kw: events.append("save_checkpoint")
        )
        trainer.save_checkpoint = save_ckpt

        volume_commit = MagicMock(side_effect=lambda: events.append("volume_commit"))

        fn = MagicMock()
        fake_call = MagicMock()
        fake_call.object_id = "fc-1"
        fn.spawn = MagicMock(
            side_effect=lambda task: events.append("spawn") or fake_call
        )
        resolver = MagicMock(return_value=fn)

        cb = AsyncEvalSpawnCallback(
            schedule=[3],
            run_uid="uid",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(manifest),
            modal_function_resolver=resolver,
            volume_commit_fn=volume_commit,
        )
        cb.on_train_start(trainer, pl_module)
        trainer.global_step = 3
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert events == [
            "save_checkpoint",
            "volume_commit",
            "spawn",
            "volume_commit",
        ]

    def test_callback_commits_manifest_row_after_spawn(self, tmp_path: Path) -> None:
        """Bug #2 regression: the spawned-row commit must fire AFTER the
        manifest write, so the eval worker's volume snapshot includes
        the row. The smoke run never wrote a ``spawned`` row to the
        worker's view because the second commit was missing."""
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        manifest = tmp_path / "eval_manifest.jsonl"
        trainer = _make_trainer(tmp_path=tmp_path)
        pl_module = MagicMock()

        # Capture the manifest contents at each volume_commit call so we
        # can prove the spawned row exists at the second commit.
        commit_snapshots: list[list[dict[str, Any]]] = []

        def _snapshot() -> None:
            commit_snapshots.append(_read_manifest(manifest))

        volume_commit = MagicMock(side_effect=_snapshot)
        resolver = _make_resolver(spawn_call_id="fc-99")

        cb = AsyncEvalSpawnCallback(
            schedule=[3],
            run_uid="uid",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(manifest),
            modal_function_resolver=resolver,
            volume_commit_fn=volume_commit,
        )
        cb.on_train_start(trainer, pl_module)
        trainer.global_step = 3
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert volume_commit.call_count == 2, (
            f"expected 2 commits (pre-spawn + post-manifest), "
            f"got {volume_commit.call_count}"
        )
        # First commit fires before the manifest row is written.
        assert commit_snapshots[0] == [], "first commit must precede the manifest write"
        # Second commit fires AFTER the manifest row, so the worker can see it.
        assert (
            len(commit_snapshots[1]) == 1
        ), f"second commit must observe the spawned row; saw {commit_snapshots[1]}"
        assert commit_snapshots[1][0]["status"] == "spawned"
        assert commit_snapshots[1][0]["modal_call_id"] == "fc-99"

    def test_callback_resolves_correct_modal_function_per_tier(
        self, tmp_path: Path
    ) -> None:
        """gpu_tier=standard → modal_evaluate_mmd_async; debug → _debug; fast → _fast."""
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        for tier, expected_name in [
            ("standard", "modal_evaluate_mmd_async"),
            ("debug", "modal_evaluate_mmd_async_debug"),
            ("fast", "modal_evaluate_mmd_async_fast"),
        ]:
            manifest = tmp_path / f"eval_manifest_{tier}.jsonl"
            trainer = _make_trainer(tmp_path=tmp_path)
            pl_module = MagicMock()
            resolver = _make_resolver()
            cb = AsyncEvalSpawnCallback(
                schedule=[3],
                run_uid="uid",
                wandb_project="graph-denoising",
                wandb_entity="your-wandb-entity",
                manifest_path=str(manifest),
                modal_function_resolver=resolver,
                volume_commit_fn=MagicMock(),
                gpu_tier=tier,
            )
            cb.on_train_start(trainer, pl_module)
            trainer.global_step = 3
            cb.on_train_batch_end(
                trainer, pl_module, outputs=None, batch=None, batch_idx=0
            )
            resolver.assert_called_once_with("graph-denoising", expected_name)

    def test_callback_spawn_payload_has_required_keys(self, tmp_path: Path) -> None:
        """Spawn must pass a task dict carrying the keys that ``evaluate_mmd_async`` reads."""
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        manifest = tmp_path / "eval_manifest.jsonl"
        trainer = _make_trainer(tmp_path=tmp_path, wandb_run_id="abc1234")
        pl_module = MagicMock()
        resolver = _make_resolver()
        cb = AsyncEvalSpawnCallback(
            schedule=[5],
            run_uid="uid-1",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(manifest),
            modal_function_resolver=resolver,
            volume_commit_fn=MagicMock(),
            num_samples=40,
            num_steps=1000,
        )
        cb.on_train_start(trainer, pl_module)
        trainer.global_step = 5
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
        fn = resolver.return_value
        assert fn.spawn.call_count == 1
        task = fn.spawn.call_args.args[0]
        for key in [
            "run_uid",
            "wandb_run_id",
            "wandb_project",
            "wandb_entity",
            "scheduled_step",
            "global_step",
            "num_samples",
            "num_steps",
            "checkpoint_path",
            "manifest_path",
        ]:
            assert key in task, f"missing key {key!r} in spawn task payload"
        assert task["wandb_run_id"] == "abc1234"
        assert task["scheduled_step"] == 5
        assert task["global_step"] == 5
        assert task["num_samples"] == 40
        assert task["num_steps"] == 1000


# ---------------------------------------------------------------------------
# on_fit_end — drain loop with 15-min progress-reset timeout
# ---------------------------------------------------------------------------


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _append_row(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


class TestOnFitEndDrain:
    def test_callback_drain_exits_immediately_when_no_spawned_rows(
        self, tmp_path: Path
    ) -> None:
        """If the manifest has no spawned rows, drain returns instantly."""
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        manifest = tmp_path / "eval_manifest.jsonl"
        _write_manifest(manifest, [])  # empty
        trainer = _make_trainer(tmp_path=tmp_path)
        pl_module = MagicMock()
        cb = AsyncEvalSpawnCallback(
            schedule=[],
            run_uid="uid",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(manifest),
            eval_drain_idle_timeout_s=30,
            eval_drain_poll_s=0.1,
        )
        cb.on_train_start(trainer, pl_module)
        t0 = time.monotonic()
        cb.on_fit_end(trainer, pl_module)
        elapsed = time.monotonic() - t0
        # No spawned rows -> exits without polling.
        assert elapsed < 1.0

    def test_callback_drain_exits_on_idle_timeout(self, tmp_path: Path) -> None:
        """One spawned row that never completes should trigger the idle timeout."""
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        manifest = tmp_path / "eval_manifest.jsonl"
        _write_manifest(
            manifest,
            [
                {
                    "kind": "eval_event",
                    "run_uid": "uid",
                    "wandb_run_id": "abc1234",
                    "scheduled_step": 100,
                    "global_step": 100,
                    "ts_utc": "2026-04-29T00:00:00+00:00",
                    "status": "spawned",
                    "modal_call_id": "fc-1",
                    "checkpoint_path": "/data/outputs/uid/checkpoints/step_100.ckpt",
                    "metrics": None,
                    "error_tail": None,
                },
            ],
        )
        trainer = _make_trainer(tmp_path=tmp_path)
        pl_module = MagicMock()
        cb = AsyncEvalSpawnCallback(
            schedule=[100],
            run_uid="uid",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(manifest),
            eval_drain_idle_timeout_s=1,
            eval_drain_poll_s=0.1,
        )
        cb.on_train_start(trainer, pl_module)
        t0 = time.monotonic()
        cb.on_fit_end(trainer, pl_module)
        elapsed = time.monotonic() - t0
        # Should exit after ~1s idle timeout, not earlier.
        assert 0.8 <= elapsed <= 3.0

    def test_callback_drain_resets_timer_on_progress(self, tmp_path: Path) -> None:
        """Three spawned rows; each completion resets the idle timer.

        We feed completions at t=0.3 and t=0.7 (well within the 1s idle
        timeout window) and t=1.2 (would normally have timed out from
        t=0, but the resets keep the drain alive). Total drain time
        should land near the 3rd completion + one poll, not the 1s
        idle timeout from t=0.
        """
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        manifest = tmp_path / "eval_manifest.jsonl"
        spawned_template = {
            "kind": "eval_event",
            "run_uid": "uid",
            "wandb_run_id": "abc1234",
            "ts_utc": "2026-04-29T00:00:00+00:00",
            "status": "spawned",
            "modal_call_id": None,
            "checkpoint_path": "/data/outputs/uid/checkpoints/step_X.ckpt",
            "metrics": None,
            "error_tail": None,
        }
        _write_manifest(
            manifest,
            [
                {
                    **spawned_template,
                    "scheduled_step": 100,
                    "global_step": 100,
                    "modal_call_id": "fc-1",
                },
                {
                    **spawned_template,
                    "scheduled_step": 200,
                    "global_step": 200,
                    "modal_call_id": "fc-2",
                },
                {
                    **spawned_template,
                    "scheduled_step": 300,
                    "global_step": 300,
                    "modal_call_id": "fc-3",
                },
            ],
        )
        trainer = _make_trainer(tmp_path=tmp_path)
        pl_module = MagicMock()
        cb = AsyncEvalSpawnCallback(
            schedule=[100, 200, 300],
            run_uid="uid",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(manifest),
            eval_drain_idle_timeout_s=1,
            eval_drain_poll_s=0.1,
        )
        cb.on_train_start(trainer, pl_module)

        # Schedule background completions via threads.
        import threading

        def _complete_after(delay: float, step: int) -> None:
            time.sleep(delay)
            _append_row(
                manifest,
                {
                    "kind": "eval_event",
                    "run_uid": "uid",
                    "wandb_run_id": "abc1234",
                    "scheduled_step": step,
                    "global_step": step,
                    "ts_utc": "2026-04-29T00:01:00+00:00",
                    "status": "completed",
                    "modal_call_id": f"fc-{step // 100}",
                    "checkpoint_path": "/data/outputs/uid/checkpoints/step_X.ckpt",
                    "metrics": {"gen-val/foo": 0.1},
                    "error_tail": None,
                },
            )

        threads = [
            threading.Thread(target=_complete_after, args=(0.3, 100)),
            threading.Thread(target=_complete_after, args=(0.7, 200)),
            threading.Thread(target=_complete_after, args=(1.2, 300)),
        ]
        for t in threads:
            t.start()

        t0 = time.monotonic()
        cb.on_fit_end(trainer, pl_module)
        elapsed = time.monotonic() - t0
        for t in threads:
            t.join()
        # Drain exits when all rows completed; total time near 1.2s + poll(0.1)
        # but well under idle_timeout_s + last_completion = 1 + 1.2 = 2.2.
        assert 1.1 <= elapsed <= 2.5, f"drain elapsed={elapsed:.2f}s"

    def test_callback_drain_exits_when_all_completed(self, tmp_path: Path) -> None:
        """Drain exits as soon as completed+failed >= total spawned, regardless of timer."""
        from tmgg.training.callbacks.async_eval_spawn import AsyncEvalSpawnCallback

        manifest = tmp_path / "eval_manifest.jsonl"
        rows: list[dict[str, Any]] = []
        for step in [100, 200]:
            rows.append(
                {
                    "kind": "eval_event",
                    "run_uid": "uid",
                    "wandb_run_id": "abc1234",
                    "scheduled_step": step,
                    "global_step": step,
                    "ts_utc": "2026-04-29T00:00:00+00:00",
                    "status": "spawned",
                    "modal_call_id": f"fc-{step}",
                    "checkpoint_path": "/data/outputs/uid/checkpoints/step_X.ckpt",
                    "metrics": None,
                    "error_tail": None,
                }
            )
            rows.append(
                {
                    "kind": "eval_event",
                    "run_uid": "uid",
                    "wandb_run_id": "abc1234",
                    "scheduled_step": step,
                    "global_step": step,
                    "ts_utc": "2026-04-29T00:00:01+00:00",
                    "status": "completed",
                    "modal_call_id": f"fc-{step}",
                    "checkpoint_path": "/data/outputs/uid/checkpoints/step_X.ckpt",
                    "metrics": {"gen-val/foo": 0.1},
                    "error_tail": None,
                }
            )
        _write_manifest(manifest, rows)
        trainer = _make_trainer(tmp_path=tmp_path)
        pl_module = MagicMock()
        cb = AsyncEvalSpawnCallback(
            schedule=[100, 200],
            run_uid="uid",
            wandb_project="graph-denoising",
            wandb_entity="your-wandb-entity",
            manifest_path=str(manifest),
            eval_drain_idle_timeout_s=60,
            eval_drain_poll_s=0.1,
        )
        cb.on_train_start(trainer, pl_module)
        t0 = time.monotonic()
        cb.on_fit_end(trainer, pl_module)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0

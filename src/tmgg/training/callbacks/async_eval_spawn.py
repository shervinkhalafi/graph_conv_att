"""Lightning callback that drives the async-eval loop from training.

Step 3 of the async-eval plan
(``internal async-eval design notes``). The trainer
fires-and-forgets a Modal eval call on every scheduled step; this
callback owns the schedule cursor, the step-stamped checkpoint write,
the volume commit, the spawn, the manifest append, and the at-fit-end
drain.

Three load-bearing decisions:

1. **W&B custom-step routing.** ``on_train_start`` calls
   ``trainer.logger.experiment.define_metric("gen-val/*",
   step_metric="trainer/global_step")`` once. The eval worker logs
   ``gen-val/*`` metrics with ``trainer/global_step`` as a value (not
   ``step=``) and W&B routes them to the correct training step. This is
   the "as if running in band" property the user asked for.
2. **Drain semantics for ``accumulate_grad_batches > 1``.** The
   callback drains the schedule list with
   ``while remaining and remaining[0] <= trainer.global_step``: each
   passed scheduled step fires once, and the manifest row records both
   the original ``scheduled_step`` and the actual ``global_step`` it
   fired at.
3. **15-min progress-reset drain.** ``on_fit_end`` polls the manifest
   every ``eval_drain_poll_s`` (default 30s). Each newly-observed
   ``completed`` or ``failed`` row resets a 15-min idle timer. If the
   timer expires without progress we exit; the post-hoc reconciler
   picks up the remaining ``spawned`` rows. Drain also exits as soon as
   ``completed + failed >= total_spawned``.

Modal coupling is kept testable via a ``modal_function_resolver``
constructor argument. In production this is bound to
``modal.Function.from_name``; tests inject a ``MagicMock``. Likewise
``volume_commit_fn`` decouples the callback from the Modal volume API
at import time — wired to ``_commit_outputs_volume`` from
``tmgg.modal._functions`` at runtime.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from pytorch_lightning.callbacks import Callback

from tmgg.sweep._eval_manifest import (
    read_manifest as eval_manifest_read,
)
from tmgg.sweep._eval_manifest import (
    write_manifest_row as eval_manifest_write_row,
)

if TYPE_CHECKING:
    import pytorch_lightning as pl


def _default_modal_function_resolver(app_name: str, fn_name: str) -> Any:
    """Default resolver: lazy-imports modal and returns ``Function.from_name``.

    The lazy import keeps Modal out of the hot import path for unit tests
    that do not exercise the spawn code path.
    """
    from modal import Function  # noqa: PLC0415 — lazy on purpose

    return Function.from_name(app_name, fn_name)


def _function_name_for_tier(gpu_tier: str) -> str:
    """Map gpu_tier to the Modal function name registered in _functions.py."""
    if gpu_tier == "standard":
        return "modal_evaluate_mmd_async"
    return f"modal_evaluate_mmd_async_{gpu_tier}"


def _write_manifest_row(manifest_path: str, row: dict[str, Any]) -> None:
    """Write a single eval-event row as an immutable JSON file.

    Replaces the legacy append-to-shared-JSONL pattern. With Modal
    Volume's last-writer-wins semantics, two concurrent appenders
    (trainer + eval worker) can clobber each other's rows when
    committing overlapping views of the same file. Per-row files keyed
    by ``{step}-{status}-{discriminator}`` eliminate that failure mode.
    The trainer's discriminator is the just-spawned ``modal_call_id``
    (``fc-...``); see ``write_manifest_row`` in
    ``tmgg.sweep._eval_manifest`` for the exact filename convention.
    """
    eval_manifest_write_row(manifest_path, row)


def _read_manifest(manifest_path: str) -> list[dict[str, Any]]:
    """Read the manifest into a list of dicts.

    Delegates to ``tmgg.sweep._eval_manifest.read_manifest`` which
    auto-detects directory vs JSONL layout. Returns an empty list when
    no manifest exists yet (drain may run before any spawns).
    """
    return eval_manifest_read(manifest_path)


class AsyncEvalSpawnCallback(Callback):
    """Spawn async MMD eval at scheduled training steps.

    Parameters
    ----------
    schedule
        Sorted list of training steps at which to fire an eval. The
        Hydra config helper converts a YAML schedule file into this
        list. Steps must be non-negative; duplicates are tolerated but
        only fire once each because we drain via list-pop.
    run_uid
        Stable per-run identifier (e.g.
        ``smallest-cfg/spectre_sbm/r1/anchor/aabbccdd``) recorded in
        every manifest row for cross-run audit.
    wandb_project, wandb_entity
        W&B project/entity passed into the spawned worker so it can
        ``init(resume="must")`` back into the trainer's run.
    manifest_path
        Absolute path to the per-run JSONL manifest, conventionally
        ``/data/outputs/{run_id}/eval_manifest.jsonl``. Append-only.
    modal_app_name
        Modal app to look the eval function up under. Defaults to
        ``graph-denoising``.
    gpu_tier
        Selects which of ``modal_evaluate_mmd_async{,_debug,_fast}`` to
        spawn. ``standard`` (A10G) is the default; ``debug`` (T4) and
        ``fast`` (A100) are available for cost/throughput tuning.
    num_samples, num_steps
        Forwarded to the worker as the eval-side knobs.
    keep_step_checkpoints
        If False, step-stamped checkpoints are deleted after spawn (not
        yet implemented; the smoke run keeps them for inspection).
    eval_drain_idle_timeout_s
        Idle-timer budget for ``on_fit_end`` drain. Default 900s
        (15 min). Each newly-observed terminal manifest row resets the
        timer. When the timer expires, drain exits and the reconciler
        picks up remaining ``spawned`` rows.
    eval_drain_poll_s
        How often to re-read the manifest during drain. Default 30s.
    modal_function_resolver
        Callable ``(app_name, fn_name) -> modal.Function``. Tests inject
        a Mock; production wires this to
        ``modal.Function.from_name`` (lazy-imported on first use).
    volume_commit_fn
        Callable ``() -> None``. When set, the callback calls it after
        the step-checkpoint write and before the spawn, so the spawned
        worker reads the just-written checkpoint from the persistent
        volume. Production wires this to ``_commit_outputs_volume``
        from ``tmgg.modal._functions``; ``None`` is allowed for
        host-side smoke tests where no volume exists.
    """

    def __init__(
        self,
        schedule: list[int],
        run_uid: str,
        wandb_project: str,
        wandb_entity: str,
        manifest_path: str,
        modal_app_name: str = "graph-denoising",
        gpu_tier: str = "standard",
        num_samples: int = 40,
        num_steps: int = 1000,
        keep_step_checkpoints: bool = True,
        eval_drain_idle_timeout_s: float = 900.0,
        eval_drain_poll_s: float = 30.0,
        modal_function_resolver: Callable[[str, str], Any] | None = None,
        volume_commit_fn: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        if any(s < 0 for s in schedule):
            raise ValueError(
                f"AsyncEvalSpawnCallback.schedule must contain non-negative integers; "
                f"got {schedule}."
            )
        if gpu_tier not in {"standard", "debug", "fast"}:
            raise ValueError(
                f"AsyncEvalSpawnCallback.gpu_tier must be one of "
                f"{{'standard','debug','fast'}}; got {gpu_tier!r}."
            )
        self.schedule = sorted(schedule)
        self.run_uid = run_uid
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.manifest_path = manifest_path
        self.modal_app_name = modal_app_name
        self.gpu_tier = gpu_tier
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.keep_step_checkpoints = keep_step_checkpoints
        self.eval_drain_idle_timeout_s = float(eval_drain_idle_timeout_s)
        self.eval_drain_poll_s = float(eval_drain_poll_s)
        self._modal_function_resolver: Callable[[str, str], Any] = (
            modal_function_resolver
            if modal_function_resolver is not None
            else _default_modal_function_resolver
        )
        self._volume_commit_fn = volume_commit_fn

        # Mutable state populated by hooks.
        self._remaining: list[int] = list(self.schedule)
        self._wandb_run_id: str | None = None

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @override
    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Declare the W&B custom-step axis and capture the run id.

        Lightning's WandbLogger has fully initialised ``experiment`` by
        this hook in Lightning 2.x; we read it once and stash the
        run id for later spawn payloads.
        """
        logger = getattr(trainer, "logger", None)
        if logger is None:
            raise RuntimeError(
                "AsyncEvalSpawnCallback requires a WandbLogger; trainer.logger is None. "
                "Either disable async eval (callbacks.async_eval_spawn.enabled=false) or "
                "configure a WandbLogger on the Trainer."
            )
        experiment = getattr(logger, "experiment", None)
        if experiment is None:
            raise RuntimeError(
                "AsyncEvalSpawnCallback requires a WandbLogger; "
                f"trainer.logger ({type(logger).__name__}) has no `experiment`. "
                "Either disable async eval or configure a WandbLogger."
            )
        define_metric = getattr(experiment, "define_metric", None)
        if define_metric is None:
            raise RuntimeError(
                "AsyncEvalSpawnCallback requires a WandbLogger; "
                f"trainer.logger.experiment ({type(experiment).__name__}) has no "
                "`define_metric` method. Wire up wandb.run before training."
            )
        define_metric("gen-val/*", step_metric="trainer/global_step")
        run_id = getattr(experiment, "id", None)
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError(
                "AsyncEvalSpawnCallback could not read trainer.logger.experiment.id; "
                f"got {run_id!r}. Cannot spawn eval workers without a wandb run id."
            )
        self._wandb_run_id = run_id

    @override
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Drain any scheduled steps that the trainer has now passed.

        Drains the head of ``self._remaining`` while the head is
        ``<= trainer.global_step``. This handles the
        ``accumulate_grad_batches > 1`` case where ``global_step`` can
        increment by more than one per batch end.
        """
        if not self._remaining:
            return
        global_step = int(trainer.global_step)
        while self._remaining and self._remaining[0] <= global_step:
            scheduled_step = self._remaining.pop(0)
            self._fire_eval(
                trainer=trainer,
                pl_module=pl_module,
                scheduled_step=scheduled_step,
                global_step=global_step,
            )

    @override
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Drain in-flight evals with a progress-reset idle timeout.

        Each newly-observed ``completed`` or ``failed`` row resets the
        idle timer. Drain exits when:

        * No ``spawned`` rows remain (everything terminal).
        * Idle timer exceeds ``eval_drain_idle_timeout_s`` without new
          progress.
        """
        rows = _read_manifest(self.manifest_path)
        total_spawned = sum(1 for r in rows if r.get("status") == "spawned")
        if total_spawned == 0:
            return
        last_completed = sum(1 for r in rows if r.get("status") == "completed")
        last_failed = sum(1 for r in rows if r.get("status") == "failed")
        last_progress_ts = time.monotonic()

        while True:
            rows = _read_manifest(self.manifest_path)
            current_completed = sum(1 for r in rows if r.get("status") == "completed")
            current_failed = sum(1 for r in rows if r.get("status") == "failed")
            current_spawned = sum(1 for r in rows if r.get("status") == "spawned")

            # Hard exit: every spawned row has terminated.
            if current_completed + current_failed >= total_spawned:
                return
            # Soft exit: no remaining spawned rows after manifest update.
            if current_spawned == 0:
                return

            if (current_completed + current_failed) > (last_completed + last_failed):
                last_completed = current_completed
                last_failed = current_failed
                last_progress_ts = time.monotonic()
            elif time.monotonic() - last_progress_ts > self.eval_drain_idle_timeout_s:
                return

            time.sleep(self.eval_drain_poll_s)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fire_eval(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        scheduled_step: int,
        global_step: int,
    ) -> None:
        """Save checkpoint, commit volume, spawn eval, append manifest row."""
        if self._wandb_run_id is None:
            raise RuntimeError(
                "AsyncEvalSpawnCallback._fire_eval called before on_train_start; "
                "_wandb_run_id is unset. Lightning hook order has been violated."
            )
        ckpt_dir = Path(trainer.default_root_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / f"step_{global_step}.ckpt"
        trainer.save_checkpoint(str(checkpoint_path))

        if self._volume_commit_fn is not None:
            self._volume_commit_fn()

        fn_name = _function_name_for_tier(self.gpu_tier)
        fn = self._modal_function_resolver(self.modal_app_name, fn_name)

        run_id = self._derive_run_id(trainer)
        task: dict[str, Any] = {
            "run_id": run_id,
            "run_uid": self.run_uid,
            "wandb_run_id": self._wandb_run_id,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "scheduled_step": scheduled_step,
            "global_step": global_step,
            "num_samples": self.num_samples,
            "num_steps": self.num_steps,
            "checkpoint_path": str(checkpoint_path),
            "manifest_path": self.manifest_path,
        }
        function_call = fn.spawn(task)
        modal_call_id = getattr(function_call, "object_id", None)

        row = {
            "kind": "eval_event",
            "run_uid": self.run_uid,
            "wandb_run_id": self._wandb_run_id,
            "scheduled_step": scheduled_step,
            "global_step": global_step,
            "ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": "spawned",
            "modal_call_id": modal_call_id,
            "checkpoint_path": str(checkpoint_path),
            "metrics": None,
            "error_tail": None,
        }
        _write_manifest_row(self.manifest_path, row)

        # Commit again so the spawned-row is visible to the eval worker's
        # snapshot of ``graph-denoising-outputs``. Modal volumes only flush on
        # function-return or explicit ``Volume.commit()``; without this
        # second commit the trainer's spawned rows never reach the eval
        # workers, and ``on_fit_end`` drain treats every spawn as
        # in-flight forever (bug #2 in the 2026-04-29 smoke).
        if self._volume_commit_fn is not None:
            self._volume_commit_fn()

    def _derive_run_id(self, trainer: pl.Trainer) -> str:
        """Best-effort derivation of the Modal run_id for the eval payload.

        The trainer's ``default_root_dir`` is conventionally
        ``/data/outputs/{run_id}`` on Modal. We strip the trailing slash
        and read the basename. On host-side test runs this returns the
        tmp_path basename; the eval worker needs ``run_id`` only for
        filesystem-naming (manifest path is already explicit).
        """
        return Path(trainer.default_root_dir).name

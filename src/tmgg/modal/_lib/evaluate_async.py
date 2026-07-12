"""Async-evaluation worker that logs back into the trainer's W&B run.

Step 1 of the async-eval plan
(``internal async-eval design notes``). The trainer
fires-and-forgets a Modal call against this module on every scheduled
eval step; the eval container then resumes the trainer's W&B run, runs
MMD evaluation, logs gen-val/* metrics under the trainer's custom step
axis (``trainer/global_step``), and appends a row to the per-run eval
manifest JSONL.

Two design decisions are load-bearing:

1. **No ``step=`` kwarg on ``wandb.log``.** Per the user's "log as if
   running in band" clarification, gen-val/* metrics travel with
   ``trainer/global_step`` as a *value* in the dict. The trainer calls
   ``wandb.define_metric("gen-val/*", step_metric="trainer/global_step")``
   once at start; W&B's custom-step routing then places these metrics
   on the correct training step. Passing ``step=`` would bypass that
   routing.
2. **Re-raise on failure.** Manifest is written with
   ``status="failed"`` first, then the exception propagates so Modal
   records the function-call failure too.

This module reuses ``run_mmd_evaluation`` from ``evaluate.py`` without
modifying it. The Modal ``@app.function`` decorator wrapping happens in
``_functions.py`` (Step 2 of the plan), not here.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import wandb

from tmgg.modal._lib.evaluate import EvaluationInput, run_mmd_evaluation
from tmgg.sweep._eval_manifest import write_manifest_row as eval_manifest_write_row

_EVAL_WORKER_ID = uuid.uuid4().hex[:12]
"""Per-process discriminator for manifest filenames.

Each eval-worker container has a unique 12-char hex ID. Combined with
``scheduled_step`` and ``status`` this guarantees a collision-free
filename in the directory-layout manifest, so the trainer container
and any concurrent eval workers never write to the same path. The
ID is generated once at module import (i.e. once per Modal call).
"""


def _write_manifest_row(manifest_path: str, row: dict[str, Any]) -> None:
    """Write a single eval-event row as an immutable JSON file.

    Replaces the legacy append-to-shared-JSONL pattern. With Modal
    Volume's last-writer-wins semantics, two concurrent appenders
    (trainer + eval worker) can clobber each other's rows when
    committing overlapping views of the same file. Per-row files
    keyed by ``{step}-{status}-{discriminator}`` eliminate that
    failure mode: every writer produces a unique path.

    Stamps ``_eval_worker_id`` on the row so the helper in
    ``tmgg.sweep._eval_manifest`` picks up a non-empty
    discriminator even when ``modal_call_id`` is null (eval workers
    don't have access to their own call ID at runtime).
    """
    row_with_disc = {**row, "_eval_worker_id": _EVAL_WORKER_ID}
    eval_manifest_write_row(manifest_path, row_with_disc)


def evaluate_mmd_async(task: dict[str, Any]) -> dict[str, Any]:
    """Run async MMD eval and log to an existing W&B run.

    Parameters
    ----------
    task
        Dict with keys ``run_id`` (str), ``run_uid`` (str),
        ``checkpoint_path`` (str), ``wandb_run_id`` (str),
        ``wandb_project`` (str), ``wandb_entity`` (str),
        ``global_step`` (int), ``num_samples`` (int), ``num_steps``
        (int), ``scheduled_step`` (int), and ``manifest_path`` (str —
        ``/data/outputs/{run_id}/eval_manifest.jsonl``).

    Returns
    -------
    dict
        The serialized ``EvaluationOutput`` returned by
        ``run_mmd_evaluation``.

    Raises
    ------
    Exception
        Any exception raised by ``run_mmd_evaluation`` or by the W&B
        attach is re-raised after the failed-row is appended to the
        manifest, so Modal records the failure too.
    """
    # TODO(smoke-run): verify whether Modal's container-start auto-reload
    # is sufficient for reading the just-written step-stamped checkpoint.
    # If a stale checkpoint is observed in the smoke run, add an explicit
    # ``volume.reload()`` here.

    manifest_path = task["manifest_path"]
    run_uid = task["run_uid"]
    wandb_run_id = task["wandb_run_id"]
    scheduled_step = task["scheduled_step"]
    global_step = task["global_step"]
    checkpoint_path = task["checkpoint_path"]

    try:
        eval_input = EvaluationInput(
            run_id=task["run_id"],
            checkpoint_path=checkpoint_path,
            num_samples=task["num_samples"],
            num_steps=task["num_steps"],
        )
        eval_output = run_mmd_evaluation(
            {
                "run_id": eval_input.run_id,
                "checkpoint_path": eval_input.checkpoint_path,
                "num_samples": eval_input.num_samples,
                "num_steps": eval_input.num_steps,
                "mmd_kernel": eval_input.mmd_kernel,
                "mmd_sigma": eval_input.mmd_sigma,
                "seed": eval_input.seed,
            }
        )
    except Exception as exc:
        # Record the failure before re-raising so the manifest stays the
        # source of truth even if Modal's traceback is the only artifact.
        failed_row = {
            "kind": "eval_event",
            "run_uid": run_uid,
            "wandb_run_id": wandb_run_id,
            "scheduled_step": scheduled_step,
            "global_step": global_step,
            "ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": "failed",
            "modal_call_id": None,
            "checkpoint_path": checkpoint_path,
            "metrics": None,
            "error_tail": str(exc)[:1000],
        }
        _write_manifest_row(manifest_path, failed_row)
        raise

    # ------------------------------------------------------------------
    # Surface CLI-level failures as exceptions: ``run_mmd_evaluation``
    # never raises -- it wraps subprocess crashes into a dict with
    # ``status="failed"``. The previous code path silently treated this
    # as a "completed" run and logged empty metrics, masking the
    # gaussian-CLI dispatch bug end-to-end.
    # ------------------------------------------------------------------
    if eval_output.get("status") != "completed":
        error_message = eval_output.get("error_message") or "unknown failure"
        failed_row = {
            "kind": "eval_event",
            "run_uid": run_uid,
            "wandb_run_id": wandb_run_id,
            "scheduled_step": scheduled_step,
            "global_step": global_step,
            "ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": "failed",
            "modal_call_id": None,
            "checkpoint_path": checkpoint_path,
            "metrics": None,
            "error_tail": str(error_message)[:1000],
        }
        _write_manifest_row(manifest_path, failed_row)
        raise RuntimeError(
            f"run_mmd_evaluation reported status={eval_output.get('status')!r}: "
            f"{error_message}"
        )

    # ------------------------------------------------------------------
    # Attach to the trainer's W&B run and log under the custom step axis.
    # ------------------------------------------------------------------
    # Flatten the nested ``results`` dict from EvaluationOutput. The
    # discrete eval CLI returns flat metric names (``degree_mmd``,
    # ``clustering_mmd``, ``spectral_mmd``, optionally ``sbm_accuracy``),
    # which we prefix with ``gen-val/`` so they match the trainer's
    # in-band logging namespace and satisfy the smoke pass criteria.
    # Already-prefixed names are passed through unchanged so future CLIs
    # that emit fully qualified keys do not get double-prefixed.
    metrics_dict: dict[str, Any] = {}
    for _label, label_metrics in eval_output.get("results", {}).items():
        for metric_name, value in label_metrics.items():
            key = (
                metric_name
                if metric_name.startswith("gen-val/")
                else f"gen-val/{metric_name}"
            )
            metrics_dict[key] = value

    wandb.init(
        id=wandb_run_id,
        project=task["wandb_project"],
        entity=task["wandb_entity"],
        resume="must",
    )
    try:
        # No step= kwarg: trainer/global_step in the payload routes via
        # define_metric (which the trainer calls at on_train_start).
        wandb.log({**metrics_dict, "trainer/global_step": global_step})
    finally:
        wandb.finish()

    completed_row = {
        "kind": "eval_event",
        "run_uid": run_uid,
        "wandb_run_id": wandb_run_id,
        "scheduled_step": scheduled_step,
        "global_step": global_step,
        "ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "completed",
        "modal_call_id": None,
        "checkpoint_path": checkpoint_path,
        "metrics": metrics_dict,
        "error_tail": None,
    }
    _write_manifest_row(manifest_path, completed_row)

    return eval_output

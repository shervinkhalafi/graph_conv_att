"""Reader helpers for the async-eval manifest.

The async-eval architecture (plan: ``async-eval design``) has
two writers record per-step events at
``/data/outputs/{run_id}/eval_manifest.d/`` (or, in legacy runs,
``eval_manifest.jsonl``): the trainer-side ``AsyncEvalSpawnCallback``
emits ``status: spawned`` when it fires a Modal eval call, and the
eval worker emits ``status: completed`` or ``status: failed`` when it
finishes.

Layout — directory of immutable per-row files (preferred)
---------------------------------------------------------

::

    eval_manifest.d/
      0000200-spawned-fc01KQDxyz.json
      0000200-completed-evalworker-<uuid>.json
      0000800-spawned-fc01KQDabc.json
      0000800-failed-evalworker-<uuid>.json

Filename: ``{scheduled_step:07d}-{status}-{discriminator}.json``.
The zero-padded step makes filename-sort = step-sort. Each file is
written once, never appended; cross-container concurrency is solved
by giving each writer a distinct filename. ``latest_status_per_step``
picks the highest-priority status per step
(``completed`` > ``failed`` > ``spawned``) so a still-pending eval
doesn't shadow a later terminal row.

Layout — single JSONL file (legacy)
-----------------------------------

Older runs wrote append-only JSONL at ``eval_manifest.jsonl`` with a
leading ``{"kind": "schema", "version": 1, ...}`` row. ``read_manifest``
auto-detects this layout and falls through to JSONL parsing for
audit-trail back-compat. New writes always use the directory layout.

Three downstream callers consume these helpers:

- ``fetch_outcomes.py`` cross-references the manifest with W&B history
  to decide whether all expected eval steps are accounted for before
  emitting ``threshold_pass``.
- ``watch_runs.py`` reads ``evals_lag`` to flag eval-starvation, which
  the recommendation flowchart uses to broaden ``extend_watch`` over
  ``kill: saturation``.
- ``reconcile_evals.py`` walks ``status: spawned`` files whose Modal
  call has since terminated and writes the missing terminal file.

Manifest row schema (per the plan)::

    {"kind": "eval_event",
     "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
     "wandb_run_id": "abc1234",
     "scheduled_step": 5237,
     "global_step": 5240,
     "ts_utc": "2026-04-29T...",
     "status": "spawned" | "completed" | "failed",
     "modal_call_id": "fc-...",
     "checkpoint_path": "/data/outputs/{run_id}/checkpoints/step_5240.ckpt",
     "metrics": {"gen-val/sbm_accuracy": 0.62, ...} | null,
     "error_tail": null | "..."}

``scheduled_step`` and ``global_step`` differ when
``accumulate_grad_batches > 1`` skips the exact scheduled step; the
eval fires at the first step >= scheduled. ``scheduled_step`` is the
key used for completeness — it matches the schedule the trainer was
given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Status priority for picking the latest-state row per step.
# ``completed`` wins over ``failed`` (a successful retry shouldn't be
# masked by an earlier failure file), and both terminal states win
# over ``spawned`` (a still-pending eval doesn't represent the
# step's resolution).
_STATUS_PRIORITY: dict[str, int] = {
    "spawned": 0,
    "failed": 1,
    "completed": 2,
}


def resolve_manifest_dir(path: str | Path) -> Path:
    """Map a manifest-path config value to the directory layout.

    Accepts either:
      - ``.../eval_manifest.jsonl`` — legacy JSONL path. Returns
        ``.../eval_manifest.d`` (sibling directory).
      - ``.../eval_manifest.d`` (or any directory path) — returned
        unchanged.

    The trainer callback and eval worker both call this helper on the
    same ``manifest_path`` task field, so they agree on the target dir
    without an explicit config change. Lets us keep the existing
    Hydra default ``manifest_path: ${paths.output_dir}/eval_manifest.jsonl``
    while writing to the directory layout transparently.
    """
    p = Path(path)
    if p.suffix == ".jsonl":
        return p.with_suffix(".d")
    return p


def manifest_filename(scheduled_step: int, status: str, discriminator: str) -> str:
    """Construct the per-row filename within an ``eval_manifest.d/``.

    Format: ``{step:07d}-{status}-{discriminator}.json``. The
    7-digit zero-pad covers training schedules up to 9_999_999 steps,
    well above any realistic budget; the lexicographic sort matches
    numeric step-sort so listing the directory yields events in
    chronological order.

    ``discriminator`` is the writer-supplied unique tag — for the
    trainer this is the Modal call ID (``fc-...``); for the eval
    worker it is a per-call UUID. Distinct discriminators per writer
    eliminate filename collisions across containers.
    """
    safe_disc = discriminator.replace("/", "_").replace(":", "_")
    return f"{scheduled_step:07d}-{status}-{safe_disc}.json"


def write_manifest_row(manifest_path: str | Path, row: dict[str, Any]) -> Path:
    """Write a single row as an immutable JSON file.

    The target directory is derived via :func:`resolve_manifest_dir` so
    callers can pass either the legacy ``eval_manifest.jsonl`` path or
    a direct ``eval_manifest.d`` path. Filename is constructed via
    :func:`manifest_filename`.

    ``discriminator`` is taken from the row in priority order:
    ``modal_call_id`` (trainer-side spawn) → ``wandb_run_id``
    (eval-worker fallback when no call ID is set in-row) → a stable
    fallback ``"unk"`` (should never be hit in production; raises a
    ``KeyError`` indirectly if it ever produces a colliding filename).

    Returns the absolute path of the file written. The Modal volume
    commit is the caller's responsibility.
    """
    target_dir = resolve_manifest_dir(manifest_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    step = int(row["scheduled_step"])
    status = str(row["status"])
    discriminator = (
        row.get("modal_call_id")
        or row.get("_eval_worker_id")
        or row.get("wandb_run_id")
        or "unk"
    )
    fname = manifest_filename(step, status, str(discriminator))
    target = target_dir / fname

    # Atomic write: serialize first, then a single ``write_text`` call.
    # Worst case (process killed mid-write) leaves a partial file that
    # readers will reject via ``json.JSONDecodeError`` and skip; the
    # discriminator means a retried writer creates a fresh path rather
    # than overwriting the partial.
    target.write_text(json.dumps(row))
    return target


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read all eval-event rows.

    Auto-detects directory vs JSONL layout:

    - If ``path`` (or its ``.d`` sibling derived via
      :func:`resolve_manifest_dir`) is a directory, lists every
      ``*.json`` file inside it, parses each, and returns the rows
      sorted by ``ts_utc`` for stable ordering.
    - Otherwise treats ``path`` as a JSONL file and reads it line by
      line, skipping the leading ``{"kind": "schema", ...}`` header
      and any blank lines. Rows with ``kind != "eval_event"`` are
      ignored so future row-types stay forward-compatible without
      breaking older readers.

    Empty list when no manifest exists yet (the trainer may call
    drain before any spawns happened).
    """
    p = Path(path)

    # Directory-layout path — either ``path`` is the dir itself, or the
    # legacy JSONL path resolves to a sibling ``.d/`` that exists.
    candidate_dir = resolve_manifest_dir(p)
    if candidate_dir.is_dir():
        rows: list[dict[str, Any]] = []
        for fp in sorted(candidate_dir.glob("*.json")):
            try:
                row = json.loads(fp.read_text())
            except json.JSONDecodeError:
                # Tolerate partial writes: skip and continue so a single
                # corrupt file doesn't blind the watcher to other rows.
                continue
            if row.get("kind") == "eval_event":
                rows.append(row)
        rows.sort(key=lambda r: str(r.get("ts_utc", "")))
        return rows

    # JSONL fallback (legacy runs).
    if p.is_file():
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "eval_event":
                rows.append(row)
        return rows

    return []


def latest_status_per_step(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Group rows by ``scheduled_step``, take the highest-priority row.

    Status priority: ``completed`` > ``failed`` > ``spawned``. This
    handles the directory-layout case where the trainer writes
    ``spawned`` and the eval worker writes ``completed`` as separate
    files — the terminal state wins regardless of which file was
    listed first. Within the same priority, ``ts_utc`` breaks ties
    (later timestamp wins) to preserve the legacy "latest write" semantics
    for retried evals.

    Keys by ``scheduled_step`` rather than ``global_step`` because
    completeness is judged against the schedule the trainer was given,
    not against the actual step the eval landed on.
    """
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = int(row["scheduled_step"])
        prev = by_step.get(step)
        if prev is None:
            by_step[step] = row
            continue
        prev_prio = _STATUS_PRIORITY.get(str(prev.get("status")), -1)
        cur_prio = _STATUS_PRIORITY.get(str(row.get("status")), -1)
        if (
            cur_prio > prev_prio
            or cur_prio == prev_prio
            and str(row.get("ts_utc", "")) > str(prev.get("ts_utc", ""))
        ):
            by_step[step] = row
    return by_step


def expected_steps(schedule: list[int]) -> set[int]:
    """Return the schedule as a set.

    Convenience wrapper so callers can phrase completeness questions as
    set algebra without re-importing builtins.
    """
    return set(schedule)


def evals_completeness(
    rows: list[dict[str, Any]],
    schedule: list[int],
) -> dict[str, int | list[int]]:
    """Summarise terminal-state coverage of ``schedule``.

    A scheduled step is ``completed`` if its latest row is
    ``status: completed``, ``failed`` if its latest row is
    ``status: failed``, and ``missing`` otherwise (no row, or only a
    ``spawned`` row). ``missing`` is sorted ascending so callers can
    print it without further work.

    Returns
    -------
    dict
        ``{"expected": int, "completed": int, "failed": int,
        "missing": list[int]}``.
    """
    latest = latest_status_per_step(rows)
    completed = 0
    failed = 0
    missing: list[int] = []
    for step in schedule:
        entry = latest.get(step)
        if entry is None:
            missing.append(step)
            continue
        status = entry.get("status")
        if status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
        else:
            # "spawned" or anything non-terminal counts as missing —
            # the eval has not produced a result yet.
            missing.append(step)
    return {
        "expected": len(schedule),
        "completed": completed,
        "failed": failed,
        "missing": sorted(missing),
    }


def evals_lag(
    rows: list[dict[str, Any]],
    schedule: list[int],
    current_step: int,
) -> int:
    """Return scheduled-and-passed steps minus completed evals.

    A positive lag means the trainer has passed eval points whose
    results have not yet been logged. ``watch_runs.py`` uses this to
    flag eval-starvation: a saturation-fit on starved data has too few
    points to be defensible, so the recommendation broadens to
    ``extend_watch`` rather than ``kill: saturation``.

    Only ``status: completed`` counts as resolved here. ``status: failed``
    is intentionally not counted: a failed eval has produced no metric
    for the saturation fit, so it still represents an information gap.
    """
    latest = latest_status_per_step(rows)
    passed = sum(1 for s in schedule if s <= current_step)
    completed = sum(
        1
        for s in schedule
        if s <= current_step and (latest.get(s) or {}).get("status") == "completed"
    )
    return passed - completed

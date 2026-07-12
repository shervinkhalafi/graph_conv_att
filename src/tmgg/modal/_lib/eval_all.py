"""Library code for the ``graph-denoising-eval-all`` Modal app.

Provides ``eval_all_checkpoints_impl``: walks the ``checkpoints/``
subdirectory of a finished training run on the shared
``graph-denoising-outputs`` volume, runs MMD evaluation on every ``*.ckpt``
sequentially in a single worker, and pushes the per-checkpoint
``gen-val/*`` metrics to a fresh W&B run with the checkpoint's
training step as the W&B step.

Persistence
-----------
Each evaluated ckpt also writes a per-ckpt subfolder under
``<experiment_dir>/eval_all/<eval_run_name>/<ckpt_stem>/`` containing
``metrics.json``, ``metrics.csv``, ``generated_graphs.json`` (edge-list),
``reference_graphs.json``, ``viz/{generated,reference}_NN.png`` ×
``viz_count``, ``val_per_batch.csv``, ``timings.json``, and a
``summary.md``. After each ckpt finishes, the worker appends one row
to ``<experiment_dir>/eval_all/<eval_run_name>/index.jsonl`` and
calls ``Volume.commit()`` so partial progress is durable: even if the
worker times out mid-panel the dumps for completed ckpts remain on
disk for downstream analysis. The ``output_dir_root`` parameter
overrides the default location (e.g. for tests using a tmp_path).

Filtering
---------
``ckpt_filter`` and ``step_filter`` narrow the walk to an explicit
subset of checkpoints. Both are matched against the entries returned
by :func:`discover_checkpoints` after the ``skip_last`` decision; if
both filters are supplied the intersection is evaluated.

Why a separate Modal app rather than another ``@app.function`` in
``_functions.py``:

- Different lifetime: the trainer-side ``graph-denoising`` app holds
  one container per running experiment, so deploying a new function
  there forces a redeploy that interrupts live training.
- Different deploy cadence: eval-all is a post-hoc tool that may be
  iterated on (new metric stacks, new evaluators) without disturbing
  the training app.
- Cleaner billing/auditing: a separate app shows up as its own
  ``modal app logs`` stream, making post-hoc ckpt evaluation runs
  trivial to find.

The module is importable from outside Modal (no ``import modal`` at
module level) so unit tests can exercise the checkpoint-walk +
step-parsing logic without touching the cloud.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ckpt filename convention from base_config: ``model-step={step}-...``
# (auto_insert_metric_name=False, plus the explicit ``{step:06d}`` /
# ``{step}`` formatter). ``last.ckpt`` has no step embedded; we read
# its step from the loaded torch payload at evaluation time.
_STEP_RE = re.compile(r"step[-_=](\d+)", re.IGNORECASE)


@dataclass
class CheckpointEntry:
    """One checkpoint queued for evaluation.

    Parameters
    ----------
    path
        Absolute path to the ``.ckpt`` file on the volume.
    step
        Training step parsed from the filename, or ``None`` if the
        filename does not embed a step (e.g. ``last.ckpt``). The
        worker resolves missing steps by inspecting the saved
        ``global_step`` in the Lightning payload.
    """

    path: Path
    step: int | None


@dataclass
class EvalAllOutput:
    """Aggregate result for one ``eval_all_checkpoints`` invocation.

    Parameters
    ----------
    experiment_dir
        Run directory walked (``/data/outputs/{experiment}/{run}``).
    wandb_run_id
        W&B run ID for the freshly-created summary run; useful so
        the caller can re-attach later for post-hoc plotting.
    num_checkpoints_total
        Count of ``*.ckpt`` files discovered before any filtering.
    num_evaluated
        Count of checkpoints actually run through MMD eval.
    per_checkpoint
        Per-step results in evaluation order. Each entry has
        ``checkpoint_name``, ``step``, ``status``
        (``"completed"`` / ``"failed"``), ``metrics`` (the flat dict
        the evaluator returned), and ``error_message`` if any.
    """

    experiment_dir: str
    wandb_run_id: str
    num_checkpoints_total: int
    num_evaluated: int
    per_checkpoint: list[dict[str, Any]] = field(default_factory=list)


def parse_step_from_ckpt_name(name: str) -> int | None:
    """Parse the training step out of a checkpoint filename.

    Recognises the project's two conventions:

    - ``model-step=040000-val_nll=...ckpt`` (discrete-NLL panel)
    - ``model-step=040000-val_loss=...ckpt`` (default panel)
    - ``last.ckpt`` -> ``None`` (caller resolves from payload).

    Returns
    -------
    int | None
        Parsed step, or ``None`` if no step substring is found.
    """
    match = _STEP_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def discover_checkpoints(
    experiment_dir: Path,
    *,
    skip_last: bool = True,
) -> list[CheckpointEntry]:
    """Walk the ``checkpoints/`` subdir under ``experiment_dir``.

    Returns the list of checkpoints sorted by step ascending
    (``None`` step entries land at the end). Does not load any
    payloads — purely filename-driven, so cheap to run before
    container startup.

    Parameters
    ----------
    experiment_dir
        Run directory holding ``checkpoints/`` and ``config.yaml``.
        On Modal this is ``/data/outputs/{experiment}/{run_id}``.
    skip_last
        Drop ``last.ckpt`` from the list. Defaults to ``True``: the
        last ckpt typically equals the final stepped ckpt and
        evaluating both wastes a worker slot.

    Raises
    ------
    FileNotFoundError
        If ``experiment_dir/checkpoints`` does not exist. Most
        likely cause: caller passed the experiment-name dir rather
        than the run-id dir below it.
    """
    ckpt_dir = experiment_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"No checkpoints directory at {ckpt_dir}. Pass the run-id "
            "directory (one level below experiment-name), e.g. "
            "/data/outputs/discrete_qm9_digress_repro/discrete_qm9_..."
        )

    entries: list[CheckpointEntry] = []
    for ckpt_path in sorted(ckpt_dir.glob("*.ckpt")):
        if skip_last and ckpt_path.name == "last.ckpt":
            continue
        step = parse_step_from_ckpt_name(ckpt_path.name)
        entries.append(CheckpointEntry(path=ckpt_path, step=step))

    # Sort by step ascending; None steps go last so they don't clobber
    # numeric-step ckpts in the W&B step axis.
    entries.sort(key=lambda e: (e.step is None, e.step or 0))
    return entries


def filter_checkpoint_entries(
    entries: list[CheckpointEntry],
    *,
    ckpt_filter: list[str] | None = None,
    step_filter: list[int] | None = None,
) -> list[CheckpointEntry]:
    """Narrow a discovered ckpt list to an explicit subset.

    Filters intersect: an entry must match every supplied filter to
    survive. ``ckpt_filter`` matches against either the full path
    string or the ckpt filename — both forms are accepted because the
    user might pass either after copying a path out of a log line.

    Parameters
    ----------
    entries
        Output of :func:`discover_checkpoints`.
    ckpt_filter
        Allowed ckpt names or paths. Falsy / empty means "no name
        filter applied". Match is exact on either ``str(entry.path)``
        or ``entry.path.name``.
    step_filter
        Allowed training steps. Entries whose step is ``None`` are
        dropped when this filter is supplied (they cannot be matched).

    Returns
    -------
    list[CheckpointEntry]
        Surviving entries in the original ascending-step order.
    """
    if not ckpt_filter and not step_filter:
        return entries
    surviving: list[CheckpointEntry] = []
    name_set = set(ckpt_filter or [])
    step_set = set(step_filter or [])
    for entry in entries:
        if name_set and (
            str(entry.path) not in name_set and entry.path.name not in name_set
        ):
            continue
        if step_set and (entry.step is None or entry.step not in step_set):
            continue
        surviving.append(entry)
    return surviving


def _resolve_step_from_payload(ckpt_path: Path) -> int | None:
    """Fallback step resolution: read ``global_step`` from torch ckpt.

    Only used for ``last.ckpt`` and other filenames missing a step
    substring. Loads the ckpt with ``weights_only=True`` to keep this
    safe on untrusted volumes.
    """
    import torch

    payload = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        return None
    step = payload.get("global_step")
    if isinstance(step, int):
        return step
    return None


def _resolve_wandb_run_name(
    experiment_dir: Path,
    wandb_run_id_override: str | None,
) -> str:
    """Build a W&B run name for the eval-all summary run."""
    if wandb_run_id_override:
        return wandb_run_id_override
    return f"eval-all/{experiment_dir.name}"


def _load_run_config(experiment_dir: Path) -> dict[str, Any]:
    """Load the saved config.yaml for the trained run as a plain dict."""
    from omegaconf import OmegaConf

    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Run directory {experiment_dir} has no config.yaml — "
            "is this really a finished training run dir?"
        )
    raw = OmegaConf.load(config_path)
    container = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(
            f"Expected dict from OmegaConf.to_container({config_path}); "
            f"got {type(container)}"
        )
    return {str(k): v for k, v in container.items()}


def _iter_evaluations(
    entries: list[CheckpointEntry],
    *,
    run_id: str,
    num_samples: int,
    num_steps: int,
    output_dir_root: Path | None,
    viz_count: int,
    val_batch_limit: int | None,
) -> Iterator[tuple[CheckpointEntry, dict[str, Any]]]:
    """Run ``run_mmd_evaluation`` on each entry, yielding the result.

    Implemented as a generator so the caller can interleave per-ckpt
    W&B logging + index.jsonl appends + volume commits without holding
    all results in memory at once. Each evaluation re-uses the same
    ``run_mmd_evaluation`` subprocess machinery the async-eval worker
    uses, so we inherit any future metric extensions (FCD, MOSES
    filters, etc.) for free.

    When ``output_dir_root`` is set, each ckpt's CLI subprocess gets
    ``--output-dir <root>/<ckpt_stem>``; the multi-format dump lands
    under that subfolder before this iterator yields the result.
    """
    from tmgg.modal._lib.evaluate import run_mmd_evaluation

    for entry in entries:
        step = entry.step
        if step is None:
            step = _resolve_step_from_payload(entry.path)
        task: dict[str, Any] = {
            "run_id": run_id,
            "checkpoint_path": str(entry.path),
            "num_samples": num_samples,
            "num_steps": num_steps,
        }
        if output_dir_root is not None:
            ckpt_subdir = output_dir_root / entry.path.stem
            task["output_dir"] = str(ckpt_subdir)
            task["viz_count"] = viz_count
            task["val_batch_limit"] = val_batch_limit
        try:
            result = run_mmd_evaluation(task)
        except Exception as exc:  # noqa: BLE001
            logger.exception("eval-all subprocess failed for %s", entry.path)
            result = {
                "status": "failed",
                "error_message": repr(exc),
                "results": {},
                "checkpoint_name": entry.path.stem,
            }
        # Attach the resolved step so the caller can use it for W&B.
        result["resolved_step"] = step
        yield entry, result


def _commit_outputs_volume() -> None:
    """Best-effort flush of the graph-denoising-outputs volume from inside Modal.

    Called after every per-ckpt write so a worker timeout mid-panel
    leaves the completed-ckpt artifacts visible from outside the
    container. Outside Modal (e.g. from local tests) the import fails
    quietly — no commits are needed because there's no Modal volume to
    flush.
    """
    try:
        import modal

        from tmgg.modal._lib.volumes import OUTPUTS_VOLUME_NAME

        vol = modal.Volume.from_name(OUTPUTS_VOLUME_NAME)
        vol.commit()
    except Exception:  # noqa: BLE001
        logger.debug("Volume commit skipped (not in Modal worker)", exc_info=True)


def _append_index_row(index_path: Path, row: dict[str, Any]) -> None:
    """Append one row to ``index.jsonl`` (creating the file if needed)."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def eval_all_checkpoints_impl(
    experiment_dir: str,
    *,
    num_samples: int = 500,
    num_steps: int = 1000,
    wandb_project_suffix: str = "-eval-all",
    wandb_run_id_override: str | None = None,
    skip_last: bool = True,
    output_dir_root: str | None = None,
    viz_count: int = 32,
    val_batch_limit: int | None = None,
    ckpt_filter: list[str] | None = None,
    step_filter: list[int] | None = None,
) -> dict[str, Any]:
    """Sequentially evaluate every ckpt in a finished run's directory.

    The function logs one fresh W&B run per invocation; gen-val/*
    metrics are emitted with the trained-checkpoint step as the W&B
    step axis. We deliberately do NOT resume the original training
    run because (a) async-eval already owns that custom-step axis
    and reattaching from a separate worker risks step collisions,
    and (b) a fresh run makes post-hoc-eval dashboards cleanly
    distinguishable from in-band training metrics.

    Parameters
    ----------
    experiment_dir
        Path to the run-id directory on the shared volume,
        e.g. ``/data/outputs/discrete_qm9_digress_repro/<run_id>``.
    num_samples, num_steps
        Forwarded to ``run_mmd_evaluation``. Defaults match the
        async-eval worker's per-step budget; bump for end-of-run
        eval if you have more budget.
    wandb_project_suffix
        Appended to the trained run's ``wandb_project`` to form the
        eval-all project name. Default ``"-eval-all"`` so a trained
        ``discrete-qm9-digress-repro`` posts to
        ``discrete-qm9-digress-repro-eval-all``.
    wandb_run_id_override
        Custom W&B run ID. Default builds ``eval-all/<run_dir_name>``.
    skip_last
        Skip ``last.ckpt`` (default) — usually a duplicate of the
        latest stepped ckpt.
    output_dir_root
        Override for the per-ckpt subfolder root. Default is
        ``<experiment_dir>/eval_all/<eval_run_name>``. Tests pass a
        ``tmp_path``; production calls leave this ``None``.
    viz_count
        Per-side viz count under each ckpt's subfolder (32 generated
        + 32 reference by default). Forwarded to the evaluate-CLI.
    val_batch_limit
        Cap on val batches walked during the per-batch capture pass.
        ``None`` walks the full val split.
    ckpt_filter, step_filter
        See :func:`filter_checkpoint_entries`. Use to evaluate only
        an explicit subset of ckpts (e.g. one ckpt for a smoke test).

    Returns
    -------
    dict
        ``EvalAllOutput`` serialised via ``dataclasses.asdict`` with
        an extra ``output_dir_root`` field on each per-ckpt entry.
    """
    import wandb

    exp_dir = Path(experiment_dir)
    config = _load_run_config(exp_dir)

    wandb_project = str(config.get("wandb_project") or "graph-denoising-eval-all")
    wandb_entity = str(config.get("wandb_entity") or "your-wandb-entity")
    eval_project = f"{wandb_project}{wandb_project_suffix}"

    run_name = _resolve_wandb_run_name(exp_dir, wandb_run_id_override)

    entries = discover_checkpoints(exp_dir, skip_last=skip_last)
    discovered_total = len(entries)
    entries = filter_checkpoint_entries(
        entries, ckpt_filter=ckpt_filter, step_filter=step_filter
    )
    logger.info(
        "eval-all: discovered %d checkpoints under %s (after filters: %d)",
        discovered_total,
        exp_dir,
        len(entries),
    )

    # Resolve the dump root before W&B init so we can include it in the
    # run config. Default puts dumps under the run dir itself so the
    # whole eval state lives next to the trained ckpts on the volume.
    if output_dir_root is None:
        dump_root = exp_dir / "eval_all" / run_name.replace("/", "_")
    else:
        dump_root = Path(output_dir_root)
    dump_root.mkdir(parents=True, exist_ok=True)
    index_path = dump_root / "index.jsonl"

    wandb_run = wandb.init(
        project=eval_project,
        entity=wandb_entity,
        name=run_name,
        config={
            "experiment_dir": str(exp_dir),
            "num_samples": num_samples,
            "num_steps": num_steps,
            "num_checkpoints_total": discovered_total,
            "num_checkpoints_after_filter": len(entries),
            "ckpt_filter": ckpt_filter,
            "step_filter": step_filter,
            "viz_count": viz_count,
            "val_batch_limit": val_batch_limit,
            "dump_root": str(dump_root),
            "trained_run_config": config,
        },
        reinit=True,
    )
    # Define the custom step axis so per-ckpt logs land at their
    # training step rather than the implicit wandb call counter.
    wandb.define_metric("trainer/global_step")
    wandb.define_metric("eval-all/*", step_metric="trainer/global_step")

    output = EvalAllOutput(
        experiment_dir=str(exp_dir),
        wandb_run_id=str(wandb_run.id),
        num_checkpoints_total=discovered_total,
        num_evaluated=0,
    )

    for entry, result in _iter_evaluations(
        entries,
        run_id=exp_dir.name,
        num_samples=num_samples,
        num_steps=num_steps,
        output_dir_root=dump_root,
        viz_count=viz_count,
        val_batch_limit=val_batch_limit,
    ):
        step = result.get("resolved_step")
        log_payload: dict[str, Any] = {}
        # ``run_mmd_evaluation`` returns nested {"results": {label: {metric: v}}}
        # for the discrete CLI (and a flat ``results`` dict in the failure
        # case). Emit both shapes through the same eval-all/* prefix.
        results_dict = result.get("results", {})
        if isinstance(results_dict, dict):
            for label, metrics in results_dict.items():
                if not isinstance(metrics, dict):
                    log_payload[f"eval-all/{label}"] = metrics
                    continue
                for metric_name, value in metrics.items():
                    log_payload[f"eval-all/{label}/{metric_name}"] = value
        log_payload["eval-all/status"] = result.get("status")
        if step is not None:
            log_payload["trainer/global_step"] = int(step)
            wandb.log(log_payload)
        else:
            # No step resolvable: log without step routing rather than
            # silently dropping the row.
            wandb.log(log_payload)

        ckpt_subdir = dump_root / entry.path.stem
        per_ckpt_entry = {
            "checkpoint_name": entry.path.name,
            "step": step,
            "status": result.get("status"),
            "metrics": results_dict,
            "error_message": result.get("error_message"),
            "output_dir": str(ckpt_subdir),
        }
        output.per_checkpoint.append(per_ckpt_entry)
        # Index row + volume commit AFTER per-ckpt artifacts are on
        # disk. A worker timeout mid-panel leaves the completed
        # ckpts' dumps + their index entries durable.
        _append_index_row(
            index_path,
            {
                "checkpoint_name": entry.path.name,
                "checkpoint_path": str(entry.path),
                "step": step,
                "status": result.get("status"),
                "output_dir": str(ckpt_subdir),
                "error_message": result.get("error_message"),
            },
        )
        _commit_outputs_volume()
        if result.get("status") == "completed":
            output.num_evaluated += 1

    wandb.finish()
    return asdict(output)

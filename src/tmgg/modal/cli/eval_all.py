"""``tmgg-modal eval-all`` CLI subcommand.

Usage::

    tmgg-modal eval-all <experiment_dir> [--gpu-tier=standard]
                                          [--num-samples=500]
                                          [--num-steps=1000]
                                          [--no-skip-last]
                                          [--detach/--no-detach]

``<experiment_dir>`` is the run-id directory **on the shared
``graph-denoising-outputs`` volume**, e.g.
``/data/outputs/discrete_qm9_digress_repro/discrete_qm9_..._fresh_...``.

Looks up the matching ``eval_all_checkpoints[_<tier>]`` function
deployed in the ``graph-denoising-eval-all`` Modal app, calls it remotely, and
prints the structured aggregate report. ``--detach`` switches to
fire-and-forget so the worker keeps running after the local command
exits — useful for long panels (one-shot evals on 50+ ckpts can take
2-3h).
"""

from __future__ import annotations

import json
from typing import Any

import click

from tmgg.modal._eval_all_functions import EVAL_ALL_APP_NAME

GPU_TIER_CHOICES = click.Choice(["debug", "standard", "fast"], case_sensitive=False)


def _function_name(gpu_tier: str) -> str:
    if gpu_tier == "standard":
        return "eval_all_checkpoints"
    return f"eval_all_checkpoints_{gpu_tier}"


def _lookup_function(name: str) -> Any:
    """Resolve the deployed Modal function handle by name."""
    import modal

    return modal.Function.from_name(EVAL_ALL_APP_NAME, name)


def _print_report(report: dict[str, Any]) -> None:
    click.echo(json.dumps(report, indent=2, sort_keys=True, default=str))


@click.command("eval-all")
@click.argument("experiment_dir", type=str)
@click.option(
    "--gpu-tier",
    type=GPU_TIER_CHOICES,
    default="standard",
    show_default=True,
    help="GPU tier — debug=T4, standard=A10G, fast=A100-40GB.",
)
@click.option(
    "--num-samples",
    type=int,
    default=500,
    show_default=True,
    help="Generated graphs per checkpoint for MMD evaluation.",
)
@click.option(
    "--num-steps",
    type=int,
    default=1000,
    show_default=True,
    help="Denoising steps per generated graph.",
)
@click.option(
    "--skip-last/--no-skip-last",
    default=True,
    show_default=True,
    help="Skip last.ckpt (usually duplicates the latest stepped ckpt).",
)
@click.option(
    "--detach/--no-detach",
    default=True,
    show_default=True,
    help=(
        "Fire-and-forget. With --no-detach the local CLI blocks until the "
        "worker finishes — fine for short panels, painful for 50+ ckpts."
    ),
)
@click.option(
    "--ckpt",
    "ckpt_filter",
    multiple=True,
    help=(
        "Restrict to these ckpts (matched against full path or filename). "
        "Repeatable: --ckpt foo.ckpt --ckpt bar.ckpt."
    ),
)
@click.option(
    "--steps",
    "step_filter_str",
    type=str,
    default=None,
    help=(
        "Restrict to these training steps (comma-sep ints, e.g. "
        "--steps 5000,10000,25000). Combine with --ckpt to intersect."
    ),
)
@click.option(
    "--viz-count",
    type=int,
    default=32,
    show_default=True,
    help=(
        "Per-side viz PNG count under each ckpt's eval_all/<run>/<ckpt>/viz/ subfolder."
    ),
)
@click.option(
    "--val-batch-limit",
    type=int,
    default=None,
    help=(
        "Cap on val batches walked during the per-batch capture pass. "
        "Default: walk the full val split."
    ),
)
@click.option(
    "--output-dir-root",
    type=str,
    default=None,
    help=(
        "Override per-ckpt subfolder root. Default: "
        "<experiment_dir>/eval_all/<eval_run_name>."
    ),
)
def eval_all(
    experiment_dir: str,
    gpu_tier: str,
    num_samples: int,
    num_steps: int,
    skip_last: bool,
    detach: bool,
    ckpt_filter: tuple[str, ...],
    step_filter_str: str | None,
    viz_count: int,
    val_batch_limit: int | None,
    output_dir_root: str | None,
) -> None:
    """Run MMD evaluation on every checkpoint in a finished run dir."""
    fn_name = _function_name(gpu_tier)
    fn = _lookup_function(fn_name)
    step_filter: list[int] | None = None
    if step_filter_str:
        step_filter = [int(x) for x in step_filter_str.split(",") if x.strip()]
    kwargs: dict[str, Any] = {
        "num_samples": num_samples,
        "num_steps": num_steps,
        "skip_last": skip_last,
        "viz_count": viz_count,
        "val_batch_limit": val_batch_limit,
        "ckpt_filter": list(ckpt_filter) if ckpt_filter else None,
        "step_filter": step_filter,
        "output_dir_root": output_dir_root,
    }
    if detach:
        call = fn.spawn(experiment_dir, **kwargs)
        click.echo(
            json.dumps(
                {
                    "spawned": True,
                    "function_call_id": call.object_id,
                    "function": fn_name,
                    "experiment_dir": experiment_dir,
                    "kwargs": kwargs,
                    "app": EVAL_ALL_APP_NAME,
                    "tail_logs": (f"modal app logs {EVAL_ALL_APP_NAME}"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    report = fn.remote(experiment_dir, **kwargs)
    _print_report(report)

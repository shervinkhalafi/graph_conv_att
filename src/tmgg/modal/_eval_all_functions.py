"""Modal deployment definitions for ``graph-denoising-eval-all``.

Single ``@app.function`` per GPU tier that walks all checkpoints in a
finished training run on the shared ``graph-denoising-outputs`` volume and pushes
gen-val/* metrics to a fresh W&B run with the ckpt step as the W&B step.

Deploy::

    uv run modal deploy -m tmgg.modal._eval_all_functions

Serve (hot-reload, only useful when iterating on the metric stack)::

    uv run modal serve -m tmgg.modal._eval_all_functions

Why a separate app:
- ``graph-denoising`` holds the long-running training containers; redeploying
  it interrupts live runs. The eval-all tool is post-hoc — it should be
  iterable without disturbing trainers.
- A separate ``modal app logs graph-denoising-eval-all`` stream is much easier to
  follow than greping eval-related lines out of mixed training logs.
"""

# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from typing import Any

import modal

from tmgg.modal._lib.image import create_tmgg_image
from tmgg.modal._lib.volumes import get_volume_mounts
from tmgg.modal.app import (
    CPU_PROFILES,
    DEFAULT_SCALEDOWN_WINDOW,
    GPU_CONFIGS,
)

EVAL_ALL_APP_NAME = "graph-denoising-eval-all"
"""Separate from ``MODAL_APP_NAME`` (``graph-denoising``)."""

# 4 hours is enough to evaluate ~30-50 checkpoints with default
# 500 samples × 1000 timesteps. Bump per-tier if you have a longer
# panel and don't want to break the run into batches.
EVAL_ALL_TIMEOUT = 14400

app = modal.App(EVAL_ALL_APP_NAME, include_source=False)

try:
    from tmgg.modal._lib.paths import discover_source_checkout_path

    # ``PYTHONOPTIMIZE=1`` is now baked into the shared image env in
    # ``_lib/image._runtime_env`` so this in-process eval-all app gets
    # the same production-faithful behaviour the training subprocess
    # has always had — strips the symmetry / row-stochastic /
    # mask-correctness ``if __debug__:`` blocks at bytecode compile
    # time. Was a silent ~10 s/eval-cycle host-sync regression before.
    experiment_image = create_tmgg_image(discover_source_checkout_path())
except (ImportError, RuntimeError):
    experiment_image = None

wandb_secret = modal.Secret.from_name(
    "wandb-credentials",
    required_keys=["WANDB_API_KEY"],
)


def _eval_all_impl(
    experiment_dir: str,
    *,
    num_samples: int,
    num_steps: int,
    wandb_project_suffix: str,
    skip_last: bool,
    output_dir_root: str | None,
    viz_count: int,
    val_batch_limit: int | None,
    ckpt_filter: list[str] | None,
    step_filter: list[int] | None,
) -> dict[str, Any]:
    """Tier-agnostic dispatch into the library implementation."""
    from tmgg.modal._lib.eval_all import eval_all_checkpoints_impl

    return eval_all_checkpoints_impl(
        experiment_dir,
        num_samples=num_samples,
        num_steps=num_steps,
        wandb_project_suffix=wandb_project_suffix,
        skip_last=skip_last,
        output_dir_root=output_dir_root,
        viz_count=viz_count,
        val_batch_limit=val_batch_limit,
        ckpt_filter=ckpt_filter,
        step_filter=step_filter,
    )


@app.function(
    name="eval_all_checkpoints",
    image=experiment_image,
    gpu=GPU_CONFIGS["standard"],
    cpu=CPU_PROFILES["standard"],
    timeout=EVAL_ALL_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[wandb_secret],
    volumes=get_volume_mounts(),
)
def eval_all_checkpoints(
    experiment_dir: str,
    *,
    num_samples: int = 500,
    num_steps: int = 1000,
    wandb_project_suffix: str = "-eval-all",
    skip_last: bool = True,
    output_dir_root: str | None = None,
    viz_count: int = 32,
    val_batch_limit: int | None = None,
    ckpt_filter: list[str] | None = None,
    step_filter: list[int] | None = None,
) -> dict[str, Any]:
    """Standard-tier (A10G) eval-all worker. See ``eval_all_checkpoints_impl``."""
    return _eval_all_impl(
        experiment_dir,
        num_samples=num_samples,
        num_steps=num_steps,
        wandb_project_suffix=wandb_project_suffix,
        skip_last=skip_last,
        output_dir_root=output_dir_root,
        viz_count=viz_count,
        val_batch_limit=val_batch_limit,
        ckpt_filter=ckpt_filter,
        step_filter=step_filter,
    )


@app.function(
    name="eval_all_checkpoints_fast",
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],
    cpu=CPU_PROFILES["fast"],
    timeout=EVAL_ALL_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[wandb_secret],
    volumes=get_volume_mounts(),
)
def eval_all_checkpoints_fast(
    experiment_dir: str,
    *,
    num_samples: int = 500,
    num_steps: int = 1000,
    wandb_project_suffix: str = "-eval-all",
    skip_last: bool = True,
    output_dir_root: str | None = None,
    viz_count: int = 32,
    val_batch_limit: int | None = None,
    ckpt_filter: list[str] | None = None,
    step_filter: list[int] | None = None,
) -> dict[str, Any]:
    """Fast-tier (A100-40GB) eval-all worker."""
    return _eval_all_impl(
        experiment_dir,
        num_samples=num_samples,
        num_steps=num_steps,
        wandb_project_suffix=wandb_project_suffix,
        skip_last=skip_last,
        output_dir_root=output_dir_root,
        viz_count=viz_count,
        val_batch_limit=val_batch_limit,
        ckpt_filter=ckpt_filter,
        step_filter=step_filter,
    )


@app.function(
    name="eval_all_checkpoints_debug",
    image=experiment_image,
    gpu=GPU_CONFIGS["debug"],
    cpu=CPU_PROFILES["debug"],
    timeout=EVAL_ALL_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[wandb_secret],
    volumes=get_volume_mounts(),
)
def eval_all_checkpoints_debug(
    experiment_dir: str,
    *,
    num_samples: int = 500,
    num_steps: int = 1000,
    wandb_project_suffix: str = "-eval-all",
    skip_last: bool = True,
    output_dir_root: str | None = None,
    viz_count: int = 32,
    val_batch_limit: int | None = None,
    ckpt_filter: list[str] | None = None,
    step_filter: list[int] | None = None,
) -> dict[str, Any]:
    """Debug-tier (T4) eval-all worker — only for sanity-checking the pipe."""
    return _eval_all_impl(
        experiment_dir,
        num_samples=num_samples,
        num_steps=num_steps,
        wandb_project_suffix=wandb_project_suffix,
        skip_last=skip_last,
        output_dir_root=output_dir_root,
        viz_count=viz_count,
        val_batch_limit=val_batch_limit,
        ckpt_filter=ckpt_filter,
        step_filter=step_filter,
    )

"""Modal app: torch.profiler wrappers around training + eval entrypoints.

Two ``@app.function``s, both on ``GPU_CONFIGS["fast"]`` (A100). Run
both in parallel via ``.spawn()`` from the launcher script::

    uv run python -m scripts.profiling.launch_profile

Each function dumps ``trace.json`` + ``summary.txt`` to::

    /outputs/profiles/<run_tag>/<train|eval>/

on the ``graph-denoising-outputs`` Modal volume; the launcher pulls those files
back to the host with ``modal volume get`` once the spawns finish.

Deploy::

    uv run modal deploy -m tmgg.modal._profile_functions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from tmgg.modal._lib.image import create_tmgg_image
from tmgg.modal._lib.paths import discover_source_checkout_path
from tmgg.modal._lib.volumes import get_volume_mounts
from tmgg.modal.app import (
    CPU_PROFILES,
    DEFAULT_SCALEDOWN_WINDOW,
    GPU_CONFIGS,
)

PROFILE_APP_NAME = "tmgg-profile"
PROFILE_TIMEOUT = 1800  # 30 minutes — both runners are intentionally short

app = modal.App(PROFILE_APP_NAME, include_source=False)

# ``PYTHONOPTIMIZE=1`` is set centrally in ``_lib/image._runtime_env``
# so the profile container Python boots with ``__debug__=False`` —
# matches production training and strips the symmetry / mask /
# row-stochastic asserts that would otherwise distort the profile.
experiment_image = create_tmgg_image(discover_source_checkout_path())

wandb_secret = modal.Secret.from_name(
    "wandb-credentials",
    required_keys=["WANDB_API_KEY"],
)


@app.function(
    name="profile_train",
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],
    cpu=CPU_PROFILES["fast"],
    timeout=PROFILE_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[wandb_secret],
    # get_volume_mounts() returns dict[str, Any] (shared infra) which
    # basedpyright rejects against modal's invariant dict[str | PurePosixPath,
    # Volume | CloudBucketMount]. Silenced per shared-helper pattern; root fix
    # would be widening the helper's return annotation across all callers.
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def profile_train(
    *,
    overrides: list[str],
    output_dir_on_volume: str,
    config_dir: str = "/app/tmgg/src/tmgg/experiments/exp_configs",
    config_name: str = "base_config_discrete_diffusion_generative",
    num_steps: int = 100,
    warmup_steps: int = 5,
    active_steps: int = 20,
) -> dict[str, Any]:
    """Profile a short training run (no validation cycles).

    Parameters mirror ``tmgg.profile.runners.run_train_profile``.
    ``output_dir_on_volume`` MUST live under ``/outputs/`` so the
    artefacts persist on the ``graph-denoising-outputs`` volume.
    """
    from tmgg.profile.runners import run_train_profile

    return run_train_profile(
        output_dir=Path(output_dir_on_volume),
        overrides=list(overrides),
        config_dir=Path(config_dir),
        config_name=config_name,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        active_steps=active_steps,
    )


@app.function(
    name="profile_eval",
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],
    cpu=CPU_PROFILES["fast"],
    timeout=PROFILE_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[wandb_secret],
    # get_volume_mounts() returns dict[str, Any] (shared infra) which
    # basedpyright rejects against modal's invariant dict[str | PurePosixPath,
    # Volume | CloudBucketMount]. Silenced per shared-helper pattern; root fix
    # would be widening the helper's return annotation across all callers.
    volumes=get_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def profile_eval(
    *,
    checkpoint_path: str,
    output_dir_on_volume: str,
    num_samples: int = 32,
    val_batch_limit: int | None = None,
    viz_count: int = 4,
    use_ema: str = "auto",
    compile_model: bool = False,
    compile_mode: str = "default",
    sample_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Profile one ``evaluate_checkpoint`` cycle on a checkpoint.

    ``checkpoint_path`` MUST be a path on the ``graph-denoising-outputs`` volume
    (e.g. ``/outputs/.../checkpoints/last.ckpt``). ``output_dir_on
    _volume`` similarly under ``/outputs/``.
    """
    from tmgg.profile.runners import run_eval_profile

    return run_eval_profile(
        output_dir=Path(output_dir_on_volume),
        checkpoint_path=Path(checkpoint_path),
        num_samples=num_samples,
        val_batch_limit=val_batch_limit,
        viz_count=viz_count,
        device="cuda",
        use_ema=use_ema,
        compile_model=compile_model,
        compile_mode=compile_mode,
        sample_chunk_size=sample_chunk_size,
    )

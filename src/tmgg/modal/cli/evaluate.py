"""Dispatch MMD evaluation of trained checkpoints to Modal GPUs.

Calls the deployed ``modal_evaluate_mmd`` function via
``modal.Function.from_name()``. Supports single-checkpoint,
all-checkpoint, and fire-and-forget modes.
"""

from __future__ import annotations

from typing import Any

import click


def _select_modal_function(gpu_tier: str) -> Any:
    """Resolve the deployed Modal evaluation function for the requested GPU tier."""
    import modal

    from tmgg.modal.app import MODAL_APP_NAME, resolve_modal_function_name

    func_name = resolve_modal_function_name("modal_evaluate_mmd", gpu_tier)
    return modal.Function.from_name(MODAL_APP_NAME, func_name)


def _get_list_checkpoints_function() -> Any:
    """Resolve the deployed ``modal_list_checkpoints`` function."""
    import modal

    from tmgg.modal.app import MODAL_APP_NAME

    return modal.Function.from_name(MODAL_APP_NAME, "modal_list_checkpoints")


def _build_task_dict(
    run_id: str,
    *,
    checkpoint_path: str | None,
    num_samples: int,
    num_steps: int,
    mmd_kernel: str,
    mmd_sigma: float,
    seed: int,
) -> dict[str, Any]:
    """Build the evaluation task payload sent to the Modal function."""
    return {
        "run_id": run_id,
        "checkpoint_path": checkpoint_path,
        "num_samples": num_samples,
        "num_steps": num_steps,
        "mmd_kernel": mmd_kernel,
        "mmd_sigma": mmd_sigma,
        "seed": seed,
    }


def _run_single(
    modal_fn: Any,
    task_dict: dict[str, Any],
    *,
    wait: bool,
    run_id: str,
    checkpoint_name: str,
) -> bool:
    """Dispatch one checkpoint evaluation. Returns True on success."""
    if not wait:
        fc = modal_fn.spawn(task_dict)
        click.echo(f"  Spawned {checkpoint_name} -> {fc.object_id}")
        return True

    click.echo(f"\nEvaluating {checkpoint_name}...")
    try:
        result = modal_fn.remote(task_dict)
    except Exception as exc:
        raise click.ClickException(f"Remote call failed: {exc}") from exc

    status = result.get("status", "unknown")
    ckpt_name = result.get("checkpoint_name", checkpoint_name)
    click.echo(f"  Status: {status}")

    if status == "completed":
        for label, metrics in result.get("results", {}).items():
            click.echo(
                f"    {label}: degree={metrics.get('degree_mmd', 0):.6f}, "
                f"clustering={metrics.get('clustering_mmd', 0):.6f}, "
                f"spectral={metrics.get('spectral_mmd', 0):.6f}"
            )
        click.echo(
            f"  Saved to: /data/outputs/{run_id}/mmd_evaluation_{ckpt_name}.json"
        )
        return True

    error = result.get("error_message", "unknown error")
    click.echo(f"  FAILED: {error}", err=True)
    return False


@click.command()
@click.option(
    "--run-id", "-r", required=True, help="Run ID whose checkpoint(s) to evaluate."
)
@click.option(
    "--checkpoint",
    "-c",
    default=None,
    help="Explicit checkpoint path. Defaults to last.ckpt.",
)
@click.option(
    "--all-checkpoints",
    is_flag=True,
    default=False,
    help="Evaluate every checkpoint in the run directory.",
)
@click.option(
    "--num-samples",
    "-n",
    type=int,
    default=500,
    show_default=True,
    help="Graphs to generate for evaluation.",
)
@click.option(
    "--num-steps",
    type=int,
    default=100,
    show_default=True,
    help="Denoising steps per sample.",
)
@click.option(
    "--mmd-kernel",
    type=click.Choice(["gaussian", "gaussian_tv"], case_sensitive=False),
    default="gaussian_tv",
    show_default=True,
    help="Kernel for MMD computation.",
)
@click.option(
    "--mmd-sigma", type=float, default=1.0, show_default=True, help="Kernel bandwidth."
)
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
@click.option(
    "--gpu",
    "-g",
    type=click.Choice(["debug", "standard", "fast"], case_sensitive=False),
    default="standard",
    show_default=True,
    help="GPU tier (debug=T4, standard=A10G, fast=A100).",
)
@click.option(
    "--wait/--no-wait",
    default=True,
    show_default=True,
    help="Block until results arrive, or fire-and-forget.",
)
def evaluate(
    run_id: str,
    checkpoint: str | None,
    all_checkpoints: bool,
    num_samples: int,
    num_steps: int,
    mmd_kernel: str,
    mmd_sigma: float,
    seed: int,
    gpu: str,
    wait: bool,
) -> None:
    """Dispatch MMD evaluation of trained checkpoints to Modal GPUs.

    \b
    Examples:
      tmgg-modal evaluate -r digress_sbm_vanilla
      tmgg-modal evaluate -r digress_sbm_vanilla --all-checkpoints --no-wait
      tmgg-modal evaluate -r digress_sbm_vanilla -c /data/.../best.ckpt -g debug
    """
    import modal

    if checkpoint and all_checkpoints:
        raise click.UsageError(
            "--checkpoint and --all-checkpoints are mutually exclusive."
        )

    try:
        modal_fn = _select_modal_function(gpu)
    except modal.exception.NotFoundError as exc:
        raise click.ClickException(
            "Modal app not deployed. Run 'mise run modal-deploy' first.\n"
            f"Details: {exc}"
        ) from exc

    click.echo(f"Evaluating run: {run_id}")
    click.echo(f"  Samples: {num_samples}, Steps: {num_steps}, GPU: {gpu}")

    # --- all checkpoints ---
    if all_checkpoints:
        click.echo("\nDiscovering checkpoints...")
        try:
            list_fn = _get_list_checkpoints_function()
            list_result = list_fn.remote(run_id)
        except modal.exception.NotFoundError as exc:
            raise click.ClickException(
                "modal_list_checkpoints not deployed. Redeploy the Modal app."
            ) from exc

        if list_result.get("status") != "completed":
            raise click.ClickException(
                list_result.get("error_message", "Unknown error")
            )

        checkpoints = list_result.get("checkpoints", [])
        if not checkpoints:
            raise click.ClickException(f"No checkpoints found for run '{run_id}'.")

        click.echo(f"Found {len(checkpoints)} checkpoint(s):")
        for ckpt in checkpoints:
            click.echo(f"  - {ckpt['name']} ({ckpt['size_mb']:.1f} MB)")

        any_failed = False
        for ckpt in checkpoints:
            task_dict = _build_task_dict(
                run_id,
                checkpoint_path=ckpt["path"],
                num_samples=num_samples,
                num_steps=num_steps,
                mmd_kernel=mmd_kernel,
                mmd_sigma=mmd_sigma,
                seed=seed,
            )
            ok = _run_single(
                modal_fn,
                task_dict,
                wait=wait,
                run_id=run_id,
                checkpoint_name=ckpt["name"],
            )
            if not ok:
                any_failed = True

        if not wait:
            click.echo("\nAll evaluations spawned. Check Modal dashboard for progress.")
        if any_failed:
            raise click.ClickException("One or more evaluations failed.")
        return

    # --- single checkpoint ---
    checkpoint_name = (
        "last"
        if not checkpoint
        else checkpoint.rsplit("/", maxsplit=1)[-1].replace(".ckpt", "")
    )
    task_dict = _build_task_dict(
        run_id,
        checkpoint_path=checkpoint,
        num_samples=num_samples,
        num_steps=num_steps,
        mmd_kernel=mmd_kernel,
        mmd_sigma=mmd_sigma,
        seed=seed,
    )

    if not wait:
        fc = modal_fn.spawn(task_dict)
        click.echo(f"\nSpawned evaluation -> {fc.object_id}")
        click.echo("Check Modal dashboard or 'modal function list' for status.")
        return

    click.echo("\nRunning evaluation (blocking)...")
    try:
        result = modal_fn.remote(task_dict)
    except Exception as exc:
        raise click.ClickException(f"Remote call failed: {exc}") from exc

    status = result.get("status", "unknown")
    ckpt_name = result.get("checkpoint_name", checkpoint_name)
    click.echo(f"\nStatus: {status}")
    click.echo(f"Checkpoint: {ckpt_name}")

    if status == "completed":
        results = result.get("results", {})
        click.echo("\nMMD Results:")
        click.echo("-" * 50)
        for label, metrics in results.items():
            click.echo(f"\n  {label}:")
            for metric, value in metrics.items():
                click.echo(f"    {metric}: {value:.6f}")
        click.echo("-" * 50)
        click.echo(
            f"\nResults saved to: /data/outputs/{run_id}/mmd_evaluation_{ckpt_name}.json"
        )
    else:
        raise click.ClickException(result.get("error_message", "Unknown error"))

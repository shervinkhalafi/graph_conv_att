"""Dispatch a single experiment to Modal with Hydra override support.

Uses Hydra's ``compose`` API to resolve the full config locally, then
hands it to ``ModalRunner`` for serialization and remote dispatch.  This
is the single-run counterpart to the ``TmggLauncher`` multirun path —
both feed into the same ``ModalRunner`` serialization boundary.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import click
from omegaconf import OmegaConf


def _write_launch_log(
    *,
    cli_cmd: str,
    config_name: str,
    overrides: list[str],
    gpu_tier: str,
    detach: bool,
    run_id: str,
    config_hash: str,
) -> None:
    """Append a launch record to ``.local-storage/modal_launches.jsonl``.

    Creates the directory if it doesn't exist.  Each line is one JSON
    object.  Failures are logged but don't abort the command.
    """
    from tmgg.modal._lib.paths import discover_tmgg_path

    log_dir = discover_tmgg_path() / ".local-storage"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "modal_launches.jsonl"

    record = {
        "launched_at": datetime.now(UTC).isoformat(),
        "cli_cmd": cli_cmd,
        "config_name": config_name,
        "overrides": overrides,
        "gpu_tier": gpu_tier,
        "detach": detach,
        "run_id": run_id,
        "config_hash": config_hash,
    }

    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        click.echo(f"Warning: could not write launch log: {exc}", err=True)


@click.command()
@click.argument("cli_cmd")
@click.argument("overrides", nargs=-1)
@click.option(
    "--gpu",
    "-g",
    type=click.Choice(
        ["debug", "standard", "fast", "multi", "h100"], case_sensitive=False
    ),
    default="standard",
    show_default=True,
    help="GPU tier (debug=T4, standard=A10G, fast=A100, h100=H100).",
)
@click.option(
    "--detach/--no-detach",
    default=False,
    show_default=True,
    help="Fire-and-forget (spawn) vs blocking (remote).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Resolve config and print summary without dispatching to Modal.",
)
def run(
    cli_cmd: str,
    overrides: tuple[str, ...],
    gpu: str,
    detach: bool,
    dry_run: bool,
) -> None:
    """Launch a single experiment on Modal.

    CLI_CMD is the experiment entry point name (e.g. tmgg-spectral-arch).
    OVERRIDES are Hydra-style key=value pairs applied on top of the base
    config.

    \b
    Examples:
      tmgg-modal run tmgg-spectral-arch model.k=16 seed=1
      tmgg-modal run tmgg-discrete-gen model.diffusion_steps=50 --gpu fast
      tmgg-modal run tmgg-spectral-arch seed=2 --detach
      tmgg-modal run tmgg-spectral-arch --dry-run
    """
    from tmgg.modal._lib.config_resolution import discover_cli_cmd_map, resolve_config

    # 1. Look up CLI command -> config name
    cmd_map = discover_cli_cmd_map()

    if cli_cmd not in cmd_map:
        known = ", ".join(sorted(cmd_map))
        raise click.UsageError(
            f"Unknown CLI command: {cli_cmd}\nKnown commands: {known}"
        )

    config_name = cmd_map[cli_cmd]
    override_list = list(overrides)

    # 2. Resolve full config via Hydra compose
    try:
        cfg = resolve_config(config_name, override_list)
    except Exception as exc:
        raise click.ClickException(f"Config resolution failed: {exc}") from exc

    # 3. Compute config hash and extract metadata
    config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    config_hash = hashlib.sha256(config_yaml.encode()).hexdigest()[:12]
    run_id = str(cfg.get("run_id", "unknown"))

    # 4. Print summary
    click.echo(f"Command:     {cli_cmd}")
    click.echo(f"Config:      {config_name}")
    click.echo(f"Run ID:      {run_id}")
    click.echo(f"Config hash: {config_hash}")
    click.echo(f"GPU tier:    {gpu}")
    if override_list:
        click.echo(f"Overrides:   {' '.join(override_list)}")
    if detach:
        click.echo("Mode:        detach (fire-and-forget)")

    # 5. Dry run: print resolved YAML and exit
    if dry_run:
        click.echo("\n--- Resolved config ---")
        click.echo(config_yaml)
        return

    # 6. Dispatch to Modal
    from tmgg.modal.runner import ModalNotDeployedError, ModalRunner

    try:
        runner = ModalRunner(gpu_type=gpu)
    except ModalNotDeployedError as exc:
        raise click.ClickException(str(exc)) from exc

    if detach:
        click.echo("\nSpawning experiment (detached)...")
        task = runner.spawn_experiment(cfg, gpu_type=gpu)
        click.echo(f"Spawned: run_id={task.run_id}, gpu={task.gpu_tier}")
        # Stable marker the launcher (``internal sweep tooling (not included)``)
        # parses to record the trainer's FunctionCall ID on the launched
        # JSONL row, enabling manual cancel via ``internal sweep tooling (not included)``.
        # ``function_call`` is None only on legacy / non-Modal paths; we
        # echo nothing in that case so the parser falls back to None.
        if task.function_call is not None:
            click.echo(f"MODAL_FUNCTION_CALL_ID={task.function_call.object_id}")
        click.echo("Check Modal dashboard or confirmation log for progress.")
    else:
        click.echo("\nDispatching experiment (blocking)...")
        result = runner.run_experiment(cfg, gpu_type=gpu)
        click.echo(f"\nStatus: {result.status}")
        if result.status == "completed":
            if result.metrics:
                click.echo("Metrics:")
                for k, v in result.metrics.items():
                    click.echo(f"  {k}: {v}")
        elif result.error_message:
            click.echo(f"Error: {result.error_message}", err=True)

    # 7. Log launch locally
    _write_launch_log(
        cli_cmd=cli_cmd,
        config_name=config_name,
        overrides=override_list,
        gpu_tier=gpu,
        detach=detach,
        run_id=run_id,
        config_hash=config_hash,
    )

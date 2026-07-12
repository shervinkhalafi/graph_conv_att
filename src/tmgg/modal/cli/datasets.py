"""Dataset prepare + validate dispatch on Modal.

Usage:

    tmgg-modal datasets prepare <name>
    tmgg-modal datasets validate <name>

Where ``<name>`` is one of: qm9, moses, guacamol, planar, sbm.

Each subcommand looks up the matching ``modal_prepare_<name>`` /
``modal_validate_<name>`` function deployed in the ``graph-denoising``
app, calls it remotely (blocking by default), and prints the structured
report. ``--detach`` switches to fire-and-forget so the worker keeps
running after the local command exits.
"""

from __future__ import annotations

import json

import click

from tmgg.modal._lib.dataset_ops import ALL_DATASETS
from tmgg.modal.app import MODAL_APP_NAME

DATASET_CHOICE = click.Choice(list(ALL_DATASETS), case_sensitive=False)


def _lookup_function(name: str):  # type: ignore[no-untyped-def]
    """Resolve the deployed Modal function handle by short name + verb."""
    import modal

    return modal.Function.from_name(MODAL_APP_NAME, name)


def _print_report(report: dict[str, object]) -> None:
    click.echo(json.dumps(report, indent=2, sort_keys=True, default=str))


@click.group("datasets")
def datasets() -> None:
    """Prepare or validate datasets on the shared ``graph-denoising-datasets`` volume."""


@datasets.command("prepare")
@click.argument("name", type=DATASET_CHOICE)
@click.option(
    "--detach/--no-detach",
    default=True,
    show_default=True,
    help=(
        "Fire-and-forget (spawn) [default] vs. block until the report "
        "returns (--no-detach). Detached calls survive a local Ctrl+C and "
        "let you tail logs via ``modal app logs graph-denoising``."
    ),
)
def prepare(name: str, detach: bool) -> None:
    """Download + preprocess one dataset on Modal."""
    fn = _lookup_function(f"modal_prepare_{name}")
    if detach:
        call = fn.spawn()  # type: ignore[attr-defined]
        click.echo(
            json.dumps(
                {
                    "status": "spawned",
                    "function": f"modal_prepare_{name}",
                    "function_call_id": getattr(call, "object_id", None),
                },
                indent=2,
            )
        )
        return
    report = fn.remote()  # type: ignore[attr-defined]
    _print_report(report)


@datasets.command("validate")
@click.argument("name", type=DATASET_CHOICE)
@click.option(
    "--detach/--no-detach",
    default=True,
    show_default=True,
    help=(
        "Fire-and-forget (spawn) [default] vs. block until the report "
        "returns (--no-detach). Detached calls survive a local Ctrl+C and "
        "let you tail logs via ``modal app logs graph-denoising``."
    ),
)
def validate(name: str, detach: bool) -> None:
    """Validate the on-volume artifacts for one dataset."""
    fn = _lookup_function(f"modal_validate_{name}")
    if detach:
        call = fn.spawn()  # type: ignore[attr-defined]
        click.echo(
            json.dumps(
                {
                    "status": "spawned",
                    "function": f"modal_validate_{name}",
                    "function_call_id": getattr(call, "object_id", None),
                },
                indent=2,
            )
        )
        return
    report = fn.remote()  # type: ignore[attr-defined]
    _print_report(report)


# Re-exported for ``tmgg.modal.cli.__init__``.
__all__ = ["datasets"]

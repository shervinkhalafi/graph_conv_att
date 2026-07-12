"""Tests for the generic Modal single-run CLI."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from click.testing import CliRunner
from omegaconf import OmegaConf

from tmgg.modal.cli.run import run as run_command


@dataclass
class _DummyFunctionCall:
    """Minimal stand-in for ``modal.functions.FunctionCall`` for tests."""

    object_id: str


@dataclass
class _DummyTask:
    """Minimal stand-in for ModalSpawnedTask.

    Carries ``function_call`` so the CLI's ``MODAL_FUNCTION_CALL_ID=...``
    marker (consumed by ``internal sweep tooling (not included)``) can be emitted
    in tests too.
    """

    run_id: str
    gpu_tier: str
    function_call: _DummyFunctionCall | None = None


class _DummyRunner:
    """Capture detached launch requests without touching Modal."""

    def __init__(self, gpu_type: str = "debug") -> None:
        self.gpu_type = gpu_type

    def spawn_experiment(self, config, gpu_type: str | None = None) -> _DummyTask:
        return _DummyTask(
            run_id=str(config.run_id),
            gpu_tier=gpu_type or self.gpu_type,
            function_call=_DummyFunctionCall(object_id="fc-01TESTCALL"),
        )


def test_detached_run_does_not_print_predicted_wandb_url(
    monkeypatch,
) -> None:
    """Detached launches should not print a locally predicted W&B URL."""
    run_module = importlib.import_module("tmgg.modal.cli.run")
    cfg = OmegaConf.create(
        {
            "run_id": "discrete-test-run",
            "paths": {"output_dir": "/tmp/output", "results_dir": "/tmp/results"},
        }
    )

    monkeypatch.setattr(
        "tmgg.modal._lib.config_resolution.discover_cli_cmd_map",
        lambda: {"tmgg-discrete-gen": "base_config_discrete_diffusion_generative"},
    )
    monkeypatch.setattr(
        "tmgg.modal._lib.config_resolution.resolve_config",
        lambda config_name, overrides: cfg,
    )
    monkeypatch.setattr("tmgg.modal.runner.ModalRunner", _DummyRunner)
    monkeypatch.setattr(run_module, "_write_launch_log", lambda **kwargs: None)

    runner = CliRunner()
    result = runner.invoke(run_command, ["tmgg-discrete-gen", "--detach"])

    assert result.exit_code == 0, result.output
    assert "W&B URL:" not in result.output
    assert "Spawned: run_id=discrete-test-run, gpu=standard" in result.output


def test_detached_run_emits_modal_function_call_id_marker(
    monkeypatch,
) -> None:
    """The detached path prints ``MODAL_FUNCTION_CALL_ID=fc-...``.

    This is the marker that ``internal sweep tooling (not included)`` parses to
    record the trainer's call ID on the launched JSONL row, enabling
    manual cancel via ``internal sweep tooling (not included)``.
    """
    run_module = importlib.import_module("tmgg.modal.cli.run")
    cfg = OmegaConf.create(
        {
            "run_id": "discrete-test-run",
            "paths": {"output_dir": "/tmp/output", "results_dir": "/tmp/results"},
        }
    )

    monkeypatch.setattr(
        "tmgg.modal._lib.config_resolution.discover_cli_cmd_map",
        lambda: {"tmgg-discrete-gen": "base_config_discrete_diffusion_generative"},
    )
    monkeypatch.setattr(
        "tmgg.modal._lib.config_resolution.resolve_config",
        lambda config_name, overrides: cfg,
    )
    monkeypatch.setattr("tmgg.modal.runner.ModalRunner", _DummyRunner)
    monkeypatch.setattr(run_module, "_write_launch_log", lambda **kwargs: None)

    runner = CliRunner()
    result = runner.invoke(run_command, ["tmgg-discrete-gen", "--detach"])

    assert result.exit_code == 0, result.output
    assert "MODAL_FUNCTION_CALL_ID=fc-01TESTCALL" in result.output

"""Tests for Modal app wiring.

Test rationale
--------------
The experiment image already installs TMGG and compiles ORCA inside the
container image. The Modal app must therefore disable automatic source
inclusion, or the runtime will mount the local ``tmgg`` package under
``/root`` and shadow the image-installed package with any checked-in native
artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tmgg.modal._functions as modal_functions


def test_modal_app_disables_source_inclusion() -> None:
    """Runtime containers should import TMGG from the image, not from `/root`."""
    assert modal_functions.app._local_state.include_source_default is False


def test_extract_wandb_run_url_returns_full_url() -> None:
    """W&B parsing should preserve the full run URL for Modal log printing."""
    output = (
        "wandb: View run digress_sbm at:\n"
        "wandb: https://wandb.ai/your-wandb-entity/discrete-diffusion/runs/abcd1234\n"
    )

    assert (
        modal_functions._extract_wandb_run_url(output)
        == "https://wandb.ai/your-wandb-entity/discrete-diffusion/runs/abcd1234"
    )
    assert modal_functions._extract_wandb_run_id(output) == "abcd1234"


def test_format_return_code_labels_signals() -> None:
    """Signal-based subprocess exits should be labeled in diagnostics."""
    assert modal_functions._format_return_code(-4) == "-4 (SIGILL)"
    assert modal_functions._format_return_code(1) == "1"


def test_run_import_preflight_reports_failing_module(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Import preflight should identify the specific module that crashed."""

    class _Completed:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    responses = iter(
        [
            _Completed(0),
            _Completed(0),
            _Completed(-4),
        ]
    )

    def _run(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr("subprocess.run", _run)
    # Patch graph_tool as required=True for this specific test so the
    # raise-path is exercised. Production config has graph_tool optional
    # (Modal worker CPUs may lack the AVX-512 the bundled binary uses).
    monkeypatch.setattr(
        modal_functions,
        "IMPORT_PREFLIGHTS",
        (
            ("torch", "import torch", True),
            ("ot", "import ot", True),
            ("graph_tool", "import graph_tool.all as gt", True),
        ),
    )

    with pytest.raises(RuntimeError, match="Module: graph_tool") as excinfo:
        modal_functions._run_import_preflight()

    captured = capsys.readouterr().out
    assert "Preflight import OK: torch" in captured
    assert "Preflight import OK: ot" in captured
    assert "SIGILL" in str(excinfo.value)


def test_run_config_preflight_reports_failure_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config preflight should surface isolated batch/setup crashes clearly."""

    class _Completed:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: _Completed(
            -6,
            stdout="Config preflight: instantiate datamodule\n",
            stderr="terminate called after throwing an instance of 'std::length_error'\n",
        ),
    )

    with pytest.raises(RuntimeError, match="Modal config preflight failed") as excinfo:
        modal_functions._run_config_preflight(Path("/tmp/run_config.yaml"))

    assert "SIGABRT" in str(excinfo.value)
    assert "std::length_error" in str(excinfo.value)


def test_run_cli_impl_forwards_output_and_prints_wandb_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Modal wrapper should surface the real W&B URL from remote subprocess output."""

    confirmations: list[dict[str, object]] = []

    class _FakeStdout:
        def __iter__(self):
            return iter(
                [
                    "starting run\n",
                    "wandb: https://wandb.ai/your-wandb-entity/discrete-diffusion/runs/abcd1234\n",
                    "done\n",
                ]
            )

    class _FakePopen:
        def __init__(self, args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.stdout = _FakeStdout()

        def wait(self) -> int:
            return 0

    def _append_confirmation(path: Path, **entry: object) -> None:
        confirmations.append({"path": path, **entry})

    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr(modal_functions, "_run_import_preflight", lambda: None)
    monkeypatch.setattr(
        modal_functions, "_run_config_preflight", lambda config_file: None
    )
    monkeypatch.setattr("subprocess.Popen", _FakePopen)
    monkeypatch.setattr(
        "tmgg.modal._lib.confirmation.append_confirmation", _append_confirmation
    )

    result = modal_functions._run_cli_impl("tmgg-discrete-gen", "run_id: abc\n", "abc")

    captured = capsys.readouterr().out

    assert (
        "wandb: https://wandb.ai/your-wandb-entity/discrete-diffusion/runs/abcd1234"
        in captured
    )
    assert (
        "W&B URL: https://wandb.ai/your-wandb-entity/discrete-diffusion/runs/abcd1234"
        in captured
    )
    assert result["status"] == "completed"
    assert result["wandb_run_id"] == "abcd1234"
    assert (
        result["wandb_url"]
        == "https://wandb.ai/your-wandb-entity/discrete-diffusion/runs/abcd1234"
    )
    assert confirmations[-1]["wandb_run_id"] == "abcd1234"
    assert (
        confirmations[-1]["wandb_url"]
        == "https://wandb.ai/your-wandb-entity/discrete-diffusion/runs/abcd1234"
    )


def test_run_cli_impl_raises_on_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Non-zero subprocess exit should fail the Modal function itself."""

    confirmations: list[dict[str, object]] = []

    class _FakeStdout:
        def __iter__(self):
            return iter(
                [
                    "starting run\n",
                    "Traceback (most recent call last):\n",
                    "ModuleNotFoundError: No module named 'torch_geometric'\n",
                ]
            )

    class _FakePopen:
        def __init__(self, args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.stdout = _FakeStdout()

        def wait(self) -> int:
            return 1

    def _append_confirmation(path: Path, **entry: object) -> None:
        confirmations.append({"path": path, **entry})

    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr(modal_functions, "_run_import_preflight", lambda: None)
    monkeypatch.setattr(
        modal_functions, "_run_config_preflight", lambda config_file: None
    )
    monkeypatch.setattr("subprocess.Popen", _FakePopen)
    monkeypatch.setattr(
        "tmgg.modal._lib.confirmation.append_confirmation", _append_confirmation
    )

    with pytest.raises(RuntimeError, match="Experiment subprocess failed"):
        modal_functions._run_cli_impl("tmgg-discrete-gen", "run_id: abc\n", "abc")

    assert confirmations[-1]["status"] == "failed"
    assert confirmations[-1]["exit_code"] == 1
    assert "torch_geometric" in str(confirmations[-1]["error"])


def test_run_cli_impl_surfaces_preflight_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preflight import failures should abort before launching the CLI."""

    confirmations: list[dict[str, object]] = []

    def _append_confirmation(path: Path, **entry: object) -> None:
        confirmations.append({"path": path, **entry})

    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr(
        modal_functions,
        "_run_import_preflight",
        lambda: (_ for _ in ()).throw(
            RuntimeError(
                "Modal import preflight failed.\n"
                "Module: graph_tool\n"
                "Exit code: -4 (SIGILL)\n"
                "Output tail:\n<no output>"
            )
        ),
    )
    monkeypatch.setattr(
        "tmgg.modal._lib.confirmation.append_confirmation", _append_confirmation
    )

    with pytest.raises(RuntimeError, match="Module: graph_tool"):
        modal_functions._run_cli_impl("tmgg-discrete-gen", "run_id: abc\n", "abc")

    assert len(confirmations) == 2
    assert confirmations[0]["status"] == "started"
    assert confirmations[1]["status"] == "failed"
    assert confirmations[1]["stage"] == "preflight"
    assert "Module: graph_tool" in str(confirmations[1]["error"])


def test_run_cli_impl_surfaces_config_preflight_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Config preflight failures should abort before launching the CLI."""

    confirmations: list[dict[str, object]] = []

    def _append_confirmation(path: Path, **entry: object) -> None:
        confirmations.append({"path": path, **entry})

    monkeypatch.setattr("tempfile.mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr(modal_functions, "_run_import_preflight", lambda: None)
    monkeypatch.setattr(
        modal_functions,
        "_run_config_preflight",
        lambda config_file: (_ for _ in ()).throw(
            RuntimeError(
                "Modal config preflight failed.\n"
                "Exit code: -6 (SIGABRT)\n"
                "Output tail:\nstd::length_error"
            )
        ),
    )
    monkeypatch.setattr(
        "tmgg.modal._lib.confirmation.append_confirmation", _append_confirmation
    )

    with pytest.raises(RuntimeError, match="Modal config preflight failed"):
        modal_functions._run_cli_impl("tmgg-discrete-gen", "run_id: abc\n", "abc")

    assert len(confirmations) == 2
    assert confirmations[0]["status"] == "started"
    assert confirmations[1]["status"] == "failed"
    assert confirmations[1]["stage"] == "config_preflight"
    assert "SIGABRT" in str(confirmations[1]["error"])

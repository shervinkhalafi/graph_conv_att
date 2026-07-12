"""Modal deployment definitions for graph-denoising.

This module is the **single entry point** for ``modal deploy``. All
``@app.function`` decorated functions and the ``@app.local_entrypoint``
are defined here. Runtime code (ModalRunner, CLI) never imports this
module — they use ``modal.Function.from_name()`` to get references
to the deployed functions.

Deploy::

    uv run modal deploy -m tmgg.modal._functions

Serve (hot-reload)::

    uv run modal serve -m tmgg.modal._functions
"""

# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportAttributeAccessIssue=false
# Modal's type stubs are incomplete for volume mount / image / volume-class attributes.

from __future__ import annotations

import re
import shlex
import signal
from pathlib import Path
from typing import Any

import modal

from tmgg.modal._lib.image import create_tmgg_image
from tmgg.modal._lib.volumes import get_volume_mounts
from tmgg.modal.app import (
    CPU_PROFILES,
    DEFAULT_SCALEDOWN_WINDOW,
    DEFAULT_TIMEOUTS,
    GPU_CONFIGS,
    MODAL_APP_NAME,
)

# ---------------------------------------------------------------------------
# App, secrets, image — only executed during `modal deploy` / `modal serve`
# ---------------------------------------------------------------------------

app = modal.App(MODAL_APP_NAME, include_source=False)

try:
    from tmgg.modal._lib.paths import discover_source_checkout_path

    experiment_image = create_tmgg_image(discover_source_checkout_path())
except (ImportError, RuntimeError):
    experiment_image = None

tigris_secret = modal.Secret.from_name(
    "tigris-credentials",
    required_keys=[
        "TMGG_TIGRIS_BUCKET",
        "TMGG_TIGRIS_ACCESS_KEY",
        "TMGG_TIGRIS_SECRET_KEY",
    ],
)

wandb_secret = modal.Secret.from_name(
    "wandb-credentials",
    required_keys=["WANDB_API_KEY"],
)

#: Preflight imports as ``(module_name, import_stmt, required)`` triples.
#: ``required=False`` modules log a warning and continue on failure;
#: ``required=True`` modules raise a clear ``RuntimeError`` so the run
#: aborts before the CLI launches. ``graph_tool`` is optional because
#: ``GraphEvaluator`` already degrades gracefully when it's missing
#: (``sbm_accuracy`` becomes ``None``); the worker CPU may lack the
#: AVX-512 instructions the bundled binary uses, triggering ``SIGILL``
#: at import time on otherwise-fine A100 hosts.
IMPORT_PREFLIGHTS: tuple[tuple[str, str, bool], ...] = (
    ("torch", "import torch", True),
    ("torch_geometric", "import torch_geometric", True),
    ("ot", "import ot", True),
    ("graph_tool", "import graph_tool.all as gt", False),
)
CONFIG_PREFLIGHT_SNIPPET = """
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

config_path = Path({config_path!r})
cfg = OmegaConf.load(config_path)

print("Config preflight: instantiate datamodule", flush=True)
dm = hydra.utils.instantiate(cfg.data)
dm.setup("fit")
print("Config preflight OK: datamodule setup", flush=True)

print("Config preflight: fetch validation batch", flush=True)
batch = next(iter(dm.val_dataloader()))

def _shape_summary(data):
    parts = [("node_mask", tuple(data.node_mask.shape))]
    for name in ("X_class", "X_feat", "E_class", "E_feat"):
        t = getattr(data, name)
        parts.append((name, tuple(t.shape) if t is not None else None))
    return parts

print(
    "Config preflight OK: validation batch",
    _shape_summary(batch),
    flush=True,
)

print("Config preflight: instantiate LightningModule", flush=True)
module = hydra.utils.instantiate(cfg.model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
module = module.to(device)
print("Config preflight OK: module to device", str(device), flush=True)

if hasattr(module, "noise_process") and module.noise_process.needs_data_initialization():
    print("Config preflight: initialize noise process", flush=True)
    module.noise_process.initialize_from_data(dm.train_dataloader_raw_pyg())
    print("Config preflight OK: noise process initialized", flush=True)

print("Config preflight: transfer batch to device", flush=True)
batch = module.transfer_batch_to_device(batch, device, 0)
print(
    "Config preflight OK: batch to device",
    str(batch.node_mask.device),
    _shape_summary(batch),
    flush=True,
)

if hasattr(module, "noise_process") and hasattr(module, "_compute_loss") and hasattr(module, "T"):
    with torch.no_grad():
        bs = int(batch.node_mask.shape[0])
        t_int = torch.randint(1, int(module.T) + 1, (bs,), device=device)

        print("Config preflight: noise forward_sample", flush=True)
        # forward_sample returns NoisedBatch (parity #17 / #18 / D-4);
        # the preflight only needs z_t for the downstream model call.
        z_t = module.noise_process.forward_sample(batch, t_int).z_t
        print(
            "Config preflight OK: noise forward_sample",
            _shape_summary(z_t),
            flush=True,
        )

        print("Config preflight: condition vector", flush=True)
        condition = module.noise_process.process_state_condition_vector(t_int)
        print("Config preflight OK: condition vector", tuple(condition.shape), flush=True)

        print("Config preflight: model forward", flush=True)
        pred = module.model(z_t, t=condition)
        print(
            "Config preflight OK: model forward",
            _shape_summary(pred),
            flush=True,
        )

        print("Config preflight: compute loss", flush=True)
        loss = module._compute_loss(pred, batch)
        print("Config preflight OK: compute loss", float(loss.item()), flush=True)
"""


# ======================================================================
# Helpers
# ======================================================================


def _extract_wandb_run_url(output: str) -> str | None:
    """Extract the W&B run URL from subprocess output.

    Looks for the run-page URL that W&B prints at the end of every run,
    e.g. ``https://wandb.ai/entity/project/runs/abcd1234``.

    Parameters
    ----------
    output
        Combined subprocess output text.

    Returns
    -------
    str or None
        The full W&B run URL, or ``None`` if not found.
    """
    matches = re.findall(r"https://wandb\.ai/[^/\s]+/[^/\s]+/runs/[a-z0-9]+", output)
    if matches:
        return matches[-1]
    return None


def _extract_wandb_run_id(output: str) -> str | None:
    """Extract the W&B run ID from subprocess output."""
    wandb_url = _extract_wandb_run_url(output)
    if wandb_url is None:
        return None
    return wandb_url.rsplit("/", 1)[-1]


def _format_return_code(return_code: int) -> str:
    """Return a human-readable subprocess exit description."""
    if return_code >= 0:
        return str(return_code)

    try:
        signame = signal.Signals(-return_code).name
    except ValueError:
        return str(return_code)
    return f"{return_code} ({signame})"


_HOST_DIAGNOSTICS_SNIPPET = r"""
# Host diagnostics printed before every native-module preflight. Dumps
# the CPU model + flags, kernel version, and (for graph_tool) the
# loaded .so path. Small, bounded, and always executed so that a SIGILL
# next time has actionable context.
import os, platform, subprocess
def _run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
        return out.strip()
    except Exception as e:
        return f"<err: {e}>"
print("[host-diag] kernel:", platform.uname().release, flush=True)
print("[host-diag] model:", _run(["bash", "-lc", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs"]), flush=True)
flags = _run(["bash", "-lc", "grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2"]).split()
interesting = sorted(f for f in flags if f in {"avx","avx2","avx512f","avx512dq","avx512cd","avx512bw","avx512vl","fma","fma4","bmi1","bmi2","adx","sse4_1","sse4_2","sha_ni"})
print("[host-diag] cpu-isa:", " ".join(interesting) or "<none>", flush=True)
print("[host-diag] libc:", _run(["bash", "-lc", "ldd --version 2>&1 | head -1"]), flush=True)
print("[host-diag] libstdc++:", _run(["bash", "-lc", "find /opt/conda /usr/lib -name 'libstdc++.so.6*' 2>/dev/null | head -3"]), flush=True)
try:
    import graph_tool  # noqa: F401  -- just read __file__, don't trigger the SIGILL path
    print("[host-diag] graph_tool.__file__:", graph_tool.__file__, flush=True)
    # Find the compiled core .so and readelf the NEEDED / RUNPATH entries.
    import os.path as _op
    gt_dir = _op.dirname(graph_tool.__file__)
    core = _run(["bash", "-lc", f"find {gt_dir} -name 'libgraph_tool*.so' -o -name '_core*.so' 2>/dev/null | head -3"])
    print("[host-diag] graph_tool core .so candidates:", core or "<none>", flush=True)
    first_so = core.splitlines()[0] if core and not core.startswith("<err") else ""
    if first_so:
        print("[host-diag] readelf -d tail:", _run(["bash", "-lc", f"readelf -d {first_so} 2>/dev/null | grep -E 'NEEDED|RUNPATH|RPATH' | head -15"]), flush=True)
except Exception as e:
    print(f"[host-diag] graph_tool pre-import metadata failed: {e}", flush=True)
"""


def _open_preflight_log() -> Path | None:
    """Create a persistent-volume log file for this container's preflight.

    Writes to ``/data/outputs/preflight_log/<UTC-stamp>_<host>_pid<pid>.log``
    so that a failed container leaves a diagnostic trail on the volume even
    when the subprocess dies with SIGILL and stdout isn't retrievable
    post-hoc. Returns ``None`` if the volume isn't mounted (local testing).
    """
    import datetime
    import os
    import socket

    log_dir = Path("/data/outputs/preflight_log")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    host = socket.gethostname().replace("/", "_")[:24]
    pid = os.getpid()
    return log_dir / f"{stamp}_{host}_pid{pid}.log"


def _append_preflight(path: Path | None, line: str) -> None:
    """Append a line to the preflight log; swallow failures (best-effort)."""
    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + ("\n" if not line.endswith("\n") else ""))
    except OSError:
        pass


def _commit_outputs_volume() -> None:
    """Force-flush the outputs volume so preflight logs survive a crashing container.

    Modal volumes buffer writes in the container filesystem; they only
    land on the persistent volume when either (a) the function returns
    cleanly or (b) the code calls ``Volume.commit()``. For preflight-log
    purposes we need (b), so a post-mortem reader on the host can fetch
    the log via ``modal volume get``.
    """
    try:
        import modal

        vol = modal.Volume.from_name("graph-denoising-outputs")
        vol.commit()
    except Exception:  # noqa: BLE001 - best effort; the preflight must not die here
        pass


def _run_host_diagnostics(log_path: Path | None = None) -> None:
    """Print host CPU + library diagnostics before the import preflight.

    Runs once per container. Mirrors its stdout into the preflight log
    on the persistent volume when a path is supplied, so the host profile
    is retrievable after a crashing container has exited.
    """
    import subprocess
    import sys

    print("=== host-diag start ===", flush=True)
    _append_preflight(log_path, "=== host-diag start ===")
    result = subprocess.run(
        [sys.executable, "-c", _HOST_DIAGNOSTICS_SNIPPET],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.stdout:
        trailer = "" if result.stdout.endswith("\n") else "\n"
        print(result.stdout, end=trailer, flush=True)
        _append_preflight(log_path, result.stdout.rstrip("\n"))
    if result.stderr:
        trailer = "" if result.stderr.endswith("\n") else "\n"
        print(result.stderr, end=trailer, flush=True)
        _append_preflight(log_path, "STDERR: " + result.stderr.rstrip("\n"))
    print("=== host-diag end ===", flush=True)
    _append_preflight(log_path, "=== host-diag end ===")


def _run_import_preflight() -> None:
    """Probe native imports in isolated subprocesses before launching the CLI.

    Native-extension import failures can terminate Python with signals like
    ``SIGILL`` before any traceback is written. Running one import per
    subprocess gives us a precise module name and exit code in Modal logs.
    A persistent volume log captures host-diag + per-stage markers so
    post-mortem diagnosis works even when ``modal container logs`` is no
    longer streamable.
    """
    import subprocess
    import sys

    log_path = _open_preflight_log()
    _append_preflight(log_path, "=== preflight start ===")
    _run_host_diagnostics(log_path)

    for module_name, import_stmt, required in IMPORT_PREFLIGHTS:
        print(f"Preflight import check: {module_name}", flush=True)
        _append_preflight(log_path, f"ATTEMPT: {module_name} :: {import_stmt}")
        # Commit before each risky import so even a SIGILL next leaves the
        # pre-attempt marker on the persistent volume.
        _commit_outputs_volume()
        result = subprocess.run(
            [sys.executable, "-c", import_stmt],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"Preflight import OK: {module_name}", flush=True)
            _append_preflight(log_path, f"OK: {module_name}")
            continue

        output = (result.stdout or "") + (result.stderr or "")
        output_tail = output[-1000:] if output else "<no output>"
        exit_description = _format_return_code(result.returncode)
        if not required:
            # Optional dep — log a clear warning and continue. Downstream
            # code that uses this module must already handle absence
            # gracefully (e.g. GraphEvaluator falls back to
            # ``sbm_accuracy=None`` when graph_tool is unavailable).
            print(
                f"Preflight import WARNING (optional): {module_name} "
                f"failed with exit={exit_description}; continuing.",
                flush=True,
            )
            _append_preflight(
                log_path,
                f"SKIP (optional): {module_name} exit={exit_description}\n"
                f"STATEMENT: {import_stmt}\nTAIL:\n{output_tail}",
            )
            _commit_outputs_volume()
            continue

        _append_preflight(
            log_path,
            f"FAIL: {module_name} exit={exit_description}\nSTATEMENT: {import_stmt}\n"
            f"TAIL:\n{output_tail}",
        )
        _commit_outputs_volume()
        raise RuntimeError(
            "Modal import preflight failed.\n"
            f"Module: {module_name}\n"
            f"Statement: {shlex.quote(import_stmt)}\n"
            f"Exit code: {exit_description}\n"
            f"Output tail:\n{output_tail}"
        )

    _append_preflight(log_path, "=== preflight done (all imports OK) ===")
    _commit_outputs_volume()


def _run_config_preflight(config_file: Path) -> None:
    """Exercise datamodule setup and first-batch transfer in isolation."""
    import subprocess
    import sys

    print("Config preflight: start", flush=True)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            CONFIG_PREFLIGHT_SNIPPET.format(config_path=str(config_file)),
        ],
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if result.returncode == 0:
        print("Config preflight OK: complete", flush=True)
        return

    output_tail = output[-1200:] if output else "<no output>"
    raise RuntimeError(
        "Modal config preflight failed.\n"
        f"Exit code: {_format_return_code(result.returncode)}\n"
        f"Output tail:\n{output_tail}"
    )


def _run_cli_impl(
    cmd: str,
    config_yaml: str,
    run_id: str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Execute a CLI experiment subprocess inside a Modal container.

    This is the shared implementation behind all ``modal_run_cli*`` functions.
    Each GPU-tier function delegates here — Modal requires a separate
    ``@app.function`` per GPU/timeout configuration, but the execution logic
    is identical.

    Parameters
    ----------
    cmd
        CLI entry point name (e.g. ``"tmgg-discrete-gen"``).
    config_yaml
        Fully resolved Hydra config serialized as YAML.
    run_id
        Experiment identifier for the confirmation log.
    overrides
        Optional Hydra CLI overrides.

    Returns
    -------
    dict
        ``{"status": "completed"|"failed", "run_id": ..., ...}``
    """
    import os
    import subprocess
    import tempfile

    from omegaconf import OmegaConf

    from tmgg.modal._lib.confirmation import (
        DEFAULT_CONFIRMATION_PATH,
        append_confirmation,
    )

    append_confirmation(
        DEFAULT_CONFIRMATION_PATH, run_id=run_id, status="started", cmd=cmd
    )

    try:
        _run_import_preflight()
    except RuntimeError as exc:
        append_confirmation(
            DEFAULT_CONFIRMATION_PATH,
            run_id=run_id,
            status="failed",
            error=str(exc)[-1000:],
            stage="preflight",
            cmd=cmd,
        )
        raise

    # Write resolved config to a temp directory so Hydra can find it
    config_dir = Path(tempfile.mkdtemp(prefix="tmgg_run_"))
    config_file = config_dir / "run_config.yaml"
    config_file.write_text(config_yaml)

    try:
        _run_config_preflight(config_file)
    except RuntimeError as exc:
        append_confirmation(
            DEFAULT_CONFIRMATION_PATH,
            run_id=run_id,
            status="failed",
            error=str(exc)[-1200:],
            stage="config_preflight",
            cmd=cmd,
        )
        raise

    cli_args = [cmd, f"--config-path={config_dir}", "--config-name=run_config"]
    if overrides:
        cli_args.extend(overrides)

    # Build subprocess env. Default: ``modal_debug=false`` → set
    # ``PYTHONOPTIMIZE=1`` so the training process skips ``assert`` and
    # ``if __debug__:`` blocks. This strips ~50 host-side ``bool(Tensor)``
    # syncs/step from the hot path (extras, mask-symmetry checks,
    # masked-softmax guards) per the 2026-04-28 sync review. Setting
    # ``modal_debug=true`` keeps asserts active for numerical investigation.
    cfg = OmegaConf.create(config_yaml)
    modal_debug = bool(cfg.get("modal_debug", False))
    cli_env = dict(os.environ)
    if not modal_debug:
        cli_env["PYTHONOPTIMIZE"] = "1"
        print("[modal] modal_debug=false → PYTHONOPTIMIZE=1", flush=True)
    else:
        cli_env.pop("PYTHONOPTIMIZE", None)
        print("[modal] modal_debug=true → asserts active", flush=True)

    process = subprocess.Popen(
        cli_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=cli_env,
    )
    assert process.stdout is not None

    output_lines: list[str] = []
    wandb_url: str | None = None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
        if wandb_url is None:
            wandb_url = _extract_wandb_run_url(line)
            if wandb_url is not None:
                print(f"W&B URL: {wandb_url}", flush=True)

    return_code = process.wait()
    combined_output = "".join(output_lines)
    if wandb_url is None:
        wandb_url = _extract_wandb_run_url(combined_output)
        if wandb_url is not None:
            print(f"W&B URL: {wandb_url}", flush=True)
    wandb_run_id = _extract_wandb_run_id(combined_output)

    if return_code == 0:
        append_confirmation(
            DEFAULT_CONFIRMATION_PATH,
            run_id=run_id,
            status="completed",
            exit_code=0,
            wandb_run_id=wandb_run_id,
            wandb_url=wandb_url,
            cmd=cmd,
        )
        return {
            "status": "completed",
            "run_id": run_id,
            "wandb_run_id": wandb_run_id,
            "wandb_url": wandb_url,
        }
    else:
        error_tail = combined_output[-500:] if combined_output else ""
        append_confirmation(
            DEFAULT_CONFIRMATION_PATH,
            run_id=run_id,
            status="failed",
            exit_code=return_code,
            error=error_tail,
            wandb_url=wandb_url,
            cmd=cmd,
        )
        raise RuntimeError(
            "Experiment subprocess failed inside Modal container.\n"
            f"Exit code: {_format_return_code(return_code)}\n"
            f"Command: {cli_args}\n"
            f"Output tail:\n{error_tail}"
        )


def _evaluate_mmd_impl(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Run MMD evaluation on a checkpoint.

    Shared implementation for all ``modal_evaluate_mmd*`` functions.
    Each GPU tier has its own ``@app.function`` (Modal requires separate
    decorators per GPU/timeout config), but delegates here.
    """
    from tmgg.modal._lib.evaluate import run_mmd_evaluation

    return run_mmd_evaluation(task_dict)


# ======================================================================
# Experiment execution functions (CLI subprocess transport)
#
# Modal requires a separate @app.function per GPU/timeout configuration —
# there is no way to parameterize GPU type at call time. Each function
# below is a thin wrapper that delegates to _run_cli_impl.
# ======================================================================


@app.function(
    name="modal_run_cli",
    image=experiment_image,
    gpu=GPU_CONFIGS["standard"],
    cpu=CPU_PROFILES["standard"],
    timeout=DEFAULT_TIMEOUTS["standard"],
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_run_cli(
    cmd: str,
    config_yaml: str,
    run_id: str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Run an experiment CLI on standard GPU (A10G)."""
    return _run_cli_impl(cmd, config_yaml, run_id, overrides)


@app.function(
    name="modal_run_cli_fast",
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],
    cpu=CPU_PROFILES["fast"],
    timeout=DEFAULT_TIMEOUTS["fast"],
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_run_cli_fast(
    cmd: str,
    config_yaml: str,
    run_id: str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Run an experiment CLI on fast GPU (A100)."""
    return _run_cli_impl(cmd, config_yaml, run_id, overrides)


@app.function(
    name="modal_run_cli_debug",
    image=experiment_image,
    gpu=GPU_CONFIGS["debug"],
    cpu=CPU_PROFILES["debug"],
    timeout=DEFAULT_TIMEOUTS["debug"],
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_run_cli_debug(
    cmd: str,
    config_yaml: str,
    run_id: str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Run an experiment CLI on debug GPU (T4)."""
    return _run_cli_impl(cmd, config_yaml, run_id, overrides)


# ======================================================================
# Evaluation functions
#
# Same pattern: one @app.function per GPU tier, all delegate to
# _evaluate_mmd_impl.
# ======================================================================


@app.function(
    name="modal_evaluate_mmd",
    image=experiment_image,
    gpu=GPU_CONFIGS["standard"],
    cpu=CPU_PROFILES["standard"],
    timeout=3600,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_evaluate_mmd(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate checkpoint MMD metrics on standard GPU (A10G)."""
    return _evaluate_mmd_impl(task_dict)


@app.function(
    name="modal_evaluate_mmd_debug",
    image=experiment_image,
    gpu=GPU_CONFIGS["debug"],
    cpu=CPU_PROFILES["debug"],
    timeout=1800,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_evaluate_mmd_debug(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate checkpoint MMD metrics on debug GPU (T4)."""
    return _evaluate_mmd_impl(task_dict)


@app.function(
    name="modal_evaluate_mmd_fast",
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],
    cpu=CPU_PROFILES["fast"],
    timeout=7200,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_evaluate_mmd_fast(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate checkpoint MMD metrics on fast GPU (A100)."""
    return _evaluate_mmd_impl(task_dict)


# ----------------------------------------------------------------------
# Async-eval wrappers (Step 2 of async-eval design plan).
#
# Same per-tier triple as ``modal_evaluate_mmd*`` above, but each
# delegates to ``evaluate_mmd_async`` from
# ``tmgg.modal._lib.evaluate_async``. The trainer fires-and-forgets
# these via ``.spawn(...)`` on every scheduled eval step; the worker
# resumes the trainer's W&B run and logs gen-val/* metrics under the
# trainer's custom step axis. ``standard`` (A10G) is the default tier
# for async eval — the cost/throughput sweet spot for MMD evaluation —
# while ``debug`` (T4) and ``fast`` (A100) cover the same tier matrix
# the rest of the code expects.
# ----------------------------------------------------------------------


@app.function(
    name="modal_evaluate_mmd_async",
    image=experiment_image,
    gpu=GPU_CONFIGS["standard"],
    cpu=CPU_PROFILES["standard"],
    timeout=3600,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_evaluate_mmd_async(task: dict[str, Any]) -> dict[str, Any]:
    """Async MMD eval that resumes the trainer's W&B run (standard, A10G)."""
    from tmgg.modal._lib.evaluate_async import evaluate_mmd_async

    return evaluate_mmd_async(task)


@app.function(
    name="modal_evaluate_mmd_async_debug",
    image=experiment_image,
    gpu=GPU_CONFIGS["debug"],
    cpu=CPU_PROFILES["debug"],
    timeout=1800,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_evaluate_mmd_async_debug(task: dict[str, Any]) -> dict[str, Any]:
    """Async MMD eval that resumes the trainer's W&B run (debug, T4)."""
    from tmgg.modal._lib.evaluate_async import evaluate_mmd_async

    return evaluate_mmd_async(task)


@app.function(
    name="modal_evaluate_mmd_async_fast",
    image=experiment_image,
    gpu=GPU_CONFIGS["fast"],
    cpu=CPU_PROFILES["fast"],
    timeout=7200,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    secrets=[tigris_secret, wandb_secret],
    volumes=get_volume_mounts(),
)
def modal_evaluate_mmd_async_fast(task: dict[str, Any]) -> dict[str, Any]:
    """Async MMD eval that resumes the trainer's W&B run (fast, A100)."""
    from tmgg.modal._lib.evaluate_async import evaluate_mmd_async

    return evaluate_mmd_async(task)


@app.function(
    name="modal_list_checkpoints",
    image=experiment_image,
    cpu=1.0,
    timeout=60,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),
)
def modal_list_checkpoints(run_id: str) -> dict[str, Any]:
    """List all checkpoints available for a run."""
    from tmgg.modal._lib.evaluate import list_checkpoints_for_run

    return list_checkpoints_for_run(run_id)


# ----------------------------------------------------------------------
# Dataset prepare + validate (CPU-only, share the graph-denoising-datasets volume).
#
# One ``@app.function`` per dataset for ``prepare`` and ``validate`` —
# matches the existing per-tier wrapper pattern; lets each call be
# launched / retried independently from the others. All delegate to
# the impls in ``tmgg.modal._lib.dataset_ops`` so the same code path
# is unit-testable locally.
# ----------------------------------------------------------------------


_DATASET_OPS_TIMEOUT = 10800
# 3h ceiling. The original 3600 was too tight: MOSES (~1.58M SMILES) hit the
# 1h mark mid-shard-write at ``cpu=4``, leaving 2/4 train shards written
# (atomic-rename kept those clean) and a stranded ``.pt.tmp``. GuacaMol
# (~1.27M SMILES) is similarly sized. 3h gives headroom for both.


@app.function(
    name="modal_prepare_qm9",
    image=experiment_image,
    cpu=4.0,
    timeout=_DATASET_OPS_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),
)
def modal_prepare_qm9() -> dict[str, Any]:
    """Download + preprocess QM9 into ``/data/datasets/qm9/``."""
    from tmgg.modal._lib.dataset_ops import prepare_dataset

    return prepare_dataset("qm9")


@app.function(
    name="modal_prepare_moses",
    image=experiment_image,
    cpu=4.0,
    timeout=_DATASET_OPS_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),
)
def modal_prepare_moses() -> dict[str, Any]:
    """Download + preprocess MOSES into ``/data/datasets/moses/``."""
    from tmgg.modal._lib.dataset_ops import prepare_dataset

    return prepare_dataset("moses")


@app.function(
    name="modal_prepare_guacamol",
    image=experiment_image,
    cpu=4.0,
    timeout=_DATASET_OPS_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),
)
def modal_prepare_guacamol() -> dict[str, Any]:
    """Download + preprocess GuacaMol into ``/data/datasets/guacamol/``."""
    from tmgg.modal._lib.dataset_ops import prepare_dataset

    return prepare_dataset("guacamol")


@app.function(
    name="modal_prepare_planar",
    image=experiment_image,
    cpu=2.0,
    timeout=_DATASET_OPS_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),
)
def modal_prepare_planar() -> dict[str, Any]:
    """Download the SPECTRE planar fixture into ``/data/datasets/spectre/``."""
    from tmgg.modal._lib.dataset_ops import prepare_dataset

    return prepare_dataset("planar")


@app.function(
    name="modal_prepare_sbm",
    image=experiment_image,
    cpu=2.0,
    timeout=_DATASET_OPS_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),
)
def modal_prepare_sbm() -> dict[str, Any]:
    """Download the SPECTRE SBM fixture into ``/data/datasets/spectre/``."""
    from tmgg.modal._lib.dataset_ops import prepare_dataset

    return prepare_dataset("sbm")


@app.function(
    name="modal_validate_qm9",
    image=experiment_image,
    cpu=2.0,
    timeout=_DATASET_OPS_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),
)
def modal_validate_qm9() -> dict[str, Any]:
    """Validate every QM9 preprocessed shard (load + schema)."""
    from tmgg.modal._lib.dataset_ops import validate_dataset

    return validate_dataset("qm9")


@app.function(
    name="modal_validate_moses",
    image=experiment_image,
    cpu=2.0,
    timeout=_DATASET_OPS_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),
)
def modal_validate_moses() -> dict[str, Any]:
    """Validate every MOSES preprocessed shard (load + schema)."""
    from tmgg.modal._lib.dataset_ops import validate_dataset

    return validate_dataset("moses")


@app.function(
    name="modal_validate_guacamol",
    image=experiment_image,
    cpu=2.0,
    timeout=_DATASET_OPS_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),
)
def modal_validate_guacamol() -> dict[str, Any]:
    """Validate every GuacaMol preprocessed shard (load + schema)."""
    from tmgg.modal._lib.dataset_ops import validate_dataset

    return validate_dataset("guacamol")


@app.function(
    name="modal_validate_planar",
    image=experiment_image,
    cpu=1.0,
    timeout=_DATASET_OPS_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),
)
def modal_validate_planar() -> dict[str, Any]:
    """Validate the SPECTRE planar fixture (file + load)."""
    from tmgg.modal._lib.dataset_ops import validate_dataset

    return validate_dataset("planar")


@app.function(
    name="modal_validate_sbm",
    image=experiment_image,
    cpu=1.0,
    timeout=_DATASET_OPS_TIMEOUT,
    scaledown_window=DEFAULT_SCALEDOWN_WINDOW,
    volumes=get_volume_mounts(),
)
def modal_validate_sbm() -> dict[str, Any]:
    """Validate the SPECTRE SBM fixture (file + load)."""
    from tmgg.modal._lib.dataset_ops import validate_dataset

    return validate_dataset("sbm")


@app.function(
    name="_warm_molecular_caches_in_image",
    image=experiment_image,
    cpu=2.0,
    timeout=600,
    volumes=get_volume_mounts(),
)
def _warm_molecular_caches_in_image() -> str:
    """Pre-load ChemNet weights into the image's filesystem cache.

    Called once at deploy time so the first validation cycle that
    instantiates ``FCDMetric`` does not pay the ChemNet load cost on
    the hot path. The maintained ``fcd>=1.2.2`` package ships ChemNet
    weights inside the wheel (``fcd/ChemNet_v0.13_pretrained.pt``,
    ~5.6 MB) and ``load_ref_model`` is ``lru_cache``'d, so calling it
    here simply primes the in-process cache and verifies the
    deserialisation path before any worker hits it.
    """
    from pathlib import Path

    from fcd import load_ref_model

    cache_dir = Path("/data/datasets/molecular/chemnet")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Calling ``load_ref_model`` materialises the wheel-bundled weights
    # into a temp file, runs the ``torch.load`` deserialisation, and
    # caches the resulting ``torch.nn.Module`` for subsequent calls.
    _ = load_ref_model()
    return f"chemnet warm; cache_dir={cache_dir}"

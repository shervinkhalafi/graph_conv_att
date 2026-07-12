"""MMD evaluation dispatch for trained DiGress checkpoints.

No ``import modal`` at module level. The ``@app.function`` decorated
wrappers live in ``_functions.py``; this module builds CLI arguments
and runs the evaluate_checkpoint CLI as a subprocess, mirroring the
training dispatch pattern in ``modal_run_cli``.

The subprocess boundary keeps ``modal/`` free of ``tmgg.experiments``
imports --- those happen inside the subprocess, not here.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from tmgg.modal._lib.volumes import OUTPUTS_MOUNT

logger = logging.getLogger(__name__)


@dataclass
class EvaluationInput:
    """Input for MMD evaluation task.

    Parameters
    ----------
    run_id
        Run ID whose checkpoint to evaluate.
    checkpoint_path
        Optional explicit checkpoint path. If None, defaults to
        ``/data/outputs/{run_id}/checkpoints/last.ckpt``.
    num_samples
        Number of graphs to generate for evaluation.
    num_steps
        Number of denoising steps for sampling.
    mmd_kernel
        Kernel type for MMD computation.
    mmd_sigma
        Bandwidth for kernel.
    seed
        Random seed for reproducibility.
    """

    run_id: str
    checkpoint_path: str | None = None
    num_samples: int = 500
    num_steps: int = 100
    mmd_kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv"
    mmd_sigma: float = 1.0
    seed: int = 42
    # Optional multi-format dump dir. When set, the evaluate CLI
    # additionally writes generated/reference graphs (edge-list JSON),
    # per-batch val diagnostics, viz PNGs, timings, and a summary.md
    # under this directory. Async-eval (training callback) leaves it
    # ``None`` to keep the per-ckpt path lean; the eval-all worker
    # supplies a fresh subfolder per ckpt.
    output_dir: str | None = None
    viz_count: int = 32
    val_batch_limit: int | None = None


@dataclass
class EvaluationOutput:
    """Output from MMD evaluation.

    Parameters
    ----------
    run_id
        Run ID that was evaluated.
    checkpoint_name
        Name of the checkpoint file evaluated (e.g., "last", "best", "epoch_50").
    status
        Completion status: 'completed' or 'failed'.
    results
        Nested dict mapping evaluation label -> metric -> value.
        E.g., {"eval": {"degree_mmd": 0.01, "clustering_mmd": 0.02, ...}}.
    error_message
        Error details if status is 'failed'.
    evaluation_params
        Parameters used for evaluation (num_samples, num_steps, etc.).
    timestamp
        ISO timestamp of when evaluation completed.
    """

    run_id: str
    checkpoint_name: str = "last"
    status: Literal["completed", "failed"] = "completed"
    results: dict[str, dict[str, float]] = field(default_factory=dict)
    error_message: str | None = None
    evaluation_params: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


def _make_failed_output(
    run_id: str,
    checkpoint_name: str,
    error_message: str,
) -> dict[str, Any]:
    """Build a serialized EvaluationOutput for a failure case."""
    return asdict(
        EvaluationOutput(
            run_id=run_id,
            checkpoint_name=checkpoint_name,
            status="failed",
            error_message=error_message,
            timestamp=datetime.now().isoformat(),
        )
    )


def _cuda_available() -> bool:
    """Check CUDA availability without importing torch at module level."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _detect_eval_cli_module(config: dict[str, Any]) -> str:
    """Pick the evaluate-CLI module path matching the trained run's diffusion family.

    The smoke run in 2026-04-29 wired a DiscreteDiffusion (``CategoricalNoiseProcess``)
    DiffusionModule but the eval dispatch hard-coded the (deleted)
    ``gaussian_diffusion_generative.evaluate_checkpoint`` module. The empty-metrics
    bug surfaced because the subprocess crashed silently and we wrapped the failure
    as a "completed" eval. This helper inspects the saved config and routes to the
    matching CLI; we only support discrete today (gaussian's CLI was removed in
    79428caa) so a non-categorical noise process is a hard error.

    Parameters
    ----------
    config
        Resolved config dict (``OmegaConf.to_container(load(config.yaml))``).

    Returns
    -------
    str
        Importable module path suitable for ``python -m``.

    Raises
    ------
    RuntimeError
        ``model.noise_process._target_`` is missing or is not a recognised
        diffusion family.
    """
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, dict):
        raise TypeError(f"Expected dict for config['model']; got {type(model_cfg)}")
    noise_cfg = model_cfg.get("noise_process", {})
    if not isinstance(noise_cfg, dict):
        raise TypeError(
            f"Expected dict for config['model']['noise_process']; got {type(noise_cfg)}"
        )
    noise_target = noise_cfg.get("_target_")
    if not isinstance(noise_target, str):
        raise RuntimeError(
            "Cannot route eval CLI: model.noise_process._target_ missing or non-string "
            f"(got {noise_target!r}). Trained config must declare a noise process."
        )
    if "Categorical" in noise_target:
        return "tmgg.experiments.discrete_diffusion_generative.evaluate_cli"
    raise RuntimeError(
        "Async-eval has no CLI for this diffusion family. "
        f"model.noise_process._target_={noise_target!r}. "
        "Only CategoricalNoiseProcess (discrete) is supported; the gaussian "
        "evaluate_checkpoint CLI was removed in 79428caa."
    )


def run_mmd_evaluation(task_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate checkpoint MMD metrics via CLI subprocess.

    Mirrors the training dispatch pattern in ``_functions.py:modal_run_cli``:
    builds CLI arguments from the evaluation parameters and calls the
    matching evaluate-CLI as a subprocess. The subprocess handles model
    loading, data reconstruction, sampling, and MMD computation. The CLI
    module is selected by inspecting ``config['model']['noise_process']._target_``
    in the run's saved ``config.yaml`` -- there is one CLI per diffusion
    family (currently only the discrete one).

    Parameters
    ----------
    task_dict
        Serialized EvaluationInput as a dictionary.

    Returns
    -------
    dict
        Serialized EvaluationOutput.
    """
    from omegaconf import OmegaConf

    task = EvaluationInput(**task_dict)
    run_id = task.run_id

    # ``output_dir`` is the run directory that holds ``config.yaml`` and
    # ``checkpoints/``. On Modal the convention is
    # ``/data/outputs/{experiment_name}/{run_id}`` -- two levels below
    # ``OUTPUTS_MOUNT``, NOT one. The trainer-side callback passes only
    # the basename ``run_id`` (it strips the experiment-name parent in
    # ``AsyncEvalSpawnCallback._derive_run_id``), so reconstructing from
    # ``run_id`` alone drops ``{experiment_name}/`` and the worker reads
    # the wrong path. When the spawn supplies an explicit
    # ``checkpoint_path`` (the async-eval call path always does), derive
    # ``output_dir`` from ``checkpoint_path.parent.parent`` so the full
    # ``{experiment_name}/{run_id}`` prefix is preserved. Only fall back
    # to ``OUTPUTS_MOUNT/run_id`` for the manual-CLI path that doesn't
    # supply a checkpoint, where the legacy single-level layout still
    # applies.
    if task.checkpoint_path:
        checkpoint_path = Path(task.checkpoint_path)
        output_dir = checkpoint_path.parent.parent
    else:
        output_dir = Path(OUTPUTS_MOUNT) / run_id
        checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    config_path = output_dir / "config.yaml"
    checkpoint_name = checkpoint_path.stem

    # ------------------------------------------------------------------
    # Fail fast on missing files (before spending time on subprocess)
    # ------------------------------------------------------------------
    if not config_path.exists():
        return _make_failed_output(
            run_id, checkpoint_name, f"Config not found: {config_path}"
        )
    if not checkpoint_path.exists():
        return _make_failed_output(
            run_id, checkpoint_name, f"Checkpoint not found: {checkpoint_path}"
        )

    # ------------------------------------------------------------------
    # Read trained config so we can route to the right CLI per diffusion
    # family (discrete vs gaussian). Empty-metrics bug surfaced because
    # the previous hard-coded path called a deleted module.
    # ------------------------------------------------------------------
    raw_config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(config, dict):
        raise TypeError(
            f"Expected dict from OmegaConf.to_container, got {type(config)}"
        )
    # OmegaConf.to_container returns a Dict[DictKeyType, Any]; the keys are
    # always strings on a real config but pyright cannot prove that, so we
    # narrow with an explicit cast.
    config_typed: dict[str, Any] = {str(k): v for k, v in config.items()}
    eval_cli_module = _detect_eval_cli_module(config_typed)

    device = "cuda" if _cuda_available() else "cpu"

    # Temp file for CLI to write JSON results into.
    # Created inside the try block so the finally cleanup always applies.
    result_path = Path(tempfile.mktemp(suffix=".json", prefix="mmd_eval_"))

    try:
        # The discrete evaluate_cli exposes a small surface intentionally:
        # ``--checkpoint``, ``--num-samples``, ``--kernel``, ``--sigma``,
        # ``--device``, ``--output`` (optionally ``--reference_set``,
        # ``--use_ema``). It pulls dataset/num_nodes back from the sibling
        # config.yaml itself, so we don't pass --dataset/--num-nodes here
        # the way the legacy gaussian CLI required.
        cli_args = [
            "python",
            "-m",
            eval_cli_module,
            "--checkpoint",
            str(checkpoint_path),
            "--num-samples",
            str(task.num_samples),
            "--kernel",
            task.mmd_kernel,
            "--sigma",
            str(task.mmd_sigma),
            "--device",
            device,
            "--output",
            str(result_path),
        ]
        if task.output_dir:
            cli_args.extend(
                [
                    "--output-dir",
                    task.output_dir,
                    "--viz-count",
                    str(task.viz_count),
                ]
            )
            if task.val_batch_limit is not None:
                cli_args.extend(["--val-batch-limit", str(task.val_batch_limit)])

        logger.info("Running evaluation CLI: %s", " ".join(cli_args))
        proc = subprocess.run(cli_args, capture_output=True, text=True)

        if proc.returncode != 0:
            error_tail = proc.stderr[-2000:] if proc.stderr else "no stderr"
            logger.error(
                "Evaluation CLI failed (exit %d): %s", proc.returncode, error_tail
            )
            return _make_failed_output(
                run_id,
                checkpoint_name,
                f"CLI exited with code {proc.returncode}: {error_tail}",
            )

        if not result_path.exists():
            return _make_failed_output(
                run_id,
                checkpoint_name,
                "CLI completed but produced no output file",
            )

        with open(result_path) as f:
            cli_results = json.load(f)

        mmd_results = cli_results.get("mmd_results", {})

        # Write combined results to the shared volume
        eval_params = {
            "num_samples": task.num_samples,
            "num_steps": task.num_steps,
            "mmd_kernel": task.mmd_kernel,
            "mmd_sigma": task.mmd_sigma,
            "seed": task.seed,
        }
        eval_output_path = output_dir / f"mmd_evaluation_{checkpoint_name}.json"
        eval_data = {
            "run_id": run_id,
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": str(checkpoint_path),
            "timestamp": datetime.now().isoformat(),
            "params": eval_params,
            "results": {"eval": mmd_results},
        }
        with open(eval_output_path, "w") as f:
            json.dump(eval_data, f, indent=2)
        logger.info("Results saved to: %s", eval_output_path)

        return asdict(
            EvaluationOutput(
                run_id=run_id,
                checkpoint_name=checkpoint_name,
                status="completed",
                results={"eval": mmd_results},
                evaluation_params=eval_params,
                timestamp=datetime.now().isoformat(),
            )
        )

    finally:
        if result_path.exists():
            result_path.unlink()


def list_checkpoints_for_run(run_id: str) -> dict[str, Any]:
    """List all checkpoints available for a run.

    Parameters
    ----------
    run_id
        Run ID to list checkpoints for.

    Returns
    -------
    dict
        Contains 'run_id', 'checkpoints' (list of checkpoint paths), and 'status'.
    """
    output_dir = Path(OUTPUTS_MOUNT) / run_id
    checkpoints_dir = output_dir / "checkpoints"

    if not checkpoints_dir.exists():
        return {
            "run_id": run_id,
            "status": "failed",
            "error_message": f"Checkpoints directory not found: {checkpoints_dir}",
            "checkpoints": [],
        }

    checkpoint_files = sorted(checkpoints_dir.glob("*.ckpt"))
    checkpoints = [
        {
            "name": ckpt.stem,
            "path": str(ckpt),
            "size_mb": ckpt.stat().st_size / (1024 * 1024),
        }
        for ckpt in checkpoint_files
    ]

    return {
        "run_id": run_id,
        "status": "completed",
        "checkpoints": checkpoints,
    }

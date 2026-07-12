"""Two profile runners reused by host CLIs and Modal wrappers.

Each runner emits a *bundle* of artefacts so a single profile run gives
full-stack coverage without re-execution. Layers, cheapest first:

1. ``trace.json`` — chrome-format ``torch.profiler`` trace
   (CUDA-kernel detail, per-op shapes, per-op memory). Streams to disk
   during the run; never aggregated in-container. Aggregate locally
   with ``scripts/profiling/aggregate_chrome_trace.py``.
2. ``cprofile.pstats`` + ``cprofile.txt`` — Python call-graph timing
   (cProfile dump + human-readable top-50). Catches Lightning hook
   overhead, dataloader fetch, callback dispatch, and anything spending
   wall-time in Python that the kernel profiler can't see.
3. ``dynamo_counters.json`` — ``torch._dynamo`` recompile / graph-break
   counters. Empty when ``compile_model=False``; populated and
   diagnostic when compile is enabled. Use to spot uncaught recompiles.

For Python line-level + GPU attribution (scalene), wrap the launcher
script externally:
``scalene --outfile scalene.html -m scripts.profiling.launch_profile``
— this only profiles the launcher process, so it captures Modal client
overhead, not in-container work. cProfile + chrome trace are the
in-container coverage.

Both runners write to ``<output_dir>/{trace.json, cprofile.pstats,
cprofile.txt, dynamo_counters.json}``. The caller is responsible for
wiring ``output_dir`` to a path that survives the run (e.g. a Modal
volume mount).
"""

from __future__ import annotations

import cProfile
import json
import pstats
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile, schedule


def _activities() -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return activities


# Dynamo recompile + graph-break logging is now enabled via the
# ``TORCH_LOGS=recompiles,graph_breaks`` env var baked into the Modal
# image (see ``tmgg/modal/_lib/image._runtime_env``). Doing it via env
# is more reliable than the ``torch._logging.set_logs`` API: the env is
# read at torch import time, so setting it post-import (which is all an
# in-process helper could do) has no effect anyway. The ``dynamo_
# counters.json`` artefact dumped at fit-end is the canonical compile-
# diagnostic surface; the live log lines are just a convenience.


def _dump_dynamo_counters(output_dir: Path) -> None:
    """Snapshot ``torch._dynamo.utils.counters`` to a JSON artefact.

    Empty / mostly-zero when ``torch.compile`` is off; rich and
    diagnostic when on (compile_count, recompile_count,
    graph_break_count, per-frame breakdown, etc.).
    """
    try:
        import torch._dynamo

        snap = {
            section: dict(counters)
            for section, counters in torch._dynamo.utils.counters.items()
        }
    except (ImportError, AttributeError):
        snap = {}
    (output_dir / "dynamo_counters.json").write_text(
        json.dumps(snap, default=str, indent=2)
    )


@contextmanager
def _cprofile_to(output_dir: Path):
    """Run a block under cProfile, dump pstats + a human-readable txt.

    The ``.pstats`` dump is the canonical artefact (loadable in any
    pstats viewer / snakeviz / py-spy speedscope); the ``.txt`` is for
    quick ``cat`` inspection.
    """
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield pr
    finally:
        pr.disable()
        pr.dump_stats(str(output_dir / "cprofile.pstats"))
        with (output_dir / "cprofile.txt").open("w") as fh:
            stats = pstats.Stats(pr, stream=fh).sort_stats("cumulative")
            stats.print_stats(50)
            fh.write("\n\n# Sorted by tottime (own time, excl. children)\n")
            stats.sort_stats("tottime").print_stats(50)


def run_train_profile(
    *,
    output_dir: Path,
    overrides: list[str],
    config_dir: Path,
    config_name: str = "base_config_discrete_diffusion_generative",
    num_steps: int = 100,
    warmup_steps: int = 5,
    active_steps: int = 20,
) -> dict[str, Any]:
    """Run a short training profile and dump trace + summary.

    Parameters
    ----------
    output_dir
        Directory to write ``trace.json`` and ``summary.txt`` into.
        Must be writable; created if missing.
    overrides
        Hydra-style overrides (e.g.
        ``["+data=spectre_sbm", "model.model.n_layers=4"]``). These
        are layered on top of ``config_name``.
    config_dir
        Absolute path to the Hydra config directory (the directory
        that *contains* ``config_name.yaml``).
    config_name
        Base config name; default matches the discrete-diffusion CLI.
    num_steps
        Total ``trainer.max_steps`` for the profiled fit. Profile
        captures only the active window (warmup + active).
    warmup_steps, active_steps
        Profiler schedule. Skip → warmup → active → repeat. We use
        ``wait=0, warmup=warmup_steps, active=active_steps, repeat=1``
        so the trace covers a clean steady-state slice.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import gc

    import hydra
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback as _PLCallback

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name=config_name, overrides=overrides)

    OmegaConf.set_struct(config, False)
    config.trainer.max_steps = num_steps
    OmegaConf.set_struct(config, True)

    # Print resolved overrides so the user can see what was profiled.
    print(f"# train profile config: max_steps={num_steps}")
    print(f"# overrides: {overrides}")

    data_module = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)

    # Use raw ``torch.profiler.profile`` instead of Lightning's
    # ``PyTorchProfiler`` wrapper. The Lightning wrapper is convenient
    # (it adds per-step span annotations to the trace) but it
    # unconditionally calls ``key_averages()`` and a Python-side summary
    # walk at fit teardown — that walk took 30-60 s on the v3 train
    # profile and is otherwise un-disable-able without subclassing.
    # We re-emit the per-step boundaries via a manual ``prof.step()``
    # call inside an ``on_train_batch_end`` callback below, which is
    # the same mechanism Lightning uses internally.
    #
    # ``enable_progress_bar=False`` skips the rich-progress teardown
    # (another 58 s on Modal's non-TTY stdout). Production training
    # keeps the bar; this only applies to profile runs.
    trainer = pl.Trainer(
        max_steps=num_steps,
        accelerator="auto",
        devices=1,
        precision=config.trainer.get("precision", "bf16-mixed"),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,  # PURE train; no eval cycles
        callbacks=[_make_profiler_step_callback(_PLCallback)],
    )

    # The torch.profiler context wraps the entire fit; the callback
    # above advances ``prof.step()`` after each training batch so the
    # warmup/active schedule fires on real per-step boundaries.
    t0 = time.time()
    with (
        _cprofile_to(output_dir),
        profile(
            activities=_activities(),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            schedule=schedule(
                wait=0,
                warmup=warmup_steps,
                active=active_steps,
                repeat=1,
            ),
        ) as prof,
    ):
        # Hand the live profile context to the per-batch callback via
        # the mutable holder it captured by closure (see
        # ``_make_profiler_step_callback``). Avoids stashing on
        # ``trainer`` (Lightning's ``Trainer`` doesn't declare
        # arbitrary attributes) and keeps the callback's reference
        # explicit.
        _profile_holder.append(prof)
        try:
            trainer.fit(model, data_module)
        finally:
            _profile_holder.clear()
    elapsed = time.time() - t0

    trace_path = output_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))

    _dump_dynamo_counters(output_dir)

    # Force-shut the Trainer + dataloader workers immediately so the
    # function can return. Without this, ``trainer.__del__`` waits on
    # the persistent dataloader workers' join (~59 s on v3) before
    # control returns to the Modal entry point.
    del trainer, model, data_module
    gc.collect()

    return {
        "kind": "train_profile",
        "output_dir": str(output_dir),
        "trace_path": str(trace_path),
        "elapsed_s": elapsed,
        "num_steps": num_steps,
        "warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


# Module-level mutable holder for the active ``torch.profiler.profile``
# context. The Lightning callback factory below captures this list by
# closure; ``run_train_profile`` appends the live ``prof`` after the
# context manager opens and clears on exit. Avoids monkey-patching
# attributes onto ``pl.Trainer`` (whose ``__init_subclass__``-style
# attribute checks reject unknown attributes under basedpyright).
_profile_holder: list[Any] = []


def _make_profiler_step_callback(callback_base: type) -> Any:
    """Build a Lightning callback that advances ``prof.step()`` per batch.

    Replaces what Lightning's ``PyTorchProfiler`` wrapper does
    internally so we can use the raw ``torch.profiler.profile`` (and
    skip its expensive end-of-fit aggregation walk). Constructed via a
    factory so the ``pytorch_lightning.callbacks.Callback`` import
    stays inside ``run_train_profile`` (consistent with the rest of the
    Lightning imports being function-local).
    """

    class _TorchProfilerStepCallback(callback_base):  # type: ignore[misc, valid-type]
        def on_train_batch_end(
            self,
            _trainer: Any,
            *_args: Any,
            **_kwargs: Any,
        ) -> None:
            if _profile_holder:
                _profile_holder[0].step()

    return _TorchProfilerStepCallback()


def run_eval_profile(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    num_samples: int = 32,
    val_batch_limit: int | None = None,
    viz_count: int = 4,
    device: str = "cuda",
    use_ema: str = "auto",
    compile_model: bool = False,
    compile_mode: str = "default",
    sample_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Run one ``evaluate_checkpoint`` cycle wrapped in ``torch.profiler``.

    Parameters
    ----------
    output_dir
        Directory to write ``trace.json`` and ``summary.txt`` into.
    checkpoint_path
        Path to a ``*.ckpt`` from a discrete-diffusion training run.
    num_samples
        Number of graphs to sample during the profiled cycle. Match
        the per-eval cost of the real sweep (32) for a comparable
        wall-time read; reduce to 8-16 if the profile run-time is
        prohibitive on cheap-tier.
    val_batch_limit, viz_count, device, use_ema
        Forwarded to ``evaluate_checkpoint`` unchanged.
    """
    from tmgg.experiments.discrete_diffusion_generative.evaluate_cli import (
        evaluate_checkpoint,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"# eval profile checkpoint: {checkpoint_path}")
    print(f"# num_samples={num_samples}, device={device}")

    t0 = time.time()
    with (
        _cprofile_to(output_dir),
        profile(
            activities=_activities(),
            # ``record_shapes`` and ``profile_memory`` stay on so the host-side
            # aggregator can attribute time by tensor shape and surface
            # memory high-water marks per op. The eval trace is large
            # (~12 GB at num_samples=32) but only the host-side aggregator
            # consumes it — Modal-side aggregation is no longer attempted.
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof,
    ):
        result = evaluate_checkpoint(
            checkpoint_path=Path(checkpoint_path),
            num_samples=num_samples,
            reference_set="val",
            use_ema=use_ema,  # type: ignore[arg-type]
            device=device,
            output_dir=output_dir / "eval_dump",
            val_batch_limit=val_batch_limit,
            viz_count=viz_count,
            compile_model=compile_model,
            compile_mode=compile_mode,
            sample_chunk_size=sample_chunk_size,
        )
    elapsed = time.time() - t0

    _dump_dynamo_counters(output_dir)

    # Chrome trace is the canonical kernel artefact. Aggregation runs on
    # the host via ``scripts/profiling/aggregate_chrome_trace.py``; the
    # cprofile.{pstats,txt} bundle covers Python-side overhead.
    trace_path = output_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))

    return {
        "kind": "eval_profile",
        "output_dir": str(output_dir),
        "trace_path": str(trace_path),
        "elapsed_s": elapsed,
        "num_samples": num_samples,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
        "metrics_keys": sorted(result.get("metrics", {}).keys())[:20]
        if isinstance(result, dict)
        else [],
    }

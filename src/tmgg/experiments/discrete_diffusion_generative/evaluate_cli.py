"""CLI for discrete diffusion checkpoint evaluation.

Loads a trained checkpoint, samples graphs, and reports the FULL graph
generation metric set (degree/clustering/spectral MMD plus orbit MMD,
SBM accuracy, planarity, uniqueness, novelty, and block-structure
telemetry) against a reference distribution drawn from the
checkpoint's own datamodule (val or test split per
``--reference_set``). Separated from ``evaluate`` to avoid an import
cycle with ``lightning_module``.

Only DiffusionModule checkpoints are supported. Legacy checkpoints from
earlier LightningModule implementations are incompatible.

Reference graphs come from the *training-time datamodule* rather than a
synthetic regenerator. The datamodule is reconstructed via
:func:`hydra.utils.instantiate` against the ``data:`` block of the
sibling ``config.yaml`` that ``run_experiment`` saves alongside every
checkpoint. The fail-loud contract holds: when ``config.yaml`` is missing
or its ``data:`` block lacks a ``_target_``, the call raises rather
than falling back to synthetic graphs.

The :class:`~tmgg.evaluation.graph_evaluator.GraphEvaluator` is also
instantiated from the sibling ``config.yaml`` (the
``model.evaluator:`` block) so the CLI computes the same metric set
that the trainer uses at validation time. The trainer's saved
``evaluator`` config carries the dataset-specific ``p_intra`` /
``p_inter`` / ``clustering_sigma`` overrides, so they survive the
round-trip into the CLI without callers having to duplicate them on
the command line. The ``--kernel`` and ``--sigma`` CLI flags, when
explicitly passed, override the corresponding fields on the
instantiated evaluator; otherwise the saved-config values win.

EMA weights swap into the loaded model when ``use_ema in ('true',
'auto')`` and the checkpoint carries an ``EMACallback`` shadow under
``checkpoint['callbacks']``. ``--use_ema true`` without a shadow raises;
``--use_ema auto`` silently uses live weights and records
``ema_active=False`` in the result dict.

Known limitations
-----------------
``novelty`` is always ``None`` in CLI results. Computing it would
require fetching the ``train`` split from the reconstructed
datamodule and threading the train graphs into ``GraphEvaluator``;
that work is tangential to the smallest-config sweep's threshold
check (which gates only on the conditionally-computed metrics like
``orbit_mmd``, ``sbm_accuracy``, and the block-structure metrics).
Lift this when the sweep configuration adds a novelty threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from tmgg.evaluation.graph_evaluator import GraphEvaluator
from tmgg.experiments.discrete_diffusion_generative._eval_dump import (
    EvalTimings,
    dump_eval_artifacts,
    run_val_pass_with_per_batch_capture,
    stage_timer,
)
from tmgg.training.callbacks.ema import EMACallback
from tmgg.training.lightning_modules.diffusion_module import DiffusionModule


def _load_diffusion_module(
    checkpoint_path: Path,
    *,
    device: str,
) -> DiffusionModule:
    """Load a diffusion checkpoint, hydra-instantiating from the sibling config.

    ``DiffusionModule.save_hyperparameters(ignore=["model", "noise_process",
    "sampler", "noise_schedule", "evaluator"])`` (see
    :class:`DiffusionModule`) drops five ``nn.Module`` constructor arguments
    from the saved hparams. Lightning's ``load_from_checkpoint`` then can't
    reconstruct the module via ``cls(**hparams)`` and raises::

        TypeError: DiffusionModule.__init__() missing 3 required keyword-only
        arguments: 'model', 'noise_process', and 'noise_schedule'

    The fix is to read the sibling ``config.yaml`` (saved by
    ``run_experiment.run_single_experiment`` next to the checkpoint),
    hydra-instantiate the full module via ``config.model`` (whose
    ``_target_`` points at :class:`DiffusionModule` and whose nested
    ``model``/``noise_process``/``noise_schedule`` sub-blocks hydra
    recursively instantiates), then load just the ``state_dict`` from the
    checkpoint.

    ``weights_only=False`` is required because Lightning checkpoints carry
    non-tensor metadata (callbacks state, hyperparameters dict, optimizer
    state). We trust the checkpoint here because we wrote it ourselves to
    our private Modal volume; the checkpoint is not third-party content.
    """
    config_path = _find_sibling_config_yaml(checkpoint_path)
    loaded = OmegaConf.load(config_path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(
            f"config.yaml at {config_path} did not parse to a DictConfig; "
            f"got {type(loaded).__name__}."
        )
    model_cfg = OmegaConf.select(loaded, "model")
    if model_cfg is None:
        raise KeyError(
            f"config.yaml at {config_path} has no `model:` block; cannot "
            f"hydra-instantiate the DiffusionModule for checkpoint loading."
        )

    module: DiffusionModule = hydra.utils.instantiate(model_cfg)

    # weights_only=False: Lightning checkpoints contain non-tensor metadata.
    # Safe here because we wrote this checkpoint ourselves on our private volume.
    checkpoint = torch.load(
        str(checkpoint_path), map_location=device, weights_only=False
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Checkpoint {checkpoint_path} has no 'state_dict' entry and "
            f"the top-level object is not a dict (got {type(state_dict).__name__})."
        )

    # CategoricalNoiseProcess pre-registers ``_limit_{x,e,y}`` as None
    # buffers in __init__; ``initialize_from_data()`` (called during
    # training-time setup) populates them with empirical-marginal probs.
    # ``state_dict`` excludes None-valued buffers, so a fresh post-init
    # module's state_dict has no ``_limit_*`` keys — yet trained
    # checkpoints DO carry populated tensor values for these keys, which
    # ``load_state_dict(strict=True)`` flags as unexpected. Pre-populate
    # the buffers directly from the checkpoint values so the strict
    # load succeeds.
    np_buf_keys = (
        "noise_process._limit_x",
        "noise_process._limit_e",
        "noise_process._limit_y",
    )
    for key in np_buf_keys:
        if key in state_dict and hasattr(module, "noise_process"):
            attr = key.split(".", 1)[1]
            module.noise_process.register_buffer(attr, state_dict[key].clone())

    module.load_state_dict(state_dict)
    return module.to(device).eval()


def _find_sibling_config_yaml(checkpoint_path: Path) -> Path:
    """Locate the ``config.yaml`` saved next to a Lightning run's checkpoints.

    ``run_experiment.run_single_experiment`` writes ``config.yaml`` into
    ``${paths.output_dir}`` and stores checkpoints under
    ``${paths.output_dir}/checkpoints/``. The sibling lookup walks up
    one directory from the checkpoint file.

    Raises
    ------
    FileNotFoundError
        Neither ``checkpoint_path.parent.parent / "config.yaml"`` nor
        ``checkpoint_path.parent / "config.yaml"`` exists. Both fail
        loudly per CLAUDE.md -- there is no synthetic-regeneration
        fallback.
    """
    candidates = [
        checkpoint_path.parent.parent / "config.yaml",
        checkpoint_path.parent / "config.yaml",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No config.yaml found alongside {checkpoint_path}; looked in: "
        f"{[str(c) for c in candidates]}. Reference-graph extraction "
        "requires the training-time data config to reconstruct the "
        "datamodule."
    )


def _load_datamodule_for_reference(
    checkpoint_path: Path,
    *,
    reference_set: Literal["val", "test"],
) -> Any:
    """Instantiate and ``setup`` the training-time datamodule.

    Reads the sibling ``config.yaml``, calls
    :func:`hydra.utils.instantiate` on the ``data:`` block, and runs the
    appropriate Lightning ``setup`` stage for the requested split.

    Parameters
    ----------
    checkpoint_path
        Path to the ``.ckpt`` file under evaluation. The sibling
        ``config.yaml`` is located one directory above.
    reference_set
        ``"val"`` runs ``setup("fit")`` (Lightning's standard stage for
        train+val); ``"test"`` runs ``setup("test")``.

    Returns
    -------
    Any
        The instantiated, setup datamodule. Typed ``Any`` because Hydra
        recursive instantiation returns ``Any`` and the concrete
        :class:`BaseGraphDataModule` subclass varies per experiment.

    Raises
    ------
    KeyError
        ``config.yaml`` lacks a ``data`` block.
    """
    config_path = _find_sibling_config_yaml(checkpoint_path)
    loaded = OmegaConf.load(config_path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(
            f"config.yaml at {config_path} did not parse to a DictConfig; "
            f"got {type(loaded).__name__}. The training-time runner saves a "
            "mapping; a list/scalar at the top level is malformed."
        )
    data_cfg = OmegaConf.select(loaded, "data")
    if data_cfg is None:
        raise KeyError(
            f"config.yaml at {config_path} has no `data:` block; cannot "
            "reconstruct the datamodule for reference-graph extraction."
        )

    datamodule = hydra.utils.instantiate(data_cfg)
    stage = "fit" if reference_set == "val" else "test"
    datamodule.setup(stage=stage)
    return datamodule


def _load_evaluator(checkpoint_path: Path) -> Any:
    """Instantiate the saved evaluator (Graph or Molecular) from config.

    Reads the sibling ``config.yaml`` and Hydra-instantiates the
    ``model.evaluator:`` block. The trainer's evaluator carries the
    dataset-specific knobs (``p_intra`` / ``p_inter`` / ``clustering_sigma``
    for graph runs, the codec + metric stack for molecular runs);
    reusing it ensures the CLI computes the same metric set the trainer
    logs at validation time.

    Returns either a :class:`GraphEvaluator` or a
    :class:`MolecularEvaluator`. Both expose ``evaluate(refs, generated)``
    and ``evaluate(...).to_dict()``; the CLI relies on that duck-typed
    surface and routes both flavours through the same code path.

    Parameters
    ----------
    checkpoint_path
        Path to the ``.ckpt`` file under evaluation. The sibling
        ``config.yaml`` is located one directory above.

    Returns
    -------
    GraphEvaluator | MolecularEvaluator
        The instantiated evaluator.

    Raises
    ------
    KeyError
        ``config.yaml`` lacks a ``model.evaluator`` block.
    """
    from tmgg.evaluation.molecular.evaluator import MolecularEvaluator

    config_path = _find_sibling_config_yaml(checkpoint_path)
    loaded = OmegaConf.load(config_path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(
            f"config.yaml at {config_path} did not parse to a DictConfig; "
            f"got {type(loaded).__name__}."
        )
    evaluator_cfg = OmegaConf.select(loaded, "model.evaluator")
    if evaluator_cfg is None:
        raise KeyError(
            f"config.yaml at {config_path} has no `model.evaluator:` block; "
            "cannot reconstruct the evaluator for the CLI metric set. "
            "Older runs without an evaluator block in the saved config are "
            "unsupported."
        )
    evaluator = hydra.utils.instantiate(evaluator_cfg)
    if not isinstance(evaluator, GraphEvaluator | MolecularEvaluator):
        raise TypeError(
            f"`model.evaluator` at {config_path} did not instantiate to a "
            f"GraphEvaluator or MolecularEvaluator; got {type(evaluator).__name__}."
        )
    return evaluator


def _collect_reference_graphs(
    datamodule: Any,
    *,
    reference_set: Literal["val", "test"],
    max_graphs: int,
) -> list[Any]:
    """Pull up to ``max_graphs`` reference graphs as ``GraphState``.

    Delegates to :meth:`BaseGraphDataModule.get_reference_graphs`,
    which (post the 2026-05-07 sparse-default refactor) returns
    per-graph ``GraphState`` slices. Callers needing nx convert at the
    leaf via :meth:`GraphState.to_networkx` (cheap, lossless).
    """
    return datamodule.get_reference_graphs(reference_set, max_graphs)


def _maybe_load_ema_state(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    """Return the EMACallback state dict from the checkpoint, if present.

    Lightning persists callback state under ``checkpoint['callbacks']``
    keyed by the callback's ``state_key`` (a string containing the
    class name and constructor arg repr). The probe matches any key
    containing ``"EMA"`` to track Lightning's default state-key naming
    convention.
    """
    callbacks = checkpoint.get("callbacks")
    if not isinstance(callbacks, dict):
        return None
    for key, value in callbacks.items():
        if "EMA" in str(key) and isinstance(value, dict):
            return value
    return None


def _maybe_swap_ema_into_module(
    module: DiffusionModule,
    checkpoint: dict[str, Any],
    use_ema: Literal["auto", "true", "false"],
) -> bool:
    """Swap EMA shadow weights into ``module.model`` when configured.

    Returns ``True`` when the swap actually fired (i.e. the result
    record's ``ema_active`` flag should be ``True``).

    Raises
    ------
    RuntimeError
        ``use_ema == "true"`` requested but the checkpoint carries no
        EMA shadow.
    """
    if use_ema == "false":
        return False
    ema_state = _maybe_load_ema_state(checkpoint)
    if ema_state is None:
        if use_ema == "true":
            raise RuntimeError(
                "use_ema=true requested but the checkpoint carries no "
                "EMA shadow under checkpoint['callbacks']."
            )
        return False
    decay = float(ema_state["decay"])
    cb = EMACallback(decay=decay)
    cb.load_state_dict(ema_state)
    cb.copy_shadow_into(module.model)
    return True


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    num_samples: int = 500,
    reference_set: Literal["val", "test"] = "val",
    use_ema: Literal["auto", "true", "false"] = "auto",
    mmd_kernel: Literal["gaussian", "gaussian_tv"] | None = None,
    mmd_sigma: float | None = None,
    device: str = "cpu",
    output_dir: str | Path | None = None,
    val_batch_limit: int | None = None,
    viz_count: int = 32,
    compile_model: bool = False,
    compile_mode: str = "default",
    sample_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Load a discrete diffusion checkpoint and compute the full metric set.

    Reference graphs come from the *training-time datamodule*: the
    sibling ``config.yaml`` is loaded, the ``data:`` block is
    Hydra-instantiated, and the appropriate split's dataloader is
    walked to collect up to ``num_samples`` references. There is no
    synthetic-regeneration fallback (CLAUDE.md "single way of doing
    things").

    The evaluator itself is instantiated from the same sibling
    ``config.yaml`` (the ``model.evaluator:`` block), so the CLI
    reports the same metric set the trainer logs at validation time:
    degree/clustering/spectral MMD, orbit MMD (when ``orca`` is
    available), SBM accuracy (when ``graph-tool`` is available),
    block-structure telemetry (modularity Q, λ₂ spectral gap,
    empirical p_in/p_out), planarity, and uniqueness. ``novelty`` is
    always ``None`` here because the CLI does not currently fetch the
    train split (see module docstring).

    Parameters
    ----------
    checkpoint_path
        Path to a Lightning checkpoint (``.ckpt``).
    num_samples
        Number of graphs to generate and compare. The reference set is
        capped at the same count.
    reference_set
        ``"val"`` (default) or ``"test"``. Selects which split's
        dataloader supplies the reference graphs.
    use_ema
        ``"auto"`` swaps shadow weights when present, otherwise uses
        live weights (recording ``ema_active=False``). ``"true"``
        requires a shadow; raises if absent. ``"false"`` skips the
        swap.
    mmd_kernel
        When non-``None``, overrides the kernel field on the
        instantiated evaluator (per the "explicit CLI flag wins"
        contract documented at module level). ``None`` keeps the
        saved-config value.
    mmd_sigma
        When non-``None``, overrides the fallback sigma field on the
        instantiated evaluator. Per-metric ``degree_sigma`` /
        ``clustering_sigma`` / ``spectral_sigma`` from the saved
        config are not overridden here -- the CLI knob is too coarse
        for that, and the dataset-specific clustering bandwidth
        matters most. ``None`` keeps the saved-config value.
    device
        Torch device.

    Returns
    -------
    dict
        Evaluation results. ``mmd_results`` carries the full flat
        dictionary produced by
        :meth:`tmgg.evaluation.graph_evaluator.EvaluationResults.to_dict`,
        with ``None`` for any conditionally-skipped metric (orca /
        graph-tool unavailable, novelty unsupported by the CLI).
    """
    checkpoint_path = Path(checkpoint_path)

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {checkpoint_path.name}")
    print(f"{'=' * 60}")

    timings = EvalTimings()
    total_t = stage_timer()

    # Load model from checkpoint (save_hyperparameters stores constructor args)
    # Only DiffusionModule checkpoints are supported.
    print("\nLoading model...")
    load_t = stage_timer()
    module = _load_diffusion_module(checkpoint_path, device=device)
    module = module.to(device)
    module.eval()

    # Optional ``torch.compile`` wrap for the eval path. Independent of
    # whether the checkpoint was trained with compile — we wrap
    # post-load so a non-compile checkpoint can still be eval-profiled
    # under compile. ``module.model`` is the inner GraphTransformer; the
    # outer LightningModule remains uncompiled to keep Lightning hooks
    # working.
    if compile_model:
        # ``dynamic=True`` is essential on the eval path: even with
        # ``sample_chunk_size`` matching train batch_size, the val
        # capture phase hits the val dataloader's tail (32 graphs / 12 =
        # 8) which would otherwise force a fresh recompile costing
        # ~80 s per shape variant. Symbolic dim-0 lets one trace cover
        # all batch sizes encountered.
        print(
            f"# wrapping module.model with torch.compile("
            f"mode={compile_mode!r}, dynamic=True)"
        )
        module.model = torch.compile(  # pyright: ignore[reportAttributeAccessIssue]
            module.model, mode=compile_mode, dynamic=True
        )

    # Re-load the raw checkpoint dict to inspect EMA callback state.
    # ``load_from_checkpoint`` does not expose the callback section.
    raw_checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    ema_active = _maybe_swap_ema_into_module(module, raw_checkpoint, use_ema)
    if ema_active:
        print("Swapped EMA shadow weights into model for evaluation.")
    else:
        print("Evaluating live (non-EMA) weights.")
    timings.load_s = load_t()

    # Pull reference graphs from the training-time datamodule. Lightning
    # `setup` reads the dataset on disk (or generates synthetic batches
    # when the datamodule does that internally), then `get_reference_graphs`
    # walks the dataloader and binarises each adjacency.
    print(
        f"\nFetching {num_samples} reference graphs from {reference_set} split "
        "via training-time datamodule..."
    )
    datamodule = _load_datamodule_for_reference(
        checkpoint_path, reference_set=reference_set
    )
    ref_graphs = _collect_reference_graphs(
        datamodule, reference_set=reference_set, max_graphs=num_samples
    )
    if not ref_graphs:
        raise RuntimeError(
            f"Reference {reference_set} split for {checkpoint_path} returned "
            "zero graphs; cannot compute the metric set against an empty reference."
        )

    # Optional val-pass-with-per-batch-capture. Only fires when
    # ``output_dir`` is set, because the per-batch CSV is the *only*
    # consumer of the captured rows; running the pass otherwise wastes
    # ~30s. Order: before sampling, so an OOM in sampling does not
    # discard val diagnostics already on disk after dump_eval_artifacts.
    val_rows = None
    if output_dir is not None:
        print("\nRunning val-pass with per-batch capture...")
        val_t = stage_timer()
        val_rows = run_val_pass_with_per_batch_capture(
            module, datamodule, max_batches=val_batch_limit
        )
        timings.val_pass_s = val_t()
        print(f"Captured {len(val_rows.rows)} val batches in {timings.val_pass_s:.1f}s")

    # Reconstruct the trainer's evaluator (Graph or Molecular) from
    # the saved config so the CLI reports the same metric set the
    # trainer logs at validation. Apply CLI overrides for kernel/sigma
    # when explicitly supplied — only meaningful for GraphEvaluator
    # (MolecularEvaluator does not carry MMD knobs).
    evaluator = _load_evaluator(checkpoint_path)
    if isinstance(evaluator, GraphEvaluator):
        if mmd_kernel is not None:
            evaluator.kernel = mmd_kernel
        if mmd_sigma is not None:
            evaluator.sigma = mmd_sigma

    # Sample from model. Generate the same count as the (possibly
    # truncated) reference list so MMD compares equally-sized
    # populations.
    num_generated = len(ref_graphs)
    print(f"Sampling {num_generated} graphs from model...")
    sample_t = stage_timer()
    with torch.no_grad():
        generated_graphs = module.generate_graphs(
            num_generated, chunk_size=sample_chunk_size
        )
    timings.sample_s = sample_t()

    eval_t = stage_timer()
    eval_results = evaluator.evaluate(ref_graphs, generated_graphs)
    if eval_results is None:
        raise RuntimeError(
            f"GraphEvaluator.evaluate returned None for {checkpoint_path}: "
            "fewer than 2 graphs in either the reference or the generated "
            "set after eval_num_samples truncation. Increase --num-samples "
            "or check the datamodule split."
        )
    metric_dict = eval_results.to_dict()
    timings.eval_s = eval_t()
    timings.total_s = total_t()

    results: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path.absolute()),
        "checkpoint_name": checkpoint_path.name,
        "reference_set": reference_set,
        "num_generated": num_generated,
        "num_reference": len(ref_graphs),
        # GraphEvaluator carries kernel/sigma; MolecularEvaluator does not.
        # Persist whichever shape the saved evaluator exposed so the
        # downstream metrics.json round-trips are evaluator-aware.
        "evaluator_type": type(evaluator).__name__,
        "mmd_kernel": getattr(evaluator, "kernel", None),
        "mmd_sigma": getattr(evaluator, "sigma", None),
        "mmd_results": metric_dict,
        "timings": timings.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "ema_active": ema_active,
    }

    # Smoke-debug print. Render every metric the evaluator returned;
    # ``None`` shows as ``n/a`` so a missing optional dependency
    # (orca, graph-tool) or skipped metric is obvious at a glance.
    print(f"\n{'=' * 60}")
    print("Evaluation Results")
    print(f"{'=' * 60}")
    for metric_name, value in metric_dict.items():
        if value is None:
            print(f"  {metric_name:<22} n/a")
        else:
            print(f"  {metric_name:<22} {value:.6f}")
    print(f"{'=' * 60}\n")

    if output_dir is not None:
        out_dir_path = Path(output_dir)
        print(f"Dumping per-checkpoint artifacts to {out_dir_path}")
        # GraphState → nx at the leaf for viz + edge-list
        # serialisation. Per spec D4-A: x_class/e_class indices ride
        # along on the resulting graphs as node/edge attributes (via
        # GraphState.to_networkx) and land in generated_graphs.json /
        # reference_graphs.json so molecular post-hoc analysis can
        # recover atom + bond types without re-loading the codec.
        # Both ``generated_graphs`` (sparse ``GraphState`` from
        # ``module.generate_graphs``) and ``ref_graphs`` (per-graph
        # ``DenseGraphState`` from ``get_reference_graphs``) carry a
        # leading batch dim of 1, so ``batch_index=0`` selects the single
        # graph in either carrier.
        nx_generated = [g.to_networkx(batch_index=0) for g in generated_graphs]
        nx_reference = [g.to_networkx(batch_index=0) for g in ref_graphs]
        artifact_paths = dump_eval_artifacts(
            out_dir_path,
            checkpoint_name=checkpoint_path.name,
            results=results,
            generated_graphs=nx_generated,
            reference_graphs=nx_reference,
            val_rows=val_rows,
            timings=timings,
            viz_count=viz_count,
        )
        results["artifact_paths"] = artifact_paths

    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for discrete diffusion checkpoint evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a discrete diffusion checkpoint with MMD metrics",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--num-samples", type=int, default=500, help="Graphs to generate/compare"
    )
    parser.add_argument(
        "--reference_set",
        default="val",
        choices=["val", "test"],
        help="Which split's dataloader supplies the reference graphs.",
    )
    parser.add_argument(
        "--use_ema",
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "Swap to EMA shadow weights before sampling. auto = swap iff "
            "the checkpoint carries a shadow."
        ),
    )
    parser.add_argument(
        "--kernel",
        default=None,
        choices=["gaussian", "gaussian_tv"],
        help=(
            "Override the MMD kernel on the saved evaluator. "
            "Default: keep the saved-config value (typically gaussian_tv)."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help=(
            "Override the fallback MMD bandwidth on the saved evaluator. "
            "Default: keep the saved-config value."
        ),
    )
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument(
        "--output", type=str, default=None, help="Write results to JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Write the full multi-format dump (metrics.json/csv, "
            "generated/reference graph edge-list JSON, viz PNGs, "
            "val_per_batch.csv, timings.json, summary.md) into this "
            "directory. Triggers an extra val-pass to capture per-batch "
            "diagnostics; orthogonal to --output (the JSON-only file)."
        ),
    )
    parser.add_argument(
        "--val-batch-limit",
        type=int,
        default=None,
        help=(
            "Cap on val-batches walked during the per-batch capture. "
            "Default: walk the full val split. Only meaningful with "
            "--output-dir."
        ),
    )
    parser.add_argument(
        "--viz-count",
        type=int,
        default=32,
        help=(
            "Per-side viz count under <output-dir>/viz/ "
            "(generated_NN.png / reference_NN.png). Default 32."
        ),
    )

    args = parser.parse_args(argv)

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        reference_set=args.reference_set,
        use_ema=args.use_ema,
        mmd_kernel=args.kernel,
        mmd_sigma=args.sigma,
        device=args.device,
        output_dir=args.output_dir,
        val_batch_limit=args.val_batch_limit,
        viz_count=args.viz_count,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"Results written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

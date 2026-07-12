"""Lightning callback that dumps generated samples at the end of fit.

At ``on_fit_end`` the callback samples ``num_samples`` graphs from the
trained model, runs the configured :class:`GraphEvaluator` against the
**test** reference set, logs the resulting metrics under the ``final/``
namespace, and persists the raw sample list to disk.

Storage location resolution:

1. An explicit ``save_path`` argument wins outright.
2. When the runtime appears to be a Modal container (``MODAL_TASK_ID``
   environment variable set), the callback writes to
   ``/data/outputs/final_samples/<run_name>.pt``. That path is the
   canonical mount point of the ``graph-denoising-outputs`` Modal volume
   (:data:`tmgg.modal._lib.volumes.OUTPUTS_MOUNT`); the constant is
   inlined here as the module-level :data:`MODAL_OUTPUTS_MOUNT` because
   tach forbids ``tmgg.training -> tmgg.modal``. A dedicated test
   pins the inlined value against the canonical constant so a future
   change to the volume layout fails loudly.
3. Otherwise the callback writes to
   ``trainer.default_root_dir / "final_samples.pt"``.

Per spec resolutions (`internal notes (not included)`):

* **Q6** -- the dump uses the **test** reference set (validation
  cadence during training stays on the val set, untouched).
* **Q7** -- if an :class:`~tmgg.training.callbacks.ema.EMACallback` is
  registered on the trainer, the callback swaps to EMA weights for the
  duration of the dump and restores live weights afterwards. Without
  EMA registered, the dump uses live weights.
* **Q8** -- Modal volume routing (see above).
* **Q9** -- single-rank v1: only ``trainer.is_global_zero`` runs the
  dump; other DDP ranks no-op so we don't double-sample.

The callback follows the :class:`EMACallback` pattern of pulling
backbone references via ``getattr`` plus fail-loud :class:`TypeError`
for missing collaborators -- Lightning's loose typing on ``pl_module``
forbids static narrowing without suppressing pyright, and CLAUDE.md
forbids new pyright suppressions.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import torch
from pytorch_lightning.callbacks import Callback

from tmgg.training.callbacks.ema import EMACallback

if TYPE_CHECKING:
    import pytorch_lightning as pl

    from tmgg.data.datasets.graph_types import GraphState

#: Mirror of :data:`tmgg.modal._lib.volumes.OUTPUTS_MOUNT`. Inlined to
#: avoid a tach-forbidden ``tmgg.training -> tmgg.modal`` import; pinned
#: by ``tests/training/test_final_sample_dump.py``.
MODAL_OUTPUTS_MOUNT = "/data/outputs"


def _require_attr(pl_module: pl.LightningModule, name: str) -> Any:
    """Return ``pl_module.<name>``, raising fail-loud if missing.

    Mirrors the pattern in :func:`tmgg.training.callbacks.ema._require_backbone_parameters`.
    Lightning's ``pl_module`` is typed as the bare LightningModule base,
    which doesn't expose ``.sampler`` / ``.evaluator`` / etc. Using
    ``getattr`` with a sentinel and an explicit None check sidesteps the
    pyright complaint without a suppression and yields a clearer error
    when the wrong module type is attached to the trainer.
    """
    sentinel = object()
    value = getattr(pl_module, name, sentinel)
    if value is sentinel or value is None:
        raise TypeError(
            f"FinalSampleDumpCallback requires the LightningModule to expose "
            f"`.{name}`; got {type(pl_module).__name__} which does not "
            "(or which exposes None). Wire the attribute on the module "
            "before registering this callback."
        )
    return value


def _resolve_dump_path(
    *,
    configured: str | os.PathLike[str] | None,
    run_name: str,
    default_root_dir: str | os.PathLike[str],
) -> Path:
    """Resolve where ``final_samples.pt`` should be written.

    Priority order documented at module level. The Modal-context check
    looks at ``MODAL_TASK_ID``; that variable is set inside Modal
    containers (and only there).

    Parameters
    ----------
    configured
        Explicit ``save_path`` from config, or ``None`` to defer to
        defaults.
    run_name
        Stable run identifier used to namespace the Modal-volume
        artefact (avoids cross-run collisions on a shared volume).
    default_root_dir
        ``trainer.default_root_dir`` -- the local fallback location.
    """
    if configured is not None:
        return Path(configured)
    if os.environ.get("MODAL_TASK_ID"):
        return Path(MODAL_OUTPUTS_MOUNT) / "final_samples" / f"{run_name}.pt"
    return Path(default_root_dir) / "final_samples.pt"


class FinalSampleDumpCallback(Callback):
    """Sample N graphs at fit-end, evaluate, and persist them.

    Parameters
    ----------
    num_samples
        Total number of graphs to draw. Must be positive (the
        orchestrator omits the callback entirely when zero).
    sample_batch_size
        Per-batch size for the generation loop. Bounds peak memory.
        Defaults to 64.
    save_path
        Optional explicit output path. When ``None`` the callback uses
        the Modal-aware resolution helper.
    run_name
        Stable run identifier for namespacing the Modal artefact.

    Raises
    ------
    ValueError
        ``num_samples <= 0`` or ``sample_batch_size <= 0``.
    """

    def __init__(
        self,
        num_samples: int,
        sample_batch_size: int = 64,
        save_path: str | os.PathLike[str] | None = None,
        run_name: str = "run",
    ) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError(
                "FinalSampleDumpCallback.num_samples must be > 0; "
                f"got {num_samples}. Disable the callback by not "
                "registering it rather than passing 0."
            )
        if sample_batch_size <= 0:
            raise ValueError(
                "FinalSampleDumpCallback.sample_batch_size must be > 0; "
                f"got {sample_batch_size}."
            )
        self.num_samples = num_samples
        self.sample_batch_size = sample_batch_size
        self.save_path = save_path
        self.run_name = run_name

    @staticmethod
    def _find_ema_callback(trainer: pl.Trainer) -> EMACallback | None:
        """Return a registered :class:`EMACallback`, or ``None``.

        ``trainer.callbacks`` is present at runtime but absent from
        Lightning's public type stubs; fetch it via ``getattr`` (whose
        ``Any`` return pyright accepts) rather than silencing the
        attribute-access error. Mirrors the getattr gate pattern used
        by ``EMACallback._require_backbone_parameters``.
        """
        callbacks = getattr(trainer, "callbacks", None)
        if callbacks is None:
            return None
        for cb in callbacks:
            if isinstance(cb, EMACallback):
                return cb
        return None

    @staticmethod
    def _log_to_experiment(
        pl_module: pl.LightningModule,
        metrics: dict[str, float],
        *,
        step: int,
    ) -> None:
        """Write metrics to the trainer logger's underlying experiment.

        Lightning forbids ``self.log()`` inside ``on_fit_end`` so we
        bypass the module-level logging shim and emit through the
        logger directly. Quiet no-op when no logger is configured (e.g.
        in unit tests run with ``logger=False``).
        """
        loggers = getattr(pl_module, "loggers", None)
        if not loggers:
            return
        for logger in loggers:
            log_metrics = getattr(logger, "log_metrics", None)
            if log_metrics is not None:
                log_metrics(metrics, step=step)

    def _sample_graphs(self, pl_module: pl.LightningModule) -> list[GraphState]:
        """Drive the sampler for ``num_samples`` graphs in batches."""
        sampler = _require_attr(pl_module, "sampler")
        noise_process = _require_attr(pl_module, "noise_process")
        model = _require_attr(pl_module, "model")
        device = next(model.parameters()).device

        # Re-use the module's per-graph node-count distribution when
        # available; otherwise fall back to the static num_nodes attr
        # (single-graph datasets).
        size_dist = getattr(pl_module, "_size_distribution", None)
        num_nodes_attr = getattr(pl_module, "num_nodes", None)
        if num_nodes_attr is None:
            raise TypeError(
                "FinalSampleDumpCallback requires `pl_module.num_nodes` "
                "to bootstrap a node-count for sampling."
            )

        graphs: list[GraphState] = []
        remaining = self.num_samples
        while remaining > 0:
            batch_size = min(remaining, self.sample_batch_size)
            if size_dist is not None and not size_dist.is_degenerate:
                num_nodes_arg: int | torch.Tensor = size_dist.sample(batch_size)
            else:
                num_nodes_arg = int(num_nodes_attr)
            graphs.extend(
                sampler.sample(
                    model=model,
                    noise_process=noise_process,
                    num_graphs=batch_size,
                    num_nodes=num_nodes_arg,
                    device=device,
                )
            )
            remaining -= batch_size
        return graphs

    @override
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # DDP rank-0 gate (spec resolution Q9). On a single-rank trainer
        # ``is_global_zero`` is True; on DDP rank-N this returns False
        # and the callback is a no-op.
        if not trainer.is_global_zero:
            return

        ema_cb = self._find_ema_callback(trainer)
        ema_active = False
        backbone_params_fn = lambda: _require_attr(pl_module, "model").parameters()  # noqa: E731

        if ema_cb is not None and ema_cb.ema is not None:
            ema_cb.ema.store(backbone_params_fn())
            ema_cb.ema.copy_to(backbone_params_fn())
            ema_active = True

        try:
            t0 = time.perf_counter()
            samples = self._sample_graphs(pl_module)
            wall_seconds = time.perf_counter() - t0

            # Resolve dump path and write.
            target = _resolve_dump_path(
                configured=self.save_path,
                run_name=self.run_name,
                default_root_dir=trainer.default_root_dir,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            global_step = int(getattr(pl_module, "global_step", -1))
            payload = {
                "graphs": samples,
                "meta": {
                    "num_samples": len(samples),
                    "global_step": global_step,
                    "wall_seconds": float(wall_seconds),
                    "ema_active": ema_active,
                    "run_name": self.run_name,
                },
            }
            torch.save(payload, target)

            # Evaluate against the test reference set (spec resolution Q6).
            # ``self.log()`` is forbidden inside ``on_fit_end`` -- write
            # directly to the logger experiment when available so the
            # final/<metric> entries still land in W&B/CSV/etc.
            evaluator = _require_attr(pl_module, "evaluator")
            # ``trainer.datamodule`` is present at runtime via Lightning's
            # attached datamodule but absent from its public type stubs;
            # use getattr to keep pyright happy without a suppression.
            datamodule = getattr(trainer, "datamodule", None)
            if datamodule is not None:
                # Both `refs` and `samples` are list[GraphState] post the
                # 2026-05-07 sparse-default refactor; evaluator.evaluate
                # internally converts to nx via to_networkx_graphs as needed.
                refs = datamodule.get_reference_graphs(
                    "test", evaluator.eval_num_samples
                )
                results = evaluator.evaluate(refs=refs, generated=samples)
                if results is not None:
                    metrics = {
                        f"final/test/{k}": float(v)
                        for k, v in results.to_dict().items()
                        if v is not None
                    }
                    self._log_to_experiment(pl_module, metrics, step=global_step)
        finally:
            if ema_active and ema_cb is not None and ema_cb.ema is not None:
                ema_cb.ema.restore(backbone_params_fn())

"""Shared LightningModule base class for all graph learning experiments.

``BaseGraphModule`` extracts pure infrastructure — optimizer/scheduler
setup, batch device transfer, parameter logging — into a single base
class. It carries **no** training logic; subclasses like
``DiffusionModule`` and ``SingleStepDenoisingModule`` implement
``training_step`` and ``forward``.

Models are passed as already-instantiated ``GraphModel`` objects,
typically constructed by Hydra's recursive ``_target_`` instantiation
from YAML config.
"""

# pyright: reportExplicitAny=false
# pyright: reportUnknownMemberType=false
# pyright: reportIncompatibleMethodOverride=false
# PyTorch Lightning's self.log(), self.trainer, self.model, and self.hparams
# have incomplete type stubs. Config dicts legitimately use Any.
# reportIncompatibleMethodOverride: LightningModule base signatures use Any,
# and narrowing batch type to GraphData is an LSP violation pyright cannot waive.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, override

import torch

from tmgg.data.datasets.graph_types import GraphState
from tmgg.models.base import GraphModel  # pyright: ignore[reportAttributeAccessIssue]
from tmgg.training.lightning_modules.optimizer_config import (
    OptimizerLRSchedulerConfig,
    SchedulerInfo,
    configure_optimizers_from_config,
)
from tmgg.training.lightning_modules.write_through_logging import (
    WriteThroughLogMixin,
)
from tmgg.training.logging import (
    log_parameter_count as _log_parameter_count,
)


class BaseGraphModule(WriteThroughLogMixin, ABC):
    """Shared infrastructure for all graph learning experiments.

    Handles optimizer and scheduler construction, ``GraphState`` batch
    transfer, and parameter-count logging. Subclasses only need to
    supply ``training_step`` and ``forward``. ``WriteThroughLogMixin``
    provides ``self.log(..., write_through=True)`` for cadence-gated
    diagnostics that must bypass Lightning's sampled/wipeable result
    collection; subclasses overriding ``on_train_batch_end`` or
    ``on_fit_end`` must call ``super()`` to keep its flush running.

    The ``model`` parameter receives an already-instantiated
    ``GraphModel``, constructed by Hydra's recursive ``_target_``
    instantiation from YAML config.

    Parameters
    ----------
    model : GraphModel
        Pre-constructed graph model (instantiated by Hydra from nested
        ``_target_`` in the YAML config).
    model_name : str
        Human-readable model identifier for logging and experiment
        naming. Typically matches the YAML config filename or
        architecture family.
    learning_rate : float
        Base learning rate for the optimizer.
    weight_decay : float
        Weight decay coefficient (only effective with AdamW).
    optimizer_type : str
        One of ``"adam"`` or ``"adamw"``.
    amsgrad : bool
        Whether to enable the AMSGrad variant.
    scheduler_config : dict[str, Any] | None
        Optional scheduler configuration. Supported ``"type"`` values:
        ``"cosine"``, ``"cosine_warmup"``, ``"step"``, ``"none"``.
    """

    def __init__(
        self,
        *,
        model: GraphModel,
        model_name: str = "",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        scheduler_config: dict[str, Any] | None = None,
        compile_model: bool = False,
        compile_mode: str = "default",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Store optimizer params
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.optimizer_type: str = optimizer_type.lower()
        self.amsgrad: bool = amsgrad
        self.scheduler_config: dict[str, Any] | None = scheduler_config

        # Assign the pre-constructed model
        if not isinstance(model, GraphModel):
            raise TypeError(f"Expected GraphModel subclass, got {type(model).__name__}")
        self.model: GraphModel = model

        # Optional ``torch.compile`` wrap. Independent of the fused-AdamW
        # optimiser path: compile fuses ops inside the model forward/backward
        # graph (LayerNorm, matmul epilogues, elementwise chains), while
        # ``fused=True`` AdamW is a single opaque CUDA kernel outside the
        # model graph. Both wins compose. ``mode="reduce-overhead"`` and
        # ``"max-autotune"`` are also valid; ``"default"`` is the safest
        # starting point.
        if compile_model:
            self.model = torch.compile(self.model, mode=compile_mode)  # pyright: ignore[reportAttributeAccessIssue]

        # Preserve model metadata in hparams for W&B logging / checkpoint inspection
        self.hparams["model_config"] = model.get_config()
        self.hparams["model_class"] = (
            f"{type(model).__module__}.{type(model).__qualname__}"
        )

        # Populated by configure_optimizers when cosine_warmup scheduler is used
        self._scheduler_info: SchedulerInfo | None = None

    @override
    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer | OptimizerLRSchedulerConfig:
        """Delegate to :func:`configure_optimizers_from_config`."""
        result, scheduler_info = configure_optimizers_from_config(
            self,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            optimizer_type=self.optimizer_type,
            amsgrad=self.amsgrad,
            scheduler_config=self.scheduler_config,
        )
        self._scheduler_info = scheduler_info
        return result

    @override
    def transfer_batch_to_device(
        self,
        batch: GraphState,
        device: torch.device,
        dataloader_idx: int,
    ) -> GraphState:
        """Move a ``GraphState`` batch to the target device."""
        return batch.to(device)

    @override
    def on_fit_start(self) -> None:
        """Log parameter count at training start."""
        _log_parameter_count(self.model, self.get_model_name(), self.logger)  # pyright: ignore[reportUnknownArgumentType]

    @override
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Log gradient and parameter L2 norms before gradient clipping.

        Fires after backward and before ``optimizer.step()`` each training
        step, so the reported ``grad_norm_l2`` is the **pre-clip** value
        (the post-clip norm is bounded by ``trainer.gradient_clip_val``
        and carries no diagnostic signal). The cost is negligible — one
        pass over parameters, the same pass Lightning already performs
        for gradient clipping when ``gradient_clip_val`` is set.

        Diagnostics:

        - ``train/diagnostics/grad_norm_l2`` — total pre-clip L2 norm of
          all parameter gradients. Stuck near zero signals
          vanishing/dead gradients; stuck at or above
          ``gradient_clip_val`` signals every step is being clipped and
          the effective LR is lower than ``learning_rate``.
        - ``train/diagnostics/grad_norm_preclip_max`` — largest
          single-parameter gradient L2. Catches one-tensor explosions
          that a global norm can mask.
        - ``train/diagnostics/param_norm_l2`` — total L2 norm of all
          parameters. Slow-moving; confirms the optimizer is actually
          updating weights between steps.
        """
        # Batch the per-tensor norms via ``torch._foreach_norm`` so all
        # parameter / gradient norms run in a single multi-tensor CUDA
        # launch group instead of N small per-tensor launches. The
        # Python-loop version was the same launch-overhead pattern that
        # hit un-fused AdamW: ~250–300 small launches per step on
        # ~50 parameter tensors, ~19 ms/step CUDA driven by host
        # dispatch latency rather than arithmetic.
        grads = [p.grad.detach() for p in self.parameters() if p.grad is not None]
        params = [p.detach() for p in self.parameters()]
        if not grads:
            return
        grad_norms = torch.stack(torch._foreach_norm(grads, 2.0))  # noqa: SLF001
        param_norms = torch.stack(torch._foreach_norm(params, 2.0))  # noqa: SLF001
        self.log_dict(
            {
                "diagnostics-train/opt-health/grad_norm_l2": grad_norms.pow(2)
                .sum()
                .sqrt(),
                "diagnostics-train/opt-health/grad_norm_preclip_max": grad_norms.max(),
                "diagnostics-train/opt-health/param_norm_l2": param_norms.pow(2)
                .sum()
                .sqrt(),
            },
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

    def get_model_name(self) -> str:
        """Return ``model_name`` from hyperparameters.

        Subclasses may override for a more descriptive name.
        """
        name: str = str(self.hparams.get("model_name", "unknown"))
        return name or "unknown"

    def get_model_config(self) -> dict[str, Any]:
        """Delegate to ``model.get_config()``.

        Returns
        -------
        dict[str, Any]
            Configuration dict as reported by the underlying model.
        """
        return self.model.get_config()  # pyright: ignore[reportUnknownVariableType]

    @abstractmethod
    @override
    def training_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> Any:
        """Execute a single training step. Subclasses must implement."""
        ...

    @abstractmethod
    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass. Subclasses must implement."""
        ...

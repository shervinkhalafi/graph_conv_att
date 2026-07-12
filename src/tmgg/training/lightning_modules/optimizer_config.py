"""Standalone optimizer and scheduler configuration functions.

Extracted from ``OptimizerMixin`` so that optimizer/scheduler construction can
be used without inheriting the mixin. The mixin remains a thin wrapper around
these functions for backward compatibility with existing Lightning modules.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig


@dataclass(frozen=True)
class SchedulerInfo:
    """Immutable record of computed scheduler parameters.

    Returned alongside the optimizer dict when a cosine-warmup scheduler is
    configured, so callers can log or inspect the values without relying on
    mutable instance attributes.

    Parameters
    ----------
    T_warmup : int
        Number of linear warmup steps.
    T_max : int
        Step at which cosine decay reaches its minimum.
    estimated_total_steps : int
        Total training steps estimated from trainer context.
    """

    T_warmup: int
    T_max: int
    estimated_total_steps: int


def configure_optimizers_from_config(
    module: pl.LightningModule,
    *,
    learning_rate: float,
    weight_decay: float,
    optimizer_type: str,
    amsgrad: bool,
    scheduler_config: dict[str, Any] | None,
) -> tuple[torch.optim.Optimizer | OptimizerLRSchedulerConfig, SchedulerInfo | None]:
    """Build an optimizer and optional LR scheduler from explicit parameters.

    Contains the same logic as ``OptimizerMixin.configure_optimizers`` but
    takes all configuration as explicit arguments rather than reading from
    ``self``, making it usable outside the mixin pattern.

    Parameters
    ----------
    module
        A ``pl.LightningModule`` whose ``parameters()`` and ``trainer``
        attribute provide the model weights and training context.
    learning_rate
        Base learning rate.
    weight_decay
        Weight decay coefficient (used only with AdamW).
    optimizer_type
        One of ``"adam"`` or ``"adamw"``.
    amsgrad
        Whether to enable the AMSGrad variant.
    scheduler_config
        Scheduler configuration dict, or ``None`` to disable scheduling.
        Supported ``"type"`` values: ``"cosine"``, ``"cosine_warmup"``,
        ``"step"``, ``"none"``.

    Returns
    -------
    tuple[torch.optim.Optimizer | OptimizerLRSchedulerConfig, SchedulerInfo | None]
        A two-element tuple. The first element is either a plain optimizer
        (when no scheduler is configured or the scheduler type is
        unrecognised) or a Lightning-format ``OptimizerLRSchedulerConfig``
        dict with ``"optimizer"`` and ``"lr_scheduler"`` keys. The second
        element is a ``SchedulerInfo`` when the ``"cosine_warmup"`` scheduler
        is used, ``None`` otherwise.
    """
    # Select optimizer based on type.
    #
    # ``fused=True`` collapses every per-parameter-tensor AdamW/Adam math op
    # into a single multi-tensor CUDA kernel. On the round-5 Greedy profile
    # the unfused path spent ~135 ms host-side per step launching ~250+
    # tiny kernels for a 28 K-param model; the fused path issues 1 launch.
    # CPU fall-through keeps tests / host-only paths working since the
    # fused kernel is CUDA-only.
    use_fused = torch.cuda.is_available()
    optimizer: torch.optim.Optimizer
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            fused=use_fused,
        )
    else:  # default to adam
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=learning_rate,
            amsgrad=amsgrad,
            fused=use_fused,
        )

    if scheduler_config is None:
        return optimizer, None

    scheduler_type: str = scheduler_config.get("type", "cosine")

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get("T_0", 20),
            T_mult=scheduler_config.get("T_mult", 2),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }, None
    elif scheduler_type == "cosine_warmup":
        result, info = _configure_cosine_warmup(module, optimizer, scheduler_config)
        return result, info
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 50),
            gamma=scheduler_config.get("gamma", 0.1),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }, None
    elif scheduler_type == "none":
        return optimizer, None
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type!r}. "
            f"Supported types: 'cosine', 'cosine_warmup', 'step', 'none'."
        )


def _configure_cosine_warmup(
    module: pl.LightningModule,
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict[str, Any],
) -> tuple[OptimizerLRSchedulerConfig, SchedulerInfo]:
    """Build a cosine-with-linear-warmup LR scheduler.

    Supports two configuration modes:

    1. **Fraction-based** (recommended): ``warmup_fraction``,
       ``decay_fraction`` -- automatically scales with ``max_steps``.
    2. **Legacy step-based**: ``T_warmup``, ``T_max`` -- requires
       manual recalculation when training length changes.

    After the decay point, progress is clamped at 1.0 so the LR stays at
    its minimum instead of oscillating.

    Requires ``trainer.max_steps`` to be set to a positive integer. This
    project uses step-based training exclusively; epoch-based scheduling
    is not supported.

    Parameters
    ----------
    module
        A ``pl.LightningModule`` whose ``trainer`` attribute provides
        ``max_steps``.
    optimizer
        Already-constructed optimizer instance.
    scheduler_config
        Scheduler configuration dict. Expected keys depend on the mode:
        fraction-based requires ``warmup_fraction`` and optionally
        ``decay_fraction``; legacy requires ``T_warmup`` and ``T_max``.

    Returns
    -------
    tuple[dict[str, Any], SchedulerInfo]
        A Lightning-format dict with optimizer and step-level scheduler,
        paired with a ``SchedulerInfo`` holding the computed parameters.

    Raises
    ------
    RuntimeError
        If total training steps cannot be estimated from trainer context.
    ValueError
        If ``T_max <= T_warmup``.
    """
    # Require step-based training: trainer.max_steps must be set.
    estimated_total_steps: int | None = None
    trainer: pl.Trainer | None = getattr(module, "trainer", None)
    if trainer is not None and trainer.max_steps and trainer.max_steps > 0:
        estimated_total_steps = trainer.max_steps

    if estimated_total_steps is None:
        raise RuntimeError(
            "Cannot configure cosine-warmup scheduler: trainer.max_steps must be "
            "set to a positive integer. This project requires step-based training; "
            "epoch-based fallback is not supported."
        )

    # Compute T_warmup and T_max from fractions or legacy values
    T_warmup: int
    T_max: int
    if "warmup_fraction" in scheduler_config:
        warmup_fraction = scheduler_config.get("warmup_fraction", 0.02)
        decay_fraction = scheduler_config.get("decay_fraction", 0.8)
        T_warmup = int(warmup_fraction * estimated_total_steps)
        T_max = int(decay_fraction * estimated_total_steps)
    else:
        T_warmup = scheduler_config.get("T_warmup", 100)
        T_max = scheduler_config.get("T_max", 1000)

    # Validation
    if T_max <= T_warmup:
        raise ValueError(
            f"Scheduler T_max ({T_max}) must be > T_warmup ({T_warmup}). "
            f"Check scheduler_config: decay_fraction must be > warmup_fraction."
        )

    if T_max < estimated_total_steps * 0.5:
        warnings.warn(
            f"Scheduler T_max={T_max} is less than 50% of estimated training "
            f"({estimated_total_steps} steps). LR will reach minimum early and "
            f"stay there for the remaining training. Consider increasing "
            f"decay_fraction or T_max.",
            UserWarning,
            stacklevel=2,
        )

    info = SchedulerInfo(
        T_warmup=T_warmup,
        T_max=T_max,
        estimated_total_steps=estimated_total_steps,
    )

    def lr_lambda(step: int) -> float:
        if step < T_warmup:
            return step / max(1, T_warmup)
        else:
            progress = (step - T_warmup) / max(1, T_max - T_warmup)
            progress = min(1.0, progress)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }, info

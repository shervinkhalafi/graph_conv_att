"""All scheduler types must use interval='step' for step-based training.

Rationale
---------
The framework uses ``max_epochs=-1`` with step-based training. When a
scheduler declares ``interval='epoch'``, Lightning only calls
``scheduler.step()`` at epoch boundaries (every ``dataset_size / batch_size``
steps) rather than every training step. This changes LR dynamics
qualitatively and inconsistently relative to ``cosine_warmup``, which was
already configured with ``interval='step'``.

Starting state: configure_optimizers_from_config with various scheduler types.
Invariant: every scheduler dict returned must contain ``interval='step'``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from tmgg.training.lightning_modules.optimizer_config import (
    configure_optimizers_from_config,
)


class _FakeModule(nn.Module):
    """Minimal host satisfying configure_optimizers_from_config's needs."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)


def _configure_with_scheduler(
    scheduler_config: dict[str, Any] | None,
) -> Any:
    """Call configure_optimizers_from_config with the given scheduler config."""
    host = _FakeModule()
    result, _ = configure_optimizers_from_config(
        host,  # type: ignore[arg-type]  # test host, not real LightningModule
        learning_rate=0.001,
        weight_decay=0.0,
        optimizer_type="adam",
        amsgrad=False,
        scheduler_config=scheduler_config,
    )
    return result


def test_all_scheduler_types_use_step_interval() -> None:
    """configure_optimizers_from_config must return interval='step' for cosine and step schedulers."""
    for sched_type in ("cosine", "step"):
        result = _configure_with_scheduler({"type": sched_type})
        assert isinstance(
            result, dict
        ), f"{sched_type}: expected dict return, got {type(result)}"
        interval = result["lr_scheduler"]["interval"]
        assert (
            interval == "step"
        ), f"Scheduler type '{sched_type}' uses interval='{interval}', must be 'step'"


def test_no_scheduler_returns_plain_optimizer() -> None:
    """When scheduler_config is None, configure_optimizers_from_config returns a bare optimizer."""
    result = _configure_with_scheduler(None)
    assert isinstance(
        result, torch.optim.Optimizer
    ), f"Expected bare optimizer, got {type(result)}"

"""Tests for the standalone optimizer configuration function.

Tests the optimizer and LR scheduler configuration logic in
``configure_optimizers_from_config``, independently of any Lightning module.
A lightweight host class provides the ``parameters()`` method the function
needs.

Testing Strategy
----------------
- OptimizerConfigHost(nn.Module) with a single nn.Linear provides
  parameters() without dragging in the full Lightning stack.
- Mock trainer/datamodule objects supply the training context that
  _configure_cosine_warmup inspects.
- Each test class targets one scheduler path or one validation branch.

Key Invariants
--------------
- optimizer_type selects Adam vs AdamW; weight_decay only affects AdamW.
- amsgrad propagates to whichever optimizer is chosen.
- Unknown scheduler types fall back to returning a bare optimizer.
- cosine_warmup: T_max must exceed T_warmup; fraction-based config scales
  with trainer.max_steps; estimation failure emits a UserWarning.
- lr_lambda returns 0 at step 0, ramps linearly during warmup, follows
  cosine decay after warmup, and clamps progress at 1.0 beyond T_max.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from tmgg.training.lightning_modules.optimizer_config import (
    SchedulerInfo,
    configure_optimizers_from_config,
)


class OptimizerConfigHost(nn.Module):
    """Minimal host for testing configure_optimizers_from_config.

    Attributes the function reads are set from kwargs so each test can
    configure only what it needs.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.learning_rate: float = kwargs.get("learning_rate", 1e-3)
        self.weight_decay: float = kwargs.get("weight_decay", 0.0)
        self.optimizer_type: str = kwargs.get("optimizer_type", "adam")
        self.amsgrad: bool = kwargs.get("amsgrad", False)
        self.scheduler_config: dict[str, Any] | None = kwargs.get("scheduler_config")


def _configure(host: OptimizerConfigHost) -> tuple[Any, SchedulerInfo | None]:
    """Call configure_optimizers_from_config with the host's attributes."""
    return configure_optimizers_from_config(
        host,  # type: ignore[arg-type]  # test host, not real LightningModule
        learning_rate=host.learning_rate,
        weight_decay=host.weight_decay,
        optimizer_type=host.optimizer_type,
        amsgrad=host.amsgrad,
        scheduler_config=host.scheduler_config,
    )


# -----------------------------------------------------------------------
# TestConfigureOptimizers
# -----------------------------------------------------------------------


class TestConfigureOptimizers:
    """Verify optimizer selection and basic scheduler dispatch."""

    def test_adam_default(self) -> None:
        """Default optimizer_type='adam' returns an Adam instance.

        Starting state: host with default kwargs.
        Invariant: Adam is the default, no scheduler means bare optimizer.
        """
        host = OptimizerConfigHost()
        result, info = _configure(host)
        assert isinstance(result, torch.optim.Adam)
        assert info is None

    def test_adamw_with_weight_decay(self) -> None:
        """optimizer_type='adamw' returns AdamW; weight_decay propagated.

        Starting state: host with adamw + weight_decay=0.01.
        Invariant: AdamW receives the declared weight_decay.
        """
        host = OptimizerConfigHost(optimizer_type="adamw", weight_decay=0.01)
        result, _ = _configure(host)
        assert isinstance(result, torch.optim.AdamW)
        assert result.defaults["weight_decay"] == pytest.approx(0.01)

    def test_amsgrad_propagated(self) -> None:
        """amsgrad=True propagates to the optimizer for both Adam and AdamW.

        Starting state: host with amsgrad=True.
        Invariant: optimizer defaults include amsgrad=True.
        """
        for opt_type, opt_cls in [
            ("adam", torch.optim.Adam),
            ("adamw", torch.optim.AdamW),
        ]:
            host = OptimizerConfigHost(optimizer_type=opt_type, amsgrad=True)
            result, _ = _configure(host)
            assert isinstance(result, torch.optim.Optimizer)
            assert isinstance(result, opt_cls)
            assert result.defaults["amsgrad"] is True

    def test_unknown_scheduler_type_raises(self) -> None:
        """Unrecognized scheduler type raises ValueError (fail-fast).

        Starting state: scheduler_config with type='unknown'.
        Invariant: unknown types crash loudly instead of silently returning
        a bare optimizer, so misconfiguration is caught early.
        """
        host = OptimizerConfigHost(scheduler_config={"type": "unknown"})
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            _configure(host)


# -----------------------------------------------------------------------
# TestCosineScheduler
# -----------------------------------------------------------------------


class TestCosineScheduler:
    """Verify cosine (CosineAnnealingWarmRestarts) scheduler path."""

    def test_creates_cosine_annealing(self) -> None:
        """scheduler_config type='cosine' creates CosineAnnealingWarmRestarts.

        Starting state: scheduler_config with T_0=15, T_mult=3.
        Invariant: scheduler instance has declared T_0 and T_mult.
        """
        host = OptimizerConfigHost(
            scheduler_config={"type": "cosine", "T_0": 15, "T_mult": 3}
        )
        result, info = _configure(host)

        assert isinstance(result, dict)
        assert info is None
        sched = result["lr_scheduler"]["scheduler"]
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
        assert sched.T_0 == 15
        assert sched.T_mult == 3


# -----------------------------------------------------------------------
# TestStepScheduler
# -----------------------------------------------------------------------


class TestStepScheduler:
    """Verify StepLR scheduler path."""

    def test_creates_step_lr(self) -> None:
        """scheduler_config type='step' creates StepLR with given params.

        Starting state: scheduler_config with step_size=25, gamma=0.5.
        Invariant: scheduler instance has declared step_size and gamma.
        """
        host = OptimizerConfigHost(
            scheduler_config={"type": "step", "step_size": 25, "gamma": 0.5}
        )
        result, info = _configure(host)

        assert isinstance(result, dict)
        assert info is None
        sched = result["lr_scheduler"]["scheduler"]
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)
        assert sched.step_size == 25
        assert sched.gamma == pytest.approx(0.5)


# -----------------------------------------------------------------------
# TestCosineWarmup
# -----------------------------------------------------------------------


class TestCosineWarmup:
    """Verify cosine_warmup scheduler path, including fraction-based config,
    legacy config, validation, and the A1 regression (warning on estimation
    failure).
    """

    @staticmethod
    def _host_with_trainer(
        scheduler_config: dict[str, Any],
        max_steps: int = 0,
        max_epochs: int | None = None,
        datamodule: Any = None,
    ) -> OptimizerConfigHost:
        """Build a host with a mock trainer attached."""
        host = OptimizerConfigHost(scheduler_config=scheduler_config)
        trainer = MagicMock()
        trainer.max_steps = max_steps
        trainer.max_epochs = max_epochs
        trainer.datamodule = datamodule
        host.trainer = trainer  # type: ignore[attr-defined]
        return host

    def test_fraction_based_with_max_steps(self) -> None:
        """Fraction-based config scales T_warmup/T_max from trainer.max_steps.

        Starting state: warmup_fraction=0.1, decay_fraction=0.9, max_steps=1000.
        Invariant: T_warmup=100, T_max=900.
        """
        host = self._host_with_trainer(
            scheduler_config={
                "type": "cosine_warmup",
                "warmup_fraction": 0.1,
                "decay_fraction": 0.9,
            },
            max_steps=1000,
        )
        result, info = _configure(host)

        assert isinstance(result, dict)
        assert info is not None
        assert info.T_warmup == 100
        assert info.T_max == 900

    def test_legacy_t_warmup_t_max(self) -> None:
        """Legacy step-based config uses T_warmup/T_max directly.

        Starting state: T_warmup=50, T_max=500, no fraction keys, max_steps=10000.
        Invariant: stored values match the explicit config.
        """
        host = self._host_with_trainer(
            scheduler_config={
                "type": "cosine_warmup",
                "T_warmup": 50,
                "T_max": 500,
            },
            max_steps=10000,
        )
        result, info = _configure(host)

        assert isinstance(result, dict)
        assert info is not None
        assert info.T_warmup == 50
        assert info.T_max == 500

    def test_t_max_le_t_warmup_raises(self) -> None:
        """T_max <= T_warmup raises ValueError.

        Starting state: T_warmup=500, T_max=200, max_steps=10000.
        Invariant: fail fast with a descriptive message.
        """
        host = self._host_with_trainer(
            scheduler_config={
                "type": "cosine_warmup",
                "T_warmup": 500,
                "T_max": 200,
            },
            max_steps=10000,
        )
        with pytest.raises(ValueError, match="T_max.*must be > T_warmup"):
            _configure(host)

    def test_missing_max_steps_raises(self) -> None:
        """Cosine-warmup requires trainer.max_steps to be a positive integer.

        Starting state: fraction-based config, trainer with max_steps=0 (unset).
        Invariant: RuntimeError with a clear message about step-based training.

        The epoch-based fallback was removed (M-29) — this project requires
        step-based training exclusively.
        """
        host = self._host_with_trainer(
            scheduler_config={
                "type": "cosine_warmup",
                "warmup_fraction": 0.1,
                "decay_fraction": 0.9,
            },
            max_steps=0,
        )

        with pytest.raises(RuntimeError, match="trainer.max_steps must be set"):
            _configure(host)

    def test_no_trainer_raises(self) -> None:
        """Cosine-warmup without a trainer attached raises RuntimeError.

        Starting state: fraction-based config, no trainer on module.
        Invariant: same step-based training error fires.
        """
        host = OptimizerConfigHost(
            scheduler_config={
                "type": "cosine_warmup",
                "warmup_fraction": 0.1,
                "decay_fraction": 0.9,
            },
        )

        with pytest.raises(RuntimeError, match="trainer.max_steps must be set"):
            _configure(host)

    def test_lr_lambda_boundaries(self) -> None:
        """lr_lambda returns correct values at warmup start, end, and beyond T_max.

        Starting state: T_warmup=100, T_max=1000, legacy config, max_steps=10000.
        Invariants:
        - step 0: lr_lambda = 0 (start of warmup)
        - step T_warmup: lr_lambda = 1.0 (end of warmup = full LR)
        - step T_max: lr_lambda = ~0 (cosine has decayed fully, progress=1.0)
        - step > T_max: progress is clamped, lr_lambda doesn't oscillate
        """
        host = self._host_with_trainer(
            scheduler_config={
                "type": "cosine_warmup",
                "T_warmup": 100,
                "T_max": 1000,
            },
            max_steps=10000,
        )
        result, info = _configure(host)
        assert isinstance(result, dict)
        assert info is not None
        assert info.T_warmup == 100
        assert info.T_max == 1000
        scheduler = result["lr_scheduler"]["scheduler"]
        assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

        # lr_lambdas is a valid runtime attribute of LambdaLR but absent
        # from the type stubs, so we use getattr to avoid a pyright error.
        lr_func = scheduler.lr_lambdas[0]

        # step 0: start of warmup -> 0/max(1,100) = 0.0
        assert lr_func(0) == pytest.approx(0.0)

        # step 100 (T_warmup): warmup just ended -> 100/100 = 1.0
        assert lr_func(100) == pytest.approx(1.0)

        # step 1000 (T_max): full cosine decay -> progress=1.0
        # 0.5*(1 + cos(pi*1.0)) = 0.5*(1 + (-1)) = 0.0
        val_at_t_max = lr_func(1000)
        assert val_at_t_max == pytest.approx(0.0, abs=1e-7)

        # step 2000 (beyond T_max): clamped at progress=1.0, same as T_max
        val_beyond = lr_func(2000)
        assert val_beyond == pytest.approx(val_at_t_max, abs=1e-7)

    def test_lr_lambda_midpoint(self) -> None:
        """lr_lambda at the midpoint of the decay phase equals 0.5.

        Starting state: T_warmup=0, T_max=1000 (no warmup), max_steps=10000.
        Invariant: at step 500, progress=0.5, cos(pi*0.5)=0,
        so lr_lambda = 0.5*(1+0) = 0.5.
        """
        host = self._host_with_trainer(
            scheduler_config={
                "type": "cosine_warmup",
                "T_warmup": 0,
                "T_max": 1000,
            },
            max_steps=10000,
        )
        result, _ = _configure(host)
        assert isinstance(result, dict)
        lr_func = result["lr_scheduler"]["scheduler"].lr_lambdas[0]

        # step 0: T_warmup=0, so we enter the else branch immediately.
        # progress = (0 - 0) / max(1, 1000 - 0) = 0.0
        # lr = 0.5*(1 + cos(0)) = 0.5*2 = 1.0
        assert lr_func(0) == pytest.approx(1.0)

        # step 500: progress = 500/1000 = 0.5
        # lr = 0.5*(1 + cos(pi*0.5)) = 0.5*(1 + 0) = 0.5
        assert lr_func(500) == pytest.approx(0.5)

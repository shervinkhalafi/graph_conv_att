"""Write-through diagnostic logging that bypasses Lightning's result collection.

Why this exists
---------------
``LightningModule.self.log(on_step=True)`` is *sampled*, not buffered:
values reach the logger only on iterations where ``global_step`` crosses
a ``log_every_n_steps`` boundary, and the pending result cache is wiped
by ``reset_results()`` at every epoch boundary. In our regime — short
epochs (~11 batches on SBM) and cadence-gated diagnostics (every
500/5000 steps) — a diagnostic logged off-boundary is silently lost.
Commit ce5cc52c worked around this by stashing values in
``on_before_optimizer_step`` and draining them via ``self.log`` from
``on_train_batch_end`` on a flush-aligned iteration, which silently
couples the diagnostic cadence to ``log_every_n_steps``.

This module removes the coupling. ``self.log(name, value,
write_through=True)`` routes the value to a :class:`WriteThroughCollector`
that calls the public ``Logger.log_metrics(metrics, step=...)`` API
directly — no result collection, no flush gate, no epoch wipe, exact
call-time step stamping. This is the docs-sanctioned "manual logging"
path; both ``WandbLogger`` (which records ``step`` as a
``trainer/global_step`` key, avoiding wandb step-monotonicity issues)
and ``CSVLogger`` implement it.

The trade-off, by construction: routed values never enter
``trainer.callback_metrics``, so they can never serve as checkpoint
monitors, early-stopping signals, or progress-bar entries. Use
write-through only for human-eyeball dashboard diagnostics. To keep
call sites honest, ``write_through=True`` combined with *any* other
logging-control argument raises ``ValueError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import pytorch_lightning as pl
import torch

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pytorch_lightning.loggers import Logger


class WriteThroughCollector:
    """Per-step buffer of diagnostic scalars, flushed straight to loggers.

    ``collect`` groups values by the step they were stamped with so one
    physical step produces one ``log_metrics`` row per logger, however
    many diagnostics it logged. ``flush`` drains the buffer
    unconditionally; only the global-zero rank writes, because routed
    values are never ``sync_dist``-reduced and per-rank views are not
    meaningful on the dashboard.
    """

    def __init__(self) -> None:
        self._pending: dict[int, dict[str, float]] = {}

    def collect(self, name: str, value: torch.Tensor | float, step: int) -> None:
        """Buffer ``value`` under ``step``.

        ``float()`` synchronises CUDA tensors at collect time; at the
        500/5000-step diagnostic cadences this is negligible.
        """
        self._pending.setdefault(step, {})[name] = float(value)

    def flush(self, loggers: Iterable[Logger], *, is_global_zero: bool) -> None:
        """Write all buffered steps (ascending) and clear the buffer.

        The buffer drains on every rank — non-zero ranks drop their
        values rather than accumulate them without bound.
        """
        if not self._pending:
            return
        pending, self._pending = self._pending, {}
        if not is_global_zero:
            return
        for step in sorted(pending):
            for logger in loggers:
                logger.log_metrics(pending[step], step=step)


class WriteThroughLogMixin(pl.LightningModule):
    """Adds ``write_through=True`` to ``self.log`` for cadence-gated diagnostics.

    Compose ahead of ``pl.LightningModule``::

        class MyModule(WriteThroughLogMixin, pl.LightningModule): ...

    Subclasses that override ``on_train_batch_end`` or ``on_fit_end``
    must call ``super()`` so the collector still flushes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._write_through_collector = WriteThroughCollector()

    @override
    def log(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        name: str,
        value: Any,
        *args: Any,
        write_through: bool = False,
        **kwargs: Any,
    ) -> None:
        """``LightningModule.log`` plus the ``write_through`` escape hatch.

        With ``write_through=False`` (default) this is exactly
        ``LightningModule.log``. With ``write_through=True`` the value
        bypasses Lightning's result collection entirely (see module
        docstring) and **no other logging argument is accepted**:
        aggregation, sync, and display semantics do not exist on the
        write-through path, so requesting them is a call-site bug.

        Raises
        ------
        ValueError
            If ``write_through=True`` is combined with any positional
            or keyword logging-control argument.
        """
        if not write_through:
            super().log(name, value, *args, **kwargs)
            return
        if args or kwargs:
            extras = [f"positional {a!r}" for a in args]
            extras += [f"{k}={v!r}" for k, v in kwargs.items()]
            raise ValueError(
                f"write_through=True logs bypass Lightning's result collection "
                f"and accept no other logging arguments, got: {', '.join(extras)}. "
                f"Drop them, or use a plain self.log() call for Lightning "
                f"aggregation/monitor semantics."
            )
        self._write_through_collector.collect(name, value, step=self.global_step)

    @override
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Flush routed diagnostics at the end of every training batch."""
        super().on_train_batch_end(outputs, batch, batch_idx)
        self._flush_write_through()

    @override
    def on_fit_end(self) -> None:
        """Backstop flush so a final-step diagnostic is never stranded."""
        super().on_fit_end()
        self._flush_write_through()

    def _flush_write_through(self) -> None:
        trainer = self.trainer
        self._write_through_collector.flush(
            trainer.loggers, is_global_zero=trainer.is_global_zero
        )

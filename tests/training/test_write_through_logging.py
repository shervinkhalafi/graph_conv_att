"""Write-through diagnostic logging: bypass Lightning's result collection.

Rationale
---------
``self.log(on_step=True)`` values are *sampled*, not buffered: Lightning
writes them to the logger only on iterations where ``global_step``
crosses a ``log_every_n_steps`` boundary, and ``reset_results()`` wipes
the pending cache at every epoch boundary. With short epochs (~11
batches on SBM) and cadence-gated diagnostics (every 500/5000 steps),
a diagnostic logged off-boundary is silently lost — the original bug
fixed by the ce5cc52c stash/drain workaround, which in turn silently
couples the diagnostic cadence to ``log_every_n_steps``.

``WriteThroughLogMixin`` removes the coupling: ``self.log(name, value,
write_through=True)`` routes to a ``WriteThroughCollector`` that calls
the public ``Logger.log_metrics(metrics, step=...)`` API directly —
no result collection, no flush gate, no epoch wipe. The contract these
tests pin:

* routed values reach every trainer logger exactly once, stamped with
  ``self.global_step`` at call time, regardless of ``log_every_n_steps``
  alignment or epoch length (the de-aligned cadence test is the
  regression test for the original bug class);
* routed keys never appear in ``trainer.callback_metrics`` — they are
  dashboard-only by construction and must not be used for checkpoint
  monitors or early stopping;
* ``write_through=True`` combined with *any* other logging-control
  argument raises ``ValueError`` — Lightning aggregation semantics
  (``on_epoch``, ``sync_dist``, ``prog_bar``, ``reduce_fx``,
  ``batch_size``, …) are not provided by the write-through path and
  asking for them is a bug at the call site;
* the collector batches all values for one step into a single
  ``log_metrics`` call per logger, only writes on the global-zero rank,
  and always clears its buffer.
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import Logger as PLLogger

from tmgg.training.lightning_modules.write_through_logging import (
    WriteThroughCollector,
    WriteThroughLogMixin,
)


class _Capture(PLLogger):
    """Minimal Lightning logger that records every ``log_metrics`` call."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[int | None, dict[str, float]]] = []

    @property
    def name(self) -> str:
        return "capture"

    @property
    def version(self) -> str:
        return "1"

    def log_metrics(
        self,
        metrics: dict[str, float] | None = None,
        step: int | None = None,
        **kw: object,
    ) -> None:
        assert metrics is not None
        self.records.append((step, dict(metrics)))

    def log_hyperparams(self, *a: object, **k: object) -> None:
        pass

    def save(self) -> None:
        pass


class _Model(WriteThroughLogMixin, pl.LightningModule):
    """Minimal module mirroring the production cadence-gated diagnostics.

    Logs one routed diagnostic from ``on_before_optimizer_step`` every
    ``cadence`` steps and one normal metric every step — the same shape
    as ``DiffusionModule``'s opt-health telemetry, minus the model.
    """

    def __init__(self, cadence: int = 5) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        self._cadence = cadence

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        loss = ((self.linear(x) - y) ** 2).mean()
        self.log("train/loss", loss, on_step=True)
        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if (self.global_step + 1) % self._cadence == 0:
            self.log(
                "diagnostics-train/opt-health/grad_snr/_total",
                torch.tensor(2.0 + self.global_step),
                write_through=True,
            )
            self.log(
                "diagnostics-train/opt-health/effective_lr",
                0.001 + self.global_step / 1000,
                write_through=True,
            )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def _fit(model: _Model, *, log_every_n_steps: int, max_steps: int = 33) -> _Capture:
    """11-batch epochs (33 samples / batch 3) — the production SBM regime
    where the original wipe bug manifested. CPU-only: these tests pin
    logging plumbing, not numerics."""
    torch.manual_seed(0)
    ds = torch.utils.data.TensorDataset(torch.randn(33, 4), torch.randn(33, 1))
    dl = torch.utils.data.DataLoader(ds, batch_size=3)
    cap = _Capture()
    trainer = pl.Trainer(
        max_steps=max_steps,
        log_every_n_steps=log_every_n_steps,
        logger=cap,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=dl)
    return cap


def test_write_through_survives_dealigned_cadence_and_short_epochs() -> None:
    """Regression for the wipe-bug class: cadence=5, log_every_n_steps=3.

    5 is not a multiple of 3, so under plain ``self.log`` (and under the
    stash/drain workaround, which relies on the cadence iter coinciding
    with a flush boundary) these values would be wiped by the epoch
    reset before any flush. Write-through must deliver them anyway.
    """
    cap = _fit(_Model(cadence=5), log_every_n_steps=3)
    flushed_keys = {k for _, m in cap.records for k in m}
    assert "diagnostics-train/opt-health/grad_snr/_total" in flushed_keys
    assert "diagnostics-train/opt-health/effective_lr" in flushed_keys


def test_write_through_stamps_call_time_global_step() -> None:
    """Routed values carry ``self.global_step`` as read at the log call.

    The cadence trigger fires in ``on_before_optimizer_step`` (pre-
    increment), at ``(global_step + 1) % cadence == 0`` — i.e. at
    global_step ∈ {4, 9, 14, ...}. Each cadence hit must appear exactly
    once with that stamp: write-through is exactly-once delivery, not
    sampled."""
    cap = _fit(_Model(cadence=5), log_every_n_steps=3)
    snr_records = [
        (step, m["diagnostics-train/opt-health/grad_snr/_total"])
        for step, m in cap.records
        if "diagnostics-train/opt-health/grad_snr/_total" in m
    ]
    expected_steps = [s for s in range(33) if (s + 1) % 5 == 0]
    assert [s for s, _ in snr_records] == expected_steps
    # Values encode the step they were computed at — confirms no
    # stale-value re-delivery.
    for (_, value), expected_step in zip(snr_records, expected_steps, strict=True):
        assert value == pytest.approx(2.0 + expected_step)


def test_write_through_batches_one_log_metrics_call_per_step() -> None:
    """Both diagnostics logged at the same step arrive in ONE
    ``log_metrics`` call (one row per step, not one row per key)."""
    cap = _fit(_Model(cadence=5), log_every_n_steps=3)
    routed = [
        m
        for _, m in cap.records
        if "diagnostics-train/opt-health/grad_snr/_total" in m
        or "diagnostics-train/opt-health/effective_lr" in m
    ]
    for m in routed:
        assert "diagnostics-train/opt-health/grad_snr/_total" in m
        assert "diagnostics-train/opt-health/effective_lr" in m


def test_write_through_keys_absent_from_callback_metrics() -> None:
    """Routed keys are dashboard-only: never in ``callback_metrics``
    (so they can never silently become checkpoint monitors), while the
    normally-logged key still lands there."""
    model = _Model(cadence=5)
    _fit(model, log_every_n_steps=3)
    cb_keys = set(model.trainer.callback_metrics)
    assert "train/loss" in cb_keys
    assert not any(k.startswith("diagnostics-train/opt-health/") for k in cb_keys)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"on_epoch": True},
        {"sync_dist": True},
        {"prog_bar": True},
        {"batch_size": 3},
        {"reduce_fx": "mean"},
        {"on_step": True},
    ],
)
def test_write_through_rejects_lightning_logging_kwargs(
    kwargs: dict[str, object],
) -> None:
    """``write_through=True`` + any logging-control kwarg raises.

    The write-through path provides none of Lightning's aggregation /
    sync / display semantics; a call site passing them (even at their
    defaults, like ``on_step=True``) is confused about which path it is
    on and must be corrected, not silently accommodated."""
    model = _Model()
    with pytest.raises(ValueError, match="write_through"):
        model.log("diagnostics-train/x", 1.0, write_through=True, **kwargs)  # pyright: ignore[reportArgumentType]


def test_collector_batches_per_step_and_clears() -> None:
    """Unit: two names at one step merge into one call per logger;
    distinct steps stay distinct calls in step order; flush drains."""
    col = WriteThroughCollector()
    col.collect("a", torch.tensor(1.0), step=10)
    col.collect("b", 2.0, step=10)
    col.collect("a", 3.0, step=15)
    lg1, lg2 = _Capture(), _Capture()
    col.flush([lg1, lg2], is_global_zero=True)
    for lg in (lg1, lg2):
        assert lg.records == [
            (10, {"a": 1.0, "b": 2.0}),
            (15, {"a": 3.0}),
        ]
    # Buffer drained: a second flush writes nothing.
    col.flush([lg1, lg2], is_global_zero=True)
    assert len(lg1.records) == 2


def test_collector_rank_guard_drops_without_writing() -> None:
    """Non-zero ranks neither write nor accumulate: values are not
    sync_dist-reduced, so only rank zero's view is meaningful, and the
    buffer must still drain to avoid unbounded growth."""
    col = WriteThroughCollector()
    col.collect("a", 1.0, step=10)
    lg = _Capture()
    col.flush([lg], is_global_zero=False)
    assert lg.records == []
    col.flush([lg], is_global_zero=True)
    assert lg.records == []  # buffer was cleared, not deferred

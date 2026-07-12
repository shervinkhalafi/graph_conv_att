"""Tests for FinalSampleDumpCallback (parity D-16b).

Test rationale
--------------
The callback fires at ``on_fit_end`` and:

* gates on ``trainer.is_global_zero`` (DDP rank-0 only -- spec Q9);
* swaps to EMA weights iff a sibling :class:`EMACallback` is registered
  (spec Q7);
* writes ``num_samples`` graphs to a path resolved through
  :func:`_resolve_dump_path` (explicit > Modal mount > local fallback,
  spec Q8);
* runs the evaluator against the test reference set (spec Q6).

We test the validation surface (constructor rejection of bad arguments),
the path-resolution helper directly, the EMA-detection helper, the
backbone-gating helper (fail-loud TypeError on missing collaborators),
and an end-to-end smoke test driving a stub LightningModule through
``trainer.fit`` so ``on_fit_end`` actually fires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from tmgg.training.callbacks.ema import EMACallback
from tmgg.training.callbacks.final_sample_dump import (
    FinalSampleDumpCallback,
    _require_attr,
    _resolve_dump_path,
)

# ---------------------------------------------------------------------------
# Constructor validity
# ---------------------------------------------------------------------------


class TestConstructorValidity:
    def test_rejects_zero_num_samples(self) -> None:
        with pytest.raises(ValueError, match="num_samples must be > 0"):
            FinalSampleDumpCallback(num_samples=0)

    def test_rejects_negative_num_samples(self) -> None:
        with pytest.raises(ValueError, match="num_samples must be > 0"):
            FinalSampleDumpCallback(num_samples=-5)

    def test_rejects_zero_batch_size(self) -> None:
        with pytest.raises(ValueError, match="sample_batch_size must be > 0"):
            FinalSampleDumpCallback(num_samples=10, sample_batch_size=0)


# ---------------------------------------------------------------------------
# Path resolution (spec Q8)
# ---------------------------------------------------------------------------


class TestResolveDumpPath:
    def test_explicit_path_wins(self, tmp_path: Path) -> None:
        explicit = tmp_path / "explicit.pt"
        out = _resolve_dump_path(
            configured=str(explicit),
            run_name="myrun",
            default_root_dir=tmp_path,
        )
        assert out == explicit

    def test_modal_context_uses_outputs_mount(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Set MODAL_TASK_ID to simulate a Modal container; the helper
        # should route to /data/outputs/final_samples/<run>.pt.
        monkeypatch.setenv("MODAL_TASK_ID", "modal-task-12345")
        out = _resolve_dump_path(
            configured=None,
            run_name="r42",
            default_root_dir=tmp_path,
        )
        # Tach forbids tmgg.training -> tmgg.modal, so the callback
        # inlines the mount path. This assertion pins the inlined
        # constant against the canonical one in tmgg.modal._lib.volumes
        # so a future re-mount fails loudly.
        from tmgg.modal._lib.volumes import OUTPUTS_MOUNT
        from tmgg.training.callbacks.final_sample_dump import (
            MODAL_OUTPUTS_MOUNT,
        )

        assert MODAL_OUTPUTS_MOUNT == OUTPUTS_MOUNT
        assert out == Path(OUTPUTS_MOUNT) / "final_samples" / "r42.pt"

    def test_local_fallback_when_no_modal_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODAL_TASK_ID", raising=False)
        out = _resolve_dump_path(
            configured=None,
            run_name="r42",
            default_root_dir=tmp_path,
        )
        assert out == tmp_path / "final_samples.pt"


# ---------------------------------------------------------------------------
# Backbone-gate helper (fail-loud on missing module attributes)
# ---------------------------------------------------------------------------


class TestRequireAttr:
    def test_raises_typeerror_on_missing_attr(self) -> None:
        bare = pl.LightningModule()
        with pytest.raises(TypeError, match="`.sampler`"):
            _require_attr(bare, "sampler")

    def test_returns_value_when_present(self) -> None:
        module = pl.LightningModule()
        sentinel = object()
        module.foo = sentinel  # type: ignore[attr-defined]
        assert _require_attr(module, "foo") is sentinel

    def test_raises_typeerror_on_none(self) -> None:
        module = pl.LightningModule()
        module.bar = None  # type: ignore[attr-defined]
        with pytest.raises(TypeError, match="`.bar`"):
            _require_attr(module, "bar")


# ---------------------------------------------------------------------------
# EMA detection (spec Q7)
# ---------------------------------------------------------------------------


class TestEmaDetection:
    def test_finds_registered_ema_callback(self) -> None:
        ema = EMACallback(decay=0.99)
        trainer = MagicMock()
        trainer.callbacks = [ema, MagicMock()]
        cb = FinalSampleDumpCallback(num_samples=4)
        assert cb._find_ema_callback(trainer) is ema

    def test_returns_none_when_absent(self) -> None:
        trainer = MagicMock()
        trainer.callbacks = [MagicMock(), MagicMock()]
        cb = FinalSampleDumpCallback(num_samples=4)
        assert cb._find_ema_callback(trainer) is None


# ---------------------------------------------------------------------------
# DDP rank gate (spec Q9)
# ---------------------------------------------------------------------------


class TestDdpRankGate:
    def test_no_op_on_non_zero_rank(self, tmp_path: Path) -> None:
        cb = FinalSampleDumpCallback(num_samples=2, save_path=str(tmp_path / "x.pt"))
        trainer = MagicMock()
        trainer.is_global_zero = False
        # If the gate did NOT fire, _require_attr would explode on the
        # bare LightningModule. The bare module here is a sentinel.
        bare = pl.LightningModule()
        cb.on_fit_end(trainer, bare)
        # No file written.
        assert not (tmp_path / "x.pt").exists()


# ---------------------------------------------------------------------------
# End-to-end smoke through trainer.fit
# ---------------------------------------------------------------------------


class _StubSampler:
    """Returns trivial GraphData payloads so the callback exercises the loop."""

    def sample(
        self,
        model: Any,
        noise_process: Any,
        num_graphs: int,
        num_nodes: Any,
        device: torch.device,
    ) -> list[Any]:
        from tmgg.data.datasets.graph_types import DenseGraphState

        _ = model, noise_process, num_nodes, device
        return [
            DenseGraphState(
                num_nodes_per_graph=torch.tensor([3], dtype=torch.long),
                y=torch.zeros(1, 0),
                E_class=torch.zeros(1, 3, 3, 2),
            )
            for _ in range(num_graphs)
        ]


class _StubEvaluator:
    eval_num_samples = 4

    def to_networkx_graphs(self, samples: list[Any]) -> list[Any]:
        import networkx as nx

        return [nx.complete_graph(3) for _ in samples]

    def evaluate(self, refs: list[Any], generated: list[Any]) -> Any:
        # Return an object exposing to_dict() with a single metric so
        # the callback's logging path runs without depending on the
        # real EvaluationResults dataclass.
        _ = refs, generated

        class _R:
            def to_dict(self) -> dict[str, float]:
                return {"degree_mmd": 0.123}

        return _R()


class _StubDataModule(pl.LightningDataModule):
    def __init__(self, loader: DataLoader[Any]) -> None:
        super().__init__()
        self._loader = loader

    def train_dataloader(self) -> DataLoader[Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._loader

    def get_reference_graphs(self, stage: str, max_graphs: int) -> list[Any]:
        import networkx as nx

        _ = stage, max_graphs
        return [nx.complete_graph(3) for _ in range(4)]


class _StubLightningModule(pl.LightningModule):
    """Minimal module exposing the attributes FinalSampleDumpCallback needs."""

    num_nodes: int = 3

    def __init__(self) -> None:
        super().__init__()
        self.model: nn.Linear = nn.Linear(2, 1)
        self.sampler: _StubSampler = _StubSampler()
        self.noise_process: object = object()
        self.evaluator: _StubEvaluator = _StubEvaluator()

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.model(x)

    def training_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        x, y = batch
        return ((self.model(x) - y) ** 2).mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:  # pyright: ignore[reportIncompatibleMethodOverride]
        return torch.optim.SGD(self.parameters(), lr=1e-2)


def test_on_fit_end_writes_dump_and_evaluates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One trainer.fit cycle + on_fit_end produces the expected artefact.

    Smoke test: verifies the callback wires through trainer.fit to
    on_fit_end, writes the dump at the resolved path with the expected
    schema, and calls the evaluator against the test reference set.
    """
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    save_path = tmp_path / "final_samples.pt"

    callback = FinalSampleDumpCallback(
        num_samples=4,
        sample_batch_size=2,
        save_path=str(save_path),
        run_name="testrun",
    )

    module = _StubLightningModule()
    x = torch.randn(8, 2)
    y = torch.randn(8, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    datamodule = _StubDataModule(loader)

    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(module, datamodule=datamodule)

    assert save_path.exists()
    payload = torch.load(save_path, weights_only=False)
    assert "graphs" in payload
    assert "meta" in payload
    assert len(payload["graphs"]) == 4
    assert payload["meta"]["num_samples"] == 4
    assert payload["meta"]["ema_active"] is False
    assert payload["meta"]["run_name"] == "testrun"


def test_on_fit_end_swaps_to_ema_when_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When EMACallback is also registered, the dump payload reports EMA active."""
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    save_path = tmp_path / "samples_ema.pt"

    ema_cb = EMACallback(decay=0.5)
    dump_cb = FinalSampleDumpCallback(
        num_samples=2,
        sample_batch_size=2,
        save_path=str(save_path),
        run_name="ema_run",
    )

    module = _StubLightningModule()
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    datamodule = _StubDataModule(loader)

    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[ema_cb, dump_cb],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(module, datamodule=datamodule)

    payload = torch.load(save_path, weights_only=False)
    assert payload["meta"]["ema_active"] is True


def test_on_fit_end_fails_loud_when_module_lacks_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module missing .sampler must surface a clear TypeError, not silently skip.

    Tested by invoking ``on_fit_end`` directly with a stub trainer
    (fail-loud behaviour belongs to the callback itself; whether
    Lightning re-surfaces the exception through ``trainer.fit`` depends
    on the Lightning version's exception-handling policy and is not
    what we're checking here).
    """
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    cb = FinalSampleDumpCallback(
        num_samples=2,
        save_path=str(tmp_path / "x.pt"),
    )

    class _Stripped(_StubLightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.sampler = None  # pyright: ignore[reportAttributeAccessIssue]

    module = _Stripped()
    trainer = MagicMock()
    trainer.is_global_zero = True
    trainer.callbacks = [cb]
    trainer.default_root_dir = str(tmp_path)

    with pytest.raises(TypeError, match="`.sampler`"):
        cb.on_fit_end(trainer, module)

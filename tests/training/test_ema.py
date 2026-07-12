"""Tests for the EMA utility and the EMACallback Lightning callback.

Test rationale
--------------
The EMA module supplies the missing piece behind upstream DiGress's
``cfg.train.ema_decay > 0`` gate (parity #45 / D-15). The contract has
two layers:

1. The :class:`~tmgg.training.ema.ExponentialMovingAverage` utility owns
   the per-parameter blend math and the store / copy_to / restore
   roundtrip. Its tests verify the blend formula, the roundtrip
   correctness, and the strict-zip mismatch behaviour.

2. The :class:`~tmgg.training.callbacks.ema.EMACallback` wires the
   utility into Lightning's lifecycle. A smoke test trains a tiny
   :class:`pl.LightningModule` for one epoch and asserts the shadow
   diverges from the live weights — proof that ``on_train_batch_end``
   actually fires the EMA update.
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from tmgg.training.callbacks.ema import EMACallback
from tmgg.training.ema import ExponentialMovingAverage


class TestExponentialMovingAverage:
    """Direct tests for the blend math and roundtrip."""

    def test_update_blends_with_decay(self) -> None:
        """One update step yields decay*shadow + (1-decay)*live."""
        param = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        ema = ExponentialMovingAverage([param], decay=0.9)
        # Mutate the live param to a fresh value before update.
        with torch.no_grad():
            param.copy_(torch.tensor([10.0, 20.0, 30.0]))
        ema.update([param])
        # shadow_t+1 = 0.9 * shadow_t + 0.1 * live = 0.9*[1,2,3] + 0.1*[10,20,30]
        expected = torch.tensor([0.9 + 1.0, 1.8 + 2.0, 2.7 + 3.0])
        torch.testing.assert_close(
            ema._shadow_params[0], expected, atol=1e-6, rtol=1e-6
        )

    def test_store_copy_to_restore_roundtrip(self) -> None:
        """store + copy_to + restore returns the live params untouched."""
        live = nn.Parameter(torch.tensor([1.0, 2.0]))
        ema = ExponentialMovingAverage([live], decay=0.5)
        # Drive the shadow away from live so copy_to is observable.
        with torch.no_grad():
            ema._shadow_params[0].copy_(torch.tensor([99.0, 99.0]))

        original = live.detach().clone()
        ema.store([live])
        ema.copy_to([live])
        torch.testing.assert_close(
            live.detach(), torch.tensor([99.0, 99.0]), atol=1e-6, rtol=1e-6
        )
        ema.restore([live])
        torch.testing.assert_close(live.detach(), original, atol=1e-6, rtol=1e-6)

    def test_restore_without_store_raises(self) -> None:
        """restore() without a preceding store() fails loudly."""
        param = nn.Parameter(torch.tensor([1.0]))
        ema = ExponentialMovingAverage([param], decay=0.9)
        with pytest.raises(RuntimeError, match="store"):
            ema.restore([param])

    def test_decay_out_of_range_raises(self) -> None:
        """Constructor rejects decay outside [0, 1]."""
        param = nn.Parameter(torch.tensor([1.0]))
        with pytest.raises(ValueError, match="decay"):
            ExponentialMovingAverage([param], decay=1.5)
        with pytest.raises(ValueError, match="decay"):
            ExponentialMovingAverage([param], decay=-0.1)

    def test_update_param_count_mismatch_raises(self) -> None:
        """Strict zip catches a mid-training change in the parameter list."""
        a = nn.Parameter(torch.tensor([1.0]))
        b = nn.Parameter(torch.tensor([2.0]))
        ema = ExponentialMovingAverage([a, b], decay=0.9)
        # Pass only one parameter on update — strict=True must raise.
        with pytest.raises(ValueError):
            ema.update([a])


class _TinyLightningModule(pl.LightningModule):
    """Two-parameter LightningModule used by the callback smoke test."""

    def __init__(self) -> None:
        super().__init__()
        self.model: nn.Linear = nn.Linear(4, 1)

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.model(x)

    def training_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        x, y = batch
        loss = ((self.model(x) - y) ** 2).mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:  # pyright: ignore[reportIncompatibleMethodOverride]
        return torch.optim.SGD(self.parameters(), lr=1e-1)


class TestEMACallback:
    """Integration tests for the Lightning callback wiring."""

    def test_callback_decay_gate_rejects_zero(self) -> None:
        """Constructor rejects decay=0 — disable by not registering instead."""
        with pytest.raises(ValueError, match="decay"):
            EMACallback(decay=0.0)

    def test_callback_decay_gate_rejects_above_one(self) -> None:
        """Constructor rejects decay > 1."""
        with pytest.raises(ValueError, match="decay"):
            EMACallback(decay=1.5)

    def test_state_dict_roundtrip_preserves_shadow_and_decay(self, tmp_path) -> None:
        """state_dict + load_state_dict on a fresh callback restore the shadow.

        Test rationale
        --------------
        Lightning auto-includes ``Callback.state_dict()`` in the
        trainer checkpoint. The eval-time CLI rebuilds the callback,
        calls ``load_state_dict`` against the saved entry, and then
        :meth:`copy_shadow_into` to swap into a freshly-loaded model.
        The roundtrip therefore has to preserve both the shadow
        tensors and the decay value bit-for-bit.
        """
        torch.manual_seed(0)
        module = _TinyLightningModule()
        x = torch.randn(16, 4)
        y = torch.randn(16, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        callback = EMACallback(decay=0.7)
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
        trainer.fit(module, loader)

        saved = callback.state_dict()
        # Sanity: the saved state carries non-empty shadow + the
        # original decay value.
        assert saved["decay"] == pytest.approx(0.7)
        assert len(saved["shadow_params"]) > 0

        fresh = EMACallback(decay=0.999)  # Different decay; load_state_dict overrides.
        fresh.load_state_dict(saved)
        assert fresh.decay == pytest.approx(0.7)
        assert fresh.ema is not None
        for restored, original in zip(
            fresh.ema._shadow_params, saved["shadow_params"], strict=True
        ):
            torch.testing.assert_close(restored, original)

    def test_copy_shadow_into_swaps_model_weights(self, tmp_path) -> None:
        """copy_shadow_into overwrites model weights with the shadow tensors.

        Test rationale
        --------------
        The CLI's ``--use_ema`` path calls ``copy_shadow_into(module.model)``
        after restoring the shadow from checkpoint state. Verify the
        in-place swap actually changes the live weights to match the
        shadow.
        """
        torch.manual_seed(0)
        module = _TinyLightningModule()
        x = torch.randn(16, 4)
        y = torch.randn(16, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        callback = EMACallback(decay=0.5)
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
        trainer.fit(module, loader)

        assert callback.ema is not None
        shadow_w_before = callback.ema._shadow_params[0].detach().clone()

        callback.copy_shadow_into(module.model)
        live_w_after = module.model.weight.detach()
        torch.testing.assert_close(live_w_after, shadow_w_before)

    def test_copy_shadow_into_fails_loud_when_not_initialized(self) -> None:
        """copy_shadow_into raises TypeError if the shadow was never built.

        Test rationale
        --------------
        Per CLAUDE.md, the helper must fail loudly when called before
        either ``on_fit_start`` (training path) or ``load_state_dict``
        (eval path). Returning silently or zero-padding would mask
        configuration bugs at the eval CLI boundary.
        """
        callback = EMACallback(decay=0.5)
        model = nn.Linear(4, 1)
        with pytest.raises(TypeError, match="copy_shadow_into called before"):
            callback.copy_shadow_into(model)

    def test_callback_updates_shadow_during_training(self, tmp_path) -> None:
        """One epoch of training drives shadow weights away from live weights.

        Asserts ``on_train_batch_end`` actually fires the EMA update
        (otherwise the shadow would equal the initial weights, while
        SGD pushes the live weights elsewhere).
        """
        torch.manual_seed(42)
        module = _TinyLightningModule()
        initial_weights = module.model.weight.detach().clone()

        # Synthetic data — 32 samples of a degenerate regression.
        x = torch.randn(32, 4)
        y = torch.randn(32, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=8)

        callback = EMACallback(decay=0.5)
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
        trainer.fit(module, loader)

        # The shadow should differ from the initial weights (update
        # fired) and from the live weights (training drifted them
        # apart).
        assert callback.ema is not None
        shadow_w = callback.ema._shadow_params[0]
        live_w = module.model.weight.detach()
        assert not torch.allclose(
            shadow_w, initial_weights, atol=1e-4
        ), "EMA shadow did not move from initial weights"
        assert not torch.allclose(
            shadow_w, live_w, atol=1e-4
        ), "EMA shadow exactly matches live weights — averaging was a no-op"

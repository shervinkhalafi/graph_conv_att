"""Lightning callback that maintains and swaps EMA model weights.

At fit start, snapshot the live model's parameters into an
:class:`~tmgg.training.ema.ExponentialMovingAverage`. After every
training batch, update the shadow. Around validation, swap EMA weights
into the live model so validation and sampling see smoothed weights,
then restore the live weights for the next training step.

Mirrors upstream DiGress's gating pattern (``main.py:181-183``:
``cfg.train.ema_decay > 0``). Configure via ``ema_decay > 0`` in the
trainer-side config; ``run_experiment.create_callbacks`` registers the
callback only when the gate fires.

Lightning auto-includes :meth:`Callback.state_dict` /
:meth:`Callback.load_state_dict` outputs in the trainer checkpoint.
The implementations below persist the shadow tensor list and the
configured ``decay`` so an evaluation CLI can rebuild the callback,
restore the shadow, and swap it into a freshly-loaded model -- the
foundation of the ``--use_ema`` flag on
``tmgg-discrete-eval-all`` (D-16c).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from pytorch_lightning.callbacks import Callback
from torch import Tensor, nn

from tmgg.training.ema import ExponentialMovingAverage

if TYPE_CHECKING:
    import pytorch_lightning as pl


def _require_backbone_parameters(pl_module: pl.LightningModule) -> Any:
    """Return ``pl_module.model.parameters()``, raising on missing backbone.

    EMACallback only needs a parameter-bearing ``.model`` backbone to
    shadow. Lightning's callback hooks type ``pl_module`` as the loose
    :class:`pytorch_lightning.LightningModule` base, which doesn't
    declare ``.model``; rather than silencing the resulting pyright
    attribute error, do the lookup via ``getattr`` (whose ``Any``
    return type pyright accepts) and raise informatively if the
    attribute is missing. Fail-loud at registration time per CLAUDE.md
    instead of erroring deep inside an EMA shadow update. A
    ``runtime_checkable`` ``Protocol`` was tried first but Python's
    isinstance check on protocols with annotated attributes only sees
    class-level annotations — instance-set ``self.model = ...``
    attributes (the universal pattern in BaseGraphModule and our test
    fixtures) are not detected.
    """
    backbone = getattr(pl_module, "model", None)
    if backbone is None:
        raise TypeError(
            "EMACallback requires a LightningModule with a `.model` backbone "
            f"to shadow; got {type(pl_module).__name__} which does not "
            "expose one."
        )
    return backbone.parameters()


class EMACallback(Callback):
    """Track EMA weights and swap them in around validation.

    Parameters
    ----------
    decay
        Smoothing constant in ``(0, 1]``. Pass ``decay`` close to 1
        (e.g. ``0.999``) for a slowly-changing shadow. ``decay = 0``
        is rejected — disable EMA by not registering the callback at
        all rather than passing a zero gate here.
    """

    def __init__(self, decay: float) -> None:
        super().__init__()
        if not 0.0 < decay <= 1.0:
            raise ValueError(
                f"EMACallback decay must be in (0, 1]; got {decay}. "
                "Disable EMA by not registering the callback rather "
                "than passing decay=0."
            )
        self.decay = decay
        self.ema: ExponentialMovingAverage | None = None

    @override
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # EMA the wrapped backbone, not the LightningModule itself, so
        # optimizer states and other non-trainable scalars stay
        # untouched.
        self.ema = ExponentialMovingAverage(
            _require_backbone_parameters(pl_module),
            decay=self.decay,
        )

    @override
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        assert self.ema is not None, "on_train_batch_end fired before on_fit_start"
        self.ema.update(_require_backbone_parameters(pl_module))

    @override
    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.ema is None:
            # Lightning runs a sanity-check validation pass before
            # on_fit_start in some configurations; skip swapping when
            # no shadow exists yet.
            return
        params = _require_backbone_parameters(pl_module)
        self.ema.store(params)
        # ``parameters()`` returns a fresh generator on every call; the
        # second iteration above would be empty if we re-used ``params``,
        # so re-fetch for copy_to.
        self.ema.copy_to(_require_backbone_parameters(pl_module))

    @override
    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.ema is None:
            return
        if self.ema._stored_params is None:
            # store() was skipped (sanity check before fit) — nothing
            # to restore.
            return
        self.ema.restore(_require_backbone_parameters(pl_module))

    @override
    def state_dict(self) -> dict[str, Any]:
        """Persist EMA shadow + decay so the checkpoint can restore them.

        Lightning auto-includes ``Callback.state_dict()`` in the trainer
        checkpoint when the method is defined. The decay is round-tripped
        so the eval-time CLI can reconstruct the callback without parsing
        the state-key string.
        """
        if self.ema is None:
            # ``state_dict`` may be called before ``on_fit_start``
            # (e.g. when Lightning persists callback state during a
            # sanity-check pass). An empty shadow is safe; the matching
            # ``load_state_dict`` rebuilds nothing.
            return {"shadow_params": [], "decay": self.decay}
        return {
            "shadow_params": [
                p.detach().cpu().clone() for p in self.ema._shadow_params
            ],
            "decay": self.decay,
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore EMA shadow from a saved state dict.

        Constructs a fresh :class:`ExponentialMovingAverage` shell when
        none is yet present (Lightning may call ``load_state_dict``
        before ``on_fit_start``), seeded with zero-tensors that the
        loaded shadow then overwrites. ``ExponentialMovingAverage``
        wants an iterable of ``nn.Parameter`` to seed its shadow list;
        we wrap each saved tensor in a throwaway :class:`nn.Parameter`
        with ``requires_grad=False`` to satisfy that contract without
        creating actual trainable parameters. The constructor
        immediately replaces the underlying shadow list with the saved
        tensors, so the wrapper parameters are short-lived.
        """
        shadow: list[Tensor] = list(state_dict["shadow_params"])
        decay: float = float(state_dict["decay"])
        self.decay = decay  # honour the saved value
        if self.ema is None:
            # Seed with the saved tensors as throwaway parameters; the
            # constructor clones them into the shadow list. ``requires_grad``
            # stays False because these are not trainable parameters.
            seed_params = [nn.Parameter(t.clone(), requires_grad=False) for t in shadow]
            self.ema = ExponentialMovingAverage(seed_params, decay=decay)
            return
        # Replace the shadow tensors in place to keep object identity
        # and tensor devices stable.
        for current, restored in zip(self.ema._shadow_params, shadow, strict=True):
            current.copy_(restored.to(current.device))

    def copy_shadow_into(self, model: nn.Module) -> None:
        """Swap EMA shadow weights into ``model`` in place.

        Reuses the existing :meth:`ExponentialMovingAverage.copy_to`
        machinery. Used by ``tmgg-discrete-eval-all`` after a checkpoint
        load to evaluate against the smoothed weights rather than the
        live training state.

        Raises
        ------
        TypeError
            The EMA shadow has not been initialised. Either run training
            (``on_fit_start`` instantiates the EMA) or call
            :meth:`load_state_dict` with a saved state first.
        """
        if self.ema is None:
            raise TypeError(
                "EMACallback.copy_shadow_into called before EMA shadow was "
                "initialised. Either run training (on_fit_start instantiates "
                "the EMA) or call load_state_dict with a saved state first."
            )
        self.ema.copy_to(model.parameters())

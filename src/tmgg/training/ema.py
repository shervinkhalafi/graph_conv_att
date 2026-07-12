"""Exponential moving average for neural-net parameters.

Standard EMA contract: ``shadow = decay * shadow + (1 - decay) * param``
applied per parameter on each :meth:`ExponentialMovingAverage.update`
call. ``store()`` / ``copy_to()`` / ``restore()`` swap shadow weights
into a model for evaluation, then restore the live weights.

Mirrors the public surface of the ``torch_ema`` package
(``ExponentialMovingAverage`` from ``torch_ema``) so a callback or
script written against this class transfers across implementations
without changes.

Upstream DiGress's ``main.py:181-183`` gates an EMA branch on
``cfg.train.ema_decay > 0`` but its ``utils.EMA`` symbol is missing
from the upstream snapshot we mirror; this module fills the gap with a
working contract.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


class ExponentialMovingAverage:
    """Maintain a shadow copy of model parameters smoothed exponentially.

    Parameters
    ----------
    parameters
        Iterable over the live :class:`torch.nn.Parameter` instances to
        track. The shadow copies are detached clones; mutating them
        does not affect the live parameters.
    decay
        Smoothing constant in ``[0, 1]``. ``decay`` close to 1 keeps
        the shadow stable; close to 0 makes it track the live weights
        almost exactly.

    Notes
    -----
    The class stores ordered tensor lists rather than referencing the
    underlying parameters by identity. Callers must therefore pass the
    parameter iterables in the same order on every call (which is the
    default for ``model.parameters()``).
    """

    def __init__(self, parameters: Iterable[nn.Parameter], decay: float):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1]; got {decay}")
        self.decay = decay
        self._shadow_params: list[Tensor] = [p.detach().clone() for p in parameters]
        self._stored_params: list[Tensor] | None = None

    def update(self, parameters: Iterable[nn.Parameter]) -> None:
        """Blend the live parameters into the shadow copies in-place."""
        with torch.no_grad():
            for shadow, param in zip(self._shadow_params, parameters, strict=True):
                shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def store(self, parameters: Iterable[nn.Parameter]) -> None:
        """Snapshot the live parameters so :meth:`restore` can revert."""
        self._stored_params = [p.detach().clone() for p in parameters]

    def copy_to(self, parameters: Iterable[nn.Parameter]) -> None:
        """Overwrite the live parameters with the shadow copies in-place."""
        with torch.no_grad():
            for shadow, param in zip(self._shadow_params, parameters, strict=True):
                param.copy_(shadow)

    def restore(self, parameters: Iterable[nn.Parameter]) -> None:
        """Write the previously stored parameters back into the model."""
        if self._stored_params is None:
            raise RuntimeError("EMA.restore() called without prior store()")
        with torch.no_grad():
            for stored, param in zip(self._stored_params, parameters, strict=True):
                param.copy_(stored)
        self._stored_params = None

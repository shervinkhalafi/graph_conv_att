"""Unified noise schedule for discrete diffusion and continuous denoising.

Wraps the schedule computation functions from ``tmgg.diffusion.diffusion_math``
into a single ``nn.Module`` that precomputes beta, alpha, and alpha-bar tensors
as registered buffers and exposes them through a uniform lookup interface.

Registered buffers ensure the tensors follow the module to the correct device
and are included in ``state_dict`` for checkpointing.
"""

# pyright: reportUninitializedInstanceVariable=false
# PyTorch register_buffer() initialises these attributes at runtime;
# pyright cannot track this pattern.

from __future__ import annotations

from typing import override

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

from .diffusion_math import (
    cosine_beta_schedule_discrete,
    custom_beta_schedule_discrete,
)

_VALID_SCHEDULE_TYPES = frozenset({"cosine_iddpm", "custom_vignac", "linear_ddpm"})


class NoiseSchedule(nn.Module):
    """Unified noise schedule (``nn.Module``) for discrete diffusion and continuous denoising.

    The schedule precomputes ``T + 1`` entries indexed by integer timestep
    ``t in {0, ..., T}``, where ``t = 0`` is the clean data and ``t = T`` is
    fully noised.  All buffer arrays (``betas``, ``alphas``, ``alpha_bar``)
    therefore have shape ``(T + 1,)``.  For cosine and custom schedules the
    arrays come from the DiGress utilities; the linear schedule provides an
    evenly-spaced alternative suitable for simpler denoising experiments.

    All precomputed tensors (``betas``, ``alphas``, ``alpha_bar``) are stored as
    registered buffers so they follow the module to the correct device and are
    included in ``state_dict``.

    Parameters
    ----------
    schedule_type : str
        One of ``"cosine_iddpm"``, ``"custom_vignac"``, or ``"linear_ddpm"``.
    timesteps : int
        Number of diffusion steps *T* (must be positive).
    average_num_nodes : int
        Average number of nodes per graph, used only by the ``"custom_vignac"``
        schedule to set a floor on per-step beta.  Ignored for other types.
    num_edge_classes : int
        Number of categorical edge classes *K*, passed to the custom schedule
        to compute the correct beta floor. Ignored for other types.
        Default is 5: K=5 is the upstream-parity default — recovers
        ``p = 1 - 1/5 = 4/5`` from upstream's hardcoded molecular case
        (``custom_beta_schedule_discrete``, ``diffusion_utils.py:77-97``).
    """

    betas: Tensor
    alphas: Tensor
    alpha_bar: Tensor

    def __init__(
        self,
        schedule_type: str,
        timesteps: int,
        average_num_nodes: int = 50,
        num_edge_classes: int = 5,
    ) -> None:
        super().__init__()

        if schedule_type not in _VALID_SCHEDULE_TYPES:
            raise ValueError(
                f"schedule_type must be one of {sorted(_VALID_SCHEDULE_TYPES)}, "
                f"got {schedule_type!r}"
            )
        if timesteps < 1:
            raise ValueError(f"timesteps must be >= 1, got {timesteps}")

        self._schedule_type = schedule_type
        self._timesteps = timesteps

        if schedule_type == "cosine_iddpm":
            self._register_schedule_from_betas(cosine_beta_schedule_discrete(timesteps))

        elif schedule_type == "custom_vignac":
            self._register_schedule_from_betas(
                custom_beta_schedule_discrete(
                    timesteps, average_num_nodes, num_edge_classes=num_edge_classes
                )
            )

        elif schedule_type == "linear_ddpm":
            # Linearly decaying signal: alpha_bar(t) = 1 - t/T, from 1.0 to 0.0.
            # Betas are derived so the DDPM identity alpha_bar = cumprod(1 - beta)
            # holds.  DiGress (github.com/cvignac/DiGress) does not have a linear
            # schedule; this is our own extension for simpler denoising experiments.
            alpha_bar_np = np.linspace(1.0, 0.0, timesteps + 1)
            self._register_schedule_from_alpha_bar(alpha_bar_np)

        # Invariant: all buffer arrays have exactly T+1 entries.
        expected = timesteps + 1
        if self.betas.shape[0] != expected:
            raise ValueError(
                f"Schedule buffer shape mismatch: betas has {self.betas.shape[0]} "
                f"entries, expected {expected} (T+1 where T={timesteps})"
            )

    def _register_schedule_from_betas(self, betas_np: NDArray[np.floating]) -> None:
        """Clip raw betas, derive alphas and alpha-bar, and register all as buffers."""
        betas_np = np.clip(betas_np, 0.0, 0.9999)
        alphas_np = 1.0 - betas_np
        alpha_bar_np = np.exp(np.cumsum(np.log(alphas_np)))

        self.register_buffer("betas", torch.from_numpy(betas_np).float())
        self.register_buffer("alphas", torch.from_numpy(alphas_np).float())
        self.register_buffer("alpha_bar", torch.from_numpy(alpha_bar_np).float())

    def _register_schedule_from_alpha_bar(
        self, alpha_bar_np: NDArray[np.floating]
    ) -> None:
        """Derive betas from a desired alpha_bar curve and register all as buffers.

        Computes ``beta[t] = 1 - alpha_bar[t] / alpha_bar[t-1]`` so the
        DDPM identity ``alpha_bar[t] = prod(1 - beta[0:t])`` holds by
        construction.  Unlike ``_register_schedule_from_betas``, which
        starts from betas and derives alpha_bar, this method starts from
        a prescribed signal-decay curve and works backwards.

        DiGress (github.com/cvignac/DiGress, Vignac et al. ICLR 2023)
        does not implement a linear schedule; its cosine and custom
        schedules both derive alpha_bar from betas.  This helper exists
        for our linear schedule extension, where the desired quantity is
        the alpha_bar curve itself.

        References
        ----------
        Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS
        2020, Eq. 4.
        """
        # Floor alpha_bar at a small positive value so log stays finite and
        # beta clipping doesn't produce surprising deviations at t=T.
        alpha_bar_np = np.clip(alpha_bar_np, 1e-5, None)
        # alpha[0] = alpha_bar[0] (no previous step), alpha[t] = alpha_bar[t]/alpha_bar[t-1]
        alphas_np = np.concatenate(
            [[alpha_bar_np[0]], alpha_bar_np[1:] / alpha_bar_np[:-1]]
        )
        betas_np = np.clip(1.0 - alphas_np, 0.0, 0.9999)
        # Recompute alpha_bar from clipped betas for exact consistency.
        alphas_np = 1.0 - betas_np
        alpha_bar_np = np.exp(np.cumsum(np.log(alphas_np)))

        self.register_buffer("betas", torch.from_numpy(betas_np).float())
        self.register_buffer("alphas", torch.from_numpy(alphas_np).float())
        self.register_buffer("alpha_bar", torch.from_numpy(alpha_bar_np).float())

    @property
    def timesteps(self) -> int:
        """Number of diffusion steps."""
        return self._timesteps

    @property
    def schedule_type(self) -> str:
        """Schedule type string passed at construction."""
        return self._schedule_type

    def _resolve_index(self, t_int: Tensor) -> tuple[Tensor, torch.Size]:
        """Flatten ``t_int`` to a 1-D long tensor and return the original shape.

        Parameters
        ----------
        t_int
            Integer timestep indices with shape ``(bs,)`` or ``(bs, 1)``.
            Valid range is ``[0, T]`` inclusive (``T + 1`` entries total).

        Returns
        -------
        tuple[Tensor, torch.Size]
            ``(flat_indices, original_shape)``

        Raises
        ------
        ValueError
            If any index falls outside ``[0, T]``.
        """
        original_shape = t_int.shape
        flat = t_int.reshape(-1).long()
        # Range-check sanity assert. ``_resolve_index`` is hit 4× per
        # training step, and each ``.item()`` syncs the GPU stream.
        # Guarded by ``__debug__`` so production runs (Python -O /
        # PYTHONOPTIMIZE=1) skip the four syncs. Indices come from
        # ``torch.randint(0, T+1, ...)`` upstream, so the range is
        # mathematically guaranteed; this catches caller bugs only.
        if __debug__:
            lo, hi = int(flat.min().item()), int(flat.max().item())
            if lo < 0 or hi > self._timesteps:
                raise ValueError(
                    f"Timestep indices must be in [0, {self._timesteps}], "
                    f"got values in [{lo}, {hi}]"
                )
        return flat, original_shape

    def get_beta(
        self,
        t_int: Tensor | None = None,
        *,
        t_normalized: Tensor | None = None,
    ) -> Tensor:
        """Beta value at integer timestep(s).

        Exactly one of *t_int* or *t_normalized* must be provided.

        Parameters
        ----------
        t_int : Tensor, optional
            Integer timestep indices, shape ``(bs,)`` or ``(bs, 1)``.
        t_normalized : Tensor, optional
            Normalised time in ``[0, 1]``, converted to integer indices via
            rounding.

        Returns
        -------
        Tensor
            Beta values with the same shape as the input.
        """
        t_int = self._resolve_t(t_int, t_normalized=t_normalized)
        flat, shape = self._resolve_index(t_int)
        return self.betas[flat].reshape(shape)

    def get_alpha_bar(
        self,
        t_int: Tensor | None = None,
        *,
        t_normalized: Tensor | None = None,
    ) -> Tensor:
        """Cumulative product of alphas up to integer timestep(s).

        Exactly one of *t_int* or *t_normalized* must be provided.

        Parameters
        ----------
        t_int : Tensor, optional
            Integer timestep indices, shape ``(bs,)`` or ``(bs, 1)``.
        t_normalized : Tensor, optional
            Normalised time in ``[0, 1]``, converted to integer indices via
            rounding.

        Returns
        -------
        Tensor
            Alpha-bar values with the same shape as the input.
        """
        t_int = self._resolve_t(t_int, t_normalized=t_normalized)
        flat, shape = self._resolve_index(t_int)
        return self.alpha_bar[flat].reshape(shape)

    def _resolve_t(
        self,
        t_int: Tensor | None,
        *,
        t_normalized: Tensor | None,
    ) -> Tensor:
        """Resolve exactly one of *t_int* / *t_normalized* to integer indices.

        Parameters
        ----------
        t_int : Tensor, optional
            Integer timestep indices.
        t_normalized : Tensor, optional
            Normalised time in ``[0, 1]``.

        Returns
        -------
        Tensor
            Integer timestep indices.

        Raises
        ------
        ValueError
            If both or neither argument is provided.
        """
        if (t_int is None) == (t_normalized is None):
            raise ValueError("Exactly one of t_int or t_normalized must be provided.")
        if t_int is None:
            if t_normalized is None:
                raise ValueError(
                    "Either t_int or t_normalized must be provided, got neither."
                )
            if (t_normalized < 0.0).any() or (t_normalized > 1.0).any():
                raise ValueError(
                    f"t_normalized must be in [0, 1], got values in "
                    f"[{t_normalized.min().item():.4f}, {t_normalized.max().item():.4f}]"
                )
            t_int = torch.round(t_normalized * self._timesteps)
        return t_int

    @override
    def forward(
        self,
        t_int: Tensor | None = None,
        *,
        t_normalized: Tensor | None = None,
    ) -> Tensor:
        """Return beta values at given timesteps.

        Exactly one of *t_int* or *t_normalized* must be provided.
        """
        return self.get_beta(t_int, t_normalized=t_normalized)

    def get_noise_level(self, t_int: Tensor) -> Tensor:
        """Noise level at integer timestep(s), defined as ``1 - alpha_bar(t)``.

        The formula is the same for all schedule types. For the
        ``linear_ddpm`` schedule alpha_bar is linear in *t*, so the noise
        level is equivalently ``t / T``; for cosine and custom schedules
        the curve is non-linear.

        Parameters
        ----------
        t_int
            Integer timestep indices, shape ``(bs,)`` or ``(bs, 1)``.

        Returns
        -------
        Tensor
            Noise levels with the same shape as *t_int*.
        """
        return 1.0 - self.get_alpha_bar(t_int)

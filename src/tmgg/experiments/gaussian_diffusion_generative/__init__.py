"""Generative graph modeling experiment with MMD evaluation.

This module provides a complete pipeline for training generative graph models
using discrete denoising diffusion and evaluating them with MMD metrics on
graph-theoretic properties (degree distribution, clustering, spectral).

``DiffusionModule`` from the shared utilities package is the training loop.
"""

from tmgg.training.lightning_modules.diffusion_module import (
    DiffusionModule,
)

__all__ = [
    "DiffusionModule",
]

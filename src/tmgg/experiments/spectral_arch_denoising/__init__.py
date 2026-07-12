"""Spectral denoising experiment module.

Provides a unified experiment interface for training spectral graph denoising
architectures: Linear PE, Graph Filter Bank, and Self-Attention denoisers.

Usage
-----
Run via Hydra:
    uv run tmgg-spectral-arch
"""

from tmgg.training.lightning_modules.denoising_module import (
    SingleStepDenoisingModule,
)

__all__ = ["SingleStepDenoisingModule"]

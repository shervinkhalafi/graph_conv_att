"""Diffusion components for discrete and continuous graph generation.

The pipeline composes three kinds of object:

- **NoiseSchedule** precomputes ``beta``, ``alpha``, and ``alpha_bar`` tensors
  that control how quickly signal decays over ``T`` diffusion steps.
- **NoiseProcess** (categorical or continuous) applies forward noise and
  computes VLB terms. Categorical processes own their stationary PMFs and
  reverse-process math directly.
- **Sampler** runs the reverse diffusion loop, denoising from the limit
  distribution back to clean graphs.

X_class / E_class synthesis convention: see
``internal notes (not included)``.
"""

from .collectors import DiffusionLikelihoodCollector, StepMetricCollector
from .noise_process import (
    CategoricalNoiseProcess,
    ContinuousNoiseProcess,
    ExactDensityNoiseProcess,
    GaussianNoiseProcess,
    NoisedBatch,
    NoiseProcess,
)
from .sampler import (
    CategoricalSampler,
    ContinuousSampler,
    DiffusionState,
    Sampler,
)
from .schedule import NoiseSchedule

__all__ = [
    "CategoricalNoiseProcess",
    "CategoricalSampler",
    "ContinuousNoiseProcess",
    "ContinuousSampler",
    "DiffusionLikelihoodCollector",
    "DiffusionState",
    "ExactDensityNoiseProcess",
    "GaussianNoiseProcess",
    "NoisedBatch",
    "NoiseProcess",
    "NoiseSchedule",
    "Sampler",
    "StepMetricCollector",
]

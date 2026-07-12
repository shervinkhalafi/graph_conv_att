"""Baseline denoising models for sanity checking training pipelines."""

from tmgg.models.baselines.linear import LinearBaseline
from tmgg.models.baselines.mlp import MLPBaseline

__all__ = ["LinearBaseline", "MLPBaseline"]

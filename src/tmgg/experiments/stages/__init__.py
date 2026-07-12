"""Unified experiment runner for all experimental stages.

Provides a single CLI entry point (tmgg-experiment) that accepts stage
configuration via Hydra's config groups:

    tmgg-experiment +stage=stage1_poc
    tmgg-experiment +stage=stage2_validation

Stages:
- stage1_poc: Proof of Concept
- stage1_sanity: Sanity check with constant noise
- stage2_validation: Cross-Dataset Validation
- stage3_diversity: Dataset Diversity (future)
- stage4_benchmarks: Real Benchmarks (future)
- stage5_full: Full Validation (future)
"""

from tmgg.experiments.stages.runner import main

__all__ = ["main"]

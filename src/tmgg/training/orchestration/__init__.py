"""Experiment execution pipeline.

Wires Hydra configuration to data modules, models, Lightning trainers,
callbacks, sanity checks, and checkpoint resumption.
"""

from tmgg.training.orchestration.run_experiment import (
    generate_run_id,
    run_experiment,
)

__all__ = [
    "generate_run_id",
    "run_experiment",
]

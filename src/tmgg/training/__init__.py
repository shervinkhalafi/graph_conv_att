"""Training infrastructure for graph denoising experiments.

Re-exports the core Lightning module abstractions and orchestration
utilities consumed by experiment ``__init__.py`` modules. Evaluation
metrics live in ``tmgg.evaluation``.
"""

from .lightning_modules.base_graph_module import BaseGraphModule
from .lightning_modules.denoising_module import SingleStepDenoisingModule
from .lightning_modules.diffusion_module import DiffusionModule
from .orchestration.run_experiment import generate_run_id, run_experiment

__all__ = [
    "BaseGraphModule",
    "DiffusionModule",
    "SingleStepDenoisingModule",
    "generate_run_id",
    "run_experiment",
]

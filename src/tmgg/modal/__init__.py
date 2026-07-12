"""Modal cloud execution for TMGG spectral denoising experiments.

This package provides Modal-specific implementations for running
TMGG experiments on cloud GPUs. The Modal function acts as a pure
CLI transport: it receives a command name, YAML config, and run ID,
then executes the corresponding entry point inside the container.

No ``import modal`` at package level. The ``modal.App`` and all
``@app.function`` decorators live in ``_functions.py``, the sole
deployment entry-point. Runtime code uses ``modal.Function.from_name()``
lazily inside method bodies.
"""

from tmgg.modal.app import GPU_CONFIGS, MODAL_APP_NAME
from tmgg.modal.runner import (
    ExperimentResult,
    ModalNotDeployedError,
    ModalRunner,
    ModalSpawnedTask,
    check_modal_deployment,
    create_runner,
)

__all__ = [
    "ExperimentResult",
    "GPU_CONFIGS",
    "MODAL_APP_NAME",
    "ModalNotDeployedError",
    "ModalRunner",
    "ModalSpawnedTask",
    "check_modal_deployment",
    "create_runner",
]

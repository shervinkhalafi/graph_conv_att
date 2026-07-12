"""Modal experiment runner.

Provides ModalRunner that executes TMGG experiments on Modal GPUs.
No ``import modal`` at module level — all Modal SDK calls happen
lazily inside method bodies. Decorated ``@app.function`` wrappers
live in ``_functions.py``; this module is pure runtime logic.

Architecture
------------
ModalRunner serializes the Hydra config to YAML and passes it to one of the
``modal_run_cli*`` functions, which write the YAML to a temp dir and run the
CLI entry point as a subprocess inside the container. The runner does not
handle storage or checkpoint uploads; those concerns live in the CLI or
in post-run analysis scripts.

Execution modes:
- Blocking: ``run_experiment()``, ``run_sweep()`` — wait for results
- Detached: ``spawn_experiment()``, ``spawn_sweep()`` — fire-and-forget
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from omegaconf import DictConfig, OmegaConf

from tmgg.modal.app import MODAL_APP_NAME, resolve_modal_function_name

if TYPE_CHECKING:
    from modal.functions import FunctionCall

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result from a single experiment run.

    Attributes
    ----------
    run_id
        Unique identifier for this run.
    config
        The configuration used for this run.
    metrics
        Final metrics from training (val_loss, test_loss, etc.).
    checkpoint_path
        Path to the best model checkpoint (local or remote).
    status
        Completion status: 'completed', 'failed', or 'timeout'.
    error_message
        Error details if status is 'failed'.
    duration_seconds
        Wall-clock time for the experiment.
    """

    run_id: str
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    status: str = "completed"
    error_message: str | None = None
    duration_seconds: float = 0.0


class ModalNotDeployedError(Exception):
    """Raised when the Modal app is not deployed."""


def check_modal_deployment() -> None:
    """Verify that the graph-denoising Modal app is deployed.

    Checks whether the Modal functions are accessible, which requires
    the app to have been deployed via ``modal deploy``. Raises
    ``ModalNotDeployedError`` with instructions if not.

    Raises
    ------
    ModalNotDeployedError
        If the Modal app is not deployed or functions are not accessible.
    """
    import modal

    try:
        fn = modal.Function.from_name(MODAL_APP_NAME, "modal_run_cli")
        fn.hydrate()
    except modal.exception.NotFoundError as e:
        raise ModalNotDeployedError(
            f"Modal app '{MODAL_APP_NAME}' is not deployed. "
            + "Run 'mise run modal-deploy' or "
            + "'uv run modal deploy -m tmgg.modal._functions' first.\n"
            + f"Original error: {e}"
        ) from e
    except Exception as e:
        raise ModalNotDeployedError(
            f"Failed to verify Modal deployment for app '{MODAL_APP_NAME}': {e}"
        ) from e


@dataclass
class ModalSpawnedTask:
    """Handle for a spawned Modal task.

    Provides the run_id for tracking and the FunctionCall handle for
    polling.

    Attributes
    ----------
    run_id
        Unique identifier for the spawned experiment.
    gpu_tier
        GPU tier used for execution.
    function_call
        Modal FunctionCall handle for polling (optional, may not be available
        after process restart).
    """

    run_id: str
    gpu_tier: str
    function_call: FunctionCall[dict[str, Any]] | None = None


class ModalRunner:
    """Modal-specific experiment runner.

    Serializes the Hydra config to YAML and dispatches it to one of the
    ``modal_run_cli*`` functions. The remote function writes the YAML to
    a temp dir and runs the CLI entry point as a subprocess.

    Requires the Modal app to be pre-deployed. If not deployed,
    instantiation will fail with ModalNotDeployedError.

    Execution Modes
    ---------------
    - **Blocking**: ``run_experiment()`` / ``run_sweep()`` wait for results
    - **Detached**: ``spawn_experiment()`` / ``spawn_sweep()`` return immediately
    """

    gpu_type: str

    def __init__(
        self,
        gpu_type: str = "debug",
        skip_deployment_check: bool = False,
    ):
        """Initialize Modal runner.

        Parameters
        ----------
        gpu_type
            Default GPU tier for experiments.
        skip_deployment_check
            If True, skip the deployment verification. Only use for testing
            or when you're certain the app is deployed.

        Raises
        ------
        ModalNotDeployedError
            If the Modal app is not deployed and skip_deployment_check is False.
        """
        if not skip_deployment_check:
            check_modal_deployment()

        self.gpu_type = gpu_type
        self._active_runs: dict[str, ModalSpawnedTask] = {}

    def _select_modal_function(self, gpu_tier: str) -> Any:
        """Select the appropriate Modal function for the GPU tier.

        Uses ``modal.Function.from_name()`` to get references to deployed
        functions, which is required when calling from outside Modal
        (i.e., from local machine).

        Tier mapping:
        - debug -> T4 (modal_run_cli_debug)
        - standard -> A10G (modal_run_cli)
        - fast, multi, h100 -> A100 (modal_run_cli_fast)
        """
        import modal

        func_name = resolve_modal_function_name("modal_run_cli", gpu_tier)
        return modal.Function.from_name(MODAL_APP_NAME, func_name)

    # =========================================================================
    # Spawn Methods (Detached Execution)
    # =========================================================================

    def spawn_experiment(
        self,
        config: DictConfig,
        gpu_type: str | None = None,
        timeout_seconds: int | None = None,
        additional_tags: list[str] | None = None,
    ) -> ModalSpawnedTask:
        """Spawn a single experiment without waiting for results.

        Uses Modal's ``.spawn()`` for fire-and-forget execution. Results
        are tracked via the confirmation log on the Modal volume.

        Parameters
        ----------
        config
            Hydra configuration.
        gpu_type
            GPU tier override.
        timeout_seconds
            Timeout override (unused, retained for interface compatibility).
        additional_tags
            Extra W&B tags (unused, retained for interface compatibility).

        Returns
        -------
        ModalSpawnedTask
            Handle with run_id for tracking.
        """
        _ = timeout_seconds, additional_tags  # reserved for future use
        gpu = gpu_type or self.gpu_type
        run_id = str(config.get("run_id", str(uuid.uuid4())[:8]))
        resolved_cmd = str(config.get("_cli_cmd", "tmgg-discrete-gen"))
        config_yaml = OmegaConf.to_yaml(config, resolve=True)

        modal_fn = self._select_modal_function(gpu)
        function_call = modal_fn.spawn(resolved_cmd, config_yaml, run_id)

        spawned = ModalSpawnedTask(
            run_id=run_id,
            gpu_tier=gpu,
            function_call=function_call,
        )
        self._active_runs[run_id] = spawned

        # ``object_id`` is the ``fc-...`` token that ``internal sweep tooling (not included)``
        # accepts for direct cancel. Logging it here gives operators a
        # second source of truth alongside the CLI's stdout marker.
        fc_id = getattr(function_call, "object_id", None)
        logger.info(
            f"Spawned experiment {run_id} on {gpu} GPU (detached); "
            f"function_call_id={fc_id}"
        )
        return spawned

    def spawn_sweep(
        self,
        configs: list[DictConfig],
        gpu_type: str | None = None,
        timeout_seconds: int | None = None,
        additional_tags: list[str] | None = None,
    ) -> list[ModalSpawnedTask]:
        """Spawn multiple experiments without waiting for results.

        Each experiment is spawned independently using ``.spawn()``.

        Parameters
        ----------
        configs
            List of configurations.
        gpu_type
            GPU tier for all experiments.
        timeout_seconds
            Timeout per experiment (unused, retained for interface compatibility).
        additional_tags
            Extra W&B tags (unused, retained for interface compatibility).

        Returns
        -------
        list[ModalSpawnedTask]
            Handles with run_ids for tracking.
        """
        gpu = gpu_type or self.gpu_type
        spawned_tasks: list[ModalSpawnedTask] = []

        for config in configs:
            task = self.spawn_experiment(config, gpu, timeout_seconds, additional_tags)
            spawned_tasks.append(task)

        logger.info(f"Spawned {len(spawned_tasks)} experiments on {gpu} GPU (detached)")
        return spawned_tasks

    # =========================================================================
    # Blocking Methods (Wait for Results)
    # =========================================================================

    def run_experiment(
        self,
        config: DictConfig,
        gpu_type: str | None = None,
        timeout_seconds: int | None = None,
        additional_tags: list[str] | None = None,
    ) -> ExperimentResult:
        """Run a single experiment and wait for results.

        Parameters
        ----------
        config
            Hydra configuration.
        gpu_type
            GPU tier override.
        timeout_seconds
            Timeout override (unused, retained for interface compatibility).
        additional_tags
            Extra W&B tags (unused, retained for interface compatibility).

        Returns
        -------
        ExperimentResult
            Result of the experiment.
        """
        _ = timeout_seconds, additional_tags  # reserved for future use
        gpu = gpu_type or self.gpu_type
        run_id = str(config.get("run_id", str(uuid.uuid4())[:8]))
        resolved_cmd = str(config.get("_cli_cmd", "tmgg-discrete-gen"))
        config_yaml = OmegaConf.to_yaml(config, resolve=True)

        modal_fn = self._select_modal_function(gpu)
        result_dict: dict[str, Any] = modal_fn.remote(resolved_cmd, config_yaml, run_id)

        config_container = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config_container, dict):
            raise TypeError(
                f"Expected dict from OmegaConf.to_container, got {type(config_container)}"
            )

        return ExperimentResult(
            run_id=run_id,
            config=cast(dict[str, Any], config_container),
            metrics=result_dict.get("metrics", {}),
            status=result_dict.get("status", "unknown"),
            error_message=result_dict.get("error"),
            duration_seconds=0.0,
        )

    def run_sweep(
        self,
        configs: list[DictConfig],
        gpu_type: str | None = None,
        timeout_seconds: int | None = None,
        additional_tags: list[str] | None = None,
    ) -> list[ExperimentResult]:
        """Run multiple experiments in parallel and wait for all results.

        Uses ``modal_fn.map()`` for parallel remote execution. Each config
        is serialized to YAML independently.

        Parameters
        ----------
        configs
            List of configurations.
        gpu_type
            GPU tier for all experiments.
        timeout_seconds
            Timeout per experiment (unused, retained for interface compatibility).
        additional_tags
            Extra W&B tags (unused, retained for interface compatibility).

        Returns
        -------
        list[ExperimentResult]
            Results from all experiments.
        """
        _ = timeout_seconds, additional_tags  # reserved for future use
        gpu = gpu_type or self.gpu_type
        modal_fn = self._select_modal_function(gpu)

        # Prepare args for each config
        call_args: list[tuple[str, str, str]] = []
        config_containers: list[dict[str, Any]] = []
        for config in configs:
            run_id = str(config.get("run_id", str(uuid.uuid4())[:8]))
            resolved_cmd = str(config.get("_cli_cmd", "tmgg-discrete-gen"))
            config_yaml = OmegaConf.to_yaml(config, resolve=True)
            call_args.append((resolved_cmd, config_yaml, run_id))

            container = OmegaConf.to_container(config, resolve=True)
            if not isinstance(container, dict):
                raise TypeError(
                    f"Expected dict from OmegaConf.to_container, got {type(container)}"
                )
            config_containers.append(cast(dict[str, Any], container))

        # modal_fn.starmap expects an iterable of arg tuples
        result_dicts = list(modal_fn.starmap(call_args))

        results: list[ExperimentResult] = []
        for rd, (_cmd, _, run_id), container in zip(
            result_dicts, call_args, config_containers, strict=True
        ):
            results.append(
                ExperimentResult(
                    run_id=run_id,
                    config=container,
                    metrics=rd.get("metrics", {}),
                    status=rd.get("status", "unknown"),
                    error_message=rd.get("error"),
                    duration_seconds=0.0,
                )
            )
        return results

    # =========================================================================
    # Status and Control
    # =========================================================================

    def get_status(self, run_id: str) -> str:
        """Get status of an experiment from local tracking.

        Parameters
        ----------
        run_id
            Run identifier.

        Returns
        -------
        str
            ``"running"`` if tracked locally, ``"unknown"`` otherwise.
        """
        if run_id in self._active_runs:
            return "running"
        return "unknown"

    def cancel(self, run_id: str) -> bool:
        """Cancel a running experiment.

        Modal doesn't support direct cancellation of spawned functions.
        This removes the local tracking but the remote task continues.

        Parameters
        ----------
        run_id
            Run identifier.

        Returns
        -------
        bool
            False (cancellation not supported).
        """
        if run_id in self._active_runs:
            del self._active_runs[run_id]
        return False


def create_runner(gpu_type: str = "debug") -> ModalRunner:
    """Factory function to create a ModalRunner.

    Parameters
    ----------
    gpu_type
        Default GPU tier.

    Returns
    -------
    ModalRunner
        Configured runner instance.
    """
    return ModalRunner(gpu_type=gpu_type)

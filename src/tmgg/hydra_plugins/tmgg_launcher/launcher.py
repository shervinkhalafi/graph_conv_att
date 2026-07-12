"""Custom Hydra launcher that dispatches experiments to Modal.

Replaces the hacky sys.argv injection pattern with proper Hydra integration.
For local sweeps use Hydra's built-in launcher.
"""

from collections.abc import Sequence
from typing import Any, cast, override

from hydra.core.utils import JobReturn, JobStatus
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf, open_dict

from tmgg.modal.runner import ExperimentResult, ModalRunner


class TmggLauncher(Launcher):
    """Custom Hydra launcher for TMGG experiments.

    Dispatches jobs to ModalRunner. For local sweeps, use Hydra's built-in
    launcher (``hydra/launcher=basic``).
    """

    def __init__(self) -> None:
        """Initialize launcher state."""
        self._runner: ModalRunner | None = None
        self._config: DictConfig | None = None
        self._hydra_context: HydraContext | None = None

    @override
    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """Configure launcher with Hydra context and select runner.

        Parameters
        ----------
        hydra_context
            Hydra's context containing config loader and other state.
        task_function
            The decorated main function (unused, we call ModalRunner directly).
        config
            The resolved Hydra configuration.
        """
        self._hydra_context = hydra_context
        self._config = config
        self._runner = self._select_runner(config)

    @override
    def launch(
        self,
        job_overrides: Sequence[Sequence[str]],
        initial_job_idx: int,
    ) -> Sequence[JobReturn]:
        """Execute jobs via ModalRunner.

        Collects all resolved configs and dispatches them as a single batch
        via ``run_sweep()``, which parallelizes via Modal's ``starmap``.

        Parameters
        ----------
        job_overrides
            List of override sequences, one per job.
        initial_job_idx
            Starting index for job numbering.

        Returns
        -------
        Sequence[JobReturn]
            Hydra JobReturn objects for each executed job.
        """
        if self._runner is None or self._config is None:
            raise RuntimeError("Launcher.setup() must be called before launch()")

        launcher_cfg = self._get_launcher_config(self._config)
        gpu_type = launcher_cfg.get("gpu_type", "debug")

        # Collect all resolved configs upfront
        configs: list[DictConfig] = []
        for overrides in job_overrides:
            configs.append(self._apply_overrides(self._config, list(overrides)))

        # Patch output paths to write to the Modal volume (consistent with
        # tmgg-modal run's resolve_config() logic)
        self._patch_modal_paths(configs)

        # Dispatch as single batch — ModalRunner parallelizes via starmap
        experiment_results = self._runner.run_sweep(
            configs,
            gpu_type=gpu_type,
        )

        # Convert to Hydra JobReturn objects
        results: list[JobReturn] = []
        for idx, (er, overrides) in enumerate(
            zip(experiment_results, job_overrides, strict=True)
        ):
            results.append(
                self._result_to_job_return(
                    er, overrides=list(overrides), idx=initial_job_idx + idx
                )
            )
        return results

    def _select_runner(self, config: DictConfig) -> ModalRunner:
        """Create ModalRunner from launcher configuration.

        Parameters
        ----------
        config
            Hydra configuration containing launcher settings.

        Returns
        -------
        ModalRunner
            Configured runner instance.

        Raises
        ------
        RuntimeError
            If ``use_modal`` is not enabled. For local sweeps use Hydra's
            built-in launcher.
        """
        launcher_cfg = self._get_launcher_config(config)

        if not launcher_cfg.get("use_modal", False):
            raise RuntimeError(
                "TmggLauncher requires use_modal=True. "
                "For local sweeps, use Hydra's built-in launcher: "
                "--multirun hydra/launcher=basic"
            )
        gpu_type = launcher_cfg.get("gpu_type", "debug")
        return ModalRunner(gpu_type=gpu_type)

    def _get_launcher_config(self, config: DictConfig) -> dict[str, Any]:
        """Extract launcher configuration from Hydra config.

        Parameters
        ----------
        config
            Full Hydra configuration.

        Returns
        -------
        dict
            Launcher-specific settings (use_modal, use_ray, gpu_type, etc.).
        """
        hydra_cfg = OmegaConf.to_container(config.get("hydra", {}), resolve=True)
        if not isinstance(hydra_cfg, dict):
            return {}
        return cast(dict[str, Any], hydra_cfg.get("launcher", {}))

    def _apply_overrides(
        self, base_config: DictConfig, overrides: list[str]
    ) -> DictConfig:
        """Apply job-specific overrides to base configuration.

        Parameters
        ----------
        base_config
            Base configuration from setup().
        overrides
            Job-specific override strings.

        Returns
        -------
        DictConfig
            New configuration with overrides applied.
        """
        container = OmegaConf.to_container(base_config, resolve=True)
        cfg = cast(DictConfig, OmegaConf.create(container))

        for ovr in overrides:
            if "=" not in ovr:
                continue
            key, value = ovr.split("=", 1)
            key = key.lstrip("+")  # Handle Hydra append syntax
            OmegaConf.update(cfg, key, value)

        return cfg

    @staticmethod
    def _patch_modal_paths(configs: list[DictConfig]) -> None:
        """Rewrite ``paths.output_dir`` and ``paths.results_dir`` to Modal volume.

        Mirrors the path-patching logic in ``config_resolution.resolve_config()``
        so that multirun-on-Modal writes to the persistent volume rather than
        ephemeral container storage.
        """
        from tmgg.modal._lib.volumes import OUTPUTS_MOUNT
        from tmgg.training.orchestration.run_experiment import (
            generate_run_id,
        )

        for cfg in configs:
            run_id = generate_run_id(cfg)
            experiment_name = cfg.get("experiment_name", "tmgg_training")
            output_dir = f"{OUTPUTS_MOUNT}/{experiment_name}/{run_id}"
            with open_dict(cfg):
                cfg.run_id = run_id
                cfg.paths.output_dir = output_dir
                cfg.paths.results_dir = f"{output_dir}/results"

    @staticmethod
    def _result_to_job_return(
        result: ExperimentResult,
        overrides: list[str],
        idx: int,
    ) -> JobReturn:
        """Convert ExperimentResult to Hydra JobReturn.

        Parameters
        ----------
        result
            Experiment result from ModalRunner.
        overrides
            The overrides used for this job.
        idx
            Job index (stored in return value for tracking).

        Returns
        -------
        JobReturn
            Hydra-compatible job result.
        """
        if result.status == "completed":
            status = JobStatus.COMPLETED
        else:
            status = JobStatus.FAILED

        # Include idx in return value for tracking
        return_value = {**result.metrics, "_job_idx": idx}

        return JobReturn(
            _return_value=return_value,
            overrides=overrides,
            cfg=OmegaConf.create(result.config),
            hydra_cfg=None,
            working_dir=None,
            task_name=None,
            status=status,
        )

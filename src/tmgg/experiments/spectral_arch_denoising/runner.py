"""Hydra entry point for spectral denoising experiments.

Run experiments with:
    uv run tmgg-spectral-arch

Override configuration with Hydra:
    uv run tmgg-spectral-arch model.k=16 data.num_nodes=100
"""

from typing import Any

import hydra
from omegaconf import DictConfig

from tmgg.training.orchestration.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_spectral_arch",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run spectral denoising experiment.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing model, data, trainer, and logger settings.
    """
    return run_experiment(cfg)


if __name__ == "__main__":
    main()

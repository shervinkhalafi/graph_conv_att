"""Hydra entry point for generative graph modeling experiments.

Run experiments with:
    uv run tmgg-gaussian-gen

Override configuration with Hydra:
    uv run tmgg-gaussian-gen model.model_name=gnn data.num_nodes=100
"""

from typing import Any

import hydra
from omegaconf import DictConfig

from tmgg.training.orchestration.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_gaussian_diffusion",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run generative graph modeling experiment.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing model, data, trainer, and logger settings.
    """
    return run_experiment(cfg)


if __name__ == "__main__":
    main()

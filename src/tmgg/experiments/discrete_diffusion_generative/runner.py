"""Hydra entry point for discrete diffusion graph generation experiments.

Run experiments with::

    tmgg-discrete-gen
    tmgg-discrete-gen model.diffusion_steps=100 data.num_nodes=12
"""

from typing import Any

import hydra
from omegaconf import DictConfig

from tmgg.training.orchestration.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_discrete_diffusion_generative",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run discrete diffusion graph generation experiment.

    Parameters
    ----------
    cfg
        Hydra configuration containing model, data, trainer, and logger settings.
    """
    return run_experiment(cfg)


if __name__ == "__main__":
    main()

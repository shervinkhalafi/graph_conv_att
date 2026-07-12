"""Hydra entry point for DiGress Transformer-based denoising experiments.

Run experiments with::

    tmgg-digress
    tmgg-digress model.n_layers=6 data.num_nodes=100
"""

from typing import Any

import hydra
from omegaconf import DictConfig

from tmgg.training.orchestration.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_digress",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run DiGress denoising experiment.

    Parameters
    ----------
    cfg
        Hydra configuration containing model, data, trainer, and logger settings.
    """
    return run_experiment(cfg)


if __name__ == "__main__":
    main()

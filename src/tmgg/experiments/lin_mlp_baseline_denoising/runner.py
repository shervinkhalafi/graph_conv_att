"""Hydra entry point for baseline denoising experiments.

Run experiments with::

    tmgg-baseline
    tmgg-baseline model=baselines/mlp model.hidden_dim=512
"""

from typing import Any

import hydra
from omegaconf import DictConfig

from tmgg.training.orchestration.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_baseline",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run baseline denoising experiment.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing model, data, trainer, and logger settings.
    """
    return run_experiment(cfg)


if __name__ == "__main__":
    main()

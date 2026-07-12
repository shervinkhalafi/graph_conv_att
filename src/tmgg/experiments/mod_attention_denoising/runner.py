# src/tmgg/experiments/my_experiment/runner.py
from typing import Any

import hydra
from omegaconf import DictConfig

from tmgg.training.orchestration.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_mod_attention",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run modified attention denoising experiment."""
    return run_experiment(cfg)


if __name__ == "__main__":
    main()

"""{{ cookiecutter.experiment_name }} experiment runner."""

from typing import Any

import hydra
from omegaconf import DictConfig

from tmgg.training.orchestration.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_{{ cookiecutter.experiment_slug }}",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run {{ cookiecutter.experiment_name }} experiment."""
    return run_experiment(cfg)


if __name__ == "__main__":
    main()

"""Runner for grid search experiments."""

import hydra
from omegaconf import DictConfig

from tmgg.training.orchestration.run_experiment import run_experiment


@hydra.main(  # pyright: ignore[reportAny]
    version_base=None, config_path="exp_configs", config_name="grid_search_base"
)
def main(config: DictConfig):
    """
    Main training function for grid search.

    Args:
        config: Hydra configuration

    Returns:
        Dictionary with training results
    """
    return run_experiment(config)


if __name__ == "__main__":
    main()

"""Hydra entry point for eigenstructure study experiments.

Run experiments with::

    tmgg-eigenstructure-exp
    tmgg-eigenstructure-exp phase=analyze dataset.name=er
"""

import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_eigenstructure",
)
def main(cfg: DictConfig) -> None:
    """Run eigenstructure study phase.

    Parameters
    ----------
    cfg
        Hydra configuration specifying phase, dataset, and analysis settings.
    """
    from tmgg.experiments.eigenstructure_study.execute import execute_eigenstructure

    execute_eigenstructure(cfg)


if __name__ == "__main__":
    main()

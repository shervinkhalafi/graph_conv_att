"""Unified CLI entry point for TMGG experiments.

Runs a single experiment, or use ``--multirun`` with the custom
``TmggLauncher`` to sweep over configurations in parallel:

    tmgg-experiment model=models/spectral/filter_bank

    tmgg-experiment --multirun \\
        hydra/launcher=tmgg_modal \\
        model=models/spectral/linear_pe,models/spectral/filter_bank
"""

from typing import Any

import hydra
from omegaconf import DictConfig

from tmgg.training.orchestration.run_experiment import run_experiment

# Config path relative to this file — resolves to src/tmgg/experiments/exp_configs/
_CONFIG_PATH = "../exp_configs"


@hydra.main(
    version_base=None,
    config_path=_CONFIG_PATH,
    config_name="base_config_denoising",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run a single experiment, or a sweep via --multirun.

    Usage
    -----
    Single experiment::

        tmgg-experiment model=models/spectral/filter_bank

    Sweep via Modal::

        tmgg-experiment --multirun \\
            hydra/launcher=tmgg_modal \\
            model=models/spectral/linear_pe,models/spectral/filter_bank
    """
    return run_experiment(cfg)


if __name__ == "__main__":
    main()

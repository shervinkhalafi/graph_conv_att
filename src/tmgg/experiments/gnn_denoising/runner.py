"""Hydra entry point for GNN-based denoising experiments.

Run experiments with::

    tmgg-gnn
    tmgg-gnn model.gnn_num_layers=4 data.num_nodes=100
"""

from typing import Any

import hydra
from omegaconf import DictConfig

from tmgg.training.orchestration.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_gnn",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run GNN denoising experiment.

    Parameters
    ----------
    cfg
        Hydra configuration containing model, data, trainer, and logger settings.
    """
    return run_experiment(cfg)


if __name__ == "__main__":
    main()

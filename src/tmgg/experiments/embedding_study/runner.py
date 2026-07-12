"""Hydra entry point for embedding dimension study experiments.

Run experiments with::

    tmgg-embedding-study-exp
    tmgg-embedding-study-exp phase=analyze datasets=[sbm]
"""

import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_embedding_study",
)
def main(cfg: DictConfig) -> None:
    """Run embedding dimension study phase.

    Parameters
    ----------
    cfg
        Hydra configuration specifying phase, datasets, and search settings.
    """
    from tmgg.experiments.embedding_study.execute import execute_embedding_study

    execute_embedding_study(cfg)


if __name__ == "__main__":
    main()

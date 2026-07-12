"""TMGG Hydra Launcher Plugin.

Custom launcher that dispatches experiments to the ModalRunner backend.

Usage
-----
Via Hydra config override::

    tmgg-experiment +stage=stage1_poc hydra/launcher=tmgg

Via Hydra multirun::

    tmgg-experiment +stage=stage1_poc --multirun \\
        hydra/launcher=tmgg \\
        model=models/spectral/linear_pe,models/spectral/filter_bank

With Modal backend::

    tmgg-experiment +stage=stage1_poc --multirun \\
        hydra/launcher=tmgg_modal \\
        model=models/spectral/linear_pe,models/spectral/filter_bank
"""

# Import config module to trigger ConfigStore registration
from tmgg.hydra_plugins.tmgg_launcher import config as _config  # noqa: F401
from tmgg.hydra_plugins.tmgg_launcher.config import TmggLauncherConf
from tmgg.hydra_plugins.tmgg_launcher.launcher import TmggLauncher

__all__ = ["TmggLauncher", "TmggLauncherConf"]

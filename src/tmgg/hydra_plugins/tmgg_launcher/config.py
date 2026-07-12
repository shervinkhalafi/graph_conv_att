"""Configuration dataclass for TmggLauncher.

Defines the configuration schema and registers it with Hydra's ConfigStore.
"""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class TmggLauncherConf:
    """Configuration for the TMGG experiment launcher.

    Attributes
    ----------
    _target_
        Fully qualified class name for Hydra instantiation.
    use_modal
        If True, dispatch jobs to Modal cloud. Default False (local).
    gpu_type
        GPU tier to request ('debug', 'standard', 'fast', 'multi').
    timeout_seconds
        Maximum runtime per experiment before termination.
    """

    _target_: str = field(
        default="tmgg.hydra_plugins.tmgg_launcher.TmggLauncher",
        metadata={"omegaconf_ignore": True},
    )
    use_modal: bool = False
    gpu_type: str = "debug"
    timeout_seconds: int = 3600


def register_configs() -> None:
    """Register TmggLauncher configuration with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(
        group="hydra/launcher",
        name="tmgg",
        node=TmggLauncherConf,
    )


# Auto-register on import
register_configs()

"""Hydra config resolution for Modal dispatch.

Provides two capabilities:

1. **CLI command discovery** — scans ``exp_configs/`` YAML files to build a
   mapping from CLI entry point names (e.g. ``tmgg-spectral-arch``) to their
   Hydra config names (e.g. ``base_config_spectral_arch``).

2. **Config composition** — uses Hydra's ``compose`` API to resolve a full
   config from a base config name and override list, then patches paths for
   Modal volume mounts.  The result is a ``DictConfig`` ready to pass to
   ``ModalRunner.run_experiment()`` / ``.spawn_experiment()``.

The compose path is the single-run counterpart to the ``TmggLauncher``
multirun path.  Both produce configs that ``ModalRunner`` serializes via
``OmegaConf.to_yaml(config, resolve=True)`` — the shared serialization
boundary.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from omegaconf import DictConfig, open_dict

from tmgg.modal._lib.volumes import OUTPUTS_MOUNT

logger = logging.getLogger(__name__)

# exp_configs/ relative to this file:
#   _lib/config_resolution.py -> _lib/ -> modal/ -> tmgg/ -> experiments/exp_configs/
_EXP_CONFIGS_DIR = Path(__file__).parent.parent.parent / "experiments" / "exp_configs"


def get_exp_configs_dir() -> Path:
    """Return the absolute path to the Hydra experiment configs directory.

    Raises
    ------
    FileNotFoundError
        If the expected directory does not exist (broken install or
        non-standard layout).
    """
    if not _EXP_CONFIGS_DIR.is_dir():
        raise FileNotFoundError(
            f"Experiment configs directory not found: {_EXP_CONFIGS_DIR}"
        )
    return _EXP_CONFIGS_DIR


def discover_cli_cmd_map() -> dict[str, str]:
    """Scan ``exp_configs/base_config_*.yaml`` for ``_cli_cmd`` declarations.

    Each experiment-type config declares its CLI entry point via a top-level
    ``_cli_cmd`` key (e.g. ``_cli_cmd: tmgg-spectral-arch``).  This function
    reads that key from each file using plain YAML (not Hydra) to build the
    reverse mapping.

    Returns
    -------
    dict[str, str]
        Maps CLI command name to config name (without ``.yaml``), e.g.
        ``{"tmgg-spectral-arch": "base_config_spectral_arch", ...}``.

    Raises
    ------
    FileNotFoundError
        If the configs directory is missing.
    """
    configs_dir = get_exp_configs_dir()
    cmd_map: dict[str, str] = {}

    for yaml_path in sorted(configs_dir.glob("base_config_*.yaml")):
        try:
            with open(yaml_path) as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError:
            logger.warning("Skipping unparseable YAML: %s", yaml_path.name)
            continue

        if not isinstance(raw, dict):
            continue

        cli_cmd = raw.get("_cli_cmd")
        if cli_cmd:
            config_name = yaml_path.stem  # e.g. "base_config_spectral_arch"
            cmd_map[str(cli_cmd)] = config_name

    return cmd_map


def resolve_config(
    config_name: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Compose a Hydra config and patch paths for Modal execution.

    Uses Hydra's ``compose`` API to resolve the full config from a base config
    name and override list.  After composition, replaces
    ``paths.output_dir`` (which references the unresolvable
    ``${hydra:runtime.output_dir}``) with an explicit path on the Modal output
    volume, and generates a ``run_id`` if one isn't already present.

    Parameters
    ----------
    config_name
        Base config filename without extension, e.g.
        ``"base_config_spectral_arch"``.
    overrides
        Hydra-style override strings (e.g. ``["model.k=16", "seed=1"]``).

    Returns
    -------
    DictConfig
        Fully composed config with Modal-appropriate paths, ready for
        ``ModalRunner``.

    Raises
    ------
    hydra.errors.ConfigCompositionError
        If config composition fails (bad override, missing config group, etc.).
    FileNotFoundError
        If the configs directory is missing.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from tmgg.training.orchestration.run_experiment import (
        generate_run_id,
    )

    configs_dir = get_exp_configs_dir()

    # Clear any existing Hydra state (compose API requires a clean slate)
    GlobalHydra.instance().clear()

    try:
        with initialize_config_dir(config_dir=str(configs_dir), version_base=None):
            cfg = compose(
                config_name=config_name,
                overrides=overrides or [],
            )
    finally:
        # Ensure we don't leak Hydra state even on failure
        GlobalHydra.instance().clear()

    # Generate run_id from the composed config (reuses the existing logic
    # that builds human-readable IDs from experiment name, model, HPs, seed)
    run_id = generate_run_id(cfg)

    with open_dict(cfg):
        cfg.run_id = run_id

        # Override paths that reference ${hydra:runtime.output_dir}, which
        # can't be resolved outside @hydra.main.  Writing to the Modal
        # volume (/data/outputs) is preferable to container-local storage.
        experiment_name = cfg.get("experiment_name", "tmgg_training")
        output_dir = f"{OUTPUTS_MOUNT}/{experiment_name}/{run_id}"
        cfg.paths.output_dir = output_dir
        cfg.paths.results_dir = f"{output_dir}/results"

    return cfg

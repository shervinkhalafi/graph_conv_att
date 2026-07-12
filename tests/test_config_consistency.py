"""Config consistency tests to prevent silent misconfiguration.

Rationale
---------
Several model config YAML files hardcode scheduler parameters instead of
inheriting from the shared base config, and some don't propagate
eval_noise_levels/fixed_noise_seed. This makes experiments inconsistent
because changes to the base config don't reach configs that hardcode their
own values. These tests enforce that model configs delegate to the shared
base where appropriate.

Starting state: model configs under exp_configs/models/{gnn,hybrid,digress,baselines}
Invariants:
  - eval_noise_levels must appear in every model config (directly or via defaults)
  - scheduler_config must reference ${scheduler_config} (not define inline dicts)
    EXCEPT for digress configs that intentionally use `type: none`
"""

import glob
import os
from pathlib import Path

import yaml

_MODELS_ROOT = Path("src/tmgg/experiments/exp_configs/models")
_DATA_CONFIG_ROOT = Path("src/tmgg/experiments/exp_configs/data")
_STAGE_CONFIG_ROOT = Path("src/tmgg/experiments/exp_configs/stage")

_CONFIG_DIRS = [
    str(_MODELS_ROOT / "gnn/"),
    str(_MODELS_ROOT / "hybrid/"),
    str(_MODELS_ROOT / "digress/"),
    str(_MODELS_ROOT / "baselines/"),
]


def _resolve_with_defaults(path: str) -> dict:
    """Load a YAML config, merging one level of Hydra defaults.

    This is a simplified resolver: it reads the ``defaults`` list, loads
    each referenced YAML from the same directory, and shallow-merges them
    (child keys override base keys). Enough for flat-key consistency checks
    but does not handle Hydra's full override semantics.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    defaults = cfg.pop("defaults", None)
    if not defaults:
        return cfg

    config_dir = os.path.dirname(path)
    merged: dict = {}
    for entry in defaults:
        if isinstance(entry, str) and entry != "_self_":
            base_path = os.path.join(config_dir, entry + ".yaml")
            if os.path.exists(base_path):
                with open(base_path) as f:
                    base_cfg = yaml.safe_load(f) or {}
                merged.update(base_cfg)
    # _self_ means child overrides base (Hydra convention)
    merged.update(cfg)
    return merged


def test_all_denoising_model_configs_propagate_eval_noise():
    """All denoising model configs must pass eval_noise_levels through.

    Checks both direct keys and keys inherited via Hydra defaults.
    """
    missing = []
    for d in _CONFIG_DIRS:
        for path in glob.glob(f"{d}*.yaml"):
            cfg = _resolve_with_defaults(path)
            if cfg and "eval_noise_levels" not in cfg:
                missing.append(path)
    assert not missing, (
        "These model configs don't propagate eval_noise_levels:\n"
        + "\n".join(f"  {p}" for p in missing)
    )


def test_no_inline_scheduler_in_model_configs():
    """Model configs should reference ${scheduler_config}, not define inline.

    Checks resolved configs (including defaults) for inline scheduler dicts.
    """
    inline = []
    for d in _CONFIG_DIRS[:3]:  # gnn, hybrid, digress only
        for path in glob.glob(f"{d}*.yaml"):
            cfg = _resolve_with_defaults(path)
            if cfg and isinstance(cfg.get("scheduler_config"), dict):
                inline.append(path)
    assert not inline, (
        "These configs define scheduler inline instead of ${scheduler_config}:\n"
        + "\n".join(f"  {p}" for p in inline)
    )


def test_grid_search_config_does_not_define_dead_final_eval_noise_levels():
    """Grid search config should not carry unread evaluation keys.

    Rationale
    ---------
    ``evaluation.final_eval_noise_levels`` had become dead config: it resolved
    correctly, but no Python consumer ever read it. Deleting the key keeps the
    config surface honest and avoids implying a non-existent evaluation phase.
    """
    path = Path("src/tmgg/experiments/exp_configs/grid_search_base.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    assert "final_eval_noise_levels" not in cfg.get("evaluation", {})


def test_future_stage_configs_only_keep_executable_settings():
    """Future stage configs should not embed documentation-only metadata.

    Stage 3-5 are executable Hydra configs. Their roadmap notes belong in
    prose comments or separate planning docs, not as keys that look live but
    have no code readers.
    """
    stale_keys = {"trigger_conditions", "ablations", "statistical_tests"}
    future_stage_paths = [
        _STAGE_CONFIG_ROOT / "stage3_diversity.yaml",
        _STAGE_CONFIG_ROOT / "stage4_benchmarks.yaml",
        _STAGE_CONFIG_ROOT / "stage5_full.yaml",
    ]

    offenders: list[str] = []
    for path in future_stage_paths:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        present = sorted(stale_keys.intersection(cfg.keys()))
        if present:
            offenders.append(f"{path}: {', '.join(present)}")

    assert not offenders, (
        "These future-stage configs still define documentation-only keys:\n"
        + "\n".join(f"  {entry}" for entry in offenders)
    )


def test_all_data_configs_declare_explicit_package_headers():
    """Every data config should state its Hydra package explicitly.

    Best practice here is explicit packaging rather than relying on directory
    inference. Normal data configs should declare ``data``; the grid-search
    overlays are intentionally ``_global_`` because they set top-level
    ``noise_type`` and ``noise_levels`` in addition to the ``data`` subtree.
    """
    global_package_files = {
        "grid_base.yaml",
        "grid_digress.yaml",
        "grid_gaussian.yaml",
        "grid_rotation.yaml",
    }

    wrong_headers: list[str] = []
    for path in sorted(_DATA_CONFIG_ROOT.glob("*.yaml")):
        with open(path) as f:
            first_nonempty = next(
                (line.strip() for line in f if line.strip()),
                "",
            )

        expected = (
            "# @package _global_"
            if path.name in global_package_files
            else "# @package data"
        )
        if first_nonempty != expected:
            wrong_headers.append(f"{path.name}: expected '{expected}'")

    assert not wrong_headers, (
        "These data configs are missing the expected explicit package header:\n"
        + "\n".join(f"  {entry}" for entry in wrong_headers)
    )


def test_deprecated_spectral_data_presets_are_removed():
    """Deprecated `_spectral` data presets should stay deleted.

    Rationale
    ---------
    These files were neither part of any defaults list nor a stable supported
    surface. Keeping them around invited drift, and one of them had already
    rotted into a setup-time failure. This test keeps both the files and the
    last public docs mention from quietly resurfacing.
    """
    removed_presets = [
        "er_spectral.yaml",
        "lfr_spectral.yaml",
        "regular_spectral.yaml",
        "tree_spectral.yaml",
    ]

    lingering_files = [
        name for name in removed_presets if (_DATA_CONFIG_ROOT / name).exists()
    ]
    assert not lingering_files, (
        "These deprecated spectral data presets still exist:\n"
        + "\n".join(f"  {name}" for name in lingering_files)
    )

    how_to_run = Path("docs/how-to-run-experiments.md").read_text()
    assert "er_spectral" not in how_to_run

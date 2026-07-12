"""Test that the baseline runner module exists and is importable.

Rationale
---------
Prior to this fix, lin_mlp_baseline_denoising had a LightningModule
but no runner.py, no base config, and no CLI entry point. This test
ensures all three exist and are wired correctly.

Starting state: runner.py does not exist, tmgg-baseline is not in pyproject.toml.
Invariants: runner.main must be callable; tmgg-baseline must appear in [project.scripts].
"""


def test_baseline_runner_importable():
    from tmgg.experiments.lin_mlp_baseline_denoising.runner import main

    assert callable(main)


def test_baseline_entry_point_registered():
    """tmgg-baseline must be in pyproject.toml scripts."""
    import tomllib
    from pathlib import Path

    pyproject = Path("pyproject.toml")
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    scripts = data["project"]["scripts"]
    assert (
        "tmgg-baseline" in scripts
    ), f"tmgg-baseline not in [project.scripts]. Found: {sorted(scripts.keys())}"

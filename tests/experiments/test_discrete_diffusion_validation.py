"""Tests for discrete diffusion baseline validation (P1.8).

Testing strategy
----------------
Verifies the validation infrastructure works correctly: the smoke test
configuration produces a passing result with finite loss and finite MMD
values, and the CLI entry point resolves without errors. Full baseline
validation (GPU, ~1h) is left to manual runs.

Key invariants:
- Quick validation completes without errors
- Loss is finite after training
- MMD values are finite and non-negative
- ValidationResult.passed is True for converged training
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "validate_discrete_diffusion.py"
)


def _import_validation_script() -> ModuleType:
    """Import the validation script by path (it's not a package)."""
    spec = importlib.util.spec_from_file_location(
        "validate_discrete_diffusion", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestValidationSmoke:
    """Run the quick validation and check results."""

    def test_quick_validation_passes(self) -> None:
        """Quick smoke test should produce finite loss and finite MMD."""
        mod = _import_validation_script()
        ValidationConfig = mod.ValidationConfig  # noqa: N806
        run_validation = mod.run_validation

        cfg = ValidationConfig(
            n_layers=2,
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 16, "dy": 16, "n_head": 2},
            diffusion_steps=10,
            num_nodes=8,
            num_graphs=32,
            batch_size=4,
            max_steps=20,
            eval_num_samples=4,
            val_check_interval=10,
            device="cpu",
            seed=42,
        )

        result = run_validation(cfg)

        assert result.loss_decreased, "Loss should be finite"
        assert result.mmd_finite, "MMD values should be finite"
        assert result.degree_mmd >= 0
        assert result.clustering_mmd >= 0
        assert result.spectral_mmd >= 0
        assert result.passed, "Smoke test should pass"


class TestValidationCLI:
    def test_cli_help(self) -> None:
        """CLI --help should exit 0."""
        mod = _import_validation_script()
        with pytest.raises(SystemExit) as exc_info:
            mod.main(["--help"])
        assert exc_info.value.code == 0

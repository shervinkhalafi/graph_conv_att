"""Tests for run_experiment preflight utilities.

These test the run_id auto-generation and W&B duplicate detection
that replaced the config_builder pipeline. The run_id generator
produces human-readable names from resolved Hydra config values,
matching the format previously handled by ExperimentConfigBuilder.generate_run_id().
The W&B check queries the API for an existing run with the same display name.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from tmgg.training.orchestration.run_experiment import (
    check_wandb_run_exists,
    generate_run_id,
)


class TestGenerateRunId:
    """Tests for generate_run_id().

    The function produces a compact, human-readable identifier from
    the resolved config, using short prefixes and scientific notation
    for small values (matching the old run_id_template convention).
    """

    def test_spectral_arch_config(self):
        """Full spectral arch config produces expected format.

        Tests the deterministic portion of the id; ``force_fresh: false``
        opts out of the post-2026-04-30 default that appends a UTC
        timestamp suffix to dodge run-dir collisions.
        """
        config = OmegaConf.create(
            {
                "experiment_name": "stage1",
                "model": {"_target_": "tmgg.models.SpectralArch", "k": 8},
                "learning_rate": 1e-4,
                "weight_decay": 1e-3,
                "seed": 1,
                "force_fresh": False,
            }
        )
        result = generate_run_id(config)
        assert result == "stage1_SpectralArch_lr1e-4_wd1e-3_k8_s1"

    def test_scientific_notation_small_values(self):
        """Values < 0.01 use scientific notation without zero-padded exponent."""
        config = OmegaConf.create(
            {
                "experiment_name": "exp",
                "model": {"_target_": "tmgg.models.Foo"},
                "learning_rate": 5e-5,
                "weight_decay": 1e-3,
                "seed": 2,
                "force_fresh": False,
            }
        )
        result = generate_run_id(config)
        assert "lr5e-5" in result
        assert "wd1e-3" in result
        assert result.endswith("_s2")

    def test_large_values_no_scientific(self):
        """Values >= 0.01 use plain string representation."""
        config = OmegaConf.create(
            {
                "experiment_name": "exp",
                "model": {"_target_": "tmgg.models.Foo"},
                "learning_rate": 0.01,
                "weight_decay": 0.1,
                "seed": 1,
                "force_fresh": False,
            }
        )
        result = generate_run_id(config)
        assert "lr0.01" in result
        assert "wd0.1" in result

    def test_missing_optional_fields(self):
        """Works with minimal config (only seed required)."""
        config = OmegaConf.create({"seed": 42, "force_fresh": False})
        result = generate_run_id(config)
        assert result == "s42"

    def test_diffusion_steps_included(self):
        """model.diffusion_steps appears as T prefix."""
        config = OmegaConf.create(
            {
                "model": {"_target_": "tmgg.models.Foo", "diffusion_steps": 500},
                "seed": 1,
                "force_fresh": False,
            }
        )
        result = generate_run_id(config)
        assert "T500" in result

    def test_dataset_discriminator_appears_in_id(self):
        """``data._target_`` short token is included as ``d<ClassName>``.

        Round-1 of the smallest-config sweep hit a contamination bug
        where ENZYMES async-eval workers loaded leftover SBM checkpoints
        from a shared run dir because the run_id formula had no
        dataset axis. This test pins that SBM and ENZYMES configs with
        identical lr/wd/seed/n_layers produce different run_ids — and
        therefore different Modal output dirs.
        """
        sbm_config = OmegaConf.create(
            {
                "experiment_name": "exp",
                "model": {"_target_": "tmgg.models.Foo"},
                "data": {"_target_": "tmgg.data.SpectreSBMDataModule"},
                "learning_rate": 2e-4,
                "seed": 0,
                "force_fresh": False,
            }
        )
        enzymes_config = OmegaConf.create(
            {
                "experiment_name": "exp",
                "model": {"_target_": "tmgg.models.Foo"},
                "data": {"_target_": "tmgg.data.GraphDataModule"},
                "learning_rate": 2e-4,
                "seed": 0,
                "force_fresh": False,
            }
        )
        sbm_id = generate_run_id(sbm_config)
        enzymes_id = generate_run_id(enzymes_config)
        assert "dSpectreSBMDataModule" in sbm_id
        assert "dGraphDataModule" in enzymes_id
        assert sbm_id != enzymes_id

    def test_n_layers_discriminator_appears_in_id(self):
        """``model.model.n_layers`` is included as ``L<n>``.

        Round-1 of the smallest-config sweep hit a related contamination
        bug where SBM n_layers=6 loaded a stale n_layers=8 checkpoint
        because the run_id formula ignored architecture depth. This
        test pins that two configs identical except for ``n_layers``
        produce different run_ids.
        """
        n6 = OmegaConf.create(
            {
                "experiment_name": "exp",
                "model": {
                    "_target_": "tmgg.models.Foo",
                    "model": {"n_layers": 6},
                },
                "seed": 0,
                "force_fresh": False,
            }
        )
        n8 = OmegaConf.create(
            {
                "experiment_name": "exp",
                "model": {
                    "_target_": "tmgg.models.Foo",
                    "model": {"n_layers": 8},
                },
                "seed": 0,
                "force_fresh": False,
            }
        )
        assert "L6" in generate_run_id(n6)
        assert "L8" in generate_run_id(n8)
        assert generate_run_id(n6) != generate_run_id(n8)

    def test_force_fresh_defaults_to_true(self):
        """Post-2026-04-30 default: every run gets a ``_fresh_<UTC>`` suffix.

        The previous default (``force_fresh: false``) shared a Modal
        output dir across successive launches that computed the same
        deterministic run_id, which caused two distinct contamination
        modes in round-1 of the smallest-config sweep:
          1. async-eval workers loaded prior-run checkpoints (saw
             evaluations against models from unrelated training runs);
          2. Lightning's resume picked up incompatible state_dicts when
             architecture changed (n_layers=6 model loaded an
             n_layers=8 ckpt and crashed).
        Defaulting force_fresh to True forces every run into a unique
        dir; the cost (no checkpoint resume after a transient crash)
        is negligible next to silent contamination.
        """
        config = OmegaConf.create(
            {
                "experiment_name": "exp",
                "model": {"_target_": "tmgg.models.Foo"},
                "seed": 1,
                # Note: no force_fresh override.
            }
        )
        result = generate_run_id(config)
        assert "_fresh_" in result, (
            "force_fresh must default to True; result was " + result
        )

    def test_force_fresh_explicit_false_still_works(self):
        """Opt-out path: ``force_fresh: false`` skips the suffix.

        For experiments that genuinely need checkpoint-resume
        semantics (e.g. continuation runs), the explicit override is
        still honored.
        """
        config = OmegaConf.create(
            {
                "experiment_name": "exp",
                "model": {"_target_": "tmgg.models.Foo"},
                "seed": 1,
                "force_fresh": False,
            }
        )
        result = generate_run_id(config)
        assert "_fresh_" not in result

    def test_existing_run_id_not_overwritten(self):
        """If config already has run_id, generate_run_id returns it unchanged."""
        config = OmegaConf.create(
            {
                "run_id": "my_custom_id",
                "seed": 1,
            }
        )
        result = generate_run_id(config)
        assert result == "my_custom_id"

    def test_force_fresh_appends_timestamp_suffix(self):
        """``force_fresh=True`` without explicit run_id appends a UTC timestamp.

        Without this suffix, ``force_fresh`` would only bypass the
        Lightning checkpoint resume but still write into the same
        directory (``outputs/<exp>/<deterministic_id>/``) as a prior
        run with identical hyperparams, risking checkpoint clobbering.
        """
        config = OmegaConf.create(
            {
                "experiment_name": "exp",
                "model": {"_target_": "tmgg.models.Foo"},
                "learning_rate": 1e-3,
                "seed": 1,
                "force_fresh": True,
            }
        )
        result = generate_run_id(config)
        # Base parts still present
        assert result.startswith("exp_Foo_lr1e-3_s1_fresh_")
        # Timestamp suffix is 15 chars: YYYYmmddTHHMMSS
        suffix = result.split("_fresh_", 1)[1]
        assert len(suffix) == 15
        assert "T" in suffix

    def test_force_fresh_does_not_override_explicit_run_id(self):
        """An explicit run_id wins over ``force_fresh``'s auto-suffix.

        ``force_fresh=True`` only fires when ``run_id`` is unset; an
        explicit override takes precedence so users who pass
        ``+run_id=...`` (e.g. via the panel runner) get exactly that id.
        """
        config = OmegaConf.create(
            {
                "run_id": "explicit_id",
                "force_fresh": True,
                "seed": 1,
            }
        )
        result = generate_run_id(config)
        assert result == "explicit_id"

    def test_force_fresh_false_no_suffix(self):
        """``force_fresh=False`` (default) leaves the deterministic id alone."""
        config = OmegaConf.create(
            {
                "experiment_name": "exp",
                "model": {"_target_": "tmgg.models.Foo"},
                "seed": 1,
                "force_fresh": False,
            }
        )
        result = generate_run_id(config)
        assert "_fresh_" not in result
        assert result == "exp_Foo_s1"


class TestCheckWandbRunExists:
    """Tests for check_wandb_run_exists().

    Uses mocked W&B API to avoid real network calls. The function
    queries by displayName filter to check for a single run efficiently.
    """

    def test_returns_true_when_run_found(self, monkeypatch):
        """Returns True when W&B API returns a matching run."""
        mock_api_cls = MagicMock()
        mock_run = MagicMock()
        mock_run.name = "stage1_linear_pe_lr1e-4_s1"
        mock_api_cls.return_value.runs.return_value = [mock_run]
        monkeypatch.setattr("wandb.Api", mock_api_cls)

        assert check_wandb_run_exists("team", "project", "stage1_linear_pe_lr1e-4_s1")
        mock_api_cls.return_value.runs.assert_called_once_with(
            "team/project",
            filters={"displayName": "stage1_linear_pe_lr1e-4_s1"},
        )

    def test_returns_false_when_no_match(self, monkeypatch):
        """Returns False when W&B API returns empty list."""
        mock_api_cls = MagicMock()
        mock_api_cls.return_value.runs.return_value = []
        monkeypatch.setattr("wandb.Api", mock_api_cls)

        assert not check_wandb_run_exists("team", "project", "nonexistent")

    def test_returns_false_on_comm_error(self, monkeypatch):
        """Returns False (not crash) on CommError -- transient network issues
        should not block experiment execution."""
        from wandb.errors import CommError

        mock_api_cls = MagicMock()
        mock_api_cls.return_value.runs.side_effect = CommError("network error")
        monkeypatch.setattr("wandb.Api", mock_api_cls)

        assert not check_wandb_run_exists("team", "project", "run_name")

    def test_reraises_non_comm_errors(self, monkeypatch):
        """Non-communication errors (auth, programming) must propagate, not silently return False.

        Rationale: the bare ``except Exception`` previously in check_wandb_run_exists
        swallowed authentication failures, programming bugs, and anything else. A
        swallowed auth error would cause the dedup check to silently return False,
        potentially duplicating hours of GPU work. Only CommError (network issues)
        should be caught; everything else must propagate so the caller sees the real
        failure.
        """
        mock_api_cls = MagicMock()
        mock_api_cls.return_value.runs.side_effect = ValueError(
            "unexpected internal error"
        )
        monkeypatch.setattr("wandb.Api", mock_api_cls)

        with pytest.raises(ValueError, match="unexpected internal error"):
            check_wandb_run_exists("entity", "project", "run_name")

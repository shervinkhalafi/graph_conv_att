"""Tests for Hydra logger/callback config variants and runtime helpers.

These tests cover the config-surface cleanup that moves experiment-specific
callback and logger policy into named Hydra config groups. They are intended
to stay fast: configs are composed in-process, callbacks are constructed
directly, and logger creation is stubbed so no network calls or filesystem
writes escape the test sandbox.

Key invariants
--------------
- Key experiment configs resolve cleanly with logger enabled.
- Grid-search and discrete diffusion select logger/callback variants through
  Hydra defaults rather than inline overrides.
- ``create_callbacks()`` builds callback objects from top-level
  ``config.callbacks`` and ignores ``trainer.callbacks``.
- ``create_loggers()`` forwards supported wandb metadata fields that appear
  in the config variants and instantiates the default CSV mirror logger.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from tmgg.training.logging import create_loggers
from tmgg.training.orchestration.run_experiment import create_callbacks

EXP_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "tmgg" / "experiments" / "exp_configs"
)


def _base_overrides(tmp_path: Path) -> list[str]:
    """Return generic overrides that keep config composition self-contained."""
    return [
        f"paths.output_dir={tmp_path}",
        f"paths.results_dir={tmp_path}/results",
        f"hydra.run.dir={tmp_path}",
        "trainer.max_steps=2",
        "trainer.val_check_interval=1",
        "trainer.accelerator=cpu",
        "data.batch_size=2",
        "data.num_workers=0",
    ]


def _compose_config(
    tmp_path: Path,
    config_name: str,
    overrides: list[str] | None = None,
    *,
    disable_logger: bool = False,
) -> DictConfig:
    """Compose a config with minimal runtime-independent overrides."""
    extra_overrides = list(overrides or [])
    if disable_logger:
        extra_overrides.append("~logger")

    with initialize_config_dir(version_base=None, config_dir=str(EXP_CONFIG_DIR)):
        return compose(
            config_name=config_name,
            overrides=_base_overrides(tmp_path) + extra_overrides,
        )


@pytest.fixture(autouse=True)
def _clear_hydra() -> Generator[None, None, None]:
    """Clear Hydra global state before and after each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.mark.integration
@pytest.mark.config
class TestLoggerAndCallbackVariants:
    """Verify config composition lands on the expected Hydra variants."""

    @pytest.mark.parametrize(
        ("config_name", "overrides"),
        [
            ("base_config_spectral_arch", []),
            ("grid_search_base", []),
            ("base_config_discrete_diffusion_generative", []),
        ],
    )
    def test_key_configs_resolve_with_logger_enabled(
        self,
        tmp_path: Path,
        config_name: str,
        overrides: list[str],
    ) -> None:
        """Critical configs should fully resolve without disabling logger.

        Some logger templates (notably ``discrete_wandb.yaml``) interpolate
        ``${run_id}`` which the orchestrator injects at runtime; we set one
        explicitly here so OmegaConf's eager resolve does not trip on it.
        """
        cfg = _compose_config(tmp_path, config_name, overrides=overrides)
        with open_dict(cfg):
            cfg.run_id = "test-resolve-run-id"
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(resolved, dict)
        assert "logger" in resolved
        assert "callbacks" in resolved

    def test_base_config_uses_wandb_and_csv_logger_variant(
        self, tmp_path: Path
    ) -> None:
        """Base infra should compose both W&B and CSV logger configs."""
        cfg = _compose_config(tmp_path, "base_config_spectral_arch")
        logger_cfg = cfg.logger

        assert len(logger_cfg) == 2
        assert list(logger_cfg[0].keys()) == ["wandb"]
        assert list(logger_cfg[1].keys()) == ["csv"]
        assert cfg.logger[1].csv.save_dir == f"{tmp_path}/csv"

    def test_grid_search_uses_top_level_variants(self, tmp_path: Path) -> None:
        """Grid search should select logger/callback variants via defaults."""
        cfg = _compose_config(tmp_path, "grid_search_base")

        assert "callbacks" in cfg
        assert "callbacks" not in cfg.trainer
        assert cfg.callbacks.early_stopping.monitor == "val/loss"
        assert cfg.callbacks.early_stopping.patience == 4
        assert cfg.callbacks.checkpoint.filename == (
            "grid_search_4k_params-step={step:06d}-val_loss={val/loss:.4f}"
        )
        assert len(cfg.logger) == 2
        assert list(cfg.logger[0].keys()) == ["wandb"]
        assert list(cfg.logger[1].keys()) == ["csv"]
        assert cfg.logger[0].wandb.group == f"unknown_{cfg.noise_type}"

    def test_discrete_config_uses_top_level_variants(self, tmp_path: Path) -> None:
        """Discrete diffusion should select NLL callbacks and discrete wandb naming."""
        cfg = _compose_config(tmp_path, "base_config_discrete_diffusion_generative")

        assert cfg.callbacks.early_stopping.monitor == "val/epoch_NLL"
        assert cfg.callbacks.checkpoint.monitor == "val/epoch_NLL"
        assert cfg.callbacks.checkpoint.filename == (
            "model-step={step:06d}-val_nll={val/epoch_NLL:.4f}"
        )
        assert len(cfg.logger) == 2
        assert list(cfg.logger[0].keys()) == ["wandb"]
        assert list(cfg.logger[1].keys()) == ["csv"]
        # The discrete wandb logger now sets ``name: ${run_id}`` (post the
        # 2026-05-01 wandb-run-id sidecar refactor at commit 44c1d5fd) so
        # the resolved name depends on a runtime-injected ``run_id``. We
        # set one explicitly here to exercise the template.
        with open_dict(cfg):
            cfg.run_id = "discrete-smoke-test"
        assert cfg.logger[0].wandb.name == "discrete-smoke-test"

    @pytest.mark.parametrize(
        "config_name",
        [
            "base_config_discrete_diffusion_generative",
            "base_config_gaussian_diffusion",
        ],
    )
    def test_generative_configs_wire_live_visualization_defaults(
        self,
        tmp_path: Path,
        config_name: str,
    ) -> None:
        """Generative configs should expose live visualization settings.

        Regression rationale
        --------------------
        The old ``visualization_interval`` knob was dead config. The new
        defaults should compose under ``evaluation.visualization`` and be
        forwarded into ``cfg.model`` for the live DiffusionModule path.
        """
        cfg = _compose_config(tmp_path, config_name)

        assert cfg.evaluation.visualization.enabled is True
        assert cfg.evaluation.visualization.num_samples == 8
        assert "visualization_interval" not in cfg
        assert "visualization_interval" not in cfg.evaluation
        assert cfg.model.visualization.enabled is True
        assert cfg.model.visualization.num_samples == 8


class TestCreateCallbacks:
    """Verify runtime callbacks are built from the top-level callback config."""

    def test_grid_search_variant_sets_monitor_and_patience(
        self,
        tmp_path: Path,
    ) -> None:
        """Grid-search callback config should drive ModelCheckpoint/EarlyStopping."""
        cfg = _compose_config(tmp_path, "grid_search_base", disable_logger=True)
        callbacks = create_callbacks(cfg)

        early_stopping = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
        checkpoint = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))

        assert early_stopping.monitor == "val/loss"
        assert early_stopping.patience == 4
        assert checkpoint.monitor == "val/loss"
        assert checkpoint.filename == (
            "grid_search_4k_params-step={step:06d}-val_loss={val/loss:.4f}"
        )

    def test_discrete_variant_sets_nll_monitor(self, tmp_path: Path) -> None:
        """Discrete callback config should swap monitor keys to epoch NLL."""
        cfg = _compose_config(
            tmp_path,
            "base_config_discrete_diffusion_generative",
            disable_logger=True,
        )
        callbacks = create_callbacks(cfg)

        early_stopping = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
        checkpoint = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))

        assert early_stopping.monitor == "val/epoch_NLL"
        assert checkpoint.monitor == "val/epoch_NLL"


class TestCreateLoggers:
    """Verify logger creation honors the config variant surface."""

    def test_grid_wandb_variant_forwards_supported_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Grid-search logger config should forward group/notes/offline."""
        cfg = _compose_config(tmp_path, "grid_search_base")
        captured: list[dict[str, Any]] = []

        class DummyWandbLogger:
            """Capture wandb init kwargs without touching the real backend."""

            def __init__(self, **kwargs: Any) -> None:
                captured.append(kwargs)

        monkeypatch.setattr(
            "tmgg.training.logging._has_wandb_credentials", lambda: True
        )
        monkeypatch.setattr("tmgg.training.logging.WandbLogger", DummyWandbLogger)

        loggers = create_loggers(cfg)

        assert len(loggers) == 2
        assert len(captured) == 1
        kwargs = captured[0]
        assert kwargs["group"] == f"unknown_{cfg.noise_type}"
        assert kwargs["notes"] == "Grid search with ~4k params across architectures"
        assert kwargs["offline"] is False

    def test_wandb_logger_uses_run_id_when_name_is_null(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Base wandb config should keep runtime name derivation via run_id."""
        cfg = _compose_config(tmp_path, "base_config_spectral_arch")
        with open_dict(cfg):
            cfg.run_id = "spectral-test-run"
        captured: list[dict[str, Any]] = []

        class DummyWandbLogger:
            """Capture runtime-derived wandb kwargs."""

            def __init__(self, **kwargs: Any) -> None:
                captured.append(kwargs)

        monkeypatch.setattr(
            "tmgg.training.logging._has_wandb_credentials", lambda: True
        )
        monkeypatch.setattr("tmgg.training.logging.WandbLogger", DummyWandbLogger)

        loggers = create_loggers(cfg)

        assert captured[0]["name"] == "spectral-test-run"
        assert len(loggers) == 2

    def test_default_config_instantiates_csv_logger_alongside_wandb(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The shared default logger policy should create both loggers.

        Regression rationale
        --------------------
        Modal runs already write checkpoints and configs to the persistent
        output volume. Adding CSV by default should mirror metrics locally
        there as well, without removing the W&B primary logger.
        """
        cfg = _compose_config(tmp_path, "base_config_spectral_arch")
        captured_wandb: list[dict[str, Any]] = []
        captured_csv: list[dict[str, Any]] = []

        class DummyWandbLogger:
            def __init__(self, **kwargs: Any) -> None:
                captured_wandb.append(kwargs)

        class DummyCSVLogger:
            def __init__(self, **kwargs: Any) -> None:
                captured_csv.append(kwargs)

        monkeypatch.setattr(
            "tmgg.training.logging._has_wandb_credentials", lambda: True
        )
        monkeypatch.setattr("tmgg.training.logging.WandbLogger", DummyWandbLogger)
        monkeypatch.setattr("tmgg.training.logging.CSVLogger", DummyCSVLogger)

        loggers = create_loggers(cfg)

        assert len(loggers) == 2
        assert len(captured_wandb) == 1
        assert len(captured_csv) == 1
        assert captured_csv[0]["save_dir"] == f"{tmp_path}/csv"
        assert captured_csv[0]["name"] == cfg.experiment_name

    def test_default_config_keeps_csv_when_wandb_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Missing W&B credentials should still leave the CSV mirror active.

        ``create_loggers`` now raises ``OSError`` when W&B credentials are
        missing unless ``allow_no_wandb=true`` is set on the config. We
        opt into the degrade-to-warning path explicitly so the CSV mirror
        survives in environments without ``WANDB_API_KEY``.
        """
        cfg = _compose_config(tmp_path, "base_config_spectral_arch")
        with open_dict(cfg):
            cfg.allow_no_wandb = True
        captured_csv: list[dict[str, Any]] = []

        class DummyCSVLogger:
            def __init__(self, **kwargs: Any) -> None:
                captured_csv.append(kwargs)

        monkeypatch.setattr(
            "tmgg.training.logging._has_wandb_credentials", lambda: False
        )
        monkeypatch.setattr("tmgg.training.logging.CSVLogger", DummyCSVLogger)

        loggers = create_loggers(cfg)

        assert len(loggers) == 1
        assert len(captured_csv) == 1
        assert captured_csv[0]["save_dir"] == f"{tmp_path}/csv"

    def test_wandb_logger_passes_id_when_resume_id_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``cfg.wandb_run_id_resume`` must propagate to ``WandbLogger(id=...)``.

        Rationale
        ---------
        ``run_experiment`` reads ``<output_dir>/wandb_run_id.txt`` before
        calling ``create_loggers`` and stashes the persisted W&B run id
        at ``cfg.wandb_run_id_resume`` when present. ``create_loggers``
        must forward that value as the ``id=`` kwarg so ``WandbLogger``
        (with its default ``resume="allow"``) appends to the same W&B
        run on the second launch — the property that makes Modal-preempt
        restart land in one continuous trajectory rather than fragmenting
        the dashboard across two same-named runs.
        """
        cfg = _compose_config(tmp_path, "base_config_spectral_arch")
        with open_dict(cfg):
            cfg.run_id = "preempt_smoke_test_1"
            cfg.wandb_run_id_resume = "wandb_internal_abc123"
        captured: list[dict[str, Any]] = []

        class DummyWandbLogger:
            def __init__(self, **kwargs: Any) -> None:
                captured.append(kwargs)

        monkeypatch.setattr(
            "tmgg.training.logging._has_wandb_credentials", lambda: True
        )
        monkeypatch.setattr("tmgg.training.logging.WandbLogger", DummyWandbLogger)

        _ = create_loggers(cfg)

        assert len(captured) == 1
        assert captured[0].get("id") == "wandb_internal_abc123", (
            "wandb_run_id_resume must be forwarded as the WandbLogger id "
            f"kwarg; got id={captured[0].get('id')!r}"
        )

    def test_wandb_logger_omits_id_when_resume_id_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """No ``wandb_run_id_resume`` → no ``id=`` kwarg, fresh W&B run.

        First launch of a named run has no sidecar yet; the absence of
        ``cfg.wandb_run_id_resume`` must NOT pass an empty / None id to
        WandbLogger (W&B would either reject or accept-and-create with a
        random id depending on version, neither is what we want). The
        kwarg should simply be omitted so Lightning's default behavior
        (auto-generate a fresh id) takes over.
        """
        cfg = _compose_config(tmp_path, "base_config_spectral_arch")
        with open_dict(cfg):
            cfg.run_id = "first_launch_named_run"
            # Intentionally do NOT set cfg.wandb_run_id_resume.
        captured: list[dict[str, Any]] = []

        class DummyWandbLogger:
            def __init__(self, **kwargs: Any) -> None:
                captured.append(kwargs)

        monkeypatch.setattr(
            "tmgg.training.logging._has_wandb_credentials", lambda: True
        )
        monkeypatch.setattr("tmgg.training.logging.WandbLogger", DummyWandbLogger)

        _ = create_loggers(cfg)

        assert len(captured) == 1
        assert "id" not in captured[0], (
            "Without cfg.wandb_run_id_resume, WandbLogger must be called "
            f"without an id= kwarg; got kwargs={list(captured[0])}"
        )

    def test_wandb_logger_honors_wandb_name_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A top-level ``wandb_name`` config key must override ``run_id``.

        Regression rationale
        --------------------
        The smallest-config sweep launcher passes
        ``+wandb_name=smallest-cfg/spectre_sbm/r0/smoke/<hash>`` as a
        Hydra override, expecting the W&B run's display name to match
        the audit identifier so post-hoc reconciliation can find the
        run by name. Before the fix, ``create_loggers`` only read
        ``logger_params.get('name')`` (left null in
        ``base/logger/wandb.yaml``) and fell back to ``run_id``,
        ignoring ``wandb_name`` entirely. The fix promotes ``wandb_name``
        to the top of the precedence chain.
        """
        cfg = _compose_config(tmp_path, "base_config_spectral_arch")
        with open_dict(cfg):
            cfg.run_id = "default-run"
            cfg.wandb_name = "custom-name"
        captured: list[dict[str, Any]] = []

        class DummyWandbLogger:
            def __init__(self, **kwargs: Any) -> None:
                captured.append(kwargs)

        monkeypatch.setattr(
            "tmgg.training.logging._has_wandb_credentials", lambda: True
        )
        monkeypatch.setattr("tmgg.training.logging.WandbLogger", DummyWandbLogger)

        _ = create_loggers(cfg)

        assert len(captured) == 1
        assert captured[0]["name"] == "custom-name", (
            f"wandb_name override must take precedence over run_id; "
            f"got name={captured[0]['name']!r}"
        )

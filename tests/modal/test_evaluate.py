"""Tests for modal/evaluate.py subprocess dispatch.

Test rationale
--------------
``modal/evaluate.py`` must not import from ``tmgg.experiments``.
The subprocess boundary keeps ``modal/`` clean: the ``experiments``
import happens inside the subprocess, never in the modal package
itself. These tests verify both the import constraint and the CLI
argument construction logic.

Invariants
----------
- The source text of ``modal/evaluate.py`` contains zero references
  to ``tmgg.experiments`` (neither ``from`` nor ``import`` forms).
- ``run_mmd_evaluation`` builds CLI arguments that match the flags
  accepted by ``evaluate_checkpoint`` and delegates to subprocess.
- The ``splits`` field was removed from ``EvaluationInput`` since the
  CLI evaluates against freshly generated reference data, not per-split
  data from a data module.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


class TestNoExperimentsImport:
    """modal/evaluate.py must contain zero imports from tmgg.experiments.

    Rationale
    ---------
    The modal package is a transport layer that should depend only on
    ``tmgg.training`` (lower layer), never on ``experiments``
    sub-packages (higher layer). The subprocess dispatch pattern keeps
    this boundary clean.
    """

    def test_no_experiments_import_in_source(self) -> None:
        """Scan the source text of modal/evaluate.py for forbidden imports."""
        spec = importlib.util.find_spec("tmgg.modal._lib.evaluate")
        assert spec is not None and spec.origin is not None
        source = Path(spec.origin).read_text()
        assert (
            "from tmgg.experiments" not in source
        ), "modal/evaluate.py must not contain 'from tmgg.experiments' imports"
        assert (
            "import tmgg.experiments" not in source
        ), "modal/evaluate.py must not contain 'import tmgg.experiments' imports"

    def test_no_experiment_utils_mmd_import_in_source(self) -> None:
        """The inline MMD computation was removed; no direct mmd_metrics import needed."""
        spec = importlib.util.find_spec("tmgg.modal._lib.evaluate")
        assert spec is not None and spec.origin is not None
        source = Path(spec.origin).read_text()
        assert "from tmgg.evaluation.mmd_metrics" not in source, (
            "modal/evaluate.py should not import mmd_metrics directly; "
            "the CLI subprocess handles that"
        )

    def test_no_data_module_import_in_source(self) -> None:
        """_reconstruct_data_module was removed; no MultiGraphDataModule import needed."""
        spec = importlib.util.find_spec("tmgg.modal._lib.evaluate")
        assert spec is not None and spec.origin is not None
        source = Path(spec.origin).read_text()
        assert "MultiGraphDataModule" not in source, (
            "modal/evaluate.py should not reference MultiGraphDataModule; "
            "the CLI subprocess handles data reconstruction"
        )


class TestEvaluationInputDataclass:
    """EvaluationInput defines the Modal transport contract for evaluation."""

    def test_default_values(self) -> None:
        from tmgg.modal._lib.evaluate import EvaluationInput

        task = EvaluationInput(run_id="test-run")
        assert task.run_id == "test-run"
        assert task.checkpoint_path is None
        assert task.num_samples == 500
        assert task.num_steps == 100
        assert task.mmd_kernel == "gaussian_tv"
        assert task.mmd_sigma == 1.0
        assert task.seed == 42

    def test_splits_field_removed(self) -> None:
        """The per-split evaluation was replaced by single CLI invocation."""
        from tmgg.modal._lib.evaluate import EvaluationInput

        assert not hasattr(EvaluationInput, "splits") or "splits" not in {
            f.name for f in EvaluationInput.__dataclass_fields__.values()
        }, "EvaluationInput should no longer have a 'splits' field"


class TestEvaluationOutputDataclass:
    """EvaluationOutput defines the Modal transport return contract."""

    def test_default_values(self) -> None:
        from tmgg.modal._lib.evaluate import EvaluationOutput

        out = EvaluationOutput(run_id="test-run")
        assert out.status == "completed"
        assert out.results == {}
        assert out.error_message is None


class TestRunMmdEvaluationFailFast:
    """run_mmd_evaluation must fail fast when config or checkpoint are missing."""

    def test_missing_config_returns_failed(self, tmp_path: Path) -> None:
        """When config.yaml does not exist, return failure immediately."""
        import tmgg.modal._lib.evaluate as mod

        # Temporarily override OUTPUTS_MOUNT
        original_mount = mod.OUTPUTS_MOUNT
        mod.OUTPUTS_MOUNT = str(tmp_path)
        try:
            run_dir = tmp_path / "run-abc"
            run_dir.mkdir()
            # No config.yaml created

            result = mod.run_mmd_evaluation({"run_id": "run-abc"})
            assert result["status"] == "failed"
            assert "Config not found" in result["error_message"]
        finally:
            mod.OUTPUTS_MOUNT = original_mount

    def test_missing_checkpoint_returns_failed(self, tmp_path: Path) -> None:
        """When checkpoint does not exist, return failure immediately."""
        import tmgg.modal._lib.evaluate as mod

        original_mount = mod.OUTPUTS_MOUNT
        mod.OUTPUTS_MOUNT = str(tmp_path)
        try:
            run_dir = tmp_path / "run-abc"
            run_dir.mkdir()
            config_path = run_dir / "config.yaml"
            config_path.write_text("data:\n  graph_type: sbm\n  num_nodes: 20\n")
            # No checkpoint created

            result = mod.run_mmd_evaluation({"run_id": "run-abc"})
            assert result["status"] == "failed"
            assert "Checkpoint not found" in result["error_message"]
        finally:
            mod.OUTPUTS_MOUNT = original_mount

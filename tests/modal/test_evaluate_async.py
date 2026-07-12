"""Tests for ``modal/_lib/evaluate_async.py`` (Step 1 of async-eval plan).

Test rationale
--------------
``evaluate_mmd_async`` is the eval-side worker for the async-eval
architecture. It runs in a separate Modal container (cheaper A10G
``standard`` tier), attaches to the trainer's existing W&B run via
``resume="must"``, runs the MMD evaluation by reusing
``run_mmd_evaluation`` from ``evaluate.py``, logs results back to W&B
under the trainer's custom step axis (``trainer/global_step``), and
appends a manifest row recording the outcome.

Invariants
----------
1. The W&B log call MUST include ``trainer/global_step`` as a value in
   the dict (NOT a ``step=`` kwarg) so W&B's custom-step routing places
   gen-val/* metrics at the right step. See the plan §2 "as if in-band".
2. ``wandb.init`` MUST be called with ``resume="must"`` so a stale
   run_id fails loudly (no silent run-creation).
3. The manifest gains exactly one row per call: ``status="completed"``
   on success, ``status="failed"`` on exception (with ``error_tail``
   set and the exception re-raised).
4. ``wandb.define_metric`` MUST NOT be called by the eval worker — the
   trainer owns that contract; double-defining is a recipe for
   confusion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Bring the manifest helper onto the test path the same way the
# Modal-side modules do at runtime, so tests can read the per-row
# directory layout via the canonical reader.
_REPO_ROOT_TEST = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_TEST) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_TEST))
from tmgg.sweep._eval_manifest import (  # noqa: E402  -- post-sys.path import
    read_manifest as _eval_manifest_read,
)
from tmgg.sweep._eval_manifest import (  # noqa: E402  -- post-sys.path import
    resolve_manifest_dir as _eval_manifest_resolve_dir,
)


def _make_task(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Build a baseline task dict for the async-eval worker."""
    base: dict[str, Any] = {
        "run_id": "smallest-cfg-r1-anchor-aabbccdd",
        "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
        "checkpoint_path": "/data/outputs/smallest-cfg-r1-anchor-aabbccdd/checkpoints/step_5240.ckpt",
        "wandb_run_id": "abc1234",
        "wandb_project": "graph-denoising",
        "wandb_entity": "your-wandb-entity",
        "global_step": 5240,
        "num_samples": 40,
        "num_steps": 100,
        "scheduled_step": 5237,
        "manifest_path": str(tmp_path / "eval_manifest.jsonl"),
    }
    base.update(overrides)
    return base


def _success_eval_output(task: dict[str, Any]) -> dict[str, Any]:
    """Build a fake EvaluationOutput dict that the worker would receive
    from ``run_mmd_evaluation`` on success."""
    return {
        "run_id": task["run_id"],
        "checkpoint_name": "step_5240",
        "status": "completed",
        "results": {
            "eval": {
                "gen-val/sbm_accuracy": 0.62,
                "gen-val/degree_mmd": 0.012,
                "gen-val/clustering_mmd": 0.034,
            }
        },
        "error_message": None,
        "evaluation_params": {
            "num_samples": task["num_samples"],
            "num_steps": task["num_steps"],
            "mmd_kernel": "gaussian_tv",
            "mmd_sigma": 1.0,
            "seed": 42,
        },
        "timestamp": "2026-04-29T12:00:00",
    }


class TestAsyncEvalLogsAtTrainerStep:
    """The eval worker MUST log gen-val metrics with trainer/global_step
    in the payload so W&B's custom-step axis routes correctly."""

    def test_evaluate_mmd_async_logs_with_trainer_global_step(
        self, tmp_path: Path
    ) -> None:
        """``wandb.log`` must be called with a dict containing
        ``trainer/global_step`` AND the gen-val/* metrics, NOT a
        separate ``step=`` kwarg."""
        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        eval_output = _success_eval_output(task)

        with (
            patch.object(
                evaluate_async, "run_mmd_evaluation", return_value=eval_output
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init") as mock_init,
            patch("tmgg.modal._lib.evaluate_async.wandb.log") as mock_log,
            patch("tmgg.modal._lib.evaluate_async.wandb.finish") as mock_finish,
        ):
            evaluate_async.evaluate_mmd_async(task)

        assert (
            mock_log.call_count == 1
        ), f"wandb.log must be called exactly once; got {mock_log.call_count}"
        call_args = mock_log.call_args
        # The single positional arg is the metrics dict
        logged_dict = (
            call_args.args[0] if call_args.args else call_args.kwargs.get("data")
        )
        assert logged_dict is not None
        assert (
            logged_dict["trainer/global_step"] == 5240
        ), "trainer/global_step must equal the task's global_step"
        assert "gen-val/sbm_accuracy" in logged_dict
        assert logged_dict["gen-val/sbm_accuracy"] == 0.62
        assert "gen-val/degree_mmd" in logged_dict

        # Crucially: no `step=` kwarg — let W&B's custom-step axis route it.
        assert "step" not in call_args.kwargs, (
            "Per the plan, log without step= kwarg; trainer/global_step in the "
            "payload routes via define_metric. step= bypasses that routing."
        )

        # init/finish must be invoked
        assert mock_init.called
        assert mock_finish.called


class TestAsyncEvalUsesResumeMust:
    """``wandb.init`` must use ``resume="must"`` to refuse stale run IDs."""

    def test_evaluate_mmd_async_uses_resume_must(self, tmp_path: Path) -> None:
        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        eval_output = _success_eval_output(task)

        with (
            patch.object(
                evaluate_async, "run_mmd_evaluation", return_value=eval_output
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init") as mock_init,
            patch("tmgg.modal._lib.evaluate_async.wandb.log"),
            patch("tmgg.modal._lib.evaluate_async.wandb.finish"),
        ):
            evaluate_async.evaluate_mmd_async(task)

        assert mock_init.called
        kwargs = mock_init.call_args.kwargs
        assert kwargs.get("id") == "abc1234"
        assert kwargs.get("project") == "graph-denoising"
        assert kwargs.get("entity") == "your-wandb-entity"
        assert kwargs.get("resume") == "must"


class TestAsyncEvalManifestWrite:
    """One manifest row per call, with the right schema."""

    def test_evaluate_mmd_async_appends_completed_manifest_row(
        self, tmp_path: Path
    ) -> None:
        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        eval_output = _success_eval_output(task)

        manifest_path = Path(task["manifest_path"])
        manifest_dir = _eval_manifest_resolve_dir(manifest_path)
        assert not manifest_dir.exists()

        with (
            patch.object(
                evaluate_async, "run_mmd_evaluation", return_value=eval_output
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init"),
            patch("tmgg.modal._lib.evaluate_async.wandb.log"),
            patch("tmgg.modal._lib.evaluate_async.wandb.finish"),
        ):
            evaluate_async.evaluate_mmd_async(task)

        assert manifest_dir.exists()
        rows = _eval_manifest_read(manifest_path)
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}: {rows}"
        row = rows[0]
        assert row["kind"] == "eval_event"
        assert row["status"] == "completed"
        assert row["run_uid"] == task["run_uid"]
        assert row["wandb_run_id"] == "abc1234"
        assert row["scheduled_step"] == 5237
        assert row["global_step"] == 5240
        assert row["checkpoint_path"] == task["checkpoint_path"]
        assert row["modal_call_id"] is None  # filled by trainer-side spawn record
        assert row["error_tail"] is None
        assert "gen-val/sbm_accuracy" in row["metrics"]
        assert "ts_utc" in row and isinstance(row["ts_utc"], str)


class TestAsyncEvalFailureSemantics:
    """On internal failure, write a failed-row AND re-raise (Modal needs
    to record the failure on its side too)."""

    def test_evaluate_mmd_async_on_failure_writes_failed_row_and_reraises(
        self, tmp_path: Path
    ) -> None:
        import pytest

        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        manifest_path = Path(task["manifest_path"])

        with (
            patch.object(
                evaluate_async,
                "run_mmd_evaluation",
                side_effect=RuntimeError("boom"),
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init"),
            patch("tmgg.modal._lib.evaluate_async.wandb.log") as mock_log,
            patch("tmgg.modal._lib.evaluate_async.wandb.finish"),
            pytest.raises(RuntimeError, match="boom"),
        ):
            evaluate_async.evaluate_mmd_async(task)

        # Manifest must record the failure.
        manifest_dir = _eval_manifest_resolve_dir(manifest_path)
        assert manifest_dir.exists()
        rows = _eval_manifest_read(manifest_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["status"] == "failed"
        assert row["error_tail"] is not None
        assert "boom" in row["error_tail"]
        assert row["global_step"] == task["global_step"]
        assert row["scheduled_step"] == task["scheduled_step"]
        # No metrics logged on failure path
        assert mock_log.call_count == 0


class TestAsyncEvalDoesNotDefineMetric:
    """The trainer owns ``define_metric``. The eval worker MUST NOT
    call it — that would race the trainer's setup."""

    def test_evaluate_mmd_async_does_not_call_define_metric(
        self, tmp_path: Path
    ) -> None:
        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        eval_output = _success_eval_output(task)

        with (
            patch.object(
                evaluate_async, "run_mmd_evaluation", return_value=eval_output
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init"),
            patch("tmgg.modal._lib.evaluate_async.wandb.log"),
            patch("tmgg.modal._lib.evaluate_async.wandb.finish"),
            patch("tmgg.modal._lib.evaluate_async.wandb.define_metric") as mock_dm,
        ):
            evaluate_async.evaluate_mmd_async(task)

        assert (
            mock_dm.call_count == 0
        ), "Eval worker must not call define_metric; trainer owns that contract."


class TestAsyncEvalPrefixesUnqualifiedMetricNames:
    """Bug #1 regression: the discrete eval CLI emits flat metric names
    (``degree_mmd`` etc.); the worker must prefix them with ``gen-val/``
    so they hit the trainer's W&B group, and the manifest row carries
    the prefixed names."""

    def test_unqualified_metrics_get_gen_val_prefix(self, tmp_path: Path) -> None:
        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        eval_output = {
            "run_id": task["run_id"],
            "checkpoint_name": "step_5240",
            "status": "completed",
            "results": {
                "eval": {
                    "degree_mmd": 0.012,
                    "clustering_mmd": 0.034,
                    "spectral_mmd": 0.056,
                    "sbm_accuracy": 0.85,
                }
            },
            "error_message": None,
            "evaluation_params": {},
            "timestamp": "2026-04-29T12:00:00",
        }

        with (
            patch.object(
                evaluate_async, "run_mmd_evaluation", return_value=eval_output
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init"),
            patch("tmgg.modal._lib.evaluate_async.wandb.log") as mock_log,
            patch("tmgg.modal._lib.evaluate_async.wandb.finish"),
        ):
            evaluate_async.evaluate_mmd_async(task)

        logged_dict = mock_log.call_args.args[0]
        # Every flat metric name picks up the gen-val/ prefix, original keys gone.
        for metric in (
            "gen-val/degree_mmd",
            "gen-val/clustering_mmd",
            "gen-val/spectral_mmd",
            "gen-val/sbm_accuracy",
        ):
            assert metric in logged_dict, f"missing prefixed key {metric!r}"
        for unprefixed in (
            "degree_mmd",
            "clustering_mmd",
            "spectral_mmd",
            "sbm_accuracy",
        ):
            assert (
                unprefixed not in logged_dict
            ), f"unprefixed key {unprefixed!r} leaked"
        assert logged_dict["gen-val/sbm_accuracy"] == 0.85
        assert logged_dict["trainer/global_step"] == task["global_step"]

        # Manifest row must carry the prefixed names too -- this is what
        # the smoke verification scripts grep for.
        rows = _eval_manifest_read(task["manifest_path"])
        assert len(rows) == 1
        assert rows[0]["status"] == "completed"
        assert "gen-val/degree_mmd" in rows[0]["metrics"]
        assert rows[0]["metrics"]["gen-val/sbm_accuracy"] == 0.85

    def test_already_prefixed_metrics_pass_through_unchanged(
        self, tmp_path: Path
    ) -> None:
        """Future CLIs that emit already-namespaced keys must not get double-prefixed."""
        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        eval_output = _success_eval_output(task)

        with (
            patch.object(
                evaluate_async, "run_mmd_evaluation", return_value=eval_output
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init"),
            patch("tmgg.modal._lib.evaluate_async.wandb.log") as mock_log,
            patch("tmgg.modal._lib.evaluate_async.wandb.finish"),
        ):
            evaluate_async.evaluate_mmd_async(task)

        logged_dict = mock_log.call_args.args[0]
        # Sanity: no "gen-val/gen-val/..." double-prefix.
        for key in logged_dict:
            assert not key.startswith("gen-val/gen-val/"), key


class TestAsyncEvalSurfacesCliFailures:
    """Bug #1 regression: ``run_mmd_evaluation`` wraps subprocess crashes
    in ``status="failed"`` dicts and never raises. The async worker
    previously logged those as "completed" with an empty metrics dict --
    masking the gaussian-CLI dispatch bug for an entire smoke run.
    Now the worker must raise and write a ``failed`` manifest row."""

    def test_failed_eval_output_produces_failed_row_and_raises(
        self, tmp_path: Path
    ) -> None:
        import pytest

        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        manifest_path = Path(task["manifest_path"])
        # run_mmd_evaluation wraps subprocess crashes here -- no exception,
        # but status="failed" + empty results.
        failed_output = {
            "run_id": task["run_id"],
            "checkpoint_name": "step_5240",
            "status": "failed",
            "results": {},
            "error_message": "CLI exited with code 1: ModuleNotFoundError",
            "evaluation_params": {},
            "timestamp": "2026-04-29T12:00:00",
        }

        with (
            patch.object(
                evaluate_async, "run_mmd_evaluation", return_value=failed_output
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init") as mock_init,
            patch("tmgg.modal._lib.evaluate_async.wandb.log") as mock_log,
            patch("tmgg.modal._lib.evaluate_async.wandb.finish"),
            pytest.raises(RuntimeError, match="ModuleNotFoundError"),
        ):
            evaluate_async.evaluate_mmd_async(task)

        # No W&B attach should fire -- the failed eval has nothing to log.
        assert mock_init.call_count == 0
        assert mock_log.call_count == 0

        # Manifest must record the failure.
        rows = _eval_manifest_read(manifest_path)
        assert len(rows) == 1
        assert rows[0]["status"] == "failed"
        assert rows[0]["error_tail"] is not None
        assert "ModuleNotFoundError" in rows[0]["error_tail"]


class TestRunMmdEvaluationDispatchesByDiffusionFamily:
    """Bug #1 regression: ``run_mmd_evaluation`` must route to the eval
    CLI matching the trained run's diffusion family. The previous build
    hard-coded the deleted ``gaussian_diffusion_generative.evaluate_checkpoint``
    module, which silently exit-1'd for every discrete-diffusion smoke."""

    def test_categorical_noise_routes_to_discrete_evaluate_cli(
        self, tmp_path: Path
    ) -> None:
        import subprocess

        import yaml

        from tmgg.modal._lib import evaluate as evaluate_mod

        # Stage a fake run dir on the volume mount: config.yaml + ckpt.
        run_id = "fake-run/discrete-smoke"
        run_dir = tmp_path / run_id
        (run_dir / "checkpoints").mkdir(parents=True)
        ckpt = run_dir / "checkpoints" / "step_200.ckpt"
        ckpt.write_bytes(b"")  # presence is enough; subprocess is mocked
        config_yaml = run_dir / "config.yaml"
        config_yaml.write_text(
            yaml.safe_dump(
                {
                    "model": {
                        "_target_": "tmgg.training.lightning_modules.diffusion_module.DiffusionModule",
                        "noise_process": {
                            "_target_": "tmgg.diffusion.noise_process.CategoricalNoiseProcess"
                        },
                    },
                    "data": {"graph_type": "sbm", "num_nodes": 20},
                }
            )
        )

        # Capture the cli_args that run_mmd_evaluation tries to subprocess.
        captured: dict[str, Any] = {}

        class _DummyCompleted:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cli_args, capture_output=True, text=True):  # noqa: ARG001
            captured["cli_args"] = list(cli_args)
            # Write the JSON the real CLI would produce.
            output_idx = cli_args.index("--output")
            output_path = Path(cli_args[output_idx + 1])
            output_path.write_text(
                json.dumps(
                    {
                        "mmd_results": {
                            "degree_mmd": 0.01,
                            "clustering_mmd": 0.02,
                            "spectral_mmd": 0.03,
                        }
                    }
                )
            )
            return _DummyCompleted()

        with (
            patch.object(evaluate_mod, "OUTPUTS_MOUNT", str(tmp_path)),
            patch.object(subprocess, "run", side_effect=fake_run),
        ):
            result = evaluate_mod.run_mmd_evaluation(
                {
                    "run_id": run_id,
                    "checkpoint_path": str(ckpt),
                    "num_samples": 40,
                    "num_steps": 100,
                    "mmd_kernel": "gaussian_tv",
                    "mmd_sigma": 1.0,
                    "seed": 42,
                }
            )

        # Routed to the discrete CLI module, not the deleted gaussian one.
        assert (
            "tmgg.experiments.discrete_diffusion_generative.evaluate_cli"
            in captured["cli_args"]
        )
        assert (
            "tmgg.experiments.gaussian_diffusion_generative.evaluate_checkpoint"
            not in captured["cli_args"]
        )
        # And the result reflects the populated mmd_results, not an empty dict.
        assert result["status"] == "completed"
        assert result["results"] == {
            "eval": {
                "degree_mmd": 0.01,
                "clustering_mmd": 0.02,
                "spectral_mmd": 0.03,
            }
        }

    def test_unknown_diffusion_family_raises_loudly(self, tmp_path: Path) -> None:
        import pytest
        import yaml

        from tmgg.modal._lib import evaluate as evaluate_mod

        run_id = "fake-run/gaussian"
        run_dir = tmp_path / run_id
        (run_dir / "checkpoints").mkdir(parents=True)
        ckpt = run_dir / "checkpoints" / "step_200.ckpt"
        ckpt.write_bytes(b"")
        config_yaml = run_dir / "config.yaml"
        config_yaml.write_text(
            yaml.safe_dump(
                {
                    "model": {
                        "noise_process": {
                            "_target_": "tmgg.diffusion.noise_process.GaussianNoiseProcess"
                        }
                    },
                    "data": {"graph_type": "sbm"},
                }
            )
        )

        with (
            patch.object(evaluate_mod, "OUTPUTS_MOUNT", str(tmp_path)),
            pytest.raises(RuntimeError, match="diffusion family"),
        ):
            evaluate_mod.run_mmd_evaluation(
                {
                    "run_id": run_id,
                    "checkpoint_path": str(ckpt),
                    "num_samples": 1,
                    "num_steps": 1,
                    "mmd_kernel": "gaussian_tv",
                    "mmd_sigma": 1.0,
                    "seed": 0,
                }
            )


class TestRunMmdEvaluationResolvesPathsFromCheckpoint:
    """Bug 6 regression (smoke run 2026-04-29): the trainer-side callback
    passes ``run_id`` as the basename only (e.g. ``MyRun_xyz``) but
    ``checkpoint_path`` carries the full ``{experiment_name}/{run_id}``
    prefix (because Lightning's ``default_root_dir`` is set to that full
    path during training). Reconstructing ``output_dir`` from
    ``OUTPUTS_MOUNT/run_id`` therefore drops the experiment-name parent
    and the worker reads ``config.yaml`` from the wrong location, so the
    eval fails with ``Config not found``. The fix derives ``output_dir``
    from ``Path(checkpoint_path).parent.parent`` whenever a
    ``checkpoint_path`` is supplied.
    """

    def test_output_dir_derived_from_checkpoint_when_path_supplied(
        self, tmp_path: Path
    ) -> None:
        """Stage a fake on-volume layout where the experiment-name parent
        is REQUIRED to find ``config.yaml``. The ``run_id`` in the task
        dict is intentionally just the basename (matching what
        ``AsyncEvalSpawnCallback._derive_run_id`` actually emits)."""
        import subprocess

        import yaml

        from tmgg.modal._lib import evaluate as evaluate_mod

        # Real-world layout: /data/outputs/{experiment_name}/{run_id}/...
        # Mirror it under tmp_path. Crucially, run_id is the basename only.
        experiment_name = "discrete_diffusion"
        run_id = (
            "discrete_diffusion_DiffusionModule_lr2e-4_wd1e-12_s0_fresh_20260429T180418"
        )
        run_dir = tmp_path / experiment_name / run_id
        (run_dir / "checkpoints").mkdir(parents=True)
        ckpt = run_dir / "checkpoints" / "step_42.ckpt"
        ckpt.write_bytes(b"")
        config_yaml = run_dir / "config.yaml"
        config_yaml.write_text(
            yaml.safe_dump(
                {
                    "model": {
                        "noise_process": {
                            "_target_": "tmgg.diffusion.noise_process.CategoricalNoiseProcess"
                        }
                    },
                    "data": {"graph_type": "sbm"},
                }
            )
        )

        # If the worker tried OUTPUTS_MOUNT/run_id (legacy bug) it would
        # look at ``tmp_path/{run_id}/config.yaml`` which does NOT exist.
        legacy_path = tmp_path / run_id / "config.yaml"
        assert (
            not legacy_path.exists()
        ), "Test setup invariant: legacy single-level path must be absent."

        captured: dict[str, Any] = {}

        class _DummyCompleted:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cli_args, capture_output=True, text=True):  # noqa: ARG001
            captured["cli_args"] = list(cli_args)
            output_idx = cli_args.index("--output")
            output_path = Path(cli_args[output_idx + 1])
            output_path.write_text(json.dumps({"mmd_results": {"degree_mmd": 0.01}}))
            return _DummyCompleted()

        with (
            patch.object(evaluate_mod, "OUTPUTS_MOUNT", str(tmp_path)),
            patch.object(subprocess, "run", side_effect=fake_run),
        ):
            result = evaluate_mod.run_mmd_evaluation(
                {
                    "run_id": run_id,  # basename only -- mirrors the trainer
                    "checkpoint_path": str(
                        ckpt
                    ),  # full {experiment_name}/{run_id} path
                    "num_samples": 40,
                    "num_steps": 100,
                    "mmd_kernel": "gaussian_tv",
                    "mmd_sigma": 1.0,
                    "seed": 42,
                }
            )

        # The eval must succeed because output_dir was derived from
        # checkpoint_path.parent.parent, not OUTPUTS_MOUNT/run_id.
        assert (
            result["status"] == "completed"
        ), f"Expected status='completed' (config resolved); got: {result}"
        # And the side-effect file (eval results JSON) lands in the right
        # run dir, which is two parents up from the checkpoint.
        eval_json = run_dir / "mmd_evaluation_step_42.json"
        assert (
            eval_json.exists()
        ), f"Combined eval results should be written to {eval_json}, but wasn't."

    def test_failed_status_message_points_to_correct_path_when_checkpoint_supplied(
        self, tmp_path: Path
    ) -> None:
        """If config.yaml truly is missing, the error must reference the
        derived path (under ``{experiment_name}/{run_id}/``) so operators
        can find the actual problem rather than chasing a phantom
        ``OUTPUTS_MOUNT/{run_id}`` location."""
        from tmgg.modal._lib import evaluate as evaluate_mod

        experiment_name = "discrete_diffusion"
        run_id = "MyRun_xyz"
        run_dir = tmp_path / experiment_name / run_id
        (run_dir / "checkpoints").mkdir(parents=True)
        ckpt = run_dir / "checkpoints" / "step_1.ckpt"
        ckpt.write_bytes(b"")
        # Deliberately do NOT create config.yaml.

        with patch.object(evaluate_mod, "OUTPUTS_MOUNT", str(tmp_path)):
            result = evaluate_mod.run_mmd_evaluation(
                {
                    "run_id": run_id,
                    "checkpoint_path": str(ckpt),
                    "num_samples": 1,
                    "num_steps": 1,
                    "mmd_kernel": "gaussian_tv",
                    "mmd_sigma": 1.0,
                    "seed": 0,
                }
            )

        assert result["status"] == "failed"
        # The error message should reference the path under the
        # experiment-name parent, NOT the legacy single-level layout.
        expected_path = str(run_dir / "config.yaml")
        legacy_path = str(tmp_path / run_id / "config.yaml")
        assert expected_path in result["error_message"], (
            f"Error must cite the derived path {expected_path!r}; "
            f"got: {result['error_message']!r}"
        )
        assert legacy_path not in result["error_message"], (
            f"Error must NOT cite the legacy single-level path {legacy_path!r}; "
            f"got: {result['error_message']!r}"
        )

    def test_legacy_path_fallback_when_no_checkpoint_supplied(
        self, tmp_path: Path
    ) -> None:
        """Manual-CLI invocations (no spawn, no checkpoint_path) keep the
        legacy ``OUTPUTS_MOUNT/{run_id}`` layout. Documents the carve-out."""
        from tmgg.modal._lib import evaluate as evaluate_mod

        run_id = "manual-run"
        run_dir = tmp_path / run_id
        run_dir.mkdir(parents=True)
        # No config, no checkpoint -- failure path is fine; we only care
        # that the *path* the failure reports matches the legacy layout.

        with patch.object(evaluate_mod, "OUTPUTS_MOUNT", str(tmp_path)):
            result = evaluate_mod.run_mmd_evaluation(
                {
                    "run_id": run_id,
                    "checkpoint_path": None,
                    "num_samples": 1,
                    "num_steps": 1,
                    "mmd_kernel": "gaussian_tv",
                    "mmd_sigma": 1.0,
                    "seed": 0,
                }
            )

        assert result["status"] == "failed"
        assert str(run_dir / "config.yaml") in result["error_message"]

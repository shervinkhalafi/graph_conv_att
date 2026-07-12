"""Tests for the tmgg-discrete-eval-all CLI (parity D-16c).

Test rationale
--------------
The CLI is an orchestration layer over
:func:`tmgg.experiments.discrete_diffusion_generative.evaluate_cli.evaluate_checkpoint`:
discover ``*.ckpt`` files, sort by training step parsed from the
filename, optionally skip work already in the CSV, call the per-
checkpoint evaluator, accumulate rows, write CSV. We test:

* Argparse flag parsing (defaults + overrides).
* Step extraction from canonical Lightning filenames (Q11-relevant).
* Device autodetect: ``--device auto`` picks ``cuda`` when available,
  ``cpu`` otherwise (spec Q10).
* EMA detection probe on a synthetic checkpoint payload (spec Q7).
* End-to-end CLI smoke: two stub checkpoints in a directory + a
  monkey-patched evaluate_checkpoint produce a 2-row CSV.
* ``--skip_existing``: pre-existing rows persist; new checkpoints
  appended; no double-eval.
* CSV column drift safety: re-running with an extended metric set
  surfaces new columns with ``NaN`` in older rows (pandas native).
* Corrupt-checkpoint fault tolerance (warning row, not abort).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tmgg.experiments.discrete_diffusion_generative.evaluate_all_cli import (
    _checkpoint_has_ema_shadow,
    _parse_step_from_filename,
    _resolve_device,
    evaluate_all_checkpoints,
    main,
)

# ---------------------------------------------------------------------------
# Filename / device helpers
# ---------------------------------------------------------------------------


class TestParseStep:
    def test_parses_canonical_lightning_filename(self) -> None:
        path = Path("model-step=000123-val_loss=0.4567.ckpt")
        assert _parse_step_from_filename(path) == 123

    def test_returns_minus_one_on_unparseable(self) -> None:
        path = Path("checkpoint_random_name.ckpt")
        assert _parse_step_from_filename(path) == -1


class TestResolveDevice:
    def test_explicit_cpu_passes_through(self) -> None:
        assert _resolve_device("cpu") == "cpu"

    def test_explicit_cuda_passes_through(self) -> None:
        assert _resolve_device("cuda") == "cuda"

    def test_auto_picks_cuda_zero_when_available(self) -> None:
        # D-16c Resolutions Q10 specifies "cuda:0" (not bare "cuda") as
        # the autodetect default, so a downstream caller selecting a
        # specific GPU index gets a deterministic, fully-qualified
        # device string.
        with patch("torch.cuda.is_available", return_value=True):
            assert _resolve_device("auto") == "cuda:0"

    def test_auto_falls_back_to_cpu(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            assert _resolve_device("auto") == "cpu"


# ---------------------------------------------------------------------------
# EMA detection probe
# ---------------------------------------------------------------------------


class TestEmaShadowProbe:
    def test_detects_ema_callback_state_key(self) -> None:
        ckpt = {
            "callbacks": {
                "EMACallback{decay=0.999}": {"shadow": []},
            }
        }
        assert _checkpoint_has_ema_shadow(ckpt)

    def test_returns_false_without_callbacks_block(self) -> None:
        ckpt = {"state_dict": {}}
        assert _checkpoint_has_ema_shadow(ckpt) is False

    def test_returns_false_when_no_ema_key(self) -> None:
        ckpt = {
            "callbacks": {
                "ModelCheckpoint{monitor=val_loss}": {},
            }
        }
        assert _checkpoint_has_ema_shadow(ckpt) is False


# ---------------------------------------------------------------------------
# Argparse parsing
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_required_args_only(self, tmp_path: Path) -> None:
        # main() returns int and triggers a real evaluation; we instead
        # exercise the argparse layer directly by importing the parser
        # via a thin invocation that fails before running. Rather than
        # invoke main, the test below exercises evaluate_all_checkpoints
        # which is the unit under test.
        _ = tmp_path

    def test_defaults_pick_up_in_main_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Run main() against an empty run dir to surface the FileNotFoundError.

        We cannot run a real eval here -- the test exercises that the
        argparse parser accepts the canonical argument set without
        crashing on flag wiring.
        """
        ckpt_dir = tmp_path
        # An empty directory (no .ckpt files) should raise
        # FileNotFoundError from evaluate_all_checkpoints.
        with pytest.raises(FileNotFoundError, match="No files matched"):
            main(
                [
                    "--run_dir",
                    str(ckpt_dir),
                    "--num_samples",
                    "1",
                ]
            )
        _ = capsys, monkeypatch


# ---------------------------------------------------------------------------
# evaluate_all_checkpoints validity gates
# ---------------------------------------------------------------------------


class TestValidityGates:
    def test_missing_run_dir_raises(self, tmp_path: Path) -> None:
        nope = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="run_dir does not exist"):
            evaluate_all_checkpoints(nope, num_samples=1)

    def test_zero_num_samples_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="num_samples must be >= 1"):
            evaluate_all_checkpoints(tmp_path, num_samples=0)


# ---------------------------------------------------------------------------
# End-to-end CLI smoke
# ---------------------------------------------------------------------------


def _make_dummy_checkpoint(path: Path) -> None:
    """Touch a placeholder .ckpt file (content doesn't matter under mock)."""
    path.write_bytes(b"\x80\x04N.")  # tiny pickle of None


def _stub_evaluate_checkpoint(
    *,
    extra_metrics: dict[str, float] | None = None,
    fail_with: type[Exception] | None = None,
) -> Any:
    """Build a stub for evaluate_cli.evaluate_checkpoint.

    Mirrors the post-D-16c-cleanup signature (no ``dataset_type`` /
    ``num_nodes`` / ``seed``; reference_set + use_ema in their place).
    The stub records every call's kwargs in a closure-captured list so
    tests can assert on argument propagation.
    """

    calls: list[dict[str, Any]] = []

    def stub(
        checkpoint_path: Any,
        num_samples: int = 500,
        reference_set: str = "val",
        use_ema: str = "auto",
        mmd_kernel: str = "gaussian_tv",
        mmd_sigma: float = 1.0,
        device: str = "cpu",
    ) -> dict[str, Any]:
        calls.append(
            {
                "checkpoint_path": Path(checkpoint_path),
                "num_samples": num_samples,
                "reference_set": reference_set,
                "use_ema": use_ema,
                "mmd_kernel": mmd_kernel,
                "mmd_sigma": mmd_sigma,
                "device": device,
            }
        )
        if fail_with is not None:
            raise fail_with("simulated failure")
        metrics = {"degree_mmd": 0.1, "clustering_mmd": 0.2, "spectral_mmd": 0.3}
        if extra_metrics:
            metrics.update(extra_metrics)
        return {
            "checkpoint_path": str(Path(checkpoint_path).absolute()),
            "checkpoint_name": Path(checkpoint_path).name,
            "reference_set": reference_set,
            "num_generated": num_samples,
            "ema_active": use_ema == "true",
            "mmd_results": metrics,
        }

    stub.calls = calls  # type: ignore[attr-defined]
    return stub


class TestEndToEndSmoke:
    def test_writes_csv_for_two_checkpoints(self, tmp_path: Path) -> None:
        run_dir = tmp_path
        _make_dummy_checkpoint(run_dir / "model-step=000010-val_loss=0.5.ckpt")
        _make_dummy_checkpoint(run_dir / "model-step=000020-val_loss=0.4.ckpt")

        stub = _stub_evaluate_checkpoint()
        with patch(
            "tmgg.experiments.discrete_diffusion_generative.evaluate_all_cli.evaluate_checkpoint",
            stub,
        ):
            df = evaluate_all_checkpoints(
                run_dir,
                num_samples=2,
                device="cpu",
                use_ema="false",
            )
        assert len(df) == 2
        # Sorted by step ascending.
        assert df["step"].tolist() == [10, 20]
        # CSV exists with the documented default name.
        assert (run_dir / "all_checkpoints_eval.csv").exists()
        assert "degree_mmd" in df.columns

    def test_skip_existing_does_not_double_eval(self, tmp_path: Path) -> None:
        run_dir = tmp_path
        _make_dummy_checkpoint(run_dir / "model-step=000010-val_loss=0.5.ckpt")
        _make_dummy_checkpoint(run_dir / "model-step=000020-val_loss=0.4.ckpt")

        stub = _stub_evaluate_checkpoint()
        with patch(
            "tmgg.experiments.discrete_diffusion_generative.evaluate_all_cli.evaluate_checkpoint",
            stub,
        ):
            evaluate_all_checkpoints(
                run_dir,
                num_samples=2,
                device="cpu",
                use_ema="false",
            )
            calls_first_pass = list(stub.calls)  # type: ignore[attr-defined]
            evaluate_all_checkpoints(
                run_dir,
                num_samples=2,
                device="cpu",
                use_ema="false",
                skip_existing=True,
            )
            calls_second_pass = list(stub.calls)  # type: ignore[attr-defined]

        # Second pass should not have invoked the stub again -- the CSV
        # already covered both checkpoints.
        assert len(calls_first_pass) == 2
        assert len(calls_second_pass) == 2  # unchanged

    def test_corrupt_checkpoint_yields_warning_row(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_dir = tmp_path
        _make_dummy_checkpoint(run_dir / "model-step=000005-val_loss=0.6.ckpt")

        stub = _stub_evaluate_checkpoint(fail_with=RuntimeError)
        with patch(
            "tmgg.experiments.discrete_diffusion_generative.evaluate_all_cli.evaluate_checkpoint",
            stub,
        ):
            df = evaluate_all_checkpoints(
                run_dir,
                num_samples=2,
                device="cpu",
                use_ema="false",
            )
        assert len(df) == 1
        assert df.iloc[0]["num_samples_generated"] == 0
        captured = capsys.readouterr()
        assert "warning" in captured.err.lower()

    def test_csv_column_drift_pandas_union(self, tmp_path: Path) -> None:
        """A future metric addition surfaces as a new column with NaN in old rows."""
        import pandas as pd

        run_dir = tmp_path
        _make_dummy_checkpoint(run_dir / "model-step=000010-val_loss=0.5.ckpt")
        _make_dummy_checkpoint(run_dir / "model-step=000020-val_loss=0.4.ckpt")

        stub_no_orbit = _stub_evaluate_checkpoint()
        with patch(
            "tmgg.experiments.discrete_diffusion_generative.evaluate_all_cli.evaluate_checkpoint",
            stub_no_orbit,
        ):
            evaluate_all_checkpoints(
                run_dir,
                num_samples=2,
                device="cpu",
                use_ema="false",
            )

        # Imagine evaluator added orbit_mmd between runs: only the
        # newly-evaluated checkpoint reports it.
        _make_dummy_checkpoint(run_dir / "model-step=000030-val_loss=0.3.ckpt")
        stub_with_orbit = _stub_evaluate_checkpoint(extra_metrics={"orbit_mmd": 0.05})
        with patch(
            "tmgg.experiments.discrete_diffusion_generative.evaluate_all_cli.evaluate_checkpoint",
            stub_with_orbit,
        ):
            evaluate_all_checkpoints(
                run_dir,
                num_samples=2,
                device="cpu",
                use_ema="false",
                skip_existing=True,
            )

        # Re-loading the CSV should expose orbit_mmd as a column with
        # NaN in the old rows. Use pure-Python record access so pyright
        # doesn't choke on pandas's loose Series typing.
        records: list[dict[str, Any]] = pd.read_csv(
            run_dir / "all_checkpoints_eval.csv"
        ).to_dict(orient="records")
        assert any("orbit_mmd" in row for row in records)
        by_step = {int(row["step"]): row for row in records}
        # Pre-existing rows have NaN (or missing) orbit_mmd.
        for step in (10, 20):
            value = by_step[step].get("orbit_mmd")
            # NaN != NaN, so the cleanest check is the float-NaN identity.
            assert value is None or (isinstance(value, float) and value != value)
        # Newly-added row has the populated value.
        assert by_step[30]["orbit_mmd"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Flag wiring (post D-16c cleanup)
# ---------------------------------------------------------------------------


class TestFlagWiring:
    """The CLI must pass --reference_set and --use_ema through to evaluate_checkpoint."""

    def test_default_reference_set_is_val(self, tmp_path: Path) -> None:
        run_dir = tmp_path
        _make_dummy_checkpoint(run_dir / "model-step=000010.ckpt")
        stub = _stub_evaluate_checkpoint()
        with patch(
            "tmgg.experiments.discrete_diffusion_generative.evaluate_all_cli.evaluate_checkpoint",
            stub,
        ):
            evaluate_all_checkpoints(
                run_dir, num_samples=2, device="cpu", use_ema="false"
            )
        calls = stub.calls  # type: ignore[attr-defined]
        assert len(calls) == 1
        assert calls[0]["reference_set"] == "val"

    def test_explicit_reference_set_test_propagates(self, tmp_path: Path) -> None:
        run_dir = tmp_path
        _make_dummy_checkpoint(run_dir / "model-step=000010.ckpt")
        stub = _stub_evaluate_checkpoint()
        with patch(
            "tmgg.experiments.discrete_diffusion_generative.evaluate_all_cli.evaluate_checkpoint",
            stub,
        ):
            evaluate_all_checkpoints(
                run_dir,
                num_samples=2,
                reference_set="test",
                device="cpu",
                use_ema="false",
            )
        calls = stub.calls  # type: ignore[attr-defined]
        assert calls[0]["reference_set"] == "test"

    def test_use_ema_flag_propagates(self, tmp_path: Path) -> None:
        run_dir = tmp_path
        _make_dummy_checkpoint(run_dir / "model-step=000010.ckpt")
        for use_ema_value in ("auto", "true", "false"):
            stub = _stub_evaluate_checkpoint()
            with patch(
                "tmgg.experiments.discrete_diffusion_generative.evaluate_all_cli.evaluate_checkpoint",
                stub,
            ):
                evaluate_all_checkpoints(
                    run_dir,
                    num_samples=2,
                    device="cpu",
                    use_ema=use_ema_value,  # type: ignore[arg-type]
                )
            calls = stub.calls  # type: ignore[attr-defined]
            assert calls[-1]["use_ema"] == use_ema_value

    def test_main_argv_passes_reference_set_to_evaluate_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """Hit the full argparse layer so the --reference_set flag is wired in."""
        run_dir = tmp_path
        _make_dummy_checkpoint(run_dir / "model-step=000010.ckpt")
        stub = _stub_evaluate_checkpoint()
        with patch(
            "tmgg.experiments.discrete_diffusion_generative.evaluate_all_cli.evaluate_checkpoint",
            stub,
        ):
            main(
                [
                    "--run_dir",
                    str(run_dir),
                    "--num_samples",
                    "1",
                    "--reference_set",
                    "test",
                    "--use_ema",
                    "false",
                    "--device",
                    "cpu",
                ]
            )
        calls = stub.calls  # type: ignore[attr-defined]
        assert calls[0]["reference_set"] == "test"
        assert calls[0]["use_ema"] == "false"

    def test_ema_active_column_reflects_evaluator_result(self, tmp_path: Path) -> None:
        """ema_active CSV column comes from the evaluator's own ema_active flag."""
        run_dir = tmp_path
        _make_dummy_checkpoint(run_dir / "model-step=000010.ckpt")
        # Stub evaluator returns ema_active=True when use_ema=='true'.
        stub = _stub_evaluate_checkpoint()
        with patch(
            "tmgg.experiments.discrete_diffusion_generative.evaluate_all_cli.evaluate_checkpoint",
            stub,
        ):
            df = evaluate_all_checkpoints(
                run_dir,
                num_samples=2,
                device="cpu",
                use_ema="true",
            )
        assert len(df) == 1
        assert bool(df.iloc[0]["ema_active"]) is True

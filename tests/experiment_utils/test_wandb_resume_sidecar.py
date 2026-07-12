"""Tests for the W&B run-id sidecar helpers used to resume the same W&B
run across Modal-preempt restarts of a named (explicit ``+run_id=``) run.

The sidecar is a single-line text file at
``<output_dir>/wandb_run_id.txt`` containing the W&B internal run id.
``run_experiment`` writes it once after the W&B logger materializes
(triggers ``wandb.init``) and reads it on the next launch into the same
output directory; ``create_loggers`` then passes ``id=`` to
``WandbLogger`` so the W&B server appends to the same run instead of
fragmenting the dashboard into multiple same-named runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tmgg.training.orchestration.run_experiment import (
    _commit_outputs_volume_if_modal,
    _persist_wandb_run_id_sidecar,
    _read_wandb_run_id_sidecar,
    _write_wandb_run_id_sidecar,
)


class TestReadWandbRunIdSidecar:
    """Cover the read path: missing, present, edge cases."""

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        """Empty output dir → no sidecar → ``None``.

        Rationale: first launch of a new named run has nothing to resume.
        The caller treats ``None`` as "do not pass ``id=`` to WandbLogger,
        let it create a fresh run."
        """
        assert _read_wandb_run_id_sidecar(tmp_path) is None

    def test_returns_value_when_present(self, tmp_path: Path) -> None:
        """Existing sidecar → trimmed run id."""
        (tmp_path / "wandb_run_id.txt").write_text("abc123\n")
        assert _read_wandb_run_id_sidecar(tmp_path) == "abc123"

    def test_strips_whitespace_and_newlines(self, tmp_path: Path) -> None:
        """Trailing whitespace introduced by editor / shell pipes is harmless."""
        (tmp_path / "wandb_run_id.txt").write_text("  xyz789  \n\n")
        assert _read_wandb_run_id_sidecar(tmp_path) == "xyz789"

    def test_returns_none_when_file_empty(self, tmp_path: Path) -> None:
        """Empty file is treated identical to a missing file.

        Rationale: a half-written sidecar after a hard preempt should not
        cause the next launch to pass an empty ``id=""`` to WandbLogger
        (which W&B would reject). Falling through to "no resume" is
        operator-recoverable: a fresh W&B run is started, written into
        the sidecar, and the resume path heals on the next preempt cycle.
        """
        (tmp_path / "wandb_run_id.txt").write_text("")
        assert _read_wandb_run_id_sidecar(tmp_path) is None


class TestWriteWandbRunIdSidecar:
    """Cover the write path: creates the file with the id and a trailing newline."""

    def test_creates_file_with_id(self, tmp_path: Path) -> None:
        _write_wandb_run_id_sidecar(tmp_path, "abc123")
        sidecar = tmp_path / "wandb_run_id.txt"
        assert sidecar.exists()
        assert sidecar.read_text() == "abc123\n"

    def test_overwrites_existing_value_idempotently(self, tmp_path: Path) -> None:
        """Writing the same id twice is a no-op on the file contents.

        Rationale: ``run_experiment`` calls the persist helper on every
        launch, including resume launches where the id is the same as
        what's already on disk. The helper must be safe to call
        repeatedly without growing the file or duplicating lines.
        """
        _write_wandb_run_id_sidecar(tmp_path, "abc123")
        _write_wandb_run_id_sidecar(tmp_path, "abc123")
        assert (tmp_path / "wandb_run_id.txt").read_text() == "abc123\n"

    def test_overwrites_with_new_value(self, tmp_path: Path) -> None:
        """A second call with a different id replaces the old contents.

        Rationale: when the user passes ``force_fresh=true`` to wipe a
        named run, the wipe gate in ``run_experiment`` rmtree's the
        directory (sidecar included), the next ``WandbLogger`` init
        creates a fresh wandb run with a new id, and the persist helper
        writes that new id. Test guards against accidental append-only
        behavior that would leave stale ids stacked in the file.
        """
        _write_wandb_run_id_sidecar(tmp_path, "abc123")
        _write_wandb_run_id_sidecar(tmp_path, "xyz789")
        assert (tmp_path / "wandb_run_id.txt").read_text() == "xyz789\n"


class TestPersistWandbRunIdSidecar:
    """Cover the logger-walking helper that's called from run_experiment.

    The helper walks the ``loggers`` list, finds the first
    ``WandbLogger`` (if any), reads ``experiment.id``, and writes the
    sidecar. Tests use a fake WandbLogger subclass injected via
    monkeypatch so we don't have to actually start a wandb process.
    """

    def _make_fake_wandb_logger(self, fake_id: str):
        """Return an instance of a fake WandbLogger that satisfies
        ``isinstance(lg, WandbLogger)`` inside the helper.

        Lightning's ``WandbLogger.__init__`` triggers ``wandb.init`` which
        we don't want in tests. Subclassing the real class but overriding
        ``__init__`` keeps the type relationship intact (so the helper's
        ``isinstance`` gate fires) while bypassing the network call.
        """
        from pytorch_lightning.loggers import WandbLogger

        class _FakeWandbLogger(WandbLogger):
            def __init__(self) -> None:  # noqa: D401 - skip parent init
                # Deliberately skip super().__init__: the real init calls
                # wandb.init() which would require credentials and a
                # network. The helper only reads ``self.experiment.id``,
                # so we provide a stub object that exposes ``.id``.
                self._fake_experiment = type("_FakeExperiment", (), {"id": fake_id})()

            @property  # type: ignore[override]
            def experiment(self) -> Any:  # type: ignore[override]
                return self._fake_experiment

        return _FakeWandbLogger()

    def test_writes_sidecar_when_wandb_logger_present(self, tmp_path: Path) -> None:
        fake = self._make_fake_wandb_logger("wandb_id_42")
        _persist_wandb_run_id_sidecar([fake], tmp_path)
        assert (tmp_path / "wandb_run_id.txt").read_text() == "wandb_id_42\n"

    def test_no_op_when_no_wandb_logger(self, tmp_path: Path) -> None:
        """List without a WandbLogger → no sidecar created, no error."""
        from pytorch_lightning.loggers import CSVLogger

        csv_only = CSVLogger(save_dir=str(tmp_path))
        _persist_wandb_run_id_sidecar([csv_only], tmp_path)
        assert not (tmp_path / "wandb_run_id.txt").exists()

    def test_no_op_with_empty_logger_list(self, tmp_path: Path) -> None:
        """Empty list → no sidecar, no error.

        Rationale: ``allow_no_wandb=true`` configurations may end up with
        zero loggers (e.g. wandb skipped, csv off). The persist helper
        must not crash that path.
        """
        _persist_wandb_run_id_sidecar([], tmp_path)
        assert not (tmp_path / "wandb_run_id.txt").exists()

    def test_swallows_experiment_id_failure(self, tmp_path: Path) -> None:
        """If ``experiment.id`` raises (e.g. wandb auth failure mid-init),
        the helper logs a warning and returns instead of faulting the
        training launch. The sidecar is best-effort, not load-bearing.
        """
        from pytorch_lightning.loggers import WandbLogger

        class _ExplodingLogger(WandbLogger):
            def __init__(self) -> None:  # noqa: D401
                pass  # skip real init

            @property  # type: ignore[override]
            def experiment(self) -> Any:  # type: ignore[override]
                raise RuntimeError("simulated wandb init failure")

        _persist_wandb_run_id_sidecar([_ExplodingLogger()], tmp_path)
        # Helper survives; sidecar was not written.
        assert not (tmp_path / "wandb_run_id.txt").exists()


class TestCommitOutputsVolumeIfModal:
    """The volume-commit helper must be a no-op outside the Modal worker."""

    def test_no_op_outside_modal_does_not_raise(self) -> None:
        """Running locally (no ``modal`` runtime) must not propagate errors.

        Rationale: ``_persist_wandb_run_id_sidecar`` calls this helper on
        every launch. In CI / local pytest, the modal client is installed
        but ``Volume.from_name`` either fails to authenticate or hits a
        local-only error path. Either way the orchestrator must not die.
        """
        _commit_outputs_volume_if_modal()  # must not raise

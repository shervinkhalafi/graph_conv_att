"""Tests for append-only experiment confirmation log.

Test rationale
--------------
The confirmation log provides a backup record of experiment execution
on the Modal volume, independent of W&B. It must be append-only (safe
for concurrent writers) and parseable for status queries.

Invariants
----------
- Each append produces exactly one newline-terminated JSON line.
- Reading back parses all lines correctly.
- Multiple appends accumulate (no overwriting).
- ``run_id`` is present in every entry.
"""

import json
from pathlib import Path

from tmgg.modal._lib.confirmation import append_confirmation, read_confirmations


class TestAppendConfirmation:
    def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        log_path = tmp_path / "confirmation.jsonl"
        append_confirmation(log_path, run_id="abc123", status="started")
        assert log_path.exists()

    def test_appends_valid_jsonl(self, tmp_path: Path) -> None:
        log_path = tmp_path / "confirmation.jsonl"
        append_confirmation(log_path, run_id="abc123", status="started")
        append_confirmation(log_path, run_id="abc123", status="completed", exit_code=0)

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        entry_0 = json.loads(lines[0])
        assert entry_0["run_id"] == "abc123"
        assert entry_0["status"] == "started"
        assert "timestamp" in entry_0

        entry_1 = json.loads(lines[1])
        assert entry_1["status"] == "completed"
        assert entry_1["exit_code"] == 0

    def test_includes_wandb_run_id_when_provided(self, tmp_path: Path) -> None:
        log_path = tmp_path / "confirmation.jsonl"
        append_confirmation(
            log_path, run_id="abc123", status="completed", wandb_run_id="wandb-xyz"
        )
        entry = json.loads(log_path.read_text().strip())
        assert entry["wandb_run_id"] == "wandb-xyz"

    def test_extra_fields_preserved(self, tmp_path: Path) -> None:
        log_path = tmp_path / "confirmation.jsonl"
        append_confirmation(
            log_path, run_id="abc123", status="completed", cmd="tmgg-discrete-gen"
        )
        entry = json.loads(log_path.read_text().strip())
        assert entry["cmd"] == "tmgg-discrete-gen"


class TestReadConfirmations:
    def test_reads_all_entries(self, tmp_path: Path) -> None:
        log_path = tmp_path / "confirmation.jsonl"
        append_confirmation(log_path, run_id="r1", status="started")
        append_confirmation(log_path, run_id="r1", status="completed")
        append_confirmation(log_path, run_id="r2", status="started")

        entries = read_confirmations(log_path)
        assert len(entries) == 3

    def test_filter_by_run_id(self, tmp_path: Path) -> None:
        log_path = tmp_path / "confirmation.jsonl"
        append_confirmation(log_path, run_id="r1", status="started")
        append_confirmation(log_path, run_id="r2", status="started")
        append_confirmation(log_path, run_id="r1", status="completed")

        entries = read_confirmations(log_path, run_id="r1")
        assert len(entries) == 2
        assert all(e["run_id"] == "r1" for e in entries)

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nonexistent.jsonl"
        assert read_confirmations(log_path) == []

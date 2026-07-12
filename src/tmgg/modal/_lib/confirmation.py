"""Append-only experiment confirmation log.

Writes one JSON line per status event to a JSONL file on the Modal volume.
Provides a lightweight backup record of experiment execution independent
of W&B. Each write appends a single newline-terminated JSON object.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_CONFIRMATION_PATH = Path("/data/outputs/confirmation.jsonl")


def append_confirmation(
    path: Path,
    *,
    run_id: str,
    status: str,
    wandb_run_id: str | None = None,
    **extra: Any,
) -> None:
    """Append a status entry to the confirmation log.

    Parameters
    ----------
    path : Path
        Path to the JSONL file.
    run_id : str
        Experiment run identifier.
    status : str
        One of ``"started"``, ``"completed"``, ``"failed"``.
    wandb_run_id : str or None
        W&B run ID if available (typically known only after completion).
    **extra
        Additional fields (``cmd``, ``exit_code``, ``error``, etc.).
    """
    entry: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
        "hostname": os.environ.get("HOSTNAME", "unknown"),
    }
    if wandb_run_id is not None:
        entry["wandb_run_id"] = wandb_run_id
    entry.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_confirmations(
    path: Path,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Read entries from the confirmation log.

    Parameters
    ----------
    path : Path
        Path to the JSONL file.
    run_id : str or None
        If provided, return only entries matching this run_id.

    Returns
    -------
    list[dict[str, Any]]
        Parsed entries, optionally filtered.
    """
    if not path.exists():
        return []

    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if run_id is None or entry.get("run_id") == run_id:
                entries.append(entry)
    return entries

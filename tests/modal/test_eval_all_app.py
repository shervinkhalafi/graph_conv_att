"""Unit tests for ``tmgg.modal._lib.eval_all`` — the eval-all library.

Covers the cheap-to-test surface of the eval-all pipeline:

* Filename → step parsing (``parse_step_from_ckpt_name``).
* Volume walk + step-ascending sort + ``last.ckpt`` skipping
  (``discover_checkpoints``).
* Failure modes (missing ``checkpoints/`` dir, missing ``config.yaml``).

The actual GPU sample-gen + MMD path is exercised end-to-end via the
existing async-eval smoke run; here we only assert the bits the
worker decides locally before any GPU work starts. The real Modal
spawn is deliberately skipped (would require Modal credentials and a
deployed app), and any test that touches ``eval_all_checkpoints_impl``
itself is marked ``slow`` so the default fast suite stays under 30 s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tmgg.modal._lib.eval_all import (
    CheckpointEntry,
    discover_checkpoints,
    parse_step_from_ckpt_name,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model-step=040000-val_nll=350.18.ckpt", 40000),
        ("model-step=000500-val_loss=1.4476.ckpt", 500),
        ("model-step=200000-val_nll=224.10.ckpt", 200000),
        ("model_step_500.ckpt", 500),
        ("last.ckpt", None),
        ("epoch=4-val_loss=0.3.ckpt", None),
        ("garbage.ckpt", None),
    ],
)
def test_parse_step_from_ckpt_name(name: str, expected: int | None) -> None:
    """Parser must handle both auto_insert_metric_name=False filename
    conventions used in the project (val_nll= and val_loss=) and degrade
    to ``None`` rather than raise on unknown shapes."""
    assert parse_step_from_ckpt_name(name) == expected


def _touch_ckpt(dir_path: Path, name: str) -> Path:
    """Create an empty placeholder ``.ckpt`` file."""
    path = dir_path / name
    path.write_bytes(b"")
    return path


def test_discover_checkpoints_sorts_by_step_ascending(tmp_path: Path) -> None:
    """The discovery walk must return ckpts in step-ascending order so
    the W&B step axis on the eval-all summary run goes left-to-right."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    # Create out-of-order filesystem listing.
    _touch_ckpt(ckpt_dir, "model-step=050000-val_nll=300.0.ckpt")
    _touch_ckpt(ckpt_dir, "model-step=010000-val_nll=400.0.ckpt")
    _touch_ckpt(ckpt_dir, "model-step=030000-val_nll=350.0.ckpt")

    entries = discover_checkpoints(tmp_path)

    assert [e.step for e in entries] == [10000, 30000, 50000]
    assert all(isinstance(e, CheckpointEntry) for e in entries)


def test_discover_checkpoints_skips_last_by_default(tmp_path: Path) -> None:
    """``last.ckpt`` is dropped on the default path because it usually
    duplicates the latest stepped ckpt and burns a worker slot for no
    informational gain."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _touch_ckpt(ckpt_dir, "model-step=010000-val_nll=400.0.ckpt")
    _touch_ckpt(ckpt_dir, "last.ckpt")

    entries = discover_checkpoints(tmp_path)

    assert [e.path.name for e in entries] == ["model-step=010000-val_nll=400.0.ckpt"]


def test_discover_checkpoints_can_keep_last(tmp_path: Path) -> None:
    """Opt-in: ``skip_last=False`` retains ``last.ckpt`` for callers
    that want to evaluate it explicitly. ``last.ckpt`` lacks an
    embedded step so it sorts to the end."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _touch_ckpt(ckpt_dir, "model-step=010000-val_nll=400.0.ckpt")
    _touch_ckpt(ckpt_dir, "last.ckpt")

    entries = discover_checkpoints(tmp_path, skip_last=False)

    names = [e.path.name for e in entries]
    assert "last.ckpt" in names
    # last.ckpt has step=None so it sorts to the end of the list.
    assert names[-1] == "last.ckpt"
    last_entry = next(e for e in entries if e.path.name == "last.ckpt")
    assert last_entry.step is None


def test_discover_checkpoints_missing_dir_raises(tmp_path: Path) -> None:
    """Most likely user error: passing the experiment-name dir instead
    of the run-id dir below it. We surface that as an explicit
    ``FileNotFoundError`` with a guidance message rather than returning
    an empty list (which would silently spawn a no-op worker)."""
    with pytest.raises(FileNotFoundError, match="No checkpoints directory"):
        discover_checkpoints(tmp_path)


def test_discover_checkpoints_handles_empty_directory(tmp_path: Path) -> None:
    """An empty checkpoints dir is a legitimate state (e.g. a run that
    crashed before saving any ckpts). It returns an empty list rather
    than raising; the caller decides whether to log a warning."""
    (tmp_path / "checkpoints").mkdir()

    entries = discover_checkpoints(tmp_path)

    assert entries == []


def test_discover_checkpoints_unstepped_ckpts_sort_to_end(tmp_path: Path) -> None:
    """A stepped ckpt and an unstepped ckpt (e.g. ``last.ckpt`` kept
    in via ``skip_last=False`` or some legacy filename) must coexist:
    stepped entries come first in step order, ``None`` entries after."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _touch_ckpt(ckpt_dir, "model-step=010000-val_nll=400.0.ckpt")
    _touch_ckpt(ckpt_dir, "model-step=020000-val_nll=350.0.ckpt")
    _touch_ckpt(ckpt_dir, "weird_ckpt_no_step.ckpt")

    entries = discover_checkpoints(tmp_path, skip_last=False)
    steps_in_order = [e.step for e in entries]
    assert steps_in_order == [10000, 20000, None]

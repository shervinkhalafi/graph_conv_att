"""Smoke tests for the ``modal_evaluate_mmd_async*`` wrapper triple.

Test rationale
--------------
Step 2 of the async-eval plan
(``internal async-eval design notes``) exposes
three Modal-decorated wrappers around
``tmgg.modal._lib.evaluate_async.evaluate_mmd_async``:

- ``modal_evaluate_mmd_async`` — default ``standard`` tier (A10G).
- ``modal_evaluate_mmd_async_debug`` — ``debug`` tier (T4).
- ``modal_evaluate_mmd_async_fast`` — ``fast`` tier (A100).

The trainer fires-and-forgets one of these on every scheduled eval
step via ``.spawn(...)``. These tests verify the wiring without
deploying:

1. The three names import cleanly and are callable.
2. Each is a real Modal ``Function`` (exposes ``.remote``, ``.spawn``,
   ``.local`` — the surface the trainer calls into).
3. Each function's ``spec.gpus`` matches the expected tier mapping
   (``A10G`` / ``T4`` / ``A100-40GB``). A drift in this mapping is the
   single most likely silent regression because Modal does not
   validate decorator args at import time.
4. ``.local(task)`` delegates to ``evaluate_mmd_async`` with the same
   task dict — guards against accidental re-routing through
   ``_evaluate_mmd_impl`` (the *sync* eval path), which would silently
   skip the W&B-resume and manifest-write logic.
"""

from __future__ import annotations

from unittest.mock import patch

import modal
import pytest

from tmgg.modal._functions import (
    modal_evaluate_mmd_async,
    modal_evaluate_mmd_async_debug,
    modal_evaluate_mmd_async_fast,
)


@pytest.mark.parametrize(
    "fn, expected_gpu",
    [
        (modal_evaluate_mmd_async, "A10G"),
        (modal_evaluate_mmd_async_debug, "T4"),
        (modal_evaluate_mmd_async_fast, "A100-40GB"),
    ],
)
def test_async_wrapper_is_modal_function_with_correct_gpu(fn, expected_gpu):
    """Each wrapper is a Modal ``Function`` decorated with the right GPU tier."""
    # ``@app.function`` replaces the def with a Modal ``Function``, which is
    # not Python-``callable`` directly — the trainer reaches it via ``.remote``,
    # ``.spawn``, or ``.local``. Assert on the type and on those entry points.
    assert isinstance(fn, modal.Function)
    assert hasattr(fn, "remote")
    assert hasattr(fn, "spawn")
    assert hasattr(fn, "local")
    assert fn.spec.gpus == expected_gpu


@pytest.mark.parametrize(
    "fn",
    [
        modal_evaluate_mmd_async,
        modal_evaluate_mmd_async_debug,
        modal_evaluate_mmd_async_fast,
    ],
)
def test_async_wrapper_delegates_to_evaluate_mmd_async(fn):
    """``.local(task)`` must hit ``evaluate_mmd_async`` with the same dict.

    Patches the symbol where it's imported (inside ``evaluate_async``)
    so the lazy ``from ... import evaluate_mmd_async`` inside the
    wrapper picks up the mock.
    """
    sentinel = {"ok": True}
    task = {"run_id": "smoke", "wandb_run_id": "abc"}

    with patch(
        "tmgg.modal._lib.evaluate_async.evaluate_mmd_async",
        return_value=sentinel,
    ) as mock_eval:
        result = fn.local(task)

    assert result is sentinel
    mock_eval.assert_called_once_with(task)

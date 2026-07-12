"""Tests for GPU tier -> Modal function name resolution.

Test rationale
--------------
GPU tier dispatch logic (``debug`` -> ``_debug`` suffix, ``fast|multi|h100`` ->
``_fast`` suffix, everything else -> base name) was duplicated across three
call-sites. ``resolve_modal_function_name`` consolidates that mapping into a
single pure function in ``modal/app.py``.

These tests verify the mapping for every known tier and for both base function
names used in the codebase (``modal_run_cli`` and ``modal_evaluate_mmd``).
"""

import pytest

from tmgg.modal.app import resolve_modal_function_name


@pytest.mark.parametrize(
    "base_name, gpu_tier, expected",
    [
        ("modal_run_cli", "debug", "modal_run_cli_debug"),
        ("modal_run_cli", "standard", "modal_run_cli"),
        ("modal_run_cli", "fast", "modal_run_cli_fast"),
        ("modal_run_cli", "multi", "modal_run_cli_fast"),
        ("modal_run_cli", "h100", "modal_run_cli_fast"),
        ("modal_evaluate_mmd", "debug", "modal_evaluate_mmd_debug"),
        ("modal_evaluate_mmd", "standard", "modal_evaluate_mmd"),
        ("modal_evaluate_mmd", "fast", "modal_evaluate_mmd_fast"),
        ("modal_evaluate_mmd", "multi", "modal_evaluate_mmd_fast"),
        ("modal_evaluate_mmd", "h100", "modal_evaluate_mmd_fast"),
    ],
)
def test_resolve_modal_function_name(base_name, gpu_tier, expected):
    assert resolve_modal_function_name(base_name, gpu_tier) == expected


def test_resolve_modal_function_name_rejects_unknown_tier():
    """Unknown GPU tiers raise ValueError rather than silently falling through."""
    with pytest.raises(ValueError, match="Unknown GPU tier"):
        resolve_modal_function_name("modal_run_cli", "typo_tier")

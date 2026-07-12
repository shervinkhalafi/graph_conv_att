"""Modal app constants and GPU configurations.

Pure constants — no ``import modal`` at module level. The actual
``modal.App`` instance lives in ``_functions.py``, the sole deployment
entry-point.
"""

# App name shared between deployment (_functions.py) and runtime
# lookups (runner.py uses modal.Function.from_name with this name).
MODAL_APP_NAME = "graph-denoising"

# GPU tier configurations
# Maps logical names to Modal GPU specifications (string format)
GPU_CONFIGS = {
    "debug": "T4",
    "standard": "A10G",
    "fast": "A100-40GB",
    "multi": "A100-40GB:2",
    "h100": "H100",
}

# Default timeouts per GPU tier (in seconds).
# 86400 s = 24 h is Modal's hard maximum
# (https://modal.com/docs/guide/timeouts: "users may specify timeout
# durations between 1 second and 24 hours"). Set to the ceiling so no
# training run is ever cut by the per-call timeout — only by
# explicit `max_steps` / `max_epochs`.
DEFAULT_TIMEOUTS = {
    "debug": 86400,
    "standard": 86400,
    "fast": 86400,
    "multi": 86400,
    "h100": 86400,
}

# How long containers stay warm after completing a task (in seconds).
# 2 s is Modal's hard minimum (validation requires the value to be in
# [2, 3600]). We pin at the floor: cold-start cost on this app is
# dominated by image fetch + GPU schedule, neither of which a longer
# warm-pool would meaningfully amortise across our typical inter-sweep
# gaps. Trading cheap container reuse for near-zero idle billing.
DEFAULT_SCALEDOWN_WINDOW = 2


def resolve_modal_function_name(base_name: str, gpu_tier: str) -> str:
    """Build the Modal function name for a given GPU tier.

    Maps logical GPU tier to the appropriate function name suffix:
    ``debug`` -> ``_debug``, ``fast|multi|h100`` -> ``_fast``,
    everything else (e.g. ``standard``) -> no suffix.

    Parameters
    ----------
    base_name : str
        Base function name without tier suffix (e.g. ``"modal_run_cli"``).
    gpu_tier : str
        Logical GPU tier from ``GPU_CONFIGS``.

    Returns
    -------
    str
        Full function name with appropriate suffix.
    """
    if gpu_tier not in GPU_CONFIGS:
        raise ValueError(
            f"Unknown GPU tier {gpu_tier!r}. Valid tiers: {sorted(GPU_CONFIGS)}"
        )
    if gpu_tier == "debug":
        return f"{base_name}_debug"
    if gpu_tier in ("fast", "multi", "h100"):
        return f"{base_name}_fast"
    return base_name


# Per-tier CPU reservations (vCPU floor; bursts up to host limit).
# Memory is intentionally left at Modal's default — RAM on Modal is
# billed for what is reserved, so let workers grow to host capacity
# instead of paying for unused headroom. CPU reservations matter more
# for guaranteeing dataloader / RDKit-preprocessing throughput.
CPU_PROFILES = {
    "debug": 2.0,
    "standard": 4.0,
    "fast": 8.0,
    "multi": 12.0,
    "h100": 8.0,
}

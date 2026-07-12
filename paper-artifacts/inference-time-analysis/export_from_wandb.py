# /// script
# dependencies = [
#   "wandb>=0.17",
#   "pandas>=2.0",
#   "numpy>=1.26",
# ]
# ///
"""Pull training-step and inference-cycle wall time from W&B into ``data/perf.csv``.

This refreshes the committed ``data/perf.csv`` that ``analyze.py`` reads. You
only need to run this if you want to re-fetch from W&B; the CSV is committed so
``analyze.py`` works offline out of the box.

What it measures
----------------
For each of the eight 2026-05-06 ``*_repro_exact`` panel runs (one model
variant per W&B project) we read the full ``impl-perf/train/step_time_s``
history plus the run summary, and derive one timing record per run:

* ``train_step_med`` (and p25/p75/mean/min) — robust per-step **training** wall
  time. The metric brackets Lightning's ``on_train_batch_start`` /
  ``on_train_batch_end`` hooks, so it covers forward + backward + optimizer and
  excludes dataloading and validation by construction.
* ``val_per_cycle_s`` — **inference (sampling) cost per validation cycle**,
  derived as ``runtime - train_step_med * final_step`` divided by the number of
  validation cycles. This is a wall-clock attribution: it bundles the
  diffusion sampling forward passes (T steps x many samples), MMD scoring,
  checkpoint I/O and logger sync. It is the only inference-time proxy the runs
  logged. Treat it as an upper bound on pure model-inference time, not a clean
  per-phase measurement. See README.md.

Authentication
--------------
Reads ``WANDB_API_KEY`` from the environment. If unset, it looks for a
``WANDB_API_KEY=...`` line in a ``.env`` file, searching this
folder and every parent directory. Run it however you like, e.g.::

    uv run export_from_wandb.py
    # or, with an explicit key:
    WANDB_API_KEY=... uv run export_from_wandb.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --- Panel definition -------------------------------------------------------
# (dataset, variant, wandb_project, run_id). Baseline variant is "vignac"
# (vanilla DiGress: ExtraFeatures(all) = cycles + Laplacian eigh). Verified
# against live W&B on 2026-06-19: these are the latest inference-time runs;
# all have compile_model=false and were preempted at ~17.7 h (state "failed").
ENTITY = "your-wandb-entity"
RUNS: list[tuple[str, str, str, str]] = [
    ("sbm", "vignac", "discrete-sbm-vignac-repro-exact", "cgfv3f85"),
    ("sbm", "pearl", "discrete-sbm-pearl-repro-exact", "k4iiw5sg"),
    ("sbm", "pearl-spectral", "discrete-sbm-pearl-spectral-repro-exact", "qukgm6zu"),
    (
        "sbm",
        "pearl-gnnconv-norm",
        "discrete-sbm-pearl-gnnconv-norm-repro-exact",
        "5qchu8c4",
    ),
    ("enzymes", "vignac", "discrete-enzymes-vignac-repro-exact", "8nhefhnl"),
    ("enzymes", "pearl", "discrete-enzymes-pearl-repro-exact", "7yi627fv"),
    (
        "enzymes",
        "pearl-spectral",
        "discrete-enzymes-pearl-spectral-repro-exact",
        "ths6e1da",
    ),
    (
        "enzymes",
        "pearl-gnnconv-norm",
        "discrete-enzymes-pearl-gnnconv-norm-repro-exact",
        "xsmz6yql",
    ),
]

STEP_KEY = "impl-perf/train/step_time_s"
HERE = Path(__file__).resolve().parent
OUT_CSV = HERE / "data" / "perf.csv"


def _load_api_key() -> None:
    """Export WANDB_API_KEY: from the env, else WANDB_API_KEY in a .env."""
    if os.environ.get("WANDB_API_KEY"):
        return
    for parent in [HERE, *HERE.parents]:
        env_path = parent / ".env"
        if not env_path.exists():
            continue
        m = re.search(
            r"^WANDB_API_KEY=(.+)$", env_path.read_text(), re.MULTILINE
        )
        if m is not None:
            os.environ["WANDB_API_KEY"] = m.group(1).strip()
            return
    raise RuntimeError(
        "WANDB_API_KEY not set and no .env with WANDB_API_KEY found "
        "in this folder or any parent. Export WANDB_API_KEY and re-run."
    )


def _eval_every_n_steps(cfg: dict) -> int:
    for key in ("eval_every_n_steps", "model/eval_every_n_steps"):
        if cfg.get(key) is not None:
            return int(cfg[key])
    raise KeyError("eval_every_n_steps not present in logged config")


def _cfg_int(cfg: dict, key: str, default: int | None = None) -> int | None:
    val = cfg.get(key)
    if val is None:
        return default
    return int(val)


def main() -> None:
    _load_api_key()
    import wandb

    api = wandb.Api()
    rows: list[dict] = []
    for dataset, variant, project, run_id in RUNS:
        run = api.run(f"{ENTITY}/{project}/{run_id}")
        cfg = {k: v for k, v in run.config.items()}
        summary = dict(run.summary)

        # Full step-time history (one value per logged training step).
        hist = run.history(keys=[STEP_KEY], samples=1_000_000, pandas=True)
        steps = hist[STEP_KEY].dropna().to_numpy(dtype=float)
        if steps.size == 0:
            raise RuntimeError(f"{run_id}: no '{STEP_KEY}' history rows")

        runtime_s = float(summary.get("_runtime", run.summary.get("_runtime")))
        final_step = int(summary.get("trainer/global_step", summary.get("_step")))
        eval_every = _eval_every_n_steps(cfg)
        n_val_cycles = max(1, final_step // eval_every)

        train_step_med = float(np.median(steps))
        # Inference (sampling + scoring + I/O) wall time, attributed per cycle.
        val_time_total = max(0.0, runtime_s - train_step_med * final_step)
        val_per_cycle = val_time_total / n_val_cycles

        rows.append(
            dict(
                dataset=dataset,
                variant=variant,
                run_id=run_id,
                wandb_project=project,
                n_max=_cfg_int(cfg, "num_nodes_max_static"),
                total_parameters=_cfg_int(cfg, "total_parameters"),
                batch_size=_cfg_int(cfg, "batch_size"),
                eval_every_n_steps=eval_every,
                runtime_s=runtime_s,
                final_step=final_step,
                n_val_cycles=n_val_cycles,
                n_history_points=int(steps.size),
                train_step_p25=float(np.percentile(steps, 25)),
                train_step_med=train_step_med,
                train_step_p75=float(np.percentile(steps, 75)),
                train_step_mean=float(np.mean(steps)),
                train_step_min=float(np.min(steps)),
                amortized_step_s=runtime_s / final_step,
                val_time_total_s=val_time_total,
                val_per_cycle_s=val_per_cycle,
            )
        )
        print(
            f"  {dataset:8s} {variant:18s} {run_id}  "
            f"train_med={train_step_med:.3f}s  val/cycle={val_per_cycle:.0f}s"
        )

    df = pd.DataFrame(rows)
    # Per-dataset ratios to the vignac baseline.
    for col, ratio in (
        ("train_step_med", "train_ratio_vs_vignac"),
        ("val_per_cycle_s", "val_ratio_vs_vignac"),
        ("amortized_step_s", "amortized_ratio_vs_vignac"),
    ):
        base = df[df.variant == "vignac"].set_index("dataset")[col]
        df[ratio] = df.apply(
            lambda r, col=col, base=base: r[col] / base[r.dataset], axis=1
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df)} rows)")


if __name__ == "__main__":
    main()

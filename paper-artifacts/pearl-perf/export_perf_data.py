# /// script
# dependencies = [
#   "wandb>=0.17",
#   "pandas>=2.0",
#   "pyarrow>=15",
#   "numpy>=1.26",
# ]
# ///
"""Pull per-step training timing from W&B into a tidy parquet.

For each of the eight 2026-05-06 ``*_repro_exact`` panel runs we read the
full ``impl-perf/train/step_time_s`` history plus the run summary, and
derive a per-run timing record:

* ``train_step_med`` / ``p25`` / ``p75`` — robust train-step wall time
  (forward + backward + optimizer; the metric excludes dataloading and
  validation by construction — see ``diffusion_module.py`` on-train-batch
  hooks).
* ``val_time_total_s`` / ``val_per_cycle_s`` — **derived**:
  ``runtime - train_step_med * final_step``, split across the number of
  validation cycles (``final_step // eval_every_n_steps``). This bundles
  validation forward + diffusion sampling + MMD + checkpoint I/O; it is
  the only validation-cost proxy the runs logged. Treat as wall-clock
  attribution, not a clean per-phase measurement.

Output schema (one row per run) is documented in ``README.md``.

Auth: reads ``WANDB_API_KEY`` from the repo-root ``.env`` and
exports it as ``WANDB_API_KEY`` before importing wandb. Run from repo root:

    uv run paper-artifacts/pearl-perf/export_perf_data.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --- Panel definition -------------------------------------------------------
# (dataset, variant, wandb_project, run_id). The baseline variant is "vignac"
# (vanilla DiGress: ExtraFeatures(all) = cycles + Laplacian eigh).
ENTITY = "your-wandb-entity"
RUNS: list[tuple[str, str, str, str]] = [
    ("sbm", "vignac",           "discrete-sbm-vignac-repro-exact",            "cgfv3f85"),
    ("sbm", "pearl",            "discrete-sbm-pearl-repro-exact",             "k4iiw5sg"),
    ("sbm", "pearl-spectral",   "discrete-sbm-pearl-spectral-repro-exact",    "qukgm6zu"),
    ("sbm", "pearl-gnnconv-norm","discrete-sbm-pearl-gnnconv-norm-repro-exact","5qchu8c4"),
    ("enzymes", "vignac",           "discrete-enzymes-vignac-repro-exact",            "8nhefhnl"),
    ("enzymes", "pearl",            "discrete-enzymes-pearl-repro-exact",             "7yi627fv"),
    ("enzymes", "pearl-spectral",   "discrete-enzymes-pearl-spectral-repro-exact",    "ths6e1da"),
    ("enzymes", "pearl-gnnconv-norm","discrete-enzymes-pearl-gnnconv-norm-repro-exact","xsmz6yql"),
]

STEP_KEY = "impl-perf/train/step_time_s"
HERE = Path(__file__).resolve().parent
OUT_PARQUET = HERE / "data" / "perf.parquet"


def _load_api_key() -> None:
    """Export WANDB_API_KEY from the repo-root .env (WANDB_API_KEY)."""
    if os.environ.get("WANDB_API_KEY"):
        return
    env_path = HERE.parent.parent / ".env"
    if not env_path.exists():
        raise FileNotFoundError(
            f"{env_path} not found and WANDB_API_KEY not set; cannot authenticate."
        )
    text = env_path.read_text()
    m = re.search(r"^WANDB_API_KEY=(.+)$", text, re.MULTILINE)
    if m is None:
        raise KeyError("WANDB_API_KEY not found in .env")
    os.environ["WANDB_API_KEY"] = m.group(1).strip()


def _eval_every_n_steps(cfg: dict) -> int:
    """Pull eval cadence from the logged config; fall back to per-dataset default."""
    # The config is logged flat under model.* in some runs; tolerate both.
    for key in ("eval_every_n_steps", "model/eval_every_n_steps"):
        if key in cfg and cfg[key] is not None:
            return int(cfg[key])
    raise KeyError("eval_every_n_steps not present in logged config")


def main() -> None:
    _load_api_key()
    import wandb

    api = wandb.Api(timeout=60)
    records: list[dict] = []

    for dataset, variant, project, run_id in RUNS:
        run = api.run(f"{ENTITY}/{project}/{run_id}")
        cfg = dict(run.config)
        summary = dict(run.summary)

        # Full step-time history (scan_history avoids the 500-sample cap).
        rows = [
            r[STEP_KEY]
            for r in run.scan_history(keys=[STEP_KEY])
            if r.get(STEP_KEY) is not None
        ]
        steps = np.asarray(rows, dtype=float)
        if steps.size == 0:
            raise RuntimeError(f"{run_id}: no {STEP_KEY} history")

        runtime_s = float(summary.get("_runtime", summary.get("_wandb", {}).get("runtime", np.nan)))
        final_step = int(summary.get("trainer/global_step", 0))
        eval_every = _eval_every_n_steps(cfg)
        n_val_cycles = max(final_step // eval_every, 1)

        train_step_med = float(np.median(steps))
        train_time_est = train_step_med * final_step
        val_time_total = runtime_s - train_time_est
        val_per_cycle = val_time_total / n_val_cycles

        # Param count + n_max for context (best-effort from config).
        total_params = cfg.get("total_parameters")
        n_max = cfg.get("num_nodes_max_static")

        records.append(
            {
                "dataset": dataset,
                "variant": variant,
                "run_id": run_id,
                "wandb_project": project,
                "n_max": n_max,
                "total_parameters": total_params,
                "batch_size": cfg.get("batch_size"),
                "eval_every_n_steps": eval_every,
                "runtime_s": runtime_s,
                "final_step": final_step,
                "n_val_cycles": n_val_cycles,
                "n_history_points": int(steps.size),
                "train_step_p25": float(np.percentile(steps, 25)),
                "train_step_med": train_step_med,
                "train_step_p75": float(np.percentile(steps, 75)),
                "train_step_mean": float(steps.mean()),
                "train_step_min": float(steps.min()),
                "amortized_step_s": runtime_s / final_step if final_step else np.nan,
                "val_time_total_s": val_time_total,
                "val_per_cycle_s": val_per_cycle,
            }
        )
        print(
            f"  {dataset:>7} / {variant:<18} {run_id}  "
            f"train_med={train_step_med:.4f}s  val/cycle={val_per_cycle:.1f}s  "
            f"(n_hist={steps.size}, steps={final_step})"
        )

    df = pd.DataFrame.from_records(records)

    # Attach per-dataset ratios vs the vignac baseline (train + val + amortized).
    out_frames = []
    for ds, grp in df.groupby("dataset"):
        base = grp.loc[grp["variant"] == "vignac"].iloc[0]
        grp = grp.copy()
        grp["train_ratio_vs_vignac"] = grp["train_step_med"] / base["train_step_med"]
        grp["val_ratio_vs_vignac"] = grp["val_per_cycle_s"] / base["val_per_cycle_s"]
        grp["amortized_ratio_vs_vignac"] = grp["amortized_step_s"] / base["amortized_step_s"]
        out_frames.append(grp)
    df = pd.concat(out_frames, ignore_index=True)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_PARQUET}")
    print(df[["dataset", "variant", "train_ratio_vs_vignac", "val_ratio_vs_vignac"]].to_string(index=False))


if __name__ == "__main__":
    main()

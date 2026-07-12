# PEARL vs vanilla-DiGress wall-time figures

Paper-bound bundle quantifying the **training vs validation** wall-time
cost of the feature/projection variants relative to the vanilla DiGress
baseline, on the two datasets we have panel data for (SBM, ENZYMES).

## Headline

The same architecture changes that cost a little (or break even) at
*training* time recover or beat parity at *validation* time. `PEARL +
GNN Q/K/V` is the clearest validation win — **−19 % on SBM, −14 % on
ENZYMES** per validation cycle — because validation is sampling-dominated
(T diffusion steps × many samples per graph) where the eigh in vanilla
DiGress is called repeatedly, whereas the GNN-projection variant avoids
eigendecomposition entirely.

| | SBM train | SBM val | ENZ train | ENZ val |
|---|---:|---:|---:|---:|
| PEARL | −5 % | −6 % | +40 % | +14 % |
| PEARL + spectral Q/K/V | +3 % | −5 % | +70 % | +2 % |
| PEARL + GNN Q/K/V | −5 % | **−19 %** | +20 % | **−14 %** |

(negative = faster than vanilla DiGress; see caveats before quoting)

## Pipeline

Two stages, both `uv` self-contained scripts (run from repo root):

```bash
# 1. W&B  ->  data/perf.parquet   (auth via .env WANDB_API_KEY)
uv run paper-artifacts/pearl-perf/export_perf_data.py

# 2. parquet  ->  figures/ + tables/
uv run paper-artifacts/pearl-perf/render.py
```

## Outputs

- `data/perf.parquet` — one row per run (8 rows). Tracked via **git LFS**.
- `figures/train_vs_val_ratio.{pdf,png}` — two-panel grouped bar chart
  (train | val), ratio to vanilla DiGress, parity line at 1.0.
- `tables/perf_summary.tex` — `booktabs` + `multirow` table (compiles
  standalone; verified with `pdflatex`). Absolute + relative numbers.

## Parquet schema

One row per `(dataset, variant)` run. Key columns:

| column | meaning |
|---|---|
| `dataset`, `variant`, `run_id`, `wandb_project` | identity |
| `n_max`, `total_parameters`, `batch_size`, `eval_every_n_steps` | config context |
| `runtime_s`, `final_step`, `n_val_cycles`, `n_history_points` | run extent |
| `train_step_{p25,med,p75,mean,min}` | per-step wall time (fwd+bwd+opt), seconds |
| `amortized_step_s` | `runtime_s / final_step` (includes val), seconds |
| `val_time_total_s`, `val_per_cycle_s` | **derived** validation cost (see caveats) |
| `train_ratio_vs_vignac`, `val_ratio_vs_vignac`, `amortized_ratio_vs_vignac` | per-dataset ratios to the `vignac` baseline |

## Caveats (read before putting numbers in the paper)

1. **Validation-cycle time is derived, not directly measured.** It is
   `runtime − train_step_med × final_step`, divided by the number of
   validation cycles. It bundles validation forward passes + diffusion
   sampling + MMD computation + checkpoint I/O + logger sync. The
   training-step number is a clean instrumented metric
   (`impl-perf/train/step_time_s`); the validation number is wall-clock
   attribution. Because the train-step distribution is right-skewed
   (periodic spikes), `median × steps` slightly *under*-counts true
   train time, so `val_time` is a mild *over*-estimate. Treat the
   validation column as directional, not precise.

2. **The 2026-05-06 runs trained with cycle features ENABLED on every
   variant.** The `use_cycles` flag that makes the PEARL configs
   "PEARL-only" was added later (commit `65aece0a`, 2026-05-21). So this
   comparison is *cycles+PEARL vs cycles+eigh*, not *PEARL-only vs eigh*.

3. **Code is from before the sparse-default-GraphData refactor.** 19
   commits to perf-relevant paths landed between the runs and current
   `main`. Numbers are not guaranteed to reproduce on current code;
   re-run a short panel before final paper submission.

4. **None of these runs finished** — all 8 were preempted at ~17.7 h
   (2–4 % of the 550k-step target on SBM, 57–91 % on ENZYMES). Step-time
   medians are stable across the run, so the timing conclusion holds, but
   these are not converged-quality runs.

5. **Single seed, single run per cell.** No error bars. The train-step
   medians are over thousands of steps (tight), but cross-run variance
   (hardware, scheduling) is not captured.

Full investigation write-up: `internal notes (not included)`
(gitignored working note).

## How to update

Re-run both scripts. If new runs are added, edit the `RUNS` list at the
top of `export_perf_data.py` (the only place run IDs live). The parquet
schema is stable; `render.py` reads only the columns listed above.

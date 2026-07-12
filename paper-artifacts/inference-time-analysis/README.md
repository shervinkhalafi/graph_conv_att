# Inference-time analysis — PEARL vs vanilla DiGress

Self-contained bundle measuring the **training-step** and **inference
(sampling) cycle** wall-time cost of the feature/projection variants relative
to vanilla DiGress, on SBM and ENZYMES. Everything needed to regenerate the
tables and figures — data, scripts, outputs — lives in this folder. You can
run the analysis offline (the data is committed); you only need W&B to refresh
the numbers.

## What is measured

Two quantities, both wall time on an A100:

- **Training step** — `impl-perf/train/step_time_s`, a directly instrumented
  metric bracketing one optimisation step (forward + backward + optimizer). It
  excludes dataloading and validation by construction. We report a robust
  median over the full run history.
- **Inference cycle** — the cost of one validation/generation cycle. This is
  what "inference time" means for a diffusion model: sampling a batch of graphs
  runs the denoiser `T` times per graph. It is **wall-clock-derived**
  (`run time − median train step × step count`, divided by the number of
  validation cycles), so it bundles the diffusion sampling forward passes, MMD
  scoring, and checkpoint I/O. Treat it as an **upper bound on pure model
  inference**, not a clean isolated measurement — see Caveats.

The four variants:

| Variant | Positional features | Q/K/V projection |
|---|---|---|
| `vignac` (baseline) | Laplacian eigendecomposition (`eigh`) | linear |
| `pearl` | R-PEARL GNN encoding | linear |
| `pearl-spectral` | R-PEARL | spectral (adds a small `eigh`) |
| `pearl-gnnconv-norm` | R-PEARL | GNN-conv |

## Headline

The architecture changes that cost a little (or much) at *training* time
mostly recover or beat parity at *inference* time. `PEARL + GNN Q/K/V` is the
clearest inference win — **−19 % on SBM, −14 % on ENZYMES per cycle** —
because inference is sampling-dominated (`T` diffusion steps × many samples),
the regime where vanilla DiGress's repeated `eigh` is expensive and the
GNN-projection variant avoids eigendecomposition entirely.

| | SBM train | SBM inference | ENZ train | ENZ inference |
|---|---:|---:|---:|---:|
| PEARL | −5 % | −6 % | +40 % | +14 % |
| PEARL + spectral Q/K/V | +3 % | −5 % | +70 % | +2 % |
| PEARL + GNN Q/K/V | −5 % | **−19 %** | +20 % | **−14 %** |

(negative = faster than vanilla DiGress. Absolute numbers in
`tables/inference_time_main.tex`.)

## How to run

Both scripts are `uv` self-contained (PEP-723 inline dependencies); run them
from anywhere.

```bash
# Regenerate tables + figures from the committed data (no W&B needed):
uv run analyze.py

# Refresh data/perf.csv from W&B (needs WANDB_API_KEY or a .env with
# WANDB_API_KEY in this folder or any parent), then re-analyze:
uv run export_from_wandb.py
uv run analyze.py
```

## What's in the folder

```
inference-time-analysis/
├── README.md                       # this file
├── BACKGROUND.md                   # investigation history + evidence (root cause, compile)
├── export_from_wandb.py            # W&B  -> data/perf.csv   (refresh step)
├── analyze.py                      # data -> tables/ + figures/
├── data/
│   └── perf.csv                    # committed: one row per run, 23 columns
├── tables/
│   ├── inference_time_main.tex     # train + inference, abs + ratio (booktabs+multirow)
│   └── inference_time_compact.tex  # inference ratio only, with param counts
└── figures/
    ├── train_vs_val_ratio.{pdf,png}    # two-panel ratio bars, parity line
    └── inference_absolute.{pdf,png}    # absolute inference seconds (log y)
```

`data/perf.csv` columns: identity (`dataset, variant, run_id, wandb_project`),
config context (`n_max, total_parameters, batch_size, eval_every_n_steps`),
run extent (`runtime_s, final_step, n_val_cycles, n_history_points`),
train-step distribution (`train_step_{p25,med,p75,mean,min}`), and derived
inference cost (`amortized_step_s, val_time_total_s, val_per_cycle_s`) plus the
per-dataset ratios to `vignac`.

## Why PEARL is slower at training (and why `torch.compile` does not fix it)

PEARL was expected to speed up *training* by replacing the `O(n³)` `eigh` with
an `O(n)` GNN. It does not, at these graph sizes, for two reasons established
in a separate root-cause investigation:

1. **The training step is launch-overhead bound, not compute-bound.** A FLOP
   accounting explains ≤3 % of observed wall time on either dataset; the rest
   is Python dispatch and kernel-launch latency on small matrices. So swapping
   one cheap feature block for another barely moves the step.
2. **PEARL adds a trainable backward pass that `eigh` does not have.**
   Forward-only PEARL beats `eigh` near `n ≈ 150`; once backward is included,
   PEARL never wins on FLOPs through `n = 500`. ENZYMES (`n = 126`) sits on the
   wrong side of both crossovers, so PEARL is 1.2–1.7× slower per train step
   there; SBM (`n = 200`) is break-even (all variants within ±5 %).

`torch.compile` does not flip this. In the recorded compile-on profile (of the
vanilla `eigh` config), compile fused the transformer glue into Triton kernels
but the cuSOLVER eigendecomposition kernel still dominated at ~32 % of CUDA —
compile cannot fuse `eigh`, and the diffusion loop's dynamic control flow
causes graph breaks. The "eigh dominates" story is therefore an *inference*
story (where `eigh` runs `T` × samples times per cycle), not a training-step
one. **No PEARL run with `compile=true` exists**; this remains a `compile=false`
comparison on both sides, which is apples-to-apples.

## Caveats (read before quoting in the paper)

1. **Inference-cycle time is derived, not directly instrumented.** It is a
   wall-clock attribution bundling sampling + MMD + I/O. Because the train-step
   distribution is mildly right-skewed, `median × steps` slightly under-counts
   train time, so the inference number is a mild *over*-estimate. Quote it as
   directional / an upper bound.
2. **Cycle features were enabled on every variant.** At run time the
   PEARL-only flag did not yet exist, so this is *cycles + PEARL vs cycles +
   eigh*, not *PEARL-only vs eigh*.
3. **Single seed, one run per cell, no error bars.** Train-step medians are
   over thousands of steps (tight); cross-run hardware variance is not
   captured.
4. **Runs were preempted at ~17.7 h** before convergence (W&B state `failed`).
   Step-time medians are stable across the run, so the timing conclusion holds,
   but these are not converged-quality runs.
5. **Code predates the sparse-default refactor.** Numbers are not guaranteed to
   reproduce byte-for-byte on current `main`; re-run a short panel before final
   submission if you want post-refactor numbers.

## How to present this

- The defensible claim is about **inference**: *the projection variants match
  or beat vanilla DiGress at sampling time, and `PEARL + GNN Q/K/V` is fastest
  (−14 % to −19 % per cycle), because it avoids the repeated eigendecomposition
  that dominates DiGress sampling.*
- Do **not** claim a training-time speedup from PEARL — at these graph sizes it
  is break-even (SBM) to slower (ENZYMES). Report that honestly; the training
  panel of `tables/inference_time_main.tex` and `figures/train_vs_val_ratio`
  show it.
- Use `tables/inference_time_main.tex` for the full picture, or
  `tables/inference_time_compact.tex` if you only have a narrow column.
  `figures/train_vs_val_ratio` is the one-glance result; `figures/inference_absolute`
  gives the raw cost (SBM ≈ 3 min, ENZYMES ≈ 24–32 min per cycle).

## Provenance

Eight W&B runs, entity `your-wandb-entity`, launched 2026-05-06, one project
per variant. **Verified against live W&B on 2026-06-19:** these are the latest
inference-time runs (the only newer project, `generation_tests_diagnostics`, is
an unrelated small-model diagnostics test); all eight log
`impl-perf/train/step_time_s` and all have `compile_model=false`.

| dataset | variant | run_id | project |
|---|---|---|---|
| sbm | vignac | `cgfv3f85` | discrete-sbm-vignac-repro-exact |
| sbm | pearl | `k4iiw5sg` | discrete-sbm-pearl-repro-exact |
| sbm | pearl-spectral | `qukgm6zu` | discrete-sbm-pearl-spectral-repro-exact |
| sbm | pearl-gnnconv-norm | `5qchu8c4` | discrete-sbm-pearl-gnnconv-norm-repro-exact |
| enzymes | vignac | `8nhefhnl` | discrete-enzymes-vignac-repro-exact |
| enzymes | pearl | `7yi627fv` | discrete-enzymes-pearl-repro-exact |
| enzymes | pearl-spectral | `ths6e1da` | discrete-enzymes-pearl-spectral-repro-exact |
| enzymes | pearl-gnnconv-norm | `xsmz6yql` | discrete-enzymes-pearl-gnnconv-norm-repro-exact |

The run IDs live in one place only: the `RUNS` list at the top of
`export_from_wandb.py`. To change the panel, edit that list and re-run both
scripts.

Background on the root cause and the `torch.compile` evidence is in
`BACKGROUND.md` (next to this file). The `torch.compile` profiling itself is
also written up in
`internal notes (not included)`
and `internal notes (not included)` in the main repo.

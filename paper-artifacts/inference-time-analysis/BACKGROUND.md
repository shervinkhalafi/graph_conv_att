# Background — investigation history & evidence

Companion to `README.md`. The README says what the numbers are and how to
regenerate them; this note records *where they came from* and the evidence
behind the two claims that matter for the paper: (1) PEARL does not speed up
**training** at these graph sizes, and (2) `torch.compile` does not change that.
Self-contained — you do not need any other file to read it.

## Two investigations feed these numbers

The result rests on two separate analyses (one on PEARL training cost, one on
`torch.compile`), which is why the "PEARL is slower even with compile" story
sits across both rather than in a single run:

1. **PEARL train-step root cause** — a FLOPs decomposition + crossover-`n`
   analysis of why PEARL is slower per training step. Runs had
   `compile_model=false`.
2. **`torch.compile` profiling** — `torch.profiler` traces of the vanilla
   `eigh` config with compile on and off. This is the only place compile was
   actually run. (In this repo: see `internal notes (not included)` and
   `internal notes (not included)`.)

There is **no run that is "PEARL + compile"** — so the comparison in this
bundle is `compile=false` on both sides, which is apples-to-apples.

## Why PEARL is slower at training (root cause)

PEARL was expected to cut *training* cost by replacing the `O(n³)` `eigh` with
an `O(n)` GNN. Three independent checks converged on why it does not, at our
graph sizes:

- **Launch-overhead bound, not compute-bound.** A FLOP accounting explains
  ≤3 % of observed training wall time on either dataset; the rest is Python
  dispatch and kernel-launch latency on small matrices. Swapping one cheap
  feature block for another barely moves the step.
- **PEARL has a trainable backward pass; `eigh` does not.** Forward-only PEARL
  beats `eigh` near `n ≈ 150`, but once backward is included PEARL never wins
  on FLOPs through `n = 500`. Crossover table (PEARL extra cost vs `eigh`,
  forward-only ratio, and forward+backward ratio):

  | n | fwd ratio | fwd+bwd ratio |
  |---:|---:|---:|
  | 100 | 1.49 | 4.47 |
  | **126 (ENZYMES)** | **1.19** | **3.57** |
  | 150 | 1.03 | 3.09 |
  | **200 (SBM)** | **0.85** | **2.54** |
  | 500 | 0.60 | 1.81 |

ENZYMES (`n=126`) is on the wrong side of both crossovers → PEARL is
1.2–1.7× slower per train step there. SBM (`n=200`) is break-even (all four
variants within ±5 %). The "`eigh` dominates" intuition is true for
**sampling/inference** (where `eigh` runs `T` × samples times per cycle), not
for the single training step.

## Why `torch.compile` does not flip it

Compile was profiled on the vanilla `eigh` config (training step), with the
relevant evidence:

- **First compile attempt:** 15 graph breaks, all from *"dynamic control flow
  is not supported"* — the diffusion loop's data-dependent conditionals.
  Compile could not capture a single graph.
- **After fixing the breaks (one captured graph):** the cuSOLVER
  eigendecomposition kernel (`syevbj_batch_32x16`) still dominated at ~32 % of
  CUDA time; the fused Triton kernels appeared at ~13 %. Reading: compile
  fuses the transformer glue (LayerNorm, elementwise, small matmuls) into
  Triton kernels but **cannot fuse `eigh`**, which stays the top cost.

So compile speeds up the glue but leaves the structural bottleneck intact —
`eigh` for vanilla DiGress, and launch-overhead + the extra backward for PEARL.
It does not change the ranking. (The raw `torch.profiler` traces are an
internal artifact, not committed to this repo.)

## The W&B recording behind `data/perf.csv`

Eight runs, entity `your-wandb-entity`, launched 2026-05-06, one project per
variant. The logged timing metric is `impl-perf/train/step_time_s` (full
history). `data/perf.csv` is the consolidation; the table below is what each
run recorded.

| dataset | variant | run_id | train_step_med (s) | inference/cycle (s) | params |
|---|---|---|---:|---:|---:|
| sbm | vignac | `cgfv3f85` | 0.513 | 214 | 7.13M |
| sbm | pearl | `k4iiw5sg` | 0.486 | 200 | 7.16M |
| sbm | pearl-spectral | `qukgm6zu` | 0.528 | 203 | 5.69M |
| sbm | pearl-gnnconv-norm | `5qchu8c4` | 0.487 | 174 | 11.87M |
| enzymes | vignac | `8nhefhnl` | 0.107 | 1662 | 7.13M |
| enzymes | pearl | `7yi627fv` | 0.150 | 1890 | 7.16M |
| enzymes | pearl-spectral | `ths6e1da` | 0.181 | 1690 | 5.69M |
| enzymes | pearl-gnnconv-norm | `xsmz6yql` | 0.128 | 1425 | 11.87M |

**Verified against live W&B on 2026-06-19:** these eight remain the latest
inference-time runs; all log `impl-perf/train/step_time_s` and all have
`compile_model=false`. The only newer project in the entity
(`generation_tests_diagnostics`, 2026-06-15) is an unrelated small-model
diagnostics test, not a perf panel. Run state is `failed` — the documented
~17.7 h preemption, not a crash mid-metric.

## Known gaps (so nothing is overclaimed)

- **No PEARL run with `compile=true`** — compile was only profiled on the
  vanilla `eigh` config.
- **Inference-cycle time is wall-clock-derived**, bundling sampling + MMD +
  I/O; it is an upper bound on pure model inference, not a directly
  instrumented sampler timer.
- **No post-sparse-refactor re-run** — numbers predate current `main`;
  directional pending a short re-run before final submission.
- **Single seed, one run per cell, no error bars.**

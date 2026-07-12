# Performance toggles

Catalog of opt-in performance flags + automatic defaults that affect
training and eval throughput. The "automatic" set lands the moment the
Modal app is redeployed; the "opt-in" set requires a config flag on the
specific run.

## Automatic — applies to every Modal-deployed training and eval run

These changed code-level defaults (or were wired into the image env). No
caller-side action needed.

### `PYTHONOPTIMIZE=1` baked into the image

Set in `src/tmgg/modal/_lib/image.py::_runtime_env`. The Modal container
Python boots with `__debug__=False`, which strips every `if __debug__:`
block in `noise_process`, `graph_types`, `transformer_model`,
`extra_features`, `diffusion_sampling`, and `schedule` at bytecode
compile time. Removes ~10 s/eval-cycle of host-GPU sync overhead from
mask-symmetry / row-stochastic / mask-correctness assertions.

The training subprocess (`_functions.py::run_cli`) keeps its existing
per-run override: setting `modal_debug=true` in the experiment config
still pops `PYTHONOPTIMIZE` from the subprocess env, leaving asserts
active for numerical investigation. The pop only affects the subprocess
— the parent container Python's already-frozen `__debug__=False` is
unaffected.

Applies automatically to: training (parent + subprocess), MMD eval
subprocess, eval-all in-process app, profile app, async eval.

### Fused AdamW / Adam (`fused=True`)

`src/tmgg/training/lightning_modules/optimizer_config.py`. CUDA-only
fall-through to unfused on CPU. Replaced ~250 small per-tensor CUDA
launches with a single multi-tensor kernel. Was the largest train-loop
host-side hotspot; ~135 ms/step host overhead → ~0.32 ms/step.

### `torch._foreach_norm` in `on_before_optimizer_step`

`src/tmgg/training/lightning_modules/base_graph_module.py`. Replaced the
per-parameter Python loop computing grad/param norms with batched
`_foreach_norm` calls. ~250 small CUDA launches per step → ~3 launches.

### `.tolist()` instead of N `.item()` for per-bin loss reporting

`src/tmgg/training/lightning_modules/diffusion_module.py` (two sites:
train per-t bin loop, val per-t bin loop). Single GPU→CPU sync instead
of N. Fires every `eval_every_n_steps`.

### Branch-free `masked_softmax`

`src/tmgg/models/layers/masked_softmax.py`. The two data-dependent
`if mask.sum() == 0:` and `if nan_mask.any():` branches are now
replaced by `torch.where`. Removes graph breaks under `torch.compile`
and eliminates two GPU→CPU syncs per attention head per step.
Behaviour preserved exactly via final
`torch.where(mask.sum() == 0, x, result)` — keeps the upstream
`return x` semantics for fully-masked rows.

### `data.y` canonical Float dtype

`src/tmgg/diffusion/diffusion_sampling.py:351`. Removed the vestigial
`uy = uy.type_as(long_mask)` cast that silently coerced the sample-time
`y` from Float (one-hot) to Long. Train, val capture, and sample-time
`y` now share a single Float dtype — one trace under `torch.compile`
instead of recompiling per dtype variant. A `__debug__`-gated assert at
`transformer_model.py` enforces the invariant in dev runs.

### `drop_last_train: bool = True` default

`src/tmgg/data/data_modules/base_data_module.py`. Train dataloader now
drops the trailing partial batch (e.g. SBM 128 / 12 = 10 full + 1 of
size 8 → 10 full only, 8 graphs unused per epoch with shuffle=True). 
Val/test loaders never drop — they iterate every sample for unbiased
metrics. Eliminates the dominant source of dynamic-batch-shape
recompiles under `torch.compile`.

### `pad_to_static_n_max: bool = True` default

`src/tmgg/data/data_modules/spectre_sbm.py`,
`spectre_planar.py`, `molecular/base.py` + the matching YAML configs
(`qm9_digress.yaml`, `moses_digress.yaml`, `guacamol_digress.yaml`,
`spectre_planar.yaml`). Every batch is now padded to the dataset's
literature-cited maximum node count (SBM 200, Planar 64, QM9 9, MOSES
30, GuacaMol 88). The model's `node_mask` zeros padded contributions,
so output is mathematically unchanged. Costs ~33 % more padded compute
on small-graph batches but enables single-trace `torch.compile`.

**Caveat for production training:** SBM batches go from
`(12, ~150_avg, …)` to `(12, 200, …)` always. Verify wall-clock per
step on a short repro before relying on the new default for long runs.

## Opt-in — requires a config flag on your specific run

### `torch.compile(model)` on training

```bash
# Hydra override on any training launch
... +model.compile_model=true
```

Toggle on `BaseGraphModule.__init__` (and threaded through
`DiffusionModule`). When True, wraps `self.model = torch.compile(self.model, mode=compile_mode)`.

Optional mode override:

```bash
... +model.compile_model=true +model.compile_mode=reduce-overhead
```

Modes: `default` (safest, recommended start), `reduce-overhead` (CUDA
graphs, can be flaky with dynamic shapes), `max-autotune` (longest
warmup, best steady-state).

**Measured impact** (round-5 Greedy SBM config, post-shape-fix):

| | Wall time, 100 steps | CUDA / step |
|---|---|---|
| compile=False | 212 s | 92 ms |
| **compile=True** | **139 s** | **55 ms** |

35 % wall-clock reduction; 40 % CUDA reduction. One-shot inductor /
autotune cost ~80 s on the first compile, amortised across the run.

**Preconditions** (all already in place via the automatic defaults
above): `drop_last_train=True`, `pad_to_static_n_max=True`, branch-free
`masked_softmax`, Float `data.y`. Without these, compile is a 2-3×
regression from recompile thrash — see `dynamo_counters.json` from any
profile run for the diagnostic.

### `torch.compile(model)` on eval

`evaluate_checkpoint(..., compile_model=True)` accepts the same flag
(in `src/tmgg/experiments/discrete_diffusion_generative/evaluate_cli.py`).
Hard-codes `dynamic=True` because the val capture phase pulls a
dataloader-tail batch (size 8) that would otherwise recompile.

If invoking from the profile launcher: `--eval-compile` flag.

**Caveat:** the eval first-compile cost (~80 s for `dynamic=True`
trace) is paid once per eval cycle. For a single 32-sample eval, the
per-call savings may not amortise the warmup. Worthwhile for many-call
eval workloads (e.g. multi-checkpoint sweeps) and/or when chained with
training in a single process.

### Profile-only runner toggles

In `src/tmgg/profile/runners.py`. These are deliberately scoped to the
profile mode and DO NOT apply to production training:

- Raw `torch.profiler.profile` instead of Lightning's `PyTorchProfiler`
  (skips Lightning's expensive end-of-fit `key_averages` aggregation).
- `enable_progress_bar=False` on the Trainer (skips ~58 s of rich-progress
  teardown on Modal's non-TTY stdout).
- `del trainer; gc.collect()` after fit (skips ~59 s of persistent-worker
  join).
- cProfile bundle (`cprofile.{pstats,txt}`) emitted alongside the chrome
  trace.
- `torch._dynamo.utils.counters` snapshotted to `dynamo_counters.json`.
- `TORCH_LOGS=recompiles,graph_breaks` enabled programmatically via
  `torch._logging.set_logs(...)`.

Production training keeps Lightning's progress bar, profiler, and worker
shutdown semantics unchanged.

## How to verify your repro picks up the changes

Redeploy the modal app:

```bash
# Training app (covers training + MMD eval)
uv run modal deploy -m tmgg.modal._functions

# Eval-all app
uv run modal deploy -m tmgg.modal._eval_all_functions

# Profile app
uv run modal deploy -m tmgg.modal._profile_functions
```

Then for compile, add `+model.compile_model=true` to your
training-launch overrides. Everything else is automatic.

For a sanity check that compile worked: pull the run's
`dynamo_counters.json` (profile runs only — production runs don't dump
it). Look for `unique_graphs: 1` and `inline_call: {}` (no graph-break
errors). If `unique_graphs > 5` or `inline_call` is non-empty, something
is recompiling — see `docs/debugging-modal.md` and the
`TORCH_LOGS=recompiles` output in the modal log.

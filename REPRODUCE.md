# Reproducing the paper

This file maps each figure and table in *Graph Convolutional Attention: A
Spectral Perspective on Graph Denoising and Diffusion* to the concrete command,
the config that fixes its hyperparameters, and the caveats you need to know. It
was written by inspecting the config tree; where the tree does not pin a value
down to a single figure-generating command, this file says so explicitly rather
than guessing.

The paper has two experiment families:

- **Denoising (§4.3, Figure 1)** — single-step graph denoising. The proposed
  method is **Graph Convolutional Attention (GCA)**; it is compared against a
  plain graph-transformer / linear-attention baseline (GT) and the other
  architecture families.
- **Generative (§4.4, Table 1 + Table 2)** — DiGress-style discrete diffusion,
  ablating the DiGress baseline against R-PEARL positional encoding and
  GCA-style (GNN / spectral) Q/K/V projections. Table 1 reports generation
  quality (MMD); Table 2 reports inference (sampling) wall-time.

## How to run anything

Every CLI uses Hydra: override any key with `key=value`, add a new key with
`+key=value`. Training is measured in **steps**, not epochs.

```bash
uv sync --all-extras          # install package + deps

# Local run of any entry point:
uv run <cli> [hydra overrides...]

# The same on Modal (what the run-*.zsh wrappers do):
uv run tmgg-modal run <cli> [hydra overrides...] --gpu <tier> [--detach]
```

`<cli>` is a `[project.scripts]` entry point in `pyproject.toml`. The entry
points used below are `tmgg-mod-attention`, `tmgg-spectral-arch`,
`tmgg-gnn-transformer`, `tmgg-gnn`, `tmgg-baseline`, `tmgg-digress` (denoising)
and `tmgg-discrete-gen` (generative).

**Hyperparameters live in the config files, not here.** This file gives you the
command and the config path; open the config for exact widths, learning rates,
schedules, and diffusion steps. Do not transcribe numbers from memory.

**`data=` vs `+data=`.** Some base configs declare `data` through a Hydra
defaults list, others inline the `data:` block. Per
`src/tmgg/experiments/exp_configs/README.md`, when the block is inline `data=...`
fails and you must use `+data=...`; the committed run wrappers use the `+data=`
form. If `data=<name>` errors with "could not override", switch to `+data=<name>`.

---

## §4.3 Denoising — Figure 1 (GT vs GCA across datasets)

### The GT-vs-GCA toggle

The paper's proposed denoiser, **GCA**, is
`tmgg.models.modified_attention.ModifiedAttentionDenoiser`, run via
`tmgg-mod-attention`. Its graph-convolution filters on the attention
projections are the literal GT-vs-GCA switch (verified in
`src/tmgg/models/modified_attention/mod_attention.py`, a K-tap filter
`Z = Σ_{i=0}^{K-1} A^i X H_i`):

| Config key | Effect |
|---|---|
| `model.filter_qk` (bool, default `false`) | graph-convolve the Q and K projections |
| `model.filter_v` (bool, default `false`) | graph-convolve the V projection |
| `model.filter_num_terms` (int, default `2`) | polynomial order `K` of the filter |

- **GT / linear-attention baseline** = filter off (the config default):
  `tmgg-mod-attention model.filter_qk=false`
- **GCAT — "GCA", the proposed method** = graph-convolutional Q/K projections on:
  `tmgg-mod-attention model.filter_qk=true`
  (Paper §4.1 Eq. 27 filters **Q and K only**; `model.filter_v` is a codebase-only
  extra ablation, not part of Figure 1. `model.filter_num_terms` = polynomial order.)

Defaults are in `src/tmgg/experiments/exp_configs/models/modified_attention/mod_attention.yaml`
(loaded by base config `base_config_mod_attention.yaml`).

> **Resolved (paper §4.1, Eq. 26–27; Figure 1 caption).** Figure 1 compares
> **GT** vs **GCAT**, both realized by `tmgg-mod-attention`: Eq. 26 (linear Q/K,
> `model.filter_qk=false`) is GT, Eq. 27 (adjacency-polynomial Q/K,
> `model.filter_qk=true`) is GCAT — the "Graph Convolutional Attention Transformer",
> i.e. the paper's proposed **GCA** mechanism. The code confirms it:
> `mod_attention.py` implements the K-tap filter `Z = Σ_i Aⁱ X Hᵢ` (Eq. 27) gated by
> `filter_qk`. `tmgg-spectral-arch` is the spectral-attention *analysis* family of
> §2–3 (Linear PE / Filter Bank / Self-Attention), and `tmgg-gnn-transformer` is the
> GNN+transformer hybrid — neither is the GT/GCAT toggle. **Residual (not a flag):**
> the committed `run-denoising-sbm-panel-a10g.zsh` wrapper launches a 5-architecture
> panel (spectral / GNN / hybrid), a *secondary* denoising comparison — not the
> Figure-1 GT-vs-GCAT pair (see the relabelled row below).

### Command → config → notes

| Paper artifact | Command | Config file | Notes |
|---|---|---|---|
| Fig 1 · GCAT (GCA, proposed) | `uv run tmgg-mod-attention model.filter_qk=true +data=<DATASET>` | `models/modified_attention/mod_attention.yaml` | The paper's method (GCAT, Eq. 27 — Q/K filtered only). `filter_num_terms` sets the polynomial order. `filter_v` is an extra ablation, not Figure 1. |
| Fig 1 · GT / linear-attention baseline | `uv run tmgg-mod-attention model.filter_qk=false model.filter_v=false +data=<DATASET>` | same file | Filters off = plain scaled-dot-product attention on eigenvectors. |
| Fig 1 · Spectral denoisers | `uv run tmgg-spectral-arch +models/spectral@model=<linear_pe\|filter_bank\|self_attention> +data=<DATASET>` | `models/spectral/*.yaml`, base `base_config_spectral_arch.yaml` | "Main focus of the paper" per README. Default model is `linear_pe`. |
| Fig 1 · GNN | `uv run tmgg-gnn +models/gnn@model=<standard_gnn\|symmetric_gnn\|nodevar_gnn> +data=<DATASET>` | `models/gnn/*.yaml`, base `base_config_gnn.yaml` | |
| Fig 1 · GNN+Transformer hybrid | `uv run tmgg-gnn-transformer +data=<DATASET>` | `models/hybrid/hybrid_with_transformer.yaml`, base `base_config_gnn_transformer.yaml` | |
| Fig 1 · DiGress transformer baseline | `uv run tmgg-digress +data=<DATASET>` | `models/digress/digress_transformer.yaml`, base `base_config_digress.yaml` | See "DiGress projection variants" below for the transformer's own GNN/spectral Q/K/V toggle. |
| Fig 1 · Linear / MLP floor | `uv run tmgg-baseline +data=<DATASET>` (or `model=baselines/mlp`) | `models/baselines/*.yaml`, base `base_config_baseline.yaml` | Structure-blind sanity floor. |
| Denoising multi-arch panel (5 archs, Modal; *not* Figure 1) | `./run-denoising-sbm-panel-a10g.zsh` | the wrapper | Launches baselines-linear, spectral linear_pe, spectral multilayer_self_attention, standard_gnn, hybrid on synthetic SBM (`+data=sbm_default`, `p_intra=0.3`, `p_inter=0.005`), `max_steps=2000`. A secondary architecture comparison — Figure 1 itself is the GT-vs-GCAT pair above (`tmgg-mod-attention model.filter_qk={false,true}`). |

Noise settings (type and levels) come from `task/denoising.yaml`
(`noise_type: digress`, `noise_levels: [0.01, 0.05, 0.1, 0.2, 0.3]`); override
with `noise_levels=[...]`.

### `<DATASET>` → `data=` option

Replace `<DATASET>` above with one of these `data/*.yaml` option names:

| Paper dataset | `data=` option | Config file |
|---|---|---|
| Synthetic SBM | `sbm_default` (also `sbm_n100`, `sbm_n200`) | `data/sbm_default.yaml` |
| SPECTRE-SBM | `spectre_sbm` | `data/spectre_sbm.yaml` |
| PROTEINS | `pyg_proteins` | `data/pyg_proteins.yaml` |
| ENZYMES | `pyg_enzymes` | `data/pyg_enzymes.yaml` |
| IMDB-BINARY | `pyg_imdb_binary` | `data/pyg_imdb_binary.yaml` |
| COLLAB | `pyg_collab` | `data/pyg_collab.yaml` |
| DEEZER-EGO-NETS | `pyg_deezer_ego_nets` | `data/pyg_deezer_ego_nets.yaml` |

**Synthetic-SBM `alpha ∈ {0.1, 1.0, 10.0}`** is the *data* α — the symmetric
Dirichlet block-size concentration (paper Appendix B.2.3, Eq. 28), named in the
code as `block_size_alpha` in `generate_sbm_batch`
(`src/tmgg/data/datasets/sbm.py`). (Do not confuse it with the *theory* α of §3,
the softmax sharpness in the outlier-shrinkage analysis — a different symbol.)
The data-module dispatch (`src/tmgg/data/data_modules/graph_generation.py`) now
forwards it, and setting it routes the paper's `sbm_n100`/`sbm_n200` presets to
the per-graph Dirichlet batch path (they otherwise use enumerated partitions and
bypass the generator). Reproduce the sweep with the paper's own presets plus an
override — this is the stable, paper-faithful form:

```bash
uv run tmgg-discrete-gen --multirun \
  data=sbm_n200 +data.graph_config.block_size_alpha=0.1,1.0,10.0   # n=200 (max block 60)
uv run tmgg-discrete-gen --multirun \
  data=sbm_n100 +data.graph_config.block_size_alpha=0.1,1.0,10.0   # n=100 (max block 50)
```

Larger α → more homogeneous block sizes → less spectral diversity across graphs.
Geometry is `p_intra=1.0`, `p_inter=0.0` (disjoint cliques), block count K uniform
in {2,3,4}; dataset size is `data.graph_config.num_graphs` (default 1000). There is
no dedicated experiment preset — use the override form above.

**Node-count caps on large graphs.** `models/spectral/linear_pe.yaml` sizes its
learnable bias term with `max_nodes` (default `200`); a spectral run on a dataset
whose largest graph exceeds `max_nodes` now raises an informative `ValueError`
rather than crashing opaquely. Raise `model.max_nodes` to at least the dataset
maximum (recorded in each `data/pyg_*.yaml` as `graph_config.num_nodes`):

| Dataset | required `model.max_nodes` ≥ |
|---|---|
| PROTEINS | 620 |
| COLLAB | 492 |
| DEEZER-EGO-NETS | 363 |
| IMDB-BINARY | 136 |
| ENZYMES | 126 |
| synthetic / SPECTRE SBM | 200 (default is sufficient) |

The exact per-cell value used for each Figure-1 panel is not separately recorded;
use the dataset maximum above unless the paper states a tighter cap.

### DiGress projection variants (the transformer's own GCA realization)

The DiGress GraphTransformer implements the *same* graph-convolutional-attention
idea via `model.projection_config`
(`src/tmgg/models/digress/transformer_model.py`; `use_gnn_*` and `use_spectral_*`
are mutually exclusive per projection). Ready-made presets:

| Preset (`model=digress/…`) | projection_config |
|---|---|
| `digress_transformer` | none — plain linear attention (GT) |
| `digress_transformer_gnn_all` / `_gnn_qk` / `_gnn_v` | `use_gnn_{q,k,v}` (GNN-conv projections) |
| `digress_transformer_spectral_all` / `_spectral_qk` | `use_spectral_{q,k,v}` (spectral projections) |

Run e.g. `uv run tmgg-digress model=digress/digress_transformer_gnn_all +data=<DATASET>`.
This same toggle is what the generative ablations below turn on.

---

## §4.4 Generative — Table 1 (quality) and Table 2 (inference time)

All generative runs use `tmgg-discrete-gen`. The paper's ablation cells are
pre-composed Hydra *experiment* configs under
`src/tmgg/experiments/exp_configs/experiment/`; run one with
`+experiment=<name>`. Each config sets data, architecture, optimizer, schedule,
and the R-PEARL / projection variant — you do not stack overrides by hand.

The four/five variants and how they differ (all layer on the DiGress SBM recipe):

| Variant | Positional features | Q/K/V projection | Experiment-config suffix |
|---|---|---|---|
| `vignac` (DiGress baseline) | Laplacian `eigh` (`ExtraFeatures`) | linear | `*_vignac_repro` |
| `pearl` (R-PEARL) | R-PEARL GNN encoding | linear | `*_pearl_repro` |
| `pearl-spectral` | R-PEARL | spectral (`use_spectral_{q,k,v}`) | `*_pearl_spectral_repro` |
| `pearl-gnnconv-norm` | R-PEARL | GNN-conv, normalized A | `*_pearl_gnnconv_norm_repro` |
| `pearl-gnnconv-raw` | R-PEARL | GNN-conv, raw A | `*_pearl_gnnconv_raw_repro` |
| `vignac-spectral` (SBM only) | `eigh` | spectral | `discrete_sbm_vignac_spectral_repro` |

`_repro` = the project's step-based schedule; `_repro_exact` = byte-for-byte
upstream-DiGress parity (the paper-anchor baseline). Not every variant has an
`_exact` twin: `*_pearl_gnnconv_raw_repro` and `discrete_enzymes_vignac_spectral`
do **not** exist; the raw-A variant is also documented as numerically unstable
(diverges) in its own config header.

### Command → config → notes

| Paper artifact | Command | Config file | Notes |
|---|---|---|---|
| Table 1 · SBM, DiGress baseline (anchor) | `uv run tmgg-discrete-gen +experiment=discrete_sbm_vignac_repro_exact` | `experiment/discrete_sbm_vignac_repro_exact.yaml` | Config header notes this is expected to land closest to DiGress Table 1. |
| Table 1 · SBM, R-PEARL | `… +experiment=discrete_sbm_pearl_repro[_exact]` | `experiment/discrete_sbm_pearl_repro*.yaml` | |
| Table 1 · SBM, R-PEARL + spectral Q/K/V | `… +experiment=discrete_sbm_pearl_spectral_repro[_exact]` | `experiment/discrete_sbm_pearl_spectral_repro*.yaml` | |
| Table 1 · SBM, R-PEARL + GNN-conv Q/K/V | `… +experiment=discrete_sbm_pearl_gnnconv_norm_repro[_exact]` | `experiment/discrete_sbm_pearl_gnnconv_norm_repro*.yaml` | |
| Table 1 · SBM, R-PEARL + GNN-conv (raw A) | `… +experiment=discrete_sbm_pearl_gnnconv_raw_repro` | `experiment/discrete_sbm_pearl_gnnconv_raw_repro.yaml` | No `_exact`; documented as numerically unstable. |
| Table 1 · SBM, DiGress + spectral Q/K/V | `… +experiment=discrete_sbm_vignac_spectral_repro[_exact]` | `experiment/discrete_sbm_vignac_spectral_repro*.yaml` | `eigh` features, spectral projections, no PEARL. |
| Table 1 · ENZYMES (each variant) | `… +experiment=discrete_enzymes_{vignac,pearl,pearl_spectral,pearl_gnnconv_norm}_repro[_exact]` and `…_pearl_gnnconv_raw_repro` | `experiment/discrete_enzymes_*_repro*.yaml` | Same variant set as SBM minus `vignac_spectral`. Data = `pyg_enzymes`. |
| Table 1 · Modal launcher (any cell) | `./scripts/run-digress-repro-modal.zsh <slug> [overrides]` | the script | Slug → experiment map is the `case` block in the script (e.g. `sbm`, `sbm-vignac-exact`, `sbm-pearl-spectral`, `enzymes-pearl-gnnconv-norm-exact`, …). Verified reference invocation. |
| Table 1 · molecular / planar DiGress repros | `… +experiment=discrete_{qm9,moses,guacamol,planar}_digress_repro` (or wrapper slugs `qm9`/`moses`/`guacamol`/`planar`) | `experiment/discrete_{qm9,moses,guacamol,planar}_digress_repro.yaml` | **Baseline DiGress only** — no R-PEARL / GCA variants exist in the tree for these datasets. |
| Table 2 · inference-time (SBM + ENZYMES) | the four `_exact` variants `vignac`, `pearl`, `pearl_spectral`, `pearl_gnnconv_norm` on both datasets | `experiment/discrete_{sbm,enzymes}_{vignac,pearl,pearl_spectral,pearl_gnnconv_norm}_repro_exact.yaml` | These are exactly the 8 runs behind the inference-time bundle. |

### Which datasets carry the ablation

The R-PEARL / spectral / GNN-conv ablation exists **only for SBM
(`spectre_sbm`) and ENZYMES (`pyg_enzymes`)**. The molecular datasets (QM9,
MOSES, GuacaMol) and Planar have DiGress-baseline repro configs only.

### Table 1 metric source

Generation quality is logged as squared MMD² (`gen-val/{degree,clustering,orbit,spectral}_mmd`).
The consolidated data + interpretation for the SBM/ENZYMES ablation ships as a
paper-bound bundle: `paper-artifacts/repro-ablations/` (long-format CSV,
per-run history parquets, pre-/post-fix configs, MMD-unit protocol, run
lineage). Read its `README.md` before comparing to DiGress/HiGen — MMD values
are squared, and pre-mask-fix runs are not paper-citable.

### Table 2 (inference time) analysis bundle

`paper-artifacts/inference-time-analysis/` is a self-contained bundle that turns
the 8 `_exact` runs into the inference-time table and figures:

```bash
uv run paper-artifacts/inference-time-analysis/analyze.py          # committed data -> tables/ + figures/
uv run paper-artifacts/inference-time-analysis/export_from_wandb.py # refresh data/perf.csv from W&B first
```

Outputs: `tables/inference_time_main.tex`, `tables/inference_time_compact.tex`,
`figures/`. The measured quantities are the instrumented training step
(`impl-perf/train/step_time_s`) and a wall-clock-derived per-cycle inference
cost. Read the bundle's `README.md` Caveats before quoting: inference-cycle time
is derived (an upper bound), runs are single-seed and were preempted before
convergence, and the numbers predate the sparse-default refactor. A parallel
bundle, `paper-artifacts/pearl-perf/`, renders the train-vs-validation wall-time
figures from the same runs.

---

## Config-reference index (where the numbers live)

| Concern | File(s) |
|---|---|
| Shared training infra (optimizer, scheduler, seed, matmul precision) | `src/tmgg/experiments/exp_configs/_base_infra.yaml` |
| Denoising task (loss, noise type/levels, data default) | `src/tmgg/experiments/exp_configs/task/denoising.yaml` |
| GCA denoiser hyperparameters | `src/tmgg/experiments/exp_configs/models/modified_attention/mod_attention.yaml` |
| Spectral denoiser hyperparameters | `src/tmgg/experiments/exp_configs/models/spectral/*.yaml` |
| DiGress transformer + projection presets | `src/tmgg/experiments/exp_configs/models/digress/*.yaml` |
| Discrete-diffusion backbone / schedule (generative) | `src/tmgg/experiments/exp_configs/models/discrete/discrete_sbm_official.yaml`, `discrete_default.yaml` |
| Generative ablation cells (Table 1/2) | `src/tmgg/experiments/exp_configs/experiment/discrete_*_repro*.yaml` |
| Dataset presets | `src/tmgg/experiments/exp_configs/data/*.yaml` |

For a broader tour of the CLI and override system see
`docs/how-to-run-experiments.md` and
`src/tmgg/experiments/exp_configs/README.md`.

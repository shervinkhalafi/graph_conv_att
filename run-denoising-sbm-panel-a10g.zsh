#!/usr/bin/env zsh

set -euo pipefail

# Single-step denoising architecture panel on small SBM graphs.
#
# Runs five architectures from the denoising research surface against
# the *same* data and noise settings, each as a separate detached Modal
# spawn on A10G. Data is synthetic SBM with the upstream DiGress
# parameters (p_intra=0.3, p_inter=0.005) at fixed n=20 so graphs fit
# cheaply on A10G (24 GB) with wide batches.
#
# The five architectures span the families surveyed on 2026-04-15:
#
#   - ``baselines/linear``          -- structure-blind sanity floor
#   - ``spectral/linear_pe``        -- simplest spectral denoiser
#   - ``spectral/multilayer_self_attention``
#                                   -- most expressive spectral variant
#   - ``gnn/standard_gnn``          -- GNN reference
#   - ``hybrid/hybrid_with_transformer``
#                                   -- GNN embedding + transformer head
#
# Fixed schedule: max_steps=2000, val every 500. Gives two validation
# checkpoints per run; enough to compare learning curves without the
# long sampling+eval phase dominating wall time. Seeds distinct so
# wandb/run-id routing is clean.
#
# Usage:
#
#   ./run-denoising-sbm-panel-a10g.zsh           # all five in sequence
#   DRY_RUN=1 ./run-denoising-sbm-panel-a10g.zsh  # print commands only
#   DEPLOY_FIRST=0 ./run-denoising-sbm-panel-a10g.zsh
#                                                 # skip modal-deploy

: "${DEPLOY_FIRST:=1}"
: "${DRY_RUN:=0}"
: "${GPU_TIER:=standard}"
: "${WANDB_ENTITY:=your-wandb-entity}"
: "${WANDB_PROJECT:=architecture-study}"
: "${MPLCONFIGDIR:=${TMPDIR:-/tmp}/tmgg-mpl-cache}"

mkdir -p "${MPLCONFIGDIR}"
export MPLCONFIGDIR

run_prefixed() {
  "$@"
}

if [[ "${DEPLOY_FIRST}" == "1" ]]; then
  print -r -- "Deploying Modal app and refreshing secrets..."
  run_prefixed mise run modal-deploy
fi

# (model_slug, hydra_override, seed)
typeset -a panel
panel=(
  "baselines-linear|+models/baselines@model=linear|101"
  "spectral-linear-pe|+models/spectral@model=linear_pe|102"
  "spectral-multilayer-sa|+models/spectral@model=multilayer_self_attention|103"
  "gnn-standard|+models/gnn@model=standard_gnn|104"
  "hybrid-transformer|+models/hybrid@model=hybrid_with_transformer|105"
)

for entry in "${panel[@]}"; do
  model_slug=${entry%%|*}
  rest=${entry#*|}
  model_override=${rest%%|*}
  seed=${rest##*|}

  print -r -- ""
  print -r -- "=== ${model_slug} (seed=${seed}) ==="

  typeset -a cmd
  cmd=(
    uv run tmgg-modal run tmgg-spectral-arch
    "${model_override}"
    +data=sbm_default
    data.graph_config.p_intra=0.3
    data.graph_config.p_inter=0.005
    seed="${seed}"
    trainer.max_steps=2000
    trainer.val_check_interval=500
    allow_no_wandb=false
    wandb_entity="${WANDB_ENTITY}"
    wandb_project="${WANDB_PROJECT}"
    --gpu
    "${GPU_TIER}"
    --detach
  )

  printf ' %q' "${cmd[@]}"
  print

  if [[ "${DRY_RUN}" == "1" ]]; then
    print -r -- "(dry run — skipping dispatch)"
    continue
  fi

  run_prefixed "${cmd[@]}"
done

print -r -- ""
print -r -- "All five spawns issued. Follow progress at:"
print -r -- "  https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"

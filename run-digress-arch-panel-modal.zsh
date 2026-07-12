#!/usr/bin/env zsh

set -euo pipefail

# DiGress architecture panel on Modal: SPECTRE SBM, discrete multi-step
# diffusion, five architecture families dispatched as detached runs.
#
# Panel members (slug, models/discrete@model override, seed):
#   - digress-transformer | discrete_sbm_official         | 201
#   - gnn-standard        | gnn/standard_gnn              | 202
#   - spectral-linear-pe  | spectral/linear_pe            | 203
#   - baseline-linear     | baselines/linear              | 204
#   - hybrid-transformer  | hybrid/hybrid_with_transformer | 205
#
# Fixed schedule: max_steps=2000 with val every 500 gives two validation
# checkpoints per run — enough to compare learning curves without the
# sampling+eval phase dominating wall time. Seeds distinct so W&B run
# routing is clean.
#
# ``+model.model.extra_features.*`` overrides are only valid for the
# upstream GraphTransformer; the GNN / spectral / baseline / hybrid
# constructors reject them (verified via Hydra compose+instantiate probe
# on 2026-04-15). The DiGress entry carries them inline; the other four
# rely on the panel yaml's default (no extra_features).
#
# Usage:
#   ./run-digress-arch-panel-modal.zsh
#   DRY_RUN=1 DEPLOY_FIRST=0 ./run-digress-arch-panel-modal.zsh

: "${DEPLOY_FIRST:=1}"
: "${DRY_RUN:=0}"
: "${GPU_TIER:=standard}"
: "${WANDB_ENTITY:=your-wandb-entity}"
: "${WANDB_PROJECT:=digress-arch-panel}"
# MODAL_DEBUG=0 (default) → container sets PYTHONOPTIMIZE=1 inside the
# training subprocess, stripping ``assert`` and ``if __debug__:`` blocks
# from the hot path (~50 host-side syncs/step removed; see
# internal notes (not included)).
# MODAL_DEBUG=1 → asserts active, for numerical investigation only.
: "${MODAL_DEBUG:=0}"
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

# Shared per-panel launch identifier. Passed as ``run_id=...`` on every
# spawn so each wave writes to a fresh /data/outputs/.../<run_id>/
# directory rather than appending to an earlier run's checkpoints,
# CSV versions, or test_results.json. Override via ``PANEL_TAG=<name>``
# (e.g. ``PANEL_TAG=no-xf`` for a no-extra-features ablation).
: "${PANEL_TAG:=$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ "${MODAL_DEBUG}" == "1" ]]; then
  modal_debug_override=true
else
  modal_debug_override=false
fi

print -r -- "Panel launch tag: ${PANEL_TAG}"

# (slug, models/discrete@model override, seed, use_extra_features)
typeset -a panel
panel=(
  "digress-transformer|discrete_sbm_official|201|1"
  "gnn-standard|gnn/standard_gnn|202|0"
  "spectral-linear-pe|spectral/linear_pe|203|0"
  "baseline-linear|baselines/linear|204|0"
  "hybrid-transformer|hybrid/hybrid_with_transformer|205|0"
)

for entry in "${panel[@]}"; do
  slug=${entry%%|*}
  rest=${entry#*|}
  override=${rest%%|*}
  rest=${rest#*|}
  seed=${rest%%|*}
  use_extra=${rest##*|}

  # Build a disambiguated run_id: panel-tag + seed + slug. The tag comes
  # first so wandb/volume listings sort all entries of a single launch
  # together even with the default alphabetical ordering.
  run_id="digress_panel_${PANEL_TAG}_s${seed}_${slug}"

  print -r -- ""
  print -r -- "=== ${slug} (seed=${seed}, run_id=${run_id}) ==="

  typeset -a cmd
  cmd=(
    uv run tmgg-modal run tmgg-discrete-gen
    "models/discrete@model=${override}"
    +data=spectre_sbm
    "+run_id=${run_id}"
    modal_debug="${modal_debug_override}"
    seed="${seed}"
    trainer.max_steps=2000
    trainer.val_check_interval=500
    model.eval_every_n_steps=500
    model.noise_schedule.timesteps=500
    allow_no_wandb=false
    wandb_entity="${WANDB_ENTITY}"
    wandb_project="${WANDB_PROJECT}"
  )

  if [[ "${use_extra}" == "1" ]]; then
    # Only the DiGress GraphTransformer accepts extra_features — GNN,
    # spectral, baseline and hybrid constructors raise TypeError when
    # the kwarg is present (confirmed via instantiate probe).
    cmd+=(
      +model.model.extra_features._target_=tmgg.models.digress.extra_features.ExtraFeatures
      +model.model.extra_features.extra_features_type=all
      +model.model.extra_features.max_n_nodes=200
    )
  fi

  cmd+=(--gpu "${GPU_TIER}" --detach)

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

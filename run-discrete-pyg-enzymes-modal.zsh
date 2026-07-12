#!/usr/bin/env zsh

set -euo pipefail

# Launch the upstream-DiGress ENZYMES training run on Modal.
#
# PyG ENZYMES dataset: ~600 graphs, 2-126 nodes, bioinformatics-derived.
# Used by HiGen Table for the DiGress comparison row.

: "${DEPLOY_FIRST:=1}"
: "${DETACH:=1}"
: "${DRY_RUN:=0}"
: "${GPU_TIER:=standard}"
: "${SEED:=1}"
: "${WANDB_ENTITY:=your-wandb-entity}"
: "${WANDB_PROJECT:=graph-denoising}"
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

if [[ "${MODAL_DEBUG}" == "1" ]]; then
  modal_debug_override=true
else
  modal_debug_override=false
fi

typeset -a cmd
cmd=(
  uv
  run
  tmgg-modal
  run
  tmgg-discrete-gen
  models/discrete@model=discrete_sbm_official
  +data=pyg_enzymes
  modal_debug="${modal_debug_override}"
  # The base config carries inline synthetic-only keys (``graph_type``,
  # ``num_nodes``, ``num_graphs``, ``train_ratio``, ``val_ratio``,
  # ``graph_config``). They are absorbed by
  # ``SpectreSBMDataModule.__init__``'s ``**_metadata`` and re-exposed
  # at the data namespace for downstream interpolation (the wandb
  # logger reads ``${data.num_nodes}``). No ``~`` deletes needed.
  learning_rate=0.0002
  weight_decay=1e-12
  amsgrad=true
  seed="${SEED}"
  trainer.max_steps=550000
  # Validate every 10 000 steps. Upstream DiGress validates per-epoch
  # (~44k steps at our batch size); we deliberately validate ~4x more
  # often for faster training-time feedback.
  #
  # NOTE (smallest-config sweep, 2026-04-29): the smallest-config
  # search is moving toward a non-uniform cosine/U-bowl eval cadence
  # (see internal design notes (not included) and internal sweep tooling (not included)). Per-round
  # round.yaml overrides will eventually replace these uniform values
  # with the inverse-CDF-placed schedule list. The training-side
  # consumer that reads the list is a deferred Phase 0.4 patch
  # (spec §10); until it lands, the schedule list is informational
  # and the 10000-step uniform cadence still governs val_check.
  trainer.val_check_interval=10000
  model.eval_every_n_steps=10000
  # timesteps=1000 now baked into discrete_sbm_official.yaml (parity #43);
  # CLI override removed.
  +model.model.extra_features._target_=tmgg.models.digress.extra_features.ExtraFeatures
  +model.model.extra_features.extra_features_type=all
  # ENZYMES graphs reach n=126; 150 gives a 19% margin for cycle-feature
  # counts that need a ceiling at-or-above the largest graph.
  +model.model.extra_features.max_n_nodes=150
  allow_no_wandb=false
  wandb_entity="${WANDB_ENTITY}"
  wandb_project="${WANDB_PROJECT}"
  --gpu
  "${GPU_TIER}"
)

if [[ "${DETACH}" == "1" ]]; then
  cmd+=(--detach)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

if (( $# > 0 )); then
  cmd+=("$@")
fi

print -r -- "Launching upstream-style DiGress SBM run on Modal..."
printf ' %q' "${cmd[@]}"
print

run_prefixed "${cmd[@]}"

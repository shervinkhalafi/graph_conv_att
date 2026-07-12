#!/usr/bin/env zsh
# DiGress repro panel launcher.
#
# Usage:
#   ./scripts/run-digress-repro-modal.zsh <sbm|planar|qm9|moses|guacamol> [hydra-overrides...]
#
# Env knobs (with defaults):
#   DEPLOY_FIRST=1, DETACH=1, DRY_RUN=0,
#   GPU_TIER=fast, PRECISION=bf16-mixed, MODAL_DEBUG=0
#
# See internal notes (not included) for context.

set -euo pipefail

: "${DEPLOY_FIRST:=1}"
: "${DETACH:=1}"
: "${DRY_RUN:=0}"
: "${GPU_TIER:=fast}"
: "${PRECISION:=bf16-mixed}"
: "${MODAL_DEBUG:=0}"
: "${MPLCONFIGDIR:=${TMPDIR:-/tmp}/tmgg-mpl-cache}"

mkdir -p "${MPLCONFIGDIR}"
export MPLCONFIGDIR

DATASET="${1:?usage: $0 <slug> [hydra-overrides...] — see case below for slugs}"
shift

case "$DATASET" in
  sbm)        EXP="discrete_sbm_vignac_repro" ;;
  sbm-vignac-exact)     EXP="discrete_sbm_vignac_repro_exact" ;;
  sbm-vignac-spectral)  EXP="discrete_sbm_vignac_spectral_repro" ;;
  sbm-pearl)  EXP="discrete_sbm_pearl_repro" ;;
  sbm-pearl-exact)             EXP="discrete_sbm_pearl_repro_exact" ;;
  sbm-pearl-spectral)  EXP="discrete_sbm_pearl_spectral_repro" ;;
  sbm-pearl-spectral-exact)    EXP="discrete_sbm_pearl_spectral_repro_exact" ;;
  sbm-pearl-gnnconv-norm)  EXP="discrete_sbm_pearl_gnnconv_norm_repro" ;;
  sbm-pearl-gnnconv-norm-exact) EXP="discrete_sbm_pearl_gnnconv_norm_repro_exact" ;;
  sbm-pearl-gnnconv-raw)   EXP="discrete_sbm_pearl_gnnconv_raw_repro" ;;
  enzymes)                   EXP="discrete_enzymes_vignac_repro" ;;
  enzymes-vignac-exact)      EXP="discrete_enzymes_vignac_repro_exact" ;;
  enzymes-pearl)             EXP="discrete_enzymes_pearl_repro" ;;
  enzymes-pearl-exact)       EXP="discrete_enzymes_pearl_repro_exact" ;;
  enzymes-pearl-spectral)    EXP="discrete_enzymes_pearl_spectral_repro" ;;
  enzymes-pearl-spectral-exact)    EXP="discrete_enzymes_pearl_spectral_repro_exact" ;;
  enzymes-pearl-gnnconv-norm) EXP="discrete_enzymes_pearl_gnnconv_norm_repro" ;;
  enzymes-pearl-gnnconv-norm-exact) EXP="discrete_enzymes_pearl_gnnconv_norm_repro_exact" ;;
  enzymes-pearl-gnnconv-raw)  EXP="discrete_enzymes_pearl_gnnconv_raw_repro" ;;
  planar)     EXP="discrete_planar_digress_repro" ;;
  qm9)        EXP="discrete_qm9_digress_repro" ;;
  moses)      EXP="discrete_moses_digress_repro" ;;
  guacamol)   EXP="discrete_guacamol_digress_repro" ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

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
  uv run tmgg-modal run tmgg-discrete-gen
  +experiment="${EXP}"
  trainer.precision="${PRECISION}"
  modal_debug="${modal_debug_override}"
  --gpu "${GPU_TIER}"
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

print -r -- "Launching DiGress repro: dataset=${DATASET} (${GPU_TIER}, ${PRECISION})"
printf ' %q' "${cmd[@]}"
print

run_prefixed "${cmd[@]}"

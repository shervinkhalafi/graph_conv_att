#!/usr/bin/env zsh
# Backward-compatible shim — delegates to the parameterised launcher.
#
# Usage:
#   ./run-discrete-sbm-vignac-repro-modal-a100.zsh [hydra-overrides...]
#
# All env knobs (DEPLOY_FIRST, DETACH, GPU_TIER, PRECISION, MODAL_DEBUG)
# are forwarded unchanged to scripts/run-digress-repro-modal.zsh.

set -euo pipefail
exec "$(dirname "${0}")/scripts/run-digress-repro-modal.zsh" sbm "$@"

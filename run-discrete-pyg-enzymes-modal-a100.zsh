#!/usr/bin/env zsh

set -euo pipefail

# Upstream DiGress SBM replication on A100 with mixed precision.
#
# Thin wrapper over ``run-upstream-digress-sbm-modal.zsh``:
#
# - GPU tier = fast (A100-40GB) so batch_size=12 fits alongside the
#   SPECTRE SBM's variable-N up to 187.
# - Precision = bf16-mixed — bf16 keeps fp32's exponent range so we
#   do not need loss-scaling and cannot overflow on extreme gradients,
#   which is the usual failure mode of fp16 AMP on deep transformers.
#   A100 has native bf16 tensor cores; speedup vs fp32 is ~2x without
#   loss of numerical stability. Lightning enables ``torch.cuda.amp``
#   under the hood when ``trainer.precision=bf16-mixed``.
#
# Any extra CLI args are passed through to the main launcher.

script_dir=${0:A:h}

GPU_TIER=fast exec "${script_dir}/run-discrete-pyg-enzymes-modal.zsh" \
  trainer.precision=bf16-mixed \
  "$@"

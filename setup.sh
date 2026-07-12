#!/usr/bin/env bash
# One-shot setup for the Graph Convolutional Attention release.
#
# Steps: check prerequisites -> install dependencies (uv) -> pre-compile the
# ORCA orbit-counter -> restore bundled dataset caches -> bootstrap .env.
#
# Usage:
#   bash setup.sh
#
# Prerequisites: uv (https://docs.astral.sh/uv/), a C++ compiler (g++), and
# zstd. The script checks for these and prints install hints if any are missing.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "1/5 Checking prerequisites"
missing=0
command -v uv   >/dev/null 2>&1 || { echo "  MISSING uv   — https://docs.astral.sh/uv/getting-started/installation/"; missing=1; }
command -v g++  >/dev/null 2>&1 || { echo "  MISSING g++  — Ubuntu/Debian: sudo apt install g++ | macOS: xcode-select --install | Arch: sudo pacman -S gcc"; missing=1; }
command -v zstd >/dev/null 2>&1 || { echo "  MISSING zstd — Ubuntu/Debian: sudo apt install zstd | macOS: brew install zstd | Arch: sudo pacman -S zstd"; missing=1; }
[ "$missing" -eq 0 ] || { echo "Install the missing tools above and re-run."; exit 1; }
echo "  ok: uv, g++, zstd"

say "2/5 Installing dependencies (uv sync)"
uv sync

say "3/5 Pre-compiling ORCA orbit counter"
uv run python -c "from tmgg.evaluation import orca; print('  ORCA binary:', orca._get_binary_path())"

say "4/5 Restoring dataset caches"
if [ -d "$SCRIPT_DIR/data-bundle" ]; then
  bash "$SCRIPT_DIR/setup_data.sh"
else
  echo "  data-bundle/ not present — skipping. Fetch it via 'git lfs pull' or the"
  echo "  out-of-band URL in the README, then run: bash setup_data.sh"
fi

say "5/5 Bootstrapping .env"
if [ -f .env ]; then
  echo "  .env already exists — leaving it untouched."
else
  cp .env.example .env
  echo "  created .env from .env.example (defaults to WANDB_MODE=offline)."
fi

say "Setup complete."
cat <<'EOF'
Try a first run:
  uv run tmgg-spectral-arch          # spectral / GCA denoising
  uv run tmgg-discrete-gen --help    # DiGress-style generative diffusion

See REPRODUCE.md for the exact commands behind each paper table/figure.
Logging is off by default (WANDB_MODE=offline). To enable W&B, fill
WANDB_API_KEY in .env and set WANDB_MODE=online.
EOF

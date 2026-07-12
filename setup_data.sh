#!/usr/bin/env bash
# Restore the bundled dataset caches so evaluations are ready to run.
#
# What it does:
#   Verifies (if SHA256SUMS is present) and extracts the zstd-compressed dataset
#   tarballs in ./data-bundle/ into the exact locations the loaders expect:
#     - TUDatasets (COLLAB, ENZYMES, PROTEINS) -> $HOME/.pyg_data/
#     - SPECTRE-SBM fixture                    -> $HOME/.cache/tmgg/spectre/
#     - repo-local caches (eval, pyg_*)        -> ./data/
#
#   The bundle provides byte-identical, version-pinned datasets. IMDB-BINARY and
#   DEEZER-EGO-NETS are not bundled and auto-download via PyG on first use. A
#   missing SPECTRE tarball is non-fatal — it auto-downloads from its source URL.
#
# Usage:
#   bash setup_data.sh
#
# Requirements: tar, zstd, and (for integrity checks) sha256sum. Idempotent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="${TMGG_DATA_BUNDLE_DIR:-$SCRIPT_DIR/data-bundle}"

PYG_DATA_ROOT="${PYG_DATA_ROOT:-$HOME/.pyg_data}"
TMGG_CACHE_ROOT="${TMGG_CACHE_ROOT:-$HOME/.cache/tmgg}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-$SCRIPT_DIR/data}"

command -v zstd >/dev/null 2>&1 || { echo "ERROR: zstd not found. Install: apt install zstd | brew install zstd | pacman -S zstd" >&2; exit 1; }
command -v tar  >/dev/null 2>&1 || { echo "ERROR: tar not found." >&2; exit 1; }

if [ ! -d "$BUNDLE_DIR" ]; then
  echo "No data-bundle/ at $BUNDLE_DIR."
  echo "Fetch it via 'git lfs pull' or the out-of-band URL in the README, then re-run."
  exit 0
fi

# Integrity check when a checksum manifest ships with the bundle.
if [ -f "$BUNDLE_DIR/SHA256SUMS" ]; then
  if command -v sha256sum >/dev/null 2>&1; then
    echo "Verifying bundle integrity (SHA256SUMS)"
    ( cd "$BUNDLE_DIR" && sha256sum --check --ignore-missing SHA256SUMS )
  else
    echo "WARN: sha256sum not found — skipping integrity check."
  fi
fi

# tarball -> target parent directory
extract() {
  local tarball="$1" target="$2"
  local path="$BUNDLE_DIR/$tarball"
  if [ ! -f "$path" ]; then
    echo "  SKIP     $tarball (not present)"
    return 0
  fi
  mkdir -p "$target"
  echo "  RESTORE  $tarball -> $target"
  zstd -dc "$path" | tar -x -C "$target"
}

echo "Restoring dataset caches from $BUNDLE_DIR"
extract "pyg_data_COLLAB.tar.zst"    "$PYG_DATA_ROOT"
extract "pyg_data_ENZYMES.tar.zst"   "$PYG_DATA_ROOT"
extract "pyg_data_PROTEINS.tar.zst"  "$PYG_DATA_ROOT"
extract "tmgg_spectre.tar.zst"       "$TMGG_CACHE_ROOT"
extract "data_local.tar.zst"         "$LOCAL_DATA_ROOT"
echo "Done. IMDB-BINARY and DEEZER-EGO-NETS auto-download on first use."

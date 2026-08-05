#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
SEED="${1:-0}"
for method in thesis hilserl eil pvp hg_dagger sac; do
  "$ROOT/scripts/generated/run_unitree_competitor_method_20260718.sh" "$method" "$SEED"
done

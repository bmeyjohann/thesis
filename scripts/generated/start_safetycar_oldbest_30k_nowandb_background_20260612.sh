#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
LOG_DIR="$ROOT/logs/manual_runs"
mkdir -p "$LOG_DIR"

nohup bash -lc 'cd /home/benjamin/thesis && DISABLE_WANDB=1 scripts/generated/run_safetycar_oldbest_framestack4_bcstrong_fixedalpha_30k_nowandb_20260612.sh' \
  > "$LOG_DIR/safetycar_oldbest_30k_nowandb_20260612.out" \
  2> "$LOG_DIR/safetycar_oldbest_30k_nowandb_20260612.err" &

echo "$!" > "$LOG_DIR/safetycar_oldbest_30k_nowandb_20260612.pid"
cat "$LOG_DIR/safetycar_oldbest_30k_nowandb_20260612.pid"

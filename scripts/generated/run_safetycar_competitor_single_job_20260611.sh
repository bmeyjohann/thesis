#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
  if [[ "$arg" != *=* ]]; then
    echo "unexpected positional argument: $arg" >&2
    exit 2
  fi
  export "$arg"
done

ROOT="${ROOT:-/home/benjamin/thesis}"
METHOD="${METHOD:-pvp}"
STEPS="${STEPS:-10000}"
SEED="${SEED:-211}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/logs/safetygym_competitor_comparison_${RUN_TS}}"
RAW_LOG_DIR="$OUT_ROOT/raw_logs"
EXP_NAME="${EXP_NAME:-safetycar_goal1_comp_${METHOD}_${STEPS}_seed${SEED}_${RUN_TS}}"
LOG_PATH="$RAW_LOG_DIR/${EXP_NAME}.log"

mkdir -p "$RAW_LOG_DIR"
cd "$ROOT"

echo "[single] method=$METHOD steps=$STEPS seed=$SEED exp=$EXP_NAME log=$LOG_PATH"
METHOD="$METHOD" \
STEPS="$STEPS" \
SEED="$SEED" \
RUN_TS="$RUN_TS" \
EXP_NAME="$EXP_NAME" \
WANDB_MODE="${WANDB_MODE:-disabled}" \
WANDB_GROUP="${WANDB_GROUP:-safetycar_competitors_${RUN_TS}}" \
  "$ROOT/scripts/generated/run_safetycar_competitor_method_20260611.sh" \
  2>&1 | tee "$LOG_PATH"

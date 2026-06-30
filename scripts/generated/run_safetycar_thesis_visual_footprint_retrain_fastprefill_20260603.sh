#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_visual_footprint_retrain_20260603.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_visualfootprint_goalinit_m024_t045_fastprefill}"
export STEPS="${STEPS:-60000}"
export SEED="${SEED:-107}"
export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-8}"
export PREFILL_MAX_STEPS_PER_EPISODE="${PREFILL_MAX_STEPS_PER_EPISODE:-200}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"

exec "$BASE"

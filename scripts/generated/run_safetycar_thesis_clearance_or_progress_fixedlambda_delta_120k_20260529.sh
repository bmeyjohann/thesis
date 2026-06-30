#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_prefill_aug_bounded_20260529.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_fixedlambda_delta_120k}"
export STEPS="${STEPS:-120000}"
export SEED="${SEED:-35}"
export PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-1.0}"
export PREF_LAMBDA_LR="0.0"
export PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-1.0}"
export PREF_ACTION_DELTA_MIN="${PREF_ACTION_DELTA_MIN:-0.35}"
export PREF_ACTION_DELTA_WEIGHT_SCALE="${PREF_ACTION_DELTA_WEIGHT_SCALE:-2.0}"
export PREF_ACTION_DELTA_WEIGHT_MAX="${PREF_ACTION_DELTA_WEIGHT_MAX:-6.0}"

exec "$BASE"

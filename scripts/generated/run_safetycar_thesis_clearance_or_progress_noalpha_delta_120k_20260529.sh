#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_prefill_aug_bounded_20260529.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_noalpha_delta_120k}"
export STEPS="${STEPS:-120000}"
export SEED="${SEED:-36}"
export ALPHA_INIT="0.000001"
export ALPHA_MIN="0.000001"
export ALPHA_MAX="0.000001"
export PREF_ACTION_DELTA_MIN="${PREF_ACTION_DELTA_MIN:-0.35}"
export PREF_ACTION_DELTA_WEIGHT_SCALE="${PREF_ACTION_DELTA_WEIGHT_SCALE:-2.0}"
export PREF_ACTION_DELTA_WEIGHT_MAX="${PREF_ACTION_DELTA_WEIGHT_MAX:-6.0}"

exec "$BASE"

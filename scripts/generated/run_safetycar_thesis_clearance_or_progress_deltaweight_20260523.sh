#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_20260523.sh"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_deltaweight}"
export PREF_ACTION_DELTA_WEIGHT_SCALE="${PREF_ACTION_DELTA_WEIGHT_SCALE:-0.5}"
export PREF_ACTION_DELTA_WEIGHT_MAX="${PREF_ACTION_DELTA_WEIGHT_MAX:-3.0}"
exec "$BASE"

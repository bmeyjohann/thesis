#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_20260523.sh"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_strongpref}"
export PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-3.0}"
export PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-15.0}"
export PREF_RANK_MARGIN="${PREF_RANK_MARGIN:-0.2}"
exec "$BASE"

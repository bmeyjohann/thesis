#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_20260523.sh"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_augpref}"
export PREF_ACTION_NOISE_COPIES="${PREF_ACTION_NOISE_COPIES:-4}"
export PREF_ACTION_NOISE_STD="${PREF_ACTION_NOISE_STD:-0.03}"
exec "$BASE"

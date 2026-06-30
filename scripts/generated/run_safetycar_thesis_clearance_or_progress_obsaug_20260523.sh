#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_20260523.sh"
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_obsaug}"
export STEPS="${STEPS:-8000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-4000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-4000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4000}"
export PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-0.5}"
export PREF_ACTION_NOISE_COPIES="${PREF_ACTION_NOISE_COPIES:-4}"
export PREF_OBS_NOISE_STD="${PREF_OBS_NOISE_STD:-0.01}"
export PREF_ACTION_NOISE_STD="0.0"
exec "$BASE"

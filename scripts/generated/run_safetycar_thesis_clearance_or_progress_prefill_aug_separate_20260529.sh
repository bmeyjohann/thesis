#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_prefill_aug_bounded_20260529.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_prefill_aug_separate}"
export SEED="${SEED:-23}"
export PREF_SAMPLING_MODE="separate"
export PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-1.0}"
export PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"

exec "$BASE"

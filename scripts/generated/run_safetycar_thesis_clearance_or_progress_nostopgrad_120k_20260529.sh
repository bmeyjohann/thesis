#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_prefill_aug_bounded_20260529.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_nostopgrad_120k}"
export STEPS="${STEPS:-120000}"
export SEED="${SEED:-34}"
export PREF_STOPGRAD_POSITIVE="0"

exec "$BASE"

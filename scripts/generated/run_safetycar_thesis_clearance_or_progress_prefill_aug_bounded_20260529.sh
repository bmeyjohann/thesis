#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_prefill_aug_20260529.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_prefill_aug_bounded}"
export STEPS="${STEPS:-120000}"
export SEED="${SEED:-22}"
export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-40}"
export PREFILL_MAX_STEPS_PER_EPISODE="${PREFILL_MAX_STEPS_PER_EPISODE:-300}"
export PREFILL_POLICY="${PREFILL_POLICY:-student}"

exec "$BASE"

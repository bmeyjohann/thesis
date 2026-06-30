#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_prefill_aug_bounded_20260529.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_stablecritic_120k}"
export STEPS="${STEPS:-120000}"
export SEED="${SEED:-32}"
export ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE:-0.0001}"
export CRITIC_LEARNING_RATE="${CRITIC_LEARNING_RATE:-0.0001}"
export NUM_UPDATES="${NUM_UPDATES:-1}"
export PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-5.0}"
export PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-0.01}"
export PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-50.0}"

exec "$BASE"

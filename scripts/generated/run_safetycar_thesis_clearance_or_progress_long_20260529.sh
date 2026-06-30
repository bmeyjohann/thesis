#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_20260523.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_long}"
export STEPS="${STEPS:-200000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
export SEED="${SEED:-1}"

# Keep the learner reward Euclidean-only; safety information enters only through interventions.
export REWARD_MODE="${REWARD_MODE:-dense}"
export DENSE_REWARD_SCALE="${DENSE_REWARD_SCALE:-1.0}"
export SUCCESS_REWARD_SCALE="${SUCCESS_REWARD_SCALE:-0.0}"
export STEP_PENALTY="${STEP_PENALTY:-0.0}"
export CLEARANCE_PENALTY_SCALE="0.0"

# Baseline thesis-method setting: linked preference loss from online teacher interventions.
export PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-0.5}"
export PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
export PREF_RANK_MARGIN="${PREF_RANK_MARGIN:-0.1}"
export PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-1.0}"
export PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-10.0}"

export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_clearance_or_progress_long_20260529}"

exec "$BASE"

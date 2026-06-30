#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_visual_footprint_retrain_t050_m018_prefonly_noprefill_20k_20260604.sh"

# Diagnostic branch after pref-only collapse:
# keep the initialized goal-reaching actor nearly fixed while the critic absorbs
# replay + preference constraints, then allow only tiny actor updates anchored to
# the original goal policy.
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_visualfootprint_goalinit_m018_t050_exit055_e008_protected_5k}"
export STEPS="${STEPS:-5000}"
export SEED="${SEED:-115}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_visual_footprint_retrain_20260604_t050_m018_protected}"

export ACTOR_UPDATE_START_STEP="${ACTOR_UPDATE_START_STEP:-4000}"
export ACTOR_REFERENCE_DISTILL_WEIGHT="${ACTOR_REFERENCE_DISTILL_WEIGHT:-5.0}"
export ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE:-0.00003}"
export CRITIC_LEARNING_RATE="${CRITIC_LEARNING_RATE:-0.0003}"
export POLICY_FREQUENCY="${POLICY_FREQUENCY:-4}"
export FREEZE_OBS_NORMALIZER_AFTER_LOAD="${FREEZE_OBS_NORMALIZER_AFTER_LOAD:-1}"

export PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-0.1}"
export PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-0.1}"
export PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-0.00005}"
export PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-0.5}"

export EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"

exec "$BASE"

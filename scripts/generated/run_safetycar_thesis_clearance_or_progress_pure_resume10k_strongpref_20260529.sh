#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_20260523.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_clearance_or_progress_pure_resume10k_strongpref}"
export STEPS="${STEPS:-150000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
export SEED="${SEED:-13}"

# Branch from the best pure-preference continuation checkpoint.
export INIT_CKPT="${INIT_CKPT:-$ROOT/models/safetygym_minimal/safetycar_goal1_thesis_scriptedgeo_reward_160000_seed11_20260529_024339_clearance_or_progress_long_resume40k/step_10000.pt}"
export LOAD_CRITIC_FROM_CHECKPOINT=1
export LOAD_CRITIC_TARGET_FROM_CHECKPOINT=1
export LOAD_ALPHA_FROM_CHECKPOINT=1
export LOAD_OPTIMIZER_STATE_FROM_CHECKPOINT=1
export LOAD_OBS_NORMALIZER_FROM_CHECKPOINT=1

export REWARD_MODE="${REWARD_MODE:-dense}"
export DENSE_REWARD_SCALE="${DENSE_REWARD_SCALE:-1.0}"
export SUCCESS_REWARD_SCALE="${SUCCESS_REWARD_SCALE:-0.0}"
export STEP_PENALTY="${STEP_PENALTY:-0.0}"
export CLEARANCE_PENALTY_SCALE="0.0"

# Test whether the remaining safety gap is just weak preference pressure.
export PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-1.0}"
export PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-1.0}"
export PREF_RANK_MARGIN="${PREF_RANK_MARGIN:-0.1}"
export PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-3.0}"
export PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-20.0}"
export ACTOR_BC_WEIGHT=0.0

export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_clearance_or_progress_long_20260529}"

exec "$BASE"

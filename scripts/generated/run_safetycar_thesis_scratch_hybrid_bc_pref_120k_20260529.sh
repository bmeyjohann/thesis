#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_clearance_or_progress_prefill_aug_bounded_20260529.sh"

export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_scratch_dense_hybrid_bc_pref_120k}"
export STEPS="${STEPS:-120000}"
export SEED="${SEED:-43}"
export NO_INIT_CHECKPOINT="1"
export PREF_STOPGRAD_POSITIVE="0"
export PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-1.0}"
export PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-0.00025}"
export PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-3.0}"
export PREF_ACTION_DELTA_MIN="${PREF_ACTION_DELTA_MIN:-0.25}"
export PREF_ACTION_DELTA_WEIGHT_SCALE="${PREF_ACTION_DELTA_WEIGHT_SCALE:-1.0}"
export PREF_ACTION_DELTA_WEIGHT_MAX="${PREF_ACTION_DELTA_WEIGHT_MAX:-4.0}"
export ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-0.25}"
export ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE="${ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE:-2.0}"
export ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE="${ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE:-2.0}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_scratch_dense_20260529}"

exec "$BASE"

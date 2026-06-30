#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_visual_footprint_retrain_20260603.sh"

# Do not run unless the matching visual-MPC teacher audit has zero visual cost
# and acceptable goal-reaching behavior.
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_visualfootprint_visualmpc_t050_rel020_m030_e012_midbc025_p20}"
export SEED="${SEED:-112}"
export STEPS="${STEPS:-120000}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_visual_footprint_retrain_20260604_visual_mpc_t050_rel020_midbc}"

export HUMAN_INPUT_DEVICE_OVERRIDE="scripted_visual_mpc"
export TEACHER_MODE_OVERRIDE="clearance_projected_release_or_progress"
export TEACHER_OVERRIDE_CLEARANCE_THRESHOLD="0.50"
export TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD="0.20"
export TEACHER_PROGRESS_SCORE_MODE="euclidean"
export TEACHER_PROGRESS_DENSE_SCALE="1.0"
export TEACHER_PROGRESS_TRIGGER_MODE="${TEACHER_PROGRESS_TRIGGER_MODE:-not_improving}"
export TEACHER_PROGRESS_RELEASE_MODE="${TEACHER_PROGRESS_RELEASE_MODE:-improve}"
export TEACHER_PROGRESS_BAD_STEPS="${TEACHER_PROGRESS_BAD_STEPS:-3}"
export TEACHER_PROGRESS_GOOD_STEPS="${TEACHER_PROGRESS_GOOD_STEPS:-5}"
export TEACHER_PROGRESS_EPSILON="${TEACHER_PROGRESS_EPSILON:-0.0005}"
export SCRIPTED_GEO_SAFETY_MARGIN="0.30"
export SCRIPTED_GEO_EMERGENCY_CLEARANCE="0.12"
export SCRIPTED_GEO_ACTION_SHIELD_STEPS="0"

export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-20}"
export ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-0.25}"
export ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE="${ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE:-1.0}"
export ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE="${ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE:-1.0}"

exec "$BASE"

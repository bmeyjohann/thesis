#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_visual_footprint_retrain_20260603.sh"

# Retrain only after the matching audit confirms zero visual-footprint cost.
# Safety takeover combines projected-release visual clearance with Euclidean
# no-progress correction; learner reward stays dense Euclidean only.
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_visualfootprint_goalinit_m030_t050_rel020_projrelease_progress_e014_shield0_midbc025_p20}"
export SEED="${SEED:-111}"
export STEPS="${STEPS:-120000}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_visual_footprint_retrain_20260604_t050_rel020_projrelease_progress_midbc}"

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
export SCRIPTED_GEO_EMERGENCY_CLEARANCE="0.14"
export SCRIPTED_GEO_ACTION_SHIELD_STEPS="0"

export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-20}"
export ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-0.25}"
export ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE="${ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE:-1.0}"
export ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE="${ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE:-1.0}"

exec "$BASE"

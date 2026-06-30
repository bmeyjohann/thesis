#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_visual_footprint_retrain_20260603.sh"

# Decision run for the visual-footprint retrain plan:
# - keep the goal-only actor initialization
# - keep learner reward Euclidean/dense-only
# - use visual-footprint cost only for safety/intervention/eval
# - avoid BC/prefill because the teacher/gate is safe but still imperfect
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_visualfootprint_goalinit_m018_t050_exit055_e008_shield0_prefonly_noprefill_20k}"
export SEED="${SEED:-115}"
export STEPS="${STEPS:-20000}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_visual_footprint_retrain_20260604_t050_m018_prefonly_noprefill}"

export HUMAN_INPUT_DEVICE_OVERRIDE="scripted_geo"
export TEACHER_MODE_OVERRIDE="clearance"
export TEACHER_CLEARANCE_SOURCE="visual_footprint"
export TEACHER_OVERRIDE_CLEARANCE_THRESHOLD="0.50"
export TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD="0.55"
export SCRIPTED_GEO_LOOKAHEAD="1.6"
export SCRIPTED_GEO_SAFETY_MARGIN="0.18"
export SCRIPTED_GEO_GRID_RESOLUTION="0.05"
export SCRIPTED_GEO_EMERGENCY_CLEARANCE="0.08"
export SCRIPTED_GEO_ACTION_SHIELD_STEPS="0"

export PREFILL_DEMO_EPISODES="0"
export PREFILL_MAX_STEPS_PER_EPISODE="0"
export ACTOR_BC_WEIGHT="0.0"
export ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE="0.0"
export ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE="0.0"
export PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-1.0}"
export PREF_LAMBDA_INIT="${PREF_LAMBDA_INIT:-1.0}"
export PREF_LAMBDA_LR="${PREF_LAMBDA_LR:-0.00025}"
export PREF_LAMBDA_MAX="${PREF_LAMBDA_MAX:-3.0}"

export EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"

exec "$BASE"

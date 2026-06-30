#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_visual_footprint_retrain_20260603.sh"

# Retrain only after the matching audit confirms zero visual-footprint cost.
# This gate enters conservatively at 0.50 clearance but releases once actual and
# one-step projected student clearance have recovered above 0.20.
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_visualfootprint_goalinit_m030_t050_rel020_projrelease_e014_shield0_midbc025_p20}"
export SEED="${SEED:-110}"
export STEPS="${STEPS:-120000}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_visual_footprint_retrain_20260604_t050_rel020_projrelease_midbc}"

export TEACHER_MODE_OVERRIDE="clearance_projected_release"
export TEACHER_OVERRIDE_CLEARANCE_THRESHOLD="0.50"
export TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD="0.20"
export SCRIPTED_GEO_SAFETY_MARGIN="0.30"
export SCRIPTED_GEO_EMERGENCY_CLEARANCE="0.14"
export SCRIPTED_GEO_ACTION_SHIELD_STEPS="0"

export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-20}"
export ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-0.25}"
export ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE="${ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE:-1.0}"
export ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE="${ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE:-1.0}"

exec "$BASE"

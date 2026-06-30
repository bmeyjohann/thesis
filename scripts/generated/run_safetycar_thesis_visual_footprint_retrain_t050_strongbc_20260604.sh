#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_visual_footprint_retrain_20260603.sh"

# Uses the strongest teacher-imitation settings from the previous best thesis
# branch, but with the stricter zero-cost-audited visual-footprint gate:
# threshold/exit 0.50/0.55, margin 0.30, emergency clearance 0.14.
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_visualfootprint_goalinit_m030_t050_exit055_directional_e014_shield0_strongbc_p40}"
export SEED="${SEED:-113}"
export STEPS="${STEPS:-120000}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_visual_footprint_retrain_20260604_t050_strongbc}"

export HUMAN_INPUT_DEVICE_OVERRIDE="scripted_geo"
export TEACHER_MODE_OVERRIDE="clearance"
export TEACHER_OVERRIDE_CLEARANCE_THRESHOLD="0.50"
export TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD="0.55"
export SCRIPTED_GEO_SAFETY_MARGIN="0.30"
export SCRIPTED_GEO_EMERGENCY_CLEARANCE="0.14"
export SCRIPTED_GEO_ACTION_SHIELD_STEPS="0"

export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-40}"
export PREFILL_MAX_STEPS_PER_EPISODE="${PREFILL_MAX_STEPS_PER_EPISODE:-300}"
export ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-1.0}"
export ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE="${ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE:-3.0}"
export ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE="${ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE:-3.0}"

exec "$BASE"

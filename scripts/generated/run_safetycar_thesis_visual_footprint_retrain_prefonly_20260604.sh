#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
BASE="$ROOT/scripts/generated/run_safetycar_thesis_visual_footprint_retrain_20260603.sh"

# Same visual-footprint-safe gate as the main retrain, but reduce imitation
# pressure so the initialized goal policy is less likely to collapse into the
# over-conservative scripted teacher behavior.
export RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)_visualfootprint_goalinit_m028_t045_exit050_directional_e012_shield0_prefonly_p10}"
export SEED="${SEED:-107}"
export STEPS="${STEPS:-120000}"
export WANDB_GROUP="${WANDB_GROUP:-safetycar_thesis_visual_footprint_retrain_20260604_prefonly}"

export PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-10}"
export ACTOR_BC_WEIGHT="${ACTOR_BC_WEIGHT:-0.0}"
export ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE="${ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE:-0.0}"
export ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE="${ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE:-0.0}"

exec "$BASE"

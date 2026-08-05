#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
MODEL="$ROOT/models/unitree_mjlab_nav_thesis/unitree_scan13_hist5_actionhist4_history_only_2500_20260714/step_2500.pt"
VARIANT="${1:?expected original or masked}"

case "$VARIANT" in
  original) MASK_GOAL_HEADING=0 ;;
  masked) MASK_GOAL_HEADING=1 ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac

export MODEL_PATH="$MODEL"
export CONTROLLER=policy
export TASK=Unitree-G1-Nav-Obstacles-Safe-Collision
export DEVICE=cuda:0
export SEED=841
export NUM_ROLLOUTS=5
export STEPS=1200
export EPISODE_LENGTH_S=60.0
export SUCCESS_DIST=0.25
export HEIGHT_SCAN_RESOLUTION=0.25
export SCAN_HISTORY=5
export ACTION_HISTORY=4
export POLICY_ACTION_SMOOTHING=0.0
export MASK_GOAL_HEADING
export POLICY_TEACHER_GATE=0
export STRICT_MIN_SIZE_OBSTACLES=0
export REQUIRE_BLOCKED_CORRIDOR=0
export DEBUG_GOAL_THROUGH_OBSTACLE=0
export RESAMPLE_TERRAIN_TILES=1
export GOAL_DISTANCE_MIN=2.8
export GOAL_DISTANCE_MAX=4.0
export MIN_GOAL_OBSTACLE_CLEARANCE=0.9
export MIN_START_OBSTACLE_CLEARANCE=0.0
export DEBUG_OBSTACLE_WIDTH_MIN=1.0
export DEBUG_OBSTACLE_WIDTH_MAX=1.4
export DEBUG_OBSTACLE_HEIGHT_MIN=1.0
export DEBUG_OBSTACLE_HEIGHT_MAX=1.0
export DEBUG_NUM_OBSTACLES=6
export DEBUG_PLATFORM_WIDTH=2.0
export DEBUG_TERRAIN_ROWS=5
export DEBUG_TERRAIN_COLS=10
export OUTPUT_DIR="$ROOT/visualizations/unitree_goal_heading_ablation_20260714/$VARIANT"

exec "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh"

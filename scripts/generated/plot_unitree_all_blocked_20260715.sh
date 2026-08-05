#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: $0 direct_goal|geom_teacher|fastsac_goal_only|policy [checkpoint]}"
MODEL_PATH_ARG="${2:-}"
ROOT=/home/benjamin/thesis
LAUNCHER="$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh"
OUT="$ROOT/visualizations/unitree_all_blocked_20260715"

common=(
  DEVICE=cuda:0 SEED=1501 NUM_ROLLOUTS=4 STEPS=1200 EPISODE_LENGTH_S=60
  SUCCESS_DIST=0.40 HEIGHT_SCAN_RESOLUTION=0.25 SCAN_HISTORY=1 ACTION_HISTORY=0
  GOAL_DISTANCE_MIN=2.8 GOAL_DISTANCE_MAX=5.0
  MIN_GOAL_OBSTACLE_CLEARANCE=0.9 MIN_START_OBSTACLE_CLEARANCE=1.0
  DEBUG_GOAL_THROUGH_OBSTACLE=1 GOAL_THROUGH_OBSTACLE_PROB=1
  REQUIRE_BLOCKED_CORRIDOR=1 BLOCKED_CORRIDOR_RADIUS=0.45
  BLOCKED_CORRIDOR_IGNORE_END_RADIUS=0.75 BLOCKED_CORRIDOR_MIN_CELLS=4
  BLOCKED_CORRIDOR_RESAMPLE_ATTEMPTS=100
  BLOCKED_GOAL_MAX_DISTANCE=6.5
  DEBUG_GOAL_OBSTACLE_MIN_DIST=0.8 DEBUG_GOAL_OBSTACLE_MAX_DIST=3.0
  DEBUG_NUM_OBSTACLES=1 DEBUG_OBSTACLE_WIDTH_MIN=1.0 DEBUG_OBSTACLE_WIDTH_MAX=1.4
  DEBUG_OBSTACLE_HEIGHT_MIN=1.0 DEBUG_OBSTACLE_HEIGHT_MAX=1.0
  DEBUG_TERRAIN_ROWS=5 DEBUG_TERRAIN_COLS=10 DEBUG_PLATFORM_WIDTH=2.0
  STRICT_MIN_SIZE_OBSTACLES=1 RESAMPLE_TERRAIN_TILES=1
)

case "$METHOD" in
  direct_goal)
    exec "$LAUNCHER" CONTROLLER=direct_goal CHECKPOINT_ENV_CONFIG=0 \
      OUTPUT_DIR="$OUT/direct_goal" "${common[@]}"
    ;;
  geom_teacher)
    exec "$LAUNCHER" CONTROLLER=geom_scan_teacher CHECKPOINT_ENV_CONFIG=0 \
      TEACHER_GEOM_CLEARANCE=0.45 TEACHER_GOAL_STOP_DIST=0.30 \
      OUTPUT_DIR="$OUT/geom_teacher" "${common[@]}"
    ;;
  fastsac_goal_only)
    exec "$LAUNCHER" CONTROLLER=policy CHECKPOINT_ENV_CONFIG=0 \
      MASK_HEIGHT_SCAN=1 MASK_GOAL_HEADING=1 \
      MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_n5_r040_linear_4k_20260715/step_4000.pt" \
      OUTPUT_DIR="$OUT/fastsac_goal_only" "${common[@]}"
    ;;
  policy)
    if [[ -z "$MODEL_PATH_ARG" ]]; then
      echo "policy requires a checkpoint path" >&2
      exit 2
    fi
    name="$(basename "$(dirname "$MODEL_PATH_ARG")")_$(basename "$MODEL_PATH_ARG" .pt)"
    exec "$LAUNCHER" CONTROLLER=policy CHECKPOINT_ENV_CONFIG=1 \
      MODEL_PATH="$MODEL_PATH_ARG" OUTPUT_DIR="$OUT/$name" "${common[@]}"
    ;;
  *)
    echo "unsupported method: $METHOD" >&2
    exit 2
    ;;
esac

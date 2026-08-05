#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?usage: $0 direct_goal|old_obstacle_free|obstacle_20k|obstacle_40k}"
ROOT=/home/benjamin/thesis
MANIFEST="$ROOT/config/unitree_eval_manifest_blocked100_seed941.json"
OUT="$ROOT/logs/unitree_mjlab/goalonly_matched_blocked100_20260731"

case "$VARIANT" in
  direct_goal)
    CONTROLLER=direct_goal
    MODEL_PATH=""
    ;;
  old_obstacle_free)
    CONTROLLER=policy
    MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalpretrain_forward_10k_20260727/step_10000.pt"
    ;;
  obstacle_20k)
    CONTROLLER=policy
    MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalonly_obstacles_forward_5k_20260729/step_2500.pt"
    ;;
  obstacle_40k)
    CONTROLLER=policy
    MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalonly_obstacles_forward_5k_20260729/final.pt"
    ;;
  *)
    echo "Unsupported variant: $VARIANT" >&2
    exit 2
    ;;
esac

[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }
if [[ -n "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
  echo "Missing checkpoint: $MODEL_PATH" >&2
  exit 2
fi

printf '%s\n' \
  "Unitree matched goal-only obstacle evaluation" \
  "variant:    $VARIANT" \
  "controller: $CONTROLLER" \
  "checkpoint: ${MODEL_PATH:-<scan-blind direct-goal controller>}" \
  "manifest:   $MANIFEST" \
  "episodes:   100"

exec "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh" \
  CONTROLLER="$CONTROLLER" \
  MODEL_PATH="$MODEL_PATH" \
  CHECKPOINT_ENV_CONFIG=0 \
  RUN_NAME="$VARIANT" \
  OUTPUT_DIR="$OUT" \
  RECORD_VIDEO=0 \
  DEVICE=cuda:0 \
  SEED=941 \
  NUM_ENVS=8 \
  NUM_EPISODES=100 \
  EPISODE_LENGTH_S=90 \
  LAYOUT_MANIFEST="$MANIFEST" \
  SUCCESS_DIST=0.40 \
  HEIGHT_SCAN_RESOLUTION=0.25 \
  SCAN_HISTORY=1 \
  ACTION_HISTORY=0 \
  MASK_GOAL_HEADING=1 \
  POLICY_ACTION_SMOOTHING=0.0 \
  HIDDEN_DIM=256 \
  GOAL_DISTANCE_MIN=4.5 \
  GOAL_DISTANCE_MAX=8.0 \
  DEBUG_GOAL_THROUGH_OBSTACLE=1 \
  GOAL_THROUGH_OBSTACLE_PROB=1.0 \
  REQUIRE_BLOCKED_CORRIDOR=1 \
  BLOCKED_CORRIDOR_RADIUS=0.45 \
  BLOCKED_CORRIDOR_MIN_CELLS=4 \
  BLOCKED_CORRIDOR_RESAMPLE_ATTEMPTS=300 \
  BLOCKED_GOAL_MAX_DISTANCE=8.0 \
  BLOCKED_GOAL_DISTANCE_SAMPLING=uniform \
  DEBUG_GOAL_OBSTACLE_MIN_DIST=1.0 \
  DEBUG_GOAL_OBSTACLE_MAX_DIST=5.5 \
  DEBUG_NUM_OBSTACLES=6 \
  DEBUG_OBSTACLE_WIDTH_MIN=1.0 \
  DEBUG_OBSTACLE_WIDTH_MAX=1.4 \
  DEBUG_OBSTACLE_HEIGHT_MIN=1.0 \
  DEBUG_OBSTACLE_HEIGHT_MAX=1.0 \
  DEBUG_PLATFORM_WIDTH=2.0 \
  DEBUG_TERRAIN_ROWS=5 \
  DEBUG_TERRAIN_COLS=10 \
  STRICT_MIN_SIZE_OBSTACLES=1 \
  RESAMPLE_TERRAIN_TILES=1 \
  MIN_START_OBSTACLE_CLEARANCE=1.0 \
  START_CLEARANCE_RESAMPLE_ATTEMPTS=100 \
  MIN_GOAL_OBSTACLE_CLEARANCE=0.9 \
  GOAL_CLEARANCE_RESAMPLE_ATTEMPTS=100

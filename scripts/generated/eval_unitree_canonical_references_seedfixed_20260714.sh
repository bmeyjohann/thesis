#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis

for arg in "$@"; do
  if [[ "$arg" == *=* ]]; then
    export "$arg"
  else
    echo "Unsupported argument: $arg" >&2
    exit 2
  fi
done

EPISODES="${EPISODES:-30}"
EVAL_TAG="${EVAL_TAG:-seedfixed}"
OUT="$ROOT/logs/unitree_mjlab/canonical_references_20260714"
COMMON=(
  OUTPUT_DIR="$OUT"
  RECORD_VIDEO=0
  SEED=101
  NUM_ENVS="${NUM_ENVS:-$EPISODES}"
  NUM_EPISODES="$EPISODES"
  EPISODE_LENGTH_S=60
  GOAL_THROUGH_OBSTACLE_PROB=0.7
  GOAL_DISTANCE_MIN=2.8
  GOAL_DISTANCE_MAX=4.0
  MIN_GOAL_OBSTACLE_CLEARANCE=0.9
  DEBUG_OBSTACLE_WIDTH_MIN=1.0
  DEBUG_OBSTACLE_WIDTH_MAX=1.4
  DEBUG_OBSTACLE_HEIGHT_MIN=1.0
  DEBUG_OBSTACLE_HEIGHT_MAX=1.0
  DEBUG_NUM_OBSTACLES=6
  DEBUG_PLATFORM_WIDTH=2.0
  DEBUG_TERRAIN_ROWS=5
  DEBUG_TERRAIN_COLS=10
)

env CONTROLLER=direct_goal \
  RUN_NAME="direct_goal_${EVAL_TAG}_${EPISODES}ep" \
  "${COMMON[@]}" \
  "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh"

env CONTROLLER=geom_scan_teacher \
  RUN_NAME="geom_teacher_clear06_${EVAL_TAG}_${EPISODES}ep" \
  TEACHER_GEOM_CLEARANCE=0.6 \
  TEACHER_GOAL_STOP_DIST=0.4 \
  "${COMMON[@]}" \
  "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh"

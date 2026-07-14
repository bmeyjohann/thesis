#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
STEP="${STEP:-10000}"
MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_student_scan_cnn_scratch_dense_geom_clear06_10k_20260714/step_${STEP}.pt"
OUT="$ROOT/visualizations/unitree_scan_cnn_student_20260714/trajectories"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing checkpoint: $MODEL_PATH" >&2
  exit 2
fi

COMMON=(
  SEED=202
  NUM_ROLLOUTS=10
  STEPS=1200
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

env CONTROLLER=policy \
  MODEL_PATH="$MODEL_PATH" \
  OUTPUT_DIR="$OUT/student_step_${STEP}" \
  "${COMMON[@]}" \
  "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh"

env CONTROLLER=geom_scan_teacher \
  TEACHER_GEOM_CLEARANCE=0.6 \
  TEACHER_GOAL_STOP_DIST=0.4 \
  OUTPUT_DIR="$OUT/geom_teacher_clear06" \
  "${COMMON[@]}" \
  "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh"

env CONTROLLER=direct_goal \
  OUTPUT_DIR="$OUT/direct_goal" \
  "${COMMON[@]}" \
  "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
OUT="$ROOT/logs/unitree_mjlab/student_eval_100ep_20260713"
COMMON=(
  OUTPUT_DIR="$OUT"
  RECORD_VIDEO=0
  SEED=101
  NUM_ENVS=32
  NUM_EPISODES=100
  EPISODE_LENGTH_S=60
  GOAL_THROUGH_OBSTACLE_PROB=0.7
  GOAL_DISTANCE_MIN=2.8
  GOAL_DISTANCE_MAX=4.0
  MIN_GOAL_OBSTACLE_CLEARANCE=0.9
  DEBUG_TERRAIN_ROWS=5
  DEBUG_TERRAIN_COLS=10
)

env CONTROLLER=direct_goal RUN_NAME=direct_goal "${COMMON[@]}" \
  "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh"

env CONTROLLER=geom_scan_teacher RUN_NAME=geom_teacher_clear06 \
  TEACHER_GEOM_CLEARANCE=0.6 TEACHER_GOAL_STOP_DIST=0.4 "${COMMON[@]}" \
  "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh"

env CONTROLLER=policy \
  MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalonly_flat_pretrain_seed17_20260713/final.pt" \
  RUN_NAME=goalonly_pretrain \
  "${COMMON[@]}" \
  "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh"

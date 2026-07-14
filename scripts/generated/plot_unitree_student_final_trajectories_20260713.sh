#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
COMMON=(
  SEED=202
  NUM_ROLLOUTS=10
  STEPS=1200
  EPISODE_LENGTH_S=60
  GOAL_THROUGH_OBSTACLE_PROB=0.7
  GOAL_DISTANCE_MIN=2.8
  GOAL_DISTANCE_MAX=4.0
  MIN_GOAL_OBSTACLE_CLEARANCE=0.9
  DEBUG_TERRAIN_ROWS=5
  DEBUG_TERRAIN_COLS=10
)

env CONTROLLER=policy \
  MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_student_scratch_dense_clear06_resume2500_warmup_to10k_20260713/step_7500.pt" \
  OUTPUT_DIR="$ROOT/visualizations/unitree_student_trajectories_20260713/scratch_effective_10k" \
  "${COMMON[@]}" \
  "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh"

env CONTROLLER=policy \
  MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_student_goalpretrained_dense_clear06_resume2500_warmup_to10k_20260713/step_7500.pt" \
  OUTPUT_DIR="$ROOT/visualizations/unitree_student_trajectories_20260713/goalpretrained_effective_10k" \
  "${COMMON[@]}" \
  "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh"

env CONTROLLER=geom_scan_teacher \
  TEACHER_GEOM_CLEARANCE=0.6 \
  TEACHER_GOAL_STOP_DIST=0.4 \
  OUTPUT_DIR="$ROOT/visualizations/unitree_student_trajectories_20260713/geom_teacher_clear06" \
  "${COMMON[@]}" \
  "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh"

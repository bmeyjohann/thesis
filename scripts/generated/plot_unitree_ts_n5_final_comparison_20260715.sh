#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
LAUNCHER="$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh"
OUT="$ROOT/visualizations/unitree_teacher_student_n5_final_20260715"

common=(
  DEVICE=cuda:0 SEED=1201 NUM_ROLLOUTS=8 STEPS=1200 EPISODE_LENGTH_S=60
  SUCCESS_DIST=0.40 HEIGHT_SCAN_RESOLUTION=0.25 SCAN_HISTORY=1 ACTION_HISTORY=0
  GOAL_DISTANCE_MIN=2.8 GOAL_DISTANCE_MAX=4.0
  MIN_GOAL_OBSTACLE_CLEARANCE=0.9 MIN_START_OBSTACLE_CLEARANCE=1.0
  DEBUG_NUM_OBSTACLES=6 DEBUG_OBSTACLE_WIDTH_MIN=1.0 DEBUG_OBSTACLE_WIDTH_MAX=1.4
  DEBUG_OBSTACLE_HEIGHT_MIN=1.0 DEBUG_OBSTACLE_HEIGHT_MAX=1.0
  DEBUG_TERRAIN_ROWS=5 DEBUG_TERRAIN_COLS=10 DEBUG_PLATFORM_WIDTH=2.0
  STRICT_MIN_SIZE_OBSTACLES=1 RESAMPLE_TERRAIN_TILES=1 GOAL_THROUGH_OBSTACLE_PROB=0
)

"$LAUNCHER" \
  CONTROLLER=direct_goal CHECKPOINT_ENV_CONFIG=0 \
  OUTPUT_DIR="$OUT/direct_goal" "${common[@]}"

"$LAUNCHER" \
  CONTROLLER=geom_scan_teacher CHECKPOINT_ENV_CONFIG=0 \
  TEACHER_GEOM_CLEARANCE=0.60 TEACHER_GOAL_STOP_DIST=0.30 \
  OUTPUT_DIR="$OUT/geom_teacher" "${common[@]}"

"$LAUNCHER" \
  CONTROLLER=policy CHECKPOINT_ENV_CONFIG=0 MASK_HEIGHT_SCAN=1 MASK_GOAL_HEADING=1 \
  MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_n5_r040_linear_4k_20260715/step_4000.pt" \
  OUTPUT_DIR="$OUT/fastsac_goal_only" "${common[@]}"

"$LAUNCHER" \
  CONTROLLER=policy CHECKPOINT_ENV_CONFIG=1 \
  MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_ts_mlp_n5_gate090_bc02_pref1_3k_20260715/step_1000.pt" \
  OUTPUT_DIR="$OUT/student_bc_pref_step1000" "${common[@]}"

"$LAUNCHER" \
  CONTROLLER=policy CHECKPOINT_ENV_CONFIG=1 \
  MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_ts_mlp_n5_gate090_bc_only_2k_20260715/step_2000.pt" \
  OUTPUT_DIR="$OUT/student_bc_only_step2000" "${common[@]}"

"$LAUNCHER" \
  CONTROLLER=policy CHECKPOINT_ENV_CONFIG=1 \
  MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_ts_mlp_n5_gate090_pref_only_2k_20260715/step_2000.pt" \
  OUTPUT_DIR="$OUT/student_pref_only_step2000" "${common[@]}"

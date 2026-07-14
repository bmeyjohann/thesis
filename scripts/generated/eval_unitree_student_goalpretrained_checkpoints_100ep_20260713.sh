#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
OUT="$ROOT/logs/unitree_mjlab/student_eval_100ep_20260713"
BASE="$ROOT/models/unitree_mjlab_nav_thesis/unitree_student_goalpretrained_dense_clear06_gatefix_10k_20260713"
CONT="$ROOT/models/unitree_mjlab_nav_thesis/unitree_student_goalpretrained_dense_clear06_resume2500_warmup_to10k_20260713"

eval_checkpoint() {
  local effective_step="$1"
  local model_path="$2"
  CONTROLLER=policy \
  MODEL_PATH="$model_path" \
  RUN_NAME="goalpretrained_effective_step_${effective_step}" \
  OUTPUT_DIR="$OUT" \
  RECORD_VIDEO=0 \
  SEED=101 \
  NUM_ENVS=32 \
  NUM_EPISODES=100 \
  EPISODE_LENGTH_S=60 \
  GOAL_THROUGH_OBSTACLE_PROB=0.7 \
  GOAL_DISTANCE_MIN=2.8 \
  GOAL_DISTANCE_MAX=4.0 \
  MIN_GOAL_OBSTACLE_CLEARANCE=0.9 \
  DEBUG_TERRAIN_ROWS=5 \
  DEBUG_TERRAIN_COLS=10 \
  "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh"
}

eval_checkpoint 2500 "$BASE/step_2500.pt"
eval_checkpoint 5000 "$CONT/step_2500.pt"
eval_checkpoint 7500 "$CONT/step_5000.pt"
eval_checkpoint 10000 "$CONT/step_7500.pt"

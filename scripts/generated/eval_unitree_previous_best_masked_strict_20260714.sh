#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
MODEL="$ROOT/models/unitree_mjlab_nav_thesis/unitree_scan13_hist5_actionhist4_history_only_2500_20260714/step_2500.pt"
exec "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh" \
  CONTROLLER=policy \
  MODEL_PATH="$MODEL" \
  MASK_GOAL_HEADING=1 \
  STRICT_MIN_SIZE_OBSTACLES=1 \
  POLICY_TEACHER_GATE=0 \
  NUM_ROLLOUTS=4 \
  STEPS=1200 \
  SEED=191 \
  OUTPUT_DIR="$ROOT/visualizations/unitree_nav_previous_best_masked_strict_eval"

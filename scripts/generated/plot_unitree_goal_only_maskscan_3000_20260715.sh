#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis

exec "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh" \
  CONTROLLER=policy \
  MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_goal_only_dense_maskscan_scan13_hist5_3000_20260715/step_3000.pt" \
  DEVICE=cuda:0 \
  SEED=973 \
  NUM_ROLLOUTS=5 \
  STEPS=1200 \
  EPISODE_LENGTH_S=60 \
  CHECKPOINT_ENV_CONFIG=1 \
  OUTPUT_DIR="$ROOT/visualizations/unitree_goal_only_dense_maskscan_diagnostic_20260715/step_3000"

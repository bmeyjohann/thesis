#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
RUN="$ROOT/models/unitree_mjlab_nav_thesis/unitree_goal_only_dense_scan13_hist5_5000_20260715"
OUT="$ROOT/visualizations/unitree_goal_only_dense_diagnostic_20260715"

for STEP in 2000 5000; do
  "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh" \
    CONTROLLER=policy \
    MODEL_PATH="$RUN/step_${STEP}.pt" \
    DEVICE=cuda:0 \
    SEED=973 \
    NUM_ROLLOUTS=5 \
    STEPS=1200 \
    EPISODE_LENGTH_S=60 \
    CHECKPOINT_ENV_CONFIG=1 \
    OUTPUT_DIR="$OUT/step_${STEP}"
done

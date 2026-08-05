#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
OUT="$ROOT/visualizations/unitree_goal_signal_diagnostic_20260715"

plot() {
  local name=$1
  local checkpoint=$2
  "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh" \
    CONTROLLER=policy \
    MODEL_PATH="$checkpoint" \
    DEVICE=cuda:0 \
    SEED=991 \
    NUM_ROLLOUTS=5 \
    STEPS=1200 \
    EPISODE_LENGTH_S=60 \
    CHECKPOINT_ENV_CONFIG=1 \
    OUTPUT_DIR="$OUT/$name"
}

plot \
  sac_utd1_scale10_step1000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalonly_terminalfix_utd1_scale10_5k_20260715/step_1000.pt"
plot \
  sac_utd1_failure2_step1000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalonly_terminalfix_utd1_scale1_failure2_5k_20260715/step_1000.pt"
plot \
  bc_only_step2000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalonly_online_bc_geom_5k_20260715/step_2000.pt"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
OUT="$ROOT/visualizations/unitree_goal_rl_minimal_20260715"

plot() {
  "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh" \
    CONTROLLER=policy MODEL_PATH="$2" DEVICE=cuda:0 SEED=991 \
    NUM_ROLLOUTS=5 STEPS=1200 EPISODE_LENGTH_S=60 CHECKPOINT_ENV_CONFIG=1 \
    OUTPUT_DIR="$OUT/$1"
}

plot kinematics_n1_step3000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_minimal_kinematics_n1_5k_20260715/step_3000.pt"
plot kinematics_n10_step2000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_minimal_kinematics_n10_5k_20260715/step_2000.pt"

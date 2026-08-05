#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
OUT="$ROOT/visualizations/unitree_goal_rl_n5_radius_reward_20260715"

plot() {
  "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh" \
    CONTROLLER=policy MODEL_PATH="$2" DEVICE=cuda:0 SEED=991 \
    NUM_ROLLOUTS=5 STEPS=1200 EPISODE_LENGTH_S=60 CHECKPOINT_ENV_CONFIG=1 \
    OUTPUT_DIR="$OUT/$1"
}

plot n5_r025_linear_step3000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_n5_r025_linear_4k_20260715/step_3000.pt"
plot n5_r040_linear_step4000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_n5_r040_linear_4k_20260715/step_4000.pt"
plot n5_r040_exp_step4000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_n5_r040_exp4_t1_4k_20260715/step_4000.pt"

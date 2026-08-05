#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
exec "$ROOT/scripts/run_unitree_mjlab_nav_plot_local.sh" \
  CONTROLLER=policy \
  CHECKPOINT_ENV_CONFIG=1 \
  MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_ts_mlp_n5_gate090_bc02_pref1_3k_20260715/step_3000.pt" \
  OUTPUT_DIR="$ROOT/visualizations/unitree_teacher_student_n5_final_20260715/student_bc_pref_step3000" \
  DEVICE=cuda:0 \
  SEED=1201 \
  NUM_ROLLOUTS=8 \
  STEPS=1200 \
  EPISODE_LENGTH_S=60

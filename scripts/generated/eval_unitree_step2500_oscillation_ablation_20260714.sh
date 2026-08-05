#!/usr/bin/env bash
set -euo pipefail

SMOOTHING="${1:-0.0}"
LABEL="${2:-raw}"
export CONTROLLER=policy
export MODEL_PATH=/home/benjamin/thesis/models/unitree_mjlab_nav_thesis/unitree_student_scan13_history5_dense_geom_20k_20260714/step_2500.pt
export RUN_NAME="unitree_step2500_${LABEL}_eval20_20260714"
export OUTPUT_DIR=/home/benjamin/thesis/logs/unitree_mjlab/oscillation_eval
export RECORD_VIDEO=0
export NUM_ENVS=8
export NUM_EPISODES=20
export SEED=83
export POLICY_ACTION_SMOOTHING="$SMOOTHING"

exec /home/benjamin/thesis/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh

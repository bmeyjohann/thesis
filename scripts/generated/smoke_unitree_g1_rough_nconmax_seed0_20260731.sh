#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT="/home/benjamin/thesis"
export TASK_ID="Unitree-G1-Rough"
export MODEL_PATH="/home/benjamin/thesis/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/supervisor_rough_model9999/model_9999.pt"
export MODE="smoke"
export DEVICE="cuda:0"
export NUM_ENVS="1"
export SEED="0"
export STEPS="2"
export NCONMAX="256"
export CHECKPOINT_OBSERVATION_MODE="task"

exec /home/benjamin/thesis/scripts/run_unitree_mjlab_velocity_inspect_local.sh

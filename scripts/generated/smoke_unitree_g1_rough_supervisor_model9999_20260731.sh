#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT="/home/benjamin/thesis"
export TASK_ID="Unitree-G1-Rough"
export MODEL_PATH="/home/benjamin/thesis/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/supervisor_rough_model9999/model_9999.pt"
export MODE="video"
export DEVICE="cuda:0"
export NUM_ENVS="1"
export SEED="31"
export STEPS="310"
export VIDEO_LENGTH="300"
export VIDEO_DIR="/home/benjamin/thesis/videos/unitree_velocity/Unitree-G1-Rough_seed31_supervisor_model9999"
export CHECKPOINT_OBSERVATION_MODE="task"
export MUJOCO_GL="egl"

exec /home/benjamin/thesis/scripts/run_unitree_mjlab_velocity_inspect_local.sh

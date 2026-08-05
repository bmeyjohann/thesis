#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT="/home/benjamin/thesis"
export TASK_ID="Unitree-G1-Rough"
export MODEL_PATH="/home/benjamin/thesis/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/pretrained_model1499/model_1499.pt"
export MODE="video"
export DEVICE="cuda:0"
export NUM_ENVS="1"
export SEED="17"
export STEPS="260"
export VIDEO_LENGTH="240"
export VIDEO_DIR="/home/benjamin/thesis/videos/unitree_velocity/Unitree-G1-Rough_seed17_model1499"
export CHECKPOINT_OBSERVATION_MODE="auto"
export MUJOCO_GL="egl"

exec /home/benjamin/thesis/scripts/run_unitree_mjlab_velocity_inspect_local.sh

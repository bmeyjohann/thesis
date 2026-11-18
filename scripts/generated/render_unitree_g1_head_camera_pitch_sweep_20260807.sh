#!/usr/bin/env bash
set -euo pipefail

for pitch in 10 25 40; do
  export REPO_ROOT="/home/benjamin/thesis"
  export TASK_ID="Unitree-G1-Rough"
  export MODEL_PATH="/home/benjamin/thesis/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/supervisor_rough_model9999/model_9999.pt"
  export MODE="smoke"
  export DEVICE="cuda:0"
  export NUM_ENVS="1"
  export SEED="7"
  export STEPS="3"
  export NCONMAX="256"
  export HEAD_CAMERA="1"
  export CAMERA_PITCH_DOWN_DEG="$pitch"
  export CAMERA_FOVY="70"
  export EGOCENTRIC_VIDEO="/home/benjamin/thesis/videos/unitree_velocity/g1_rough_head_rgbd_pitch${pitch}_seed7_probe.mp4"
  export CHECKPOINT_OBSERVATION_MODE="task"
  export MUJOCO_GL="egl"

  /home/benjamin/thesis/scripts/run_unitree_mjlab_velocity_inspect_local.sh
done

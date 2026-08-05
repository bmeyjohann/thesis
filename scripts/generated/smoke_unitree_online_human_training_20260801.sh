#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
export MPLCONFIGDIR=/tmp/mplconfig
export WARP_CACHE_PATH=/tmp/warp-cache
export XDG_CACHE_HOME=/tmp/unitree-cache
export MUJOCO_GL=egl
export SDL_VIDEODRIVER=dummy
mkdir -p "$MPLCONFIGDIR" "$WARP_CACHE_PATH" "$XDG_CACHE_HOME"

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  "$ROOT/eval_interactive_unitree_nav.py" \
  --controller policy \
  --model-path "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalonly_obstacles_forward_5k_20260729/step_2500.pt" \
  --device cuda:0 \
  --human-input-device none \
  --human-dataset-dir "/tmp/unitree-human-online-smoke-dataset-$$" \
  --online-train \
  --online-run-name "unitree_human_online_smoke_$$" \
  --online-output-dir /tmp \
  --online-learning-starts 2 \
  --online-batch-size 2 \
  --online-updates-per-step 1 \
  --online-n-step 1 \
  --online-log-interval 1 \
  --online-checkpoint-interval 10 \
  --online-total-steps 4 \
  --online-wandb-mode disabled \
  --sim-fps 0 \
  --fps 1000 \
  --display-every 1000 \
  --no-show-scan-samples \
  --no-start-paused \
  --auto-reset \
  --checkpoint-env-config \
  --force-obstacles \
  --force-obstacle-profile strict_blocked_v1

#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
MPLCONFIGDIR=/tmp/mplconfig \
MUJOCO_GL=egl \
WARP_CACHE_PATH=/tmp/warp-cache \
XDG_CACHE_HOME=/tmp/unitree-cache \
/home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/eval_interactive_unitree_nav.py \
  --controller scan_teacher \
  --device cuda:0 \
  --sim-fps 0 \
  --fps 30 \
  --show-rgb \
  --rgb-every 2 \
  --show-reconstructed-scan \
  --require-blocked-corridor \
  --blocked-corridor-radius 0.8 \
  --blocked-corridor-min-cells 4 \
  --blocked-corridor-resample-attempts 1000 \
  --debug-goal-through-obstacle \
  --debug-num-obstacles 3 \
  --debug-obstacle-width-min 1.15 \
  --debug-obstacle-width-max 1.55 \
  --debug-obstacle-height-min 1.0 \
  --debug-obstacle-height-max 1.0 \
  --debug-platform-width 2.6 \
  --debug-goal-distance 4.2 \
  --debug-goal-obstacle-min-dist 0.8 \
  --debug-goal-obstacle-max-dist 1.8 \
  --min-start-obstacle-clearance 1.0 \
  --min-goal-obstacle-clearance 0.7 \
  --goal-clearance-resample-attempts 1000 \
  "$@"

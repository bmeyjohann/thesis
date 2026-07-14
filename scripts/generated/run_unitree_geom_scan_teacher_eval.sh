#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

RUN_NAME="${RUN_NAME:-unitree_geom_scan_teacher_astar_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
NUM_EPISODES="${NUM_EPISODES:-50}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-16.0}"

echo "[geom-scan-teacher-eval] run=${RUN_NAME} episodes=${NUM_EPISODES} episode_length_s=${EPISODE_LENGTH_S} device=${DEVICE}"

MPLCONFIGDIR=/tmp/mplconfig \
MUJOCO_GL=egl \
WARP_CACHE_PATH=/tmp/warp-cache \
XDG_CACHE_HOME=/tmp/unitree-cache \
"${PYTHON_BIN}" -u /home/benjamin/thesis/eval_unitree_nav_baselines.py \
  --controller geom_scan_teacher \
  --device "${DEVICE}" \
  --num-envs 1 \
  --num-episodes "${NUM_EPISODES}" \
  --episode-length-s "${EPISODE_LENGTH_S}" \
  --run-name "${RUN_NAME}" \
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
  --teacher-max-vx 0.65 \
  --teacher-max-vy 0.45 \
  --teacher-align-angle 0.55 \
  --teacher-min-forward-scale 0.12 \
  --teacher-goal-stop-dist 0.55 \
  --teacher-geom-planner astar \
  --teacher-geom-clearance 0.7 \
  --teacher-geom-grid-resolution 0.15 \
  --teacher-geom-waypoint-index 3 \
  --teacher-geom-emergency-radius 0.9 \
  "$@"

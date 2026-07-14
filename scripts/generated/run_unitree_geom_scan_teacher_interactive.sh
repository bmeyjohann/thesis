#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
LOW_LEVEL_POLICY_PATH="${LOW_LEVEL_POLICY_PATH:-/home/benjamin/thesis/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-07-12_10-35-19_omni_finetune_model1499_20260712}"

MPLCONFIGDIR=/tmp/mplconfig \
MUJOCO_GL=egl \
WARP_CACHE_PATH=/tmp/warp-cache \
XDG_CACHE_HOME=/tmp/unitree-cache \
"${PYTHON_BIN}" /home/benjamin/thesis/eval_interactive_unitree_nav.py \
  --controller geom_scan_teacher \
  --device "${DEVICE}" \
  --low-level-policy-path "${LOW_LEVEL_POLICY_PATH}" \
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
  --debug-terrain-rows "${DEBUG_TERRAIN_ROWS:-5}" \
  --debug-terrain-cols "${DEBUG_TERRAIN_COLS:-10}" \
  --debug-goal-distance 4.2 \
  --debug-goal-obstacle-min-dist 0.8 \
  --debug-goal-obstacle-max-dist 1.8 \
  --min-start-obstacle-clearance 1.0 \
  --min-goal-obstacle-clearance 0.7 \
  --teacher-max-vx 0.95 \
  --teacher-max-vy 0.45 \
  --teacher-align-angle 0.55 \
  --teacher-min-forward-scale 0.12 \
  --teacher-goal-stop-dist "${TEACHER_GOAL_STOP_DIST:-0.40}" \
  --teacher-geom-planner astar \
  --teacher-geom-clearance "${TEACHER_GEOM_CLEARANCE:-0.4}" \
  --teacher-geom-grid-resolution 0.15 \
  --teacher-geom-waypoint-index 3 \
  --teacher-geom-emergency-radius 0.9 \
  --teacher-geom-side-penalty 8.0 \
  --teacher-geom-side-frame body \
  --teacher-geom-disengage-clear-steps 20 \
  --episode-length-s 60.0 \
  "$@"

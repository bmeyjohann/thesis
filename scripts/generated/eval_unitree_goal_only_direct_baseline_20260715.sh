#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python

export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export WANDB_MODE=disabled

exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
  --controller direct_goal \
  --task Unitree-G1-Nav-Obstacles-Safe-Collision \
  --device cuda:0 \
  --seed 973 \
  --num-envs 8 \
  --num-episodes 40 \
  --episode-length-s 60 \
  --resample-terrain-tiles \
  --success-dist 0.25 \
  --goal-distance-min 2.8 \
  --goal-distance-max 4.0 \
  --disable-obstacles \
  --debug-terrain-rows 5 \
  --debug-terrain-cols 10 \
  --height-scan-resolution 0.25 \
  --scan-history 5 \
  --action-history 4 \
  --teacher-max-vx 0.95 \
  --teacher-max-vy 0.45 \
  --teacher-yaw-gain 1.2 \
  --teacher-align-angle 0.55 \
  --output-dir "$ROOT/logs/unitree_mjlab/goal_only_diagnostic" \
  --run-name direct_goal_no_obstacles_seed973_20260715

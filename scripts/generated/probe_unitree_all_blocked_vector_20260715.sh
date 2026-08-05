#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
OUT="$ROOT/logs/unitree_mjlab/all_blocked_probe_20260715"
export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl WANDB_MODE=disabled

exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
  --controller direct_goal \
  --run-name direct_goal_vector8 \
  --device cuda:0 \
  --seed 1501 \
  --num-envs 8 \
  --num-episodes 16 \
  --episode-length-s 60 \
  --success-dist 0.40 \
  --height-scan-resolution 0.25 \
  --goal-distance-min 2.8 \
  --goal-distance-max 5.0 \
  --min-start-obstacle-clearance 1.0 \
  --min-goal-obstacle-clearance 0.9 \
  --debug-goal-through-obstacle \
  --goal-through-obstacle-prob 1.0 \
  --require-blocked-corridor \
  --blocked-corridor-radius 0.45 \
  --blocked-corridor-ignore-end-radius 0.75 \
  --blocked-corridor-min-cells 4 \
  --blocked-corridor-resample-attempts 100 \
  --debug-goal-obstacle-min-dist 0.8 \
  --debug-goal-obstacle-max-dist 3.0 \
  --debug-num-obstacles 6 \
  --debug-obstacle-width-min 1.0 \
  --debug-obstacle-width-max 1.4 \
  --debug-obstacle-height-min 1.0 \
  --debug-obstacle-height-max 1.0 \
  --debug-terrain-rows 5 \
  --debug-terrain-cols 10 \
  --debug-platform-width 2.0 \
  --strict-min-size-obstacles \
  --resample-terrain-tiles \
  --output-dir "$OUT"

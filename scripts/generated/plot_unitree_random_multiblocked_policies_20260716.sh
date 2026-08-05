#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:?usage: $0 label checkpoint [rollouts]}"
MODEL="${2:?usage: $0 label checkpoint [rollouts]}"
ROLLOUTS="${3:-4}"
ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
OUT="$ROOT/visualizations/unitree_random_multiblocked_20260716/$LABEL"

export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl

exec "$PYTHON" "$ROOT/plot_unitree_nav_rollout.py" \
  --controller policy --model-path "$MODEL" --no-checkpoint-env-config \
  --device cuda:0 --seed 1607 --num-envs 1 --num-rollouts "$ROLLOUTS" \
  --steps 1800 --episode-length-s 90 --success-dist 0.40 \
  --height-scan-resolution 0.25 --scan-history 1 --action-history 0 \
  --mask-goal-heading --use-layer-norm \
  --goal-distance-min 4.5 --goal-distance-max 8.0 \
  --min-start-obstacle-clearance 1.0 --min-goal-obstacle-clearance 0.9 \
  --debug-goal-through-obstacle --goal-through-obstacle-prob 1.0 \
  --require-blocked-corridor --blocked-corridor-radius 0.45 \
  --blocked-corridor-ignore-end-radius 0.75 --blocked-corridor-min-cells 4 \
  --blocked-corridor-resample-attempts 200 --blocked-goal-max-distance 8.0 \
  --blocked-goal-distance-sampling uniform \
  --debug-goal-obstacle-min-dist 1.0 --debug-goal-obstacle-max-dist 5.5 \
  --debug-num-obstacles 6 --debug-obstacle-width-min 1.0 --debug-obstacle-width-max 1.4 \
  --debug-obstacle-height-min 1.0 --debug-obstacle-height-max 1.0 \
  --debug-terrain-rows 5 --debug-terrain-cols 10 --debug-platform-width 2.0 \
  --strict-min-size-obstacles --resample-terrain-tiles --output-dir "$OUT"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl WANDB_MODE=disabled

exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
  --controller policy \
  --model-path "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_n5_r040_linear_4k_20260715/step_4000.pt" \
  --no-checkpoint-env-config --scan-history 1 --action-history 0 \
  --mask-height-scan --mask-goal-heading --use-layer-norm \
  --device cuda:0 --seed 1201 --num-envs 8 --num-episodes 40 \
  --episode-length-s 60 --success-dist 0.40 --height-scan-resolution 0.25 \
  --goal-distance-min 2.8 --goal-distance-max 4.0 \
  --debug-num-obstacles 6 --debug-obstacle-width-min 1.0 --debug-obstacle-width-max 1.4 \
  --debug-obstacle-height-min 1.0 --debug-obstacle-height-max 1.0 \
  --debug-terrain-rows 5 --debug-terrain-cols 10 --debug-platform-width 2.0 \
  --strict-min-size-obstacles --resample-terrain-tiles \
  --output-dir "$ROOT/logs/unitree_mjlab/teacher_student_obstacle_baselines_20260715" \
  --run-name fastsac_goal_only_mlp

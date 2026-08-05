#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
RUN="$ROOT/models/unitree_mjlab_nav_thesis/unitree_goal_only_dense_scan13_hist5_5000_20260715"
OUT="$ROOT/logs/unitree_mjlab/goal_only_diagnostic/reeval_fixed_20260715"

export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export WANDB_MODE=disabled

for STEP in 1000 2000 3000 4000 5000; do
  "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
    --controller policy \
    --model-path "$RUN/step_${STEP}.pt" \
    --checkpoint-env-config \
    --device cuda:0 \
    --seed 973 \
    --num-envs 4 \
    --num-episodes 8 \
    --output-dir "$OUT" \
    --run-name "step_${STEP}"
done

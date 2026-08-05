#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
OUT="$ROOT/logs/unitree_mjlab/goal_only_diagnostic/robust_40ep_20260715"

export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export WANDB_MODE=disabled

evaluate() {
  local run_name=$1
  local checkpoint=$2
  "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
    --controller policy \
    --model-path "$checkpoint" \
    --checkpoint-env-config \
    --device cuda:0 \
    --seed 991 \
    --num-envs 8 \
    --num-episodes 40 \
    --output-dir "$OUT" \
    --run-name "$run_name"
}

evaluate \
  unmasked_step_2000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goal_only_dense_scan13_hist5_5000_20260715/step_2000.pt"

evaluate \
  masked_step_3000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goal_only_dense_maskscan_scan13_hist5_3000_20260715/step_3000.pt"

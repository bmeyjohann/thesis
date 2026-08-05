#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
OUT="$ROOT/logs/unitree_mjlab/goal_rl_minimal_40ep_20260715"

export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl WANDB_MODE=disabled

evaluate() {
  "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
    --controller policy --model-path "$2" --checkpoint-env-config \
    --device cuda:0 --seed 991 --num-envs 8 --num-episodes 40 \
    --output-dir "$OUT" --run-name "$1"
}

evaluate kinematics_n1_step3000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_minimal_kinematics_n1_5k_20260715/step_3000.pt"
evaluate kinematics_n10_step2000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalrl_minimal_kinematics_n10_5k_20260715/step_2000.pt"

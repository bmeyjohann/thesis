#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
OUT="$ROOT/logs/unitree_mjlab/goal_signal_diagnostic_40ep_20260715"

export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export WANDB_MODE=disabled

evaluate() {
  local name=$1
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
    --run-name "$name"
}

evaluate \
  sac_utd1_scale10_step1000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalonly_terminalfix_utd1_scale10_5k_20260715/step_1000.pt"
evaluate \
  sac_utd1_failure2_step1000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalonly_terminalfix_utd1_scale1_failure2_5k_20260715/step_1000.pt"
evaluate \
  bc_only_step2000 \
  "$ROOT/models/unitree_mjlab_nav_thesis/unitree_goalonly_online_bc_geom_5k_20260715/step_2000.pt"

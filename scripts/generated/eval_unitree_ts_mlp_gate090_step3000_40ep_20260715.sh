#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl WANDB_MODE=disabled

exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
  --controller policy \
  --model-path "$ROOT/models/unitree_mjlab_nav_thesis/unitree_ts_mlp_n5_gate090_bc02_pref1_3k_20260715/step_3000.pt" \
  --checkpoint-env-config \
  --device cuda:0 --seed 1201 --num-envs 8 --num-episodes 40 \
  --output-dir "$ROOT/logs/unitree_mjlab/teacher_student_n5_final_20260715" \
  --run-name gate090_bc02_pref1_step3000

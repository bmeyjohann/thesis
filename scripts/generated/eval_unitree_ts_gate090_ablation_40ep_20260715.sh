#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: $0 bc_only|pref_only [step]}"
STEP="${2:-1000}"
case "$METHOD" in
  bc_only|pref_only) ;;
  *) echo "unsupported method: $METHOD" >&2; exit 2 ;;
esac

ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
MODEL="$ROOT/models/unitree_mjlab_nav_thesis/unitree_ts_mlp_n5_gate090_${METHOD}_2k_20260715/step_${STEP}.pt"
export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl WANDB_MODE=disabled

exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
  --controller policy --model-path "$MODEL" --checkpoint-env-config \
  --device cuda:0 --seed 1201 --num-envs 8 --num-episodes 40 \
  --output-dir "$ROOT/logs/unitree_mjlab/teacher_student_n5_final_20260715" \
  --run-name "gate090_${METHOD}_step${STEP}"

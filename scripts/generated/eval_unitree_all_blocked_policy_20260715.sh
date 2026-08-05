#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: $0 checkpoint [episodes] [run_name]}"
EPISODES="${2:-16}"
RUN_NAME="${3:-$(basename "$(dirname "$MODEL_PATH")")_$(basename "$MODEL_PATH" .pt)_${EPISODES}ep}"
ROOT=/home/benjamin/thesis
PYTHON=/home/benjamin/miniconda3/envs/fasttd3/bin/python
OUT="$ROOT/logs/unitree_mjlab/all_blocked_policy_eval_20260715"

export PYTHONPATH="$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab:$ROOT/external/unitree_rl_mjlab/source/unitree_rl_mjlab_tasks:${PYTHONPATH:-}"
export MUJOCO_GL=egl WANDB_MODE=disabled

exec "$PYTHON" "$ROOT/eval_unitree_nav_baselines.py" \
  --controller policy \
  --model-path "$MODEL_PATH" \
  --run-name "$RUN_NAME" \
  --device cuda:0 \
  --seed 1501 \
  --num-envs 8 \
  --num-episodes "$EPISODES" \
  --output-dir "$OUT"

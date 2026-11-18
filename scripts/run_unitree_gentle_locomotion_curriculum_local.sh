#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/external/unitree_rl_mjlab${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" "$ROOT_DIR/train_unitree_gentle_locomotion.py" \
  --seed "${SEED:-0}" \
  --num-envs "${NUM_ENVS:-256}" \
  --iterations "${ITERATIONS:-6000}" \
  --save-interval "${SAVE_INTERVAL:-250}" \
  --logger "${LOGGER:-tensorboard}" \
  --run-name "${RUN_NAME:-g1_gentle_turn_curriculum_seed${SEED:-0}}" \
  "$@"

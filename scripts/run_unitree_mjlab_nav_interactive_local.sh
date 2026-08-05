#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${REPO_ROOT:-/home/benjamin/thesis}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
: "${MODEL_PATH:?Set MODEL_PATH to a Unitree navigation checkpoint}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/warp-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/unitree-cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$MPLCONFIGDIR" "$WARP_CACHE_PATH" "$XDG_CACHE_HOME"

exec "$PYTHON_BIN" "$ROOT_DIR/eval_interactive_unitree_nav.py" \
  --controller policy \
  --model-path "$MODEL_PATH" \
  --device "${DEVICE:-cuda:0}" \
  --fps "${FPS:-30}" \
  --sim-fps "${SIM_FPS:-0}" \
  --show-rgb \
  --no-start-paused \
  --auto-reset \
  --checkpoint-env-config

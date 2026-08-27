#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$ROOT_DIR/external/unitree_rl_mjlab"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"

TASK="${TASK:-Unitree-G1-Nav-Obstacles}"
DEVICE="${DEVICE:-cpu}"
NUM_ENVS="${NUM_ENVS:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-120}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT_DIR/logs/unitree_mjlab/nav_smoke_video}"

UNITREE_CACHE_ROOT="${UNITREE_CACHE_ROOT:-$HOME/.cache/unitree-nav}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$UNITREE_CACHE_ROOT/matplotlib}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$UNITREE_CACHE_ROOT/warp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$UNITREE_CACHE_ROOT/xdg}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "$MPLCONFIGDIR" "$WARP_CACHE_PATH" "$XDG_CACHE_HOME" "$VIDEO_DIR"

cd "$REPO_DIR"
exec "$PYTHON_BIN" scripts/smoke_test_nav_forward.py \
  --task "$TASK" \
  --device "$DEVICE" \
  --num-envs "$NUM_ENVS" \
  --record \
  --video-length "$VIDEO_LENGTH" \
  --video-dir "$VIDEO_DIR"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/benjamin/thesis}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/visualizations/unitree_target_terrain_variants}"
ARENA_SIZE="${ARENA_SIZE:-24}"
VIDEO_SECONDS="${VIDEO_SECONDS:-3}"

cd "$REPO_ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"

"$PYTHON_BIN" tools/render_unitree_target_terrains.py \
  --seeds 3 11 \
  --presets balanced traversal navigation \
  --arena-size "$ARENA_SIZE" \
  --width 960 \
  --height 720 \
  --video-seconds "$VIDEO_SECONDS" \
  --output-dir "$OUTPUT_DIR"

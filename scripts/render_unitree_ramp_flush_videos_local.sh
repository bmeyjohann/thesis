#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-visualizations/unitree_ramp_calibration_flush/videos_14s}"
DEVICE="${DEVICE:-cuda:0}"
mkdir -p "$OUTPUT_ROOT"

# WSL Mesa otherwise selects CPU llvmpipe for EGL even when CUDA is available.
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"

render_ramp() {
  local name="$1"
  local rise="$2"
  MPLCONFIGDIR=/tmp/mplconfig MUJOCO_GL=egl "$PYTHON_BIN" \
    eval_unitree_target_terrain_policy.py \
    --terrain-mode ramp --geometry-rise "$rise" --geometry-side-length 2 \
    --num-paths 1 --steps 700 --device "$DEVICE" --video \
    --output-dir "$OUTPUT_ROOT/$name" >"$OUTPUT_ROOT/$name.log" 2>&1
  echo "[ramp-video] $name complete"
}

# Independent one-environment renders can safely overlap and are much faster
# than rendering the three clips serially.
render_ramp ramp_05deg 0.175 &
pid_5=$!
render_ramp ramp_10deg 0.353 &
pid_10=$!
render_ramp ramp_15deg 0.536 &
pid_15=$!
wait "$pid_5"
wait "$pid_10"
wait "$pid_15"
echo "[ramp-video] complete: $OUTPUT_ROOT"

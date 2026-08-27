#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-visualizations/unitree_ramp_calibration_flush}"
DEVICE="${DEVICE:-cuda:0}"
mkdir -p "$OUTPUT_ROOT"

run_ramp() {
  local name="$1"
  local rise="$2"
  "$PYTHON_BIN" eval_unitree_target_terrain_policy.py \
    --num-paths 1 --steps 700 --device "$DEVICE" --terrain-mode ramp \
    --geometry-rise "$rise" --geometry-side-length 2.0 \
    --output-dir "$OUTPUT_ROOT/$name" >"$OUTPUT_ROOT/$name.log" 2>&1
  echo "[ramp-flush] $name"
}

run_ramp ramp_05deg 0.175
run_ramp ramp_10deg 0.353
run_ramp ramp_15deg 0.536
run_ramp ramp_20deg 0.728
run_ramp ramp_25deg 0.933
run_ramp ramp_30deg 1.155
echo "[ramp-flush] complete: $OUTPUT_ROOT"

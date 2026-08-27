#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-visualizations/unitree_geometry_calibration}"
DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-700}"
mkdir -p "$OUTPUT_ROOT"

run_geometry() {
  local name="$1"
  shift
  echo "[geometry-calibration] $name"
  "$PYTHON_BIN" eval_unitree_target_terrain_policy.py \
    --num-paths 1 --steps "$STEPS" --device "$DEVICE" \
    --output-dir "$OUTPUT_ROOT/$name" "$@" \
    >"$OUTPUT_ROOT/$name.log" 2>&1
}

# Two-meter ramp sides: rises correspond approximately to 5-30 degrees.
run_geometry ramp_05deg --terrain-mode ramp --geometry-rise 0.175 --geometry-side-length 2.0
run_geometry ramp_10deg --terrain-mode ramp --geometry-rise 0.353 --geometry-side-length 2.0
run_geometry ramp_15deg --terrain-mode ramp --geometry-rise 0.536 --geometry-side-length 2.0
run_geometry ramp_20deg --terrain-mode ramp --geometry-rise 0.728 --geometry-side-length 2.0
run_geometry ramp_25deg --terrain-mode ramp --geometry-rise 0.933 --geometry-side-length 2.0
run_geometry ramp_30deg --terrain-mode ramp --geometry-rise 1.155 --geometry-side-length 2.0

# Five steps per side and 2 m per side: 5-25 cm individual step heights.
run_geometry stairs_05cm --terrain-mode stairs --geometry-rise 0.25 --geometry-side-length 2.0 --geometry-steps-per-side 5
run_geometry stairs_10cm --terrain-mode stairs --geometry-rise 0.50 --geometry-side-length 2.0 --geometry-steps-per-side 5
run_geometry stairs_15cm --terrain-mode stairs --geometry-rise 0.75 --geometry-side-length 2.0 --geometry-steps-per-side 5
run_geometry stairs_20cm --terrain-mode stairs --geometry-rise 1.00 --geometry-side-length 2.0 --geometry-steps-per-side 5
run_geometry stairs_25cm --terrain-mode stairs --geometry-rise 1.25 --geometry-side-length 2.0 --geometry-steps-per-side 5

echo "[geometry-calibration] complete: $OUTPUT_ROOT"

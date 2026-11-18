#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/benjamin/thesis}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:?Set CHECKPOINT_FILE to a gentle locomotion checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/visualizations/unitree_gentle_surface_stress_20260903}"
STEPS="${STEPS:-1000}"
NUM_PATHS="${NUM_PATHS:-8}"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_ROOT"

run_case() {
  local name="$1"
  shift
  echo "[gentle-stress] $name"
  "$PYTHON_BIN" eval_unitree_target_terrain_policy.py \
    --checkpoint-file "$CHECKPOINT_FILE" \
    --num-paths "$NUM_PATHS" \
    --steps "$STEPS" \
    --device cuda:0 \
    --output-dir "$OUTPUT_ROOT/$name" \
    "$@" >"$OUTPUT_ROOT/$name.log" 2>&1
}

run_case rigid --terrain-mode surface --surface rigid
run_case sand_contact --terrain-mode surface --surface sand
run_case sand_drag15 --terrain-mode surface --surface sand --surface-linear-drag 15
run_case sand_drag40 --terrain-mode surface --surface sand --surface-linear-drag 40
run_case sand_drag80 --terrain-mode surface --surface sand --surface-linear-drag 80
run_case sand_drag120 --terrain-mode surface --surface sand --surface-linear-drag 120
run_case ice_friction010 --terrain-mode surface --surface ice --surface-sliding-friction 0.10
run_case ice_friction020 --terrain-mode surface --surface ice --surface-sliding-friction 0.20
run_case ice_friction030 --terrain-mode surface --surface ice --surface-sliding-friction 0.30
run_case ice_friction045 --terrain-mode surface --surface ice --surface-sliding-friction 0.45
run_case ice_friction060 --terrain-mode surface --surface ice --surface-sliding-friction 0.60
run_case cobblestone --terrain-mode cobblestone
run_case stairs_6cm --terrain-mode stairs --geometry-rise 0.30 --geometry-steps-per-side 5
run_case stairs_10cm --terrain-mode stairs --geometry-rise 0.50 --geometry-steps-per-side 5
run_case ramp_10deg --terrain-mode ramp --geometry-rise 0.353 --geometry-side-length 2.0
run_case ramp_15deg --terrain-mode ramp --geometry-rise 0.536 --geometry-side-length 2.0

"$PYTHON_BIN" tools/plot_unitree_gentle_surface_stress.py \
  --input-dir "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/comparison.png" \
  --csv-output "$OUTPUT_ROOT/comparison.csv"
echo "[gentle-stress] complete: $OUTPUT_ROOT"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-visualizations/unitree_sand_drag_refinement}"
DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-1000}"
mkdir -p "$OUTPUT_ROOT"

run_drag() {
  local value="$1"
  "$PYTHON_BIN" eval_unitree_target_terrain_policy.py \
    --terrain-mode surface --surface sand --surface-linear-drag "$value" \
    --num-paths 8 --steps "$STEPS" --device "$DEVICE" \
    --output-dir "$OUTPUT_ROOT/drag_$value" \
    >"$OUTPUT_ROOT/drag_$value.log" 2>&1
}

"$PYTHON_BIN" eval_unitree_target_terrain_policy.py \
  --terrain-mode surface --surface rigid --num-paths 8 --steps "$STEPS" \
  --device "$DEVICE" --output-dir "$OUTPUT_ROOT/rigid_reference" \
  >"$OUTPUT_ROOT/rigid_reference.log" 2>&1
for value in 10 15 20 25 30 35; do
  echo "[sand-drag-refinement] drag=$value"
  run_drag "$value"
done
echo "[sand-drag-refinement] complete: $OUTPUT_ROOT"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-visualizations/unitree_sand_sink_refinement}"
DEVICE="${DEVICE:-cuda:0}"
mkdir -p "$OUTPUT_ROOT"

run_variant() {
  local name="$1"
  shift
  "$PYTHON_BIN" eval_unitree_target_terrain_policy.py \
    --terrain-mode surface --surface sand --num-paths 8 --steps 1000 \
    --device "$DEVICE" --output-dir "$OUTPUT_ROOT/$name" "$@" \
    >"$OUTPUT_ROOT/$name.log" 2>&1
}

"$PYTHON_BIN" eval_unitree_target_terrain_policy.py \
  --terrain-mode surface --surface rigid --num-paths 8 --steps 1000 \
  --device "$DEVICE" --output-dir "$OUTPUT_ROOT/rigid_reference" \
  >"$OUTPUT_ROOT/rigid_reference.log" 2>&1
run_variant drag_15 --surface-linear-drag 15
run_variant soft_mild \
  --surface-solref 0.05 1.0 \
  --surface-solimp 0.7 0.9 0.03 0.5 2.0 \
  --surface-contact-priority 1
run_variant soft_drag_10 \
  --surface-solref 0.05 1.0 \
  --surface-solimp 0.7 0.9 0.03 0.5 2.0 \
  --surface-contact-priority 1 --surface-linear-drag 10
run_variant soft_drag_15 \
  --surface-solref 0.05 1.0 \
  --surface-solimp 0.7 0.9 0.03 0.5 2.0 \
  --surface-contact-priority 1 --surface-linear-drag 15
echo "[sand-sink-refinement] complete: $OUTPUT_ROOT"

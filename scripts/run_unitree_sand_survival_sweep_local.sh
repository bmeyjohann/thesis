#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-visualizations/unitree_sand_survival_sweep}"
DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-1000}"
NUM_PATHS="${NUM_PATHS:-8}"
mkdir -p "$OUTPUT_ROOT"

run_variant() {
  local name="$1"
  shift
  echo "[sand-survival] $name"
  "$PYTHON_BIN" eval_unitree_target_terrain_policy.py \
    --terrain-mode surface --surface sand --num-paths "$NUM_PATHS" --steps "$STEPS" \
    --device "$DEVICE" --output-dir "$OUTPUT_ROOT/$name" "$@" \
    >"$OUTPUT_ROOT/$name.log" 2>&1
}

"$PYTHON_BIN" eval_unitree_target_terrain_policy.py \
  --terrain-mode surface --surface rigid --num-paths "$NUM_PATHS" --steps "$STEPS" \
  --device "$DEVICE" --output-dir "$OUTPUT_ROOT/rigid_reference" \
  >"$OUTPUT_ROOT/rigid_reference.log" 2>&1

for drag in 40 60 80 100 120; do
  run_variant "drag_${drag}" --surface-linear-drag "$drag"
done

for drag in 40 80 120; do
  run_variant "soft_mild_drag_${drag}" \
    --surface-solref 0.05 1.0 \
    --surface-solimp 0.7 0.9 0.03 0.5 2.0 \
    --surface-contact-priority 1 --surface-linear-drag "$drag"
  run_variant "soft_deep_drag_${drag}" \
    --surface-solref 0.12 0.8 \
    --surface-solimp 0.3 0.75 0.06 0.5 2.0 \
    --surface-contact-priority 1 --surface-linear-drag "$drag"
done

echo "[sand-survival] complete: $OUTPUT_ROOT"

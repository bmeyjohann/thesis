#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-visualizations/unitree_sand_calibration}"
STEPS="${STEPS:-500}"
NUM_PATHS="${NUM_PATHS:-8}"
DEVICE="${DEVICE:-cuda:0}"

run_variant() {
  local name="$1"
  shift
  echo "[sand-calibration] ${name}"
  "$PYTHON_BIN" eval_unitree_target_terrain_policy.py \
    --terrain-mode surface \
    --surface sand \
    --num-paths "$NUM_PATHS" \
    --steps "$STEPS" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_ROOT/$name" \
    "$@" \
    >"$OUTPUT_ROOT/$name.log" 2>&1
}

mkdir -p "$OUTPUT_ROOT"

# Reference and contact-only variants.
"$PYTHON_BIN" eval_unitree_target_terrain_policy.py \
  --terrain-mode surface --surface rigid --num-paths "$NUM_PATHS" \
  --steps "$STEPS" --device "$DEVICE" --output-dir "$OUTPUT_ROOT/rigid_reference" \
  >"$OUTPUT_ROOT/rigid_reference.log" 2>&1
run_variant friction_1p25
run_variant friction_2p0 --surface-sliding-friction 2.0
run_variant friction_4p0 --surface-sliding-friction 4.0
run_variant soft_mild \
  --surface-solref 0.05 1.0 \
  --surface-solimp 0.7 0.9 0.03 0.5 2.0 \
  --surface-contact-priority 1
run_variant soft_deep \
  --surface-solref 0.12 0.8 \
  --surface-solimp 0.3 0.75 0.06 0.5 2.0 \
  --surface-contact-priority 1
run_variant margin_2cm --surface-margin 0.02 --surface-contact-priority 1

# Dissipative proxies. Damping is global in this homogeneous calibration;
# linear drag can later be gated by the visible local material.
run_variant damping_1p8 --joint-damping-scale 1.8
run_variant damping_3p0 --joint-damping-scale 3.0
run_variant drag_25 --surface-linear-drag 25
run_variant drag_50 --surface-linear-drag 50
run_variant drag_80 --surface-linear-drag 80
run_variant soft_drag_35 \
  --surface-solref 0.05 1.0 \
  --surface-solimp 0.7 0.9 0.03 0.5 2.0 \
  --surface-contact-priority 1 \
  --surface-linear-drag 35

echo "[sand-calibration] complete: $OUTPUT_ROOT"

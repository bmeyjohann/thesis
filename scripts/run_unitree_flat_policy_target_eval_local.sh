#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-$REPO_ROOT/external/unitree_rl_mjlab/logs/velocity/g1_flat/model_1499.pt}"
TERRAIN_SEED="${TERRAIN_SEED:-3}"
TERRAIN_PRESET="${TERRAIN_PRESET:-balanced}"
COMMAND_SPEED="${COMMAND_SPEED:-0.6}"
STEPS="${STEPS:-600}"
DEVICE="${DEVICE:-cuda:0}"
VIDEO="${VIDEO:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/visualizations/unitree_flat_policy_target_eval/seed_${TERRAIN_SEED}_${TERRAIN_PRESET}}"

cd "$REPO_ROOT"
EXTRA_ARGS=()
if [[ "$VIDEO" == "1" ]]; then
  EXTRA_ARGS+=(--video)
fi
MUJOCO_GL="${MUJOCO_GL:-egl}" \
  "$PYTHON_BIN" "$REPO_ROOT/eval_unitree_target_terrain_policy.py" \
  --checkpoint-file "$CHECKPOINT_FILE" \
  --terrain-seed "$TERRAIN_SEED" \
  --preset "$TERRAIN_PRESET" \
  --command-speed "$COMMAND_SPEED" \
  --steps "$STEPS" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"

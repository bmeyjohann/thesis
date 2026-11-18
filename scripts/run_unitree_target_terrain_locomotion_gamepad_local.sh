#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/benjamin/thesis}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
GAMEPAD_ENDPOINT_FILE="${GAMEPAD_ENDPOINT_FILE:-/mnt/c/Data/thesis/local/unitree_gamepad_endpoint.json}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"

if [[ -z "$CHECKPOINT_FILE" ]]; then
  RUN_ROOT="$ROOT_DIR/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity"
  CHECKPOINT_FILE="$(find "$RUN_ROOT" -maxdepth 2 -type f -path '*g1_gentle_turn_curriculum_seed0_resume13000_20260903/model_*.pt' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
fi
if [[ ! -f "$CHECKPOINT_FILE" ]]; then
  echo "Missing gentle locomotion checkpoint: $CHECKPOINT_FILE" >&2
  exit 1
fi
if [[ ! -f "$GAMEPAD_ENDPOINT_FILE" ]]; then
  echo "Start C:\\Data\\thesis\\scripts\\run_unitree_nav_gamepad_sender_windows.ps1 first." >&2
  exit 1
fi
mapfile -t endpoint < <(
  "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8-sig")); print(p["host"]); print(p.get("port", 8794))' "$GAMEPAD_ENDPOINT_FILE"
)

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/external/unitree_rl_mjlab${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
echo "checkpoint: $CHECKPOINT_FILE"
echo "gamepad:    ${endpoint[0]}:${endpoint[1]}"
echo "viewer:     http://localhost:8080"
exec "$PYTHON_BIN" eval_unitree_target_terrain_policy.py \
  --checkpoint-file "$CHECKPOINT_FILE" \
  --terrain-mode target \
  --preset traversal \
  --terrain-seed "${TERRAIN_SEED:-11}" \
  --num-paths 1 \
  --steps 1000 \
  --device cuda:0 \
  --output-dir /tmp/unitree_target_terrain_manual \
  --viewer viser \
  --gamepad \
  --gamepad-host "${endpoint[0]}" \
  --gamepad-port "${endpoint[1]}"

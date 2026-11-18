#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${STATE_DIR:-$ROOT_DIR/artifacts/unitree_gentle_locomotion}"
RUN_TAG="${RUN_TAG:-seed${SEED:-0}_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="$STATE_DIR/train_${RUN_TAG}.log"
PID_FILE="$STATE_DIR/train_${RUN_TAG}.pid"

mkdir -p "$STATE_DIR"
if [[ -e "$PID_FILE" ]]; then
  existing="$(cat "$PID_FILE")"
  if kill -0 "$existing" 2>/dev/null; then
    echo "Training already running: pid=$existing log=$LOG_FILE" >&2
    exit 1
  fi
fi

nohup env \
  SEED="${SEED:-0}" \
  NUM_ENVS="${NUM_ENVS:-256}" \
  ITERATIONS="${ITERATIONS:-6000}" \
  SAVE_INTERVAL="${SAVE_INTERVAL:-250}" \
  RUN_NAME="${RUN_NAME:-g1_gentle_turn_curriculum_seed${SEED:-0}}" \
  "$ROOT_DIR/scripts/run_unitree_gentle_locomotion_curriculum_local.sh" \
  >"$LOG_FILE" 2>&1 </dev/null &

pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
printf 'pid=%s\nlog=%s\npid_file=%s\n' "$pid" "$LOG_FILE" "$PID_FILE"

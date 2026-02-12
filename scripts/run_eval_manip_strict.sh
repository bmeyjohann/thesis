#!/usr/bin/env bash
set -euo pipefail

# Strict intervention eval launcher for manipulation tasks.
#
# Usage:
#   bash scripts/run_eval_manip_strict.sh
#   STRICTNESS=hard bash scripts/run_eval_manip_strict.sh
#   STRICTNESS=medium ENV_NAME=cube-double-v0 CONTROLLER=human bash scripts/run_eval_manip_strict.sh
#
# Knobs:
#   STRICTNESS: hard|medium|soft   (default: hard)
#   ENV_NAME:   OGBench manip env  (default: cube-double-v0)
#   CONTROLLER: human|keyboard|random|idle (default: human)
#   MAX_EPISODE_STEPS: per-episode cap (default: 1000)
#   KEYBOARD_SCALE: teleop magnitude (default: 0.5)

STRICTNESS="${STRICTNESS:-hard}"
ENV_NAME="${ENV_NAME:-cube-double-v0}"
CONTROLLER="${CONTROLLER:-human}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
KEYBOARD_SCALE="${KEYBOARD_SCALE:-0.5}"

case "$STRICTNESS" in
  hard)
    TOLERANCE_TYPE="l2"
    TOLERANCE_VALUE="0.0"
    ;;
  medium)
    TOLERANCE_TYPE="l2"
    TOLERANCE_VALUE="0.01"
    ;;
  soft)
    TOLERANCE_TYPE="l2"
    TOLERANCE_VALUE="0.05"
    ;;
  *)
    echo "Unknown STRICTNESS='$STRICTNESS' (expected: hard|medium|soft)" >&2
    exit 2
    ;;
esac

python eval_interactive_manip.py \
  --env_name "$ENV_NAME" \
  --controller "$CONTROLLER" \
  --render_mode human \
  --intervention_mode agent \
  --teacher_type cube_plan \
  --tolerance_type "$TOLERANCE_TYPE" \
  --tolerance_value "$TOLERANCE_VALUE" \
  --max_episode_steps "$MAX_EPISODE_STEPS" \
  --keyboard_scale "$KEYBOARD_SCALE"

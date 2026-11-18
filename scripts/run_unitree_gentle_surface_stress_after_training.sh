#!/usr/bin/env bash
set -euo pipefail

TRAIN_SERVICE="${TRAIN_SERVICE:-unitree-gentle-resume13000.service}"
while systemctl --user is-active --quiet "$TRAIN_SERVICE"; do
  sleep 60
done

RUN_ROOT="/home/benjamin/thesis/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity"
RUN_DIR="$(find "$RUN_ROOT" -maxdepth 1 -type d -name '*g1_gentle_turn_curriculum_seed0_resume13000_20260903' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
if [[ -z "$RUN_DIR" ]]; then
  echo "Could not resolve resumed gentle-locomotion run" >&2
  exit 1
fi
CHECKPOINT_FILE="$(find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sort -V | tail -1)"
if [[ -z "$CHECKPOINT_FILE" ]]; then
  echo "No checkpoint found in $RUN_DIR" >&2
  exit 1
fi

export CHECKPOINT_FILE="$RUN_DIR/$CHECKPOINT_FILE"
export OUTPUT_ROOT="/home/benjamin/thesis/visualizations/unitree_gentle_surface_stress_20260903"
exec /home/benjamin/thesis/scripts/run_unitree_gentle_surface_stress_local.sh

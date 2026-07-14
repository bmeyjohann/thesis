#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis

for arg in "$@"; do
  if [[ "$arg" == *=* ]]; then
    export "$arg"
  else
    echo "Unsupported argument: $arg" >&2
    exit 2
  fi
done

STEP="${STEP:-2500}"
EPISODES="${EPISODES:-100}"
EVAL_TAG="${EVAL_TAG:-canonical}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/unitree_mjlab_nav_thesis/unitree_student_scan_cnn_scratch_dense_geom_clear06_10k_20260714/step_${STEP}.pt}"
RUN_STEM="${RUN_STEM:-scan_cnn_scratch}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/logs/unitree_mjlab/scan_cnn_student_canonical_20260714}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing checkpoint: $MODEL_PATH" >&2
  exit 2
fi

CONTROLLER=policy \
MODEL_PATH="$MODEL_PATH" \
RUN_NAME="${RUN_STEM}_step_${STEP}_${EVAL_TAG}_${EPISODES}ep" \
OUTPUT_DIR="$OUTPUT_DIR" \
RECORD_VIDEO=0 \
SEED=101 \
NUM_ENVS="${NUM_ENVS:-$EPISODES}" \
NUM_EPISODES="$EPISODES" \
EPISODE_LENGTH_S=60 \
GOAL_THROUGH_OBSTACLE_PROB=0.7 \
GOAL_DISTANCE_MIN=2.8 \
GOAL_DISTANCE_MAX=4.0 \
MIN_GOAL_OBSTACLE_CLEARANCE=0.9 \
DEBUG_OBSTACLE_WIDTH_MIN=1.0 \
DEBUG_OBSTACLE_WIDTH_MAX=1.4 \
DEBUG_OBSTACLE_HEIGHT_MIN=1.0 \
DEBUG_OBSTACLE_HEIGHT_MAX=1.0 \
DEBUG_NUM_OBSTACLES=6 \
DEBUG_PLATFORM_WIDTH=2.0 \
DEBUG_TERRAIN_ROWS=5 \
DEBUG_TERRAIN_COLS=10 \
"$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh"

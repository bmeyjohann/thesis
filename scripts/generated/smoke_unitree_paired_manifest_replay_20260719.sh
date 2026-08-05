#!/usr/bin/env bash
set -euo pipefail

CONTROLLER_ARG="${1:?usage: $0 direct_goal|policy}"
case "$CONTROLLER_ARG" in direct_goal|policy) ;; *) exit 2 ;; esac

ROOT=/home/benjamin/thesis
export REPO_ROOT="$ROOT"
export CONTROLLER="$CONTROLLER_ARG"
export DEVICE=cuda:0
export SEED=123
export NUM_ENVS=8
export NUM_EPISODES=16
export EPISODE_LENGTH_S=90
export RECORD_VIDEO=0
export OUTPUT_DIR="$ROOT/logs/unitree_mjlab/paired_manifest_smoke"
export RUN_NAME="${CONTROLLER_ARG}_replay"
export LAYOUT_MANIFEST="$ROOT/config/unitree_eval_manifest_blocked100_seed941.json"
export MODEL_PATH="$ROOT/models/unitree_mjlab_nav_thesis/unitree_compare_thesis_seed0_20260718_v2/step_5000.pt"
export CHECKPOINT_ENV_CONFIG="$([[ "$CONTROLLER_ARG" == policy ]] && echo 1 || echo 0)"

export SUCCESS_DIST=0.40
export GOAL_DISTANCE_MIN=4.5
export GOAL_DISTANCE_MAX=8.0
export REQUIRE_BLOCKED_CORRIDOR=1
export BLOCKED_CORRIDOR_RADIUS=0.45
export BLOCKED_CORRIDOR_MIN_CELLS=4
export DEBUG_NUM_OBSTACLES=6
export DEBUG_OBSTACLE_WIDTH_MIN=1.0
export DEBUG_OBSTACLE_WIDTH_MAX=1.4
export DEBUG_TERRAIN_ROWS=5
export DEBUG_TERRAIN_COLS=10
export DEBUG_PLATFORM_WIDTH=2.0
export STRICT_MIN_SIZE_OBSTACLES=1
export RESAMPLE_TERRAIN_TILES=1
export MIN_START_OBSTACLE_CLEARANCE=1.0
export MIN_GOAL_OBSTACLE_CLEARANCE=0.9

exec "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh"

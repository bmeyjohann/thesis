#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
MODEL="$ROOT/models/unitree_mjlab_nav_human/unitree_human_scratch_20260803_092823_retry02/final.pt"
MANIFEST="$ROOT/local/eval_manifests/unitree_human_rectscan_blocked_100_seed0.json"

exec "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh" \
  CONTROLLER=geom_scan_teacher \
  MODEL_PATH="$MODEL" \
  CHECKPOINT_ENV_CONFIG=1 \
  NUM_ENVS=20 \
  NUM_EPISODES=100 \
  RUN_NAME=unitree_geom_teacher_human_rectscan_matched_100 \
  OUTPUT_DIR="$ROOT/logs/unitree_mjlab/human_eval" \
  LAYOUT_MANIFEST="$MANIFEST" \
  RECORD_VIDEO=0 \
  REQUIRE_BLOCKED_CORRIDOR=1 \
  DEBUG_GOAL_THROUGH_OBSTACLE=1 \
  STRICT_MIN_SIZE_OBSTACLES=1 \
  HEIGHT_SCAN_RESOLUTION=0.25 \
  HEIGHT_SCAN_FORWARD_SIZE=5.0 \
  HEIGHT_SCAN_LATERAL_SIZE=3.0 \
  SUCCESS_DIST=0.4 \
  TEACHER_GEOM_PLANNER=astar \
  TEACHER_GEOM_CLEARANCE=0.62 \
  TEACHER_GEOM_GRID_RESOLUTION=0.18 \
  TEACHER_GEOM_LOOKAHEAD=1.6 \
  TEACHER_GEOM_WAYPOINT_INDEX=3 \
  TEACHER_GEOM_SIDE_PENALTY=4.0 \
  TEACHER_GEOM_SIDE_FRAME=body \
  TEACHER_GEOM_DISENGAGE_CLEAR_STEPS=12 \
  TEACHER_GEOM_EMERGENCY_RADIUS=0.8 \
  TEACHER_GOAL_STOP_DIST=0.35

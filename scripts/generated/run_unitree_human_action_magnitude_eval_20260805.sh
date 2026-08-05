#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/benjamin/thesis
KIND="${1:?usage: $0 human|sac|teacher}"
MANIFEST="$ROOT/local/eval_manifests/unitree_human_rectscan_blocked_100_seed0.json"

case "$KIND" in
  human)
    CONTROLLER=policy
    MODEL="$ROOT/models/unitree_mjlab_nav_human/unitree_human_scratch_20260803_092823_retry02/final.pt"
    ;;
  sac)
    CONTROLLER=policy
    MODEL="$ROOT/models/unitree_mjlab_nav_thesis/unitree_human_matched_sac_reward_only_seed0_28k_20260804/final.pt"
    ;;
  teacher)
    CONTROLLER=geom_scan_teacher
    MODEL=""
    ;;
  *) echo "unknown kind: $KIND" >&2; exit 2 ;;
esac

exec "$ROOT/scripts/run_unitree_mjlab_nav_baseline_eval_local.sh" \
  CONTROLLER="$CONTROLLER" \
  MODEL_PATH="$MODEL" \
  CHECKPOINT_ENV_CONFIG=1 \
  NUM_ENVS=20 \
  NUM_EPISODES=100 \
  RUN_NAME="unitree_${KIND}_human_rectscan_matched_magnitude_100" \
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
  GOAL_DISTANCE_MIN=4.5 \
  GOAL_DISTANCE_MAX=8.0 \
  MIN_GOAL_OBSTACLE_CLEARANCE=0.9 \
  MIN_START_OBSTACLE_CLEARANCE=1.0 \
  BLOCKED_CORRIDOR_MIN_CELLS=4 \
  BLOCKED_CORRIDOR_RESAMPLE_ATTEMPTS=300 \
  BLOCKED_GOAL_MAX_DISTANCE=8.0 \
  BLOCKED_GOAL_DISTANCE_SAMPLING=uniform \
  BLOCKED_GOAL_PLACEMENT_MODE=obstacle_multiplier \
  BLOCKED_GOAL_DISTANCE_MULTIPLIER_MIN=1.0 \
  BLOCKED_GOAL_DISTANCE_MULTIPLIER_MAX=2.0 \
  DEBUG_GOAL_OBSTACLE_MIN_DIST=1.0 \
  DEBUG_GOAL_OBSTACLE_MAX_DIST=5.5 \
  GOAL_THROUGH_OBSTACLE_PROB=1.0 \
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

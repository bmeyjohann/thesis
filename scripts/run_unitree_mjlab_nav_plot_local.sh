#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${REPO_ROOT:-$PWD}"
if [[ ! -f "$ROOT_DIR/plot_unitree_nav_rollout.py" ]]; then
  ROOT_DIR="$SCRIPT_ROOT"
fi
if [[ ! -f "$ROOT_DIR/plot_unitree_nav_rollout.py" ]]; then
  echo "Could not resolve thesis repo root; set REPO_ROOT=/home/benjamin/thesis" >&2
  exit 2
fi

for arg in "$@"; do
  if [[ "$arg" == *=* ]]; then
    export "$arg"
  else
    echo "Unsupported positional argument: $arg" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/warp-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/unitree-cache}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "$MPLCONFIGDIR" "$WARP_CACHE_PATH" "$XDG_CACHE_HOME"

ESCAPE_ALL_DIRECTIONS_FLAG=()
if [[ "${TEACHER_ESCAPE_ALL_DIRECTIONS:-0}" == "1" ]]; then
  ESCAPE_ALL_DIRECTIONS_FLAG=(--teacher-escape-all-directions)
fi

REQUIRE_BLOCKED_CORRIDOR_FLAG=()
if [[ "${REQUIRE_BLOCKED_CORRIDOR:-0}" == "1" ]]; then
  REQUIRE_BLOCKED_CORRIDOR_FLAG=(--require-blocked-corridor)
fi

DEBUG_GOAL_THROUGH_OBSTACLE_FLAG=()
if [[ "${DEBUG_GOAL_THROUGH_OBSTACLE:-0}" == "1" ]]; then
  DEBUG_GOAL_THROUGH_OBSTACLE_FLAG=(--debug-goal-through-obstacle)
fi

RESAMPLE_TERRAIN_TILES_FLAG=()
if [[ "${RESAMPLE_TERRAIN_TILES:-1}" == "1" ]]; then
  RESAMPLE_TERRAIN_TILES_FLAG=(--resample-terrain-tiles)
fi

exec "$PYTHON_BIN" "$ROOT_DIR/plot_unitree_nav_rollout.py" \
  --controller "${CONTROLLER:-scan_teacher}" \
  --model-path "${MODEL_PATH:-}" \
  --task "${TASK:-Unitree-G1-Nav-Obstacles-Safe-Collision}" \
  --device "${DEVICE:-cuda:0}" \
  --seed "${SEED:-0}" \
  --episode-length-s "${EPISODE_LENGTH_S:-60.0}" \
  --steps "${STEPS:-360}" \
  --num-rollouts "${NUM_ROLLOUTS:-1}" \
  --layout-generation-attempts "${LAYOUT_GENERATION_ATTEMPTS:-20}" \
  --success-dist "${SUCCESS_DIST:-0.5}" \
  --low-level-policy-path "${LOW_LEVEL_POLICY_PATH:-$ROOT_DIR/external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/2026-07-12_10-35-19_omni_finetune_model1499_20260712}" \
  --output-dir "${OUTPUT_DIR:-$ROOT_DIR/visualizations/unitree_nav_debug}" \
  --teacher-scan-block-threshold "${TEACHER_SCAN_BLOCK_THRESHOLD:-0.12}" \
  --teacher-scan-block-delta "${TEACHER_SCAN_BLOCK_DELTA:-0.0}" \
  --teacher-scan-planner "${TEACHER_SCAN_PLANNER:-heuristic}" \
  --teacher-scan-astar-clearance "${TEACHER_SCAN_ASTAR_CLEARANCE:-0.6}" \
  --teacher-scan-astar-cell-padding "${TEACHER_SCAN_ASTAR_CELL_PADDING:-0.35}" \
  --teacher-scan-astar-resolution "${TEACHER_SCAN_ASTAR_RESOLUTION:-0.25}" \
  --teacher-scan-astar-waypoint-index "${TEACHER_SCAN_ASTAR_WAYPOINT_INDEX:-3}" \
  --teacher-scan-astar-side-penalty "${TEACHER_SCAN_ASTAR_SIDE_PENALTY:-8.0}" \
  --teacher-scan-astar-commit-steps "${TEACHER_SCAN_ASTAR_COMMIT_STEPS:-30}" \
  --teacher-sector-half-width "${TEACHER_SECTOR_HALF_WIDTH:-0.35}" \
  --teacher-align-angle "${TEACHER_ALIGN_ANGLE:-0.55}" \
  --teacher-max-vx "${TEACHER_MAX_VX:-0.95}" \
  --teacher-max-vy "${TEACHER_MAX_VY:-0.45}" \
  --teacher-yaw-gain "${TEACHER_YAW_GAIN:-1.2}" \
  --teacher-clearance-weight "${TEACHER_CLEARANCE_WEIGHT:-0.0}" \
  --teacher-clearance-power "${TEACHER_CLEARANCE_POWER:-2.0}" \
  --teacher-speed-clearance-scale "${TEACHER_SPEED_CLEARANCE_SCALE:-0.0}" \
  --teacher-num-sectors "${TEACHER_NUM_SECTORS:-13}" \
  --teacher-min-forward-scale "${TEACHER_MIN_FORWARD_SCALE:-0.12}" \
  --teacher-escape-risk-threshold "${TEACHER_ESCAPE_RISK_THRESHOLD:-0.0}" \
  --teacher-escape-forward-scale "${TEACHER_ESCAPE_FORWARD_SCALE:-0.0}" \
  --teacher-escape-lateral-scale "${TEACHER_ESCAPE_LATERAL_SCALE:-1.0}" \
  --teacher-escape-radius "${TEACHER_ESCAPE_RADIUS:-1.0}" \
  --teacher-bypass-angle "${TEACHER_BYPASS_ANGLE:-0.0}" \
  --teacher-goal-stop-dist "${TEACHER_GOAL_STOP_DIST:-0.0}" \
  --teacher-wall-follow-steps "${TEACHER_WALL_FOLLOW_STEPS:-0}" \
  --teacher-wall-follow-angle "${TEACHER_WALL_FOLLOW_ANGLE:-0.9}" \
  --teacher-wall-follow-clear-risk "${TEACHER_WALL_FOLLOW_CLEAR_RISK:-0.15}" \
  --teacher-rollout-horizon "${TEACHER_ROLLOUT_HORIZON:-0.0}" \
  --teacher-rollout-clearance "${TEACHER_ROLLOUT_CLEARANCE:-0.65}" \
  --teacher-rollout-samples "${TEACHER_ROLLOUT_SAMPLES:-8}" \
  --teacher-rollout-clearance-weight "${TEACHER_ROLLOUT_CLEARANCE_WEIGHT:-20.0}" \
  --teacher-rollout-forward-bias "${TEACHER_ROLLOUT_FORWARD_BIAS:-0.05}" \
  --teacher-geom-planner "${TEACHER_GEOM_PLANNER:-astar}" \
  --teacher-geom-scan-margin "${TEACHER_GEOM_SCAN_MARGIN:-0.15}" \
  --teacher-geom-back-margin "${TEACHER_GEOM_BACK_MARGIN:-0.35}" \
  --teacher-geom-max-forward "${TEACHER_GEOM_MAX_FORWARD:-1.75}" \
  --teacher-geom-max-lateral "${TEACHER_GEOM_MAX_LATERAL:-1.75}" \
  --teacher-geom-lookahead "${TEACHER_GEOM_LOOKAHEAD:-1.6}" \
  --teacher-geom-clearance "${TEACHER_GEOM_CLEARANCE:-0.5}" \
  --teacher-geom-grid-resolution "${TEACHER_GEOM_GRID_RESOLUTION:-0.15}" \
  --teacher-geom-waypoint-index "${TEACHER_GEOM_WAYPOINT_INDEX:-3}" \
  --teacher-geom-side-penalty "${TEACHER_GEOM_SIDE_PENALTY:-8.0}" \
  --teacher-geom-side-frame "${TEACHER_GEOM_SIDE_FRAME:-body}" \
  --teacher-geom-disengage-clear-steps "${TEACHER_GEOM_DISENGAGE_CLEAR_STEPS:-20}" \
  --teacher-geom-emergency-radius "${TEACHER_GEOM_EMERGENCY_RADIUS:-0.8}" \
  --min-goal-obstacle-clearance "${MIN_GOAL_OBSTACLE_CLEARANCE:-0.9}" \
  --goal-clearance-resample-attempts "${GOAL_CLEARANCE_RESAMPLE_ATTEMPTS:-50}" \
  --min-start-obstacle-clearance "${MIN_START_OBSTACLE_CLEARANCE:-0.0}" \
  --start-clearance-resample-attempts "${START_CLEARANCE_RESAMPLE_ATTEMPTS:-20}" \
  --blocked-corridor-radius "${BLOCKED_CORRIDOR_RADIUS:-0.45}" \
  --blocked-corridor-ignore-end-radius "${BLOCKED_CORRIDOR_IGNORE_END_RADIUS:-0.75}" \
  --blocked-corridor-min-cells "${BLOCKED_CORRIDOR_MIN_CELLS:-1}" \
  --blocked-corridor-resample-attempts "${BLOCKED_CORRIDOR_RESAMPLE_ATTEMPTS:-100}" \
  --debug-obstacle-width-min "${DEBUG_OBSTACLE_WIDTH_MIN:-1.0}" \
  --debug-obstacle-width-max "${DEBUG_OBSTACLE_WIDTH_MAX:-1.4}" \
  --debug-obstacle-height-min "${DEBUG_OBSTACLE_HEIGHT_MIN:-1.0}" \
  --debug-obstacle-height-max "${DEBUG_OBSTACLE_HEIGHT_MAX:-1.0}" \
  --debug-num-obstacles "${DEBUG_NUM_OBSTACLES:-6}" \
  --debug-platform-width "${DEBUG_PLATFORM_WIDTH:-2.0}" \
  --debug-obstacle-border-width "${DEBUG_OBSTACLE_BORDER_WIDTH:-0.0}" \
  --debug-terrain-rows "${DEBUG_TERRAIN_ROWS:-5}" \
  --debug-terrain-cols "${DEBUG_TERRAIN_COLS:-10}" \
  --debug-goal-distance "${DEBUG_GOAL_DISTANCE:-3.2}" \
  --debug-goal-obstacle-min-dist "${DEBUG_GOAL_OBSTACLE_MIN_DIST:-0.8}" \
  --debug-goal-obstacle-max-dist "${DEBUG_GOAL_OBSTACLE_MAX_DIST:-2.2}" \
  --goal-through-obstacle-prob "${GOAL_THROUGH_OBSTACLE_PROB:-0.7}" \
  --goal-distance-min "${GOAL_DISTANCE_MIN:-2.8}" \
  --goal-distance-max "${GOAL_DISTANCE_MAX:-4.0}" \
  --arrow-every "${ARROW_EVERY:-12}" \
  "${ESCAPE_ALL_DIRECTIONS_FLAG[@]}" \
  "${REQUIRE_BLOCKED_CORRIDOR_FLAG[@]}" \
  "${DEBUG_GOAL_THROUGH_OBSTACLE_FLAG[@]}" \
  "${RESAMPLE_TERRAIN_TILES_FLAG[@]}"

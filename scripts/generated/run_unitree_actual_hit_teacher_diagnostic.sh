#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
export MPLCONFIGDIR=/tmp/mplconfig
export MUJOCO_GL=egl
export WARP_CACHE_PATH=/tmp/warp-cache
export XDG_CACHE_HOME=/tmp/unitree-cache

RUN_NAME="${RUN_NAME:-actual_hit_teacher_blocked_$(date +%Y%m%d_%H%M%S)}"

python -u /home/benjamin/thesis/tools/diagnose_unitree_actual_scan_teacher.py \
  --device "${DEVICE:-cuda:0}" \
  --num-envs 1 \
  --num-episodes "${NUM_EPISODES:-4}" \
  --episode-length-s "${EPISODE_LENGTH_S:-16.0}" \
  --run-name "$RUN_NAME" \
  --require-blocked-corridor \
  --blocked-corridor-radius "${BLOCKED_CORRIDOR_RADIUS:-0.55}" \
  --blocked-corridor-min-cells "${BLOCKED_CORRIDOR_MIN_CELLS:-1}" \
  --blocked-corridor-ignore-end-radius "${BLOCKED_CORRIDOR_IGNORE_END_RADIUS:-0.75}" \
  --debug-obstacle-width-min "${DEBUG_OBSTACLE_WIDTH_MIN:-0.75}" \
  --debug-obstacle-width-max "${DEBUG_OBSTACLE_WIDTH_MAX:-1.25}" \
  --debug-obstacle-height-min "${DEBUG_OBSTACLE_HEIGHT_MIN:-1.0}" \
  --debug-obstacle-height-max "${DEBUG_OBSTACLE_HEIGHT_MAX:-1.0}" \
  --debug-num-obstacles "${DEBUG_NUM_OBSTACLES:-12}" \
  --debug-platform-width "${DEBUG_PLATFORM_WIDTH:-1.8}" \
  --debug-goal-through-obstacle \
  --min-start-obstacle-clearance "${MIN_START_OBSTACLE_CLEARANCE:-0.75}" \
  --min-goal-obstacle-clearance "${MIN_GOAL_OBSTACLE_CLEARANCE:-0.80}" \
  --scan-block-threshold "${SCAN_BLOCK_THRESHOLD:-0.12}" \
  --scan-block-delta "${SCAN_BLOCK_DELTA:-0.025}" \
  --lookahead "${LOOKAHEAD:-2.4}" \
  --corridor-width "${CORRIDOR_WIDTH:-0.78}" \
  --clearance-soft-width "${CLEARANCE_SOFT_WIDTH:-1.05}" \
  --risk-weight "${RISK_WEIGHT:-5.5}" \
  --blocked-penalty "${BLOCKED_PENALTY:-24.0}" \
  --angle-weight "${ANGLE_WEIGHT:-0.18}" \
  --forward-bias "${FORWARD_BIAS:-0.35}" \
  --max-vx "${MAX_VX:-0.72}" \
  --max-vy "${MAX_VY:-0.45}" \
  --yaw-gain "${YAW_GAIN:-1.2}" \
  --align-angle "${ALIGN_ANGLE:-0.75}" \
  --min-forward-scale "${MIN_FORWARD_SCALE:-0.12}" \
  --goal-stop-dist "${GOAL_STOP_DIST:-0.55}" \
  --emergency-radius "${EMERGENCY_RADIUS:-0.95}" \
  --emergency-speed-scale "${EMERGENCY_SPEED_SCALE:-0.75}" \
  --emergency-repulsion-weight "${EMERGENCY_REPULSION_WEIGHT:-0.8}" \
  --emergency-tangent-weight "${EMERGENCY_TANGENT_WEIGHT:-1.2}" \
  --emergency-goal-weight "${EMERGENCY_GOAL_WEIGHT:-0.4}" \
  --snapshot-steps "${SNAPSHOT_STEPS:-0,40,80,120}"

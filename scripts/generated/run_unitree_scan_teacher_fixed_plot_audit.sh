#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
export MPLCONFIGDIR=/tmp/mplconfig
export MUJOCO_GL=egl
export WARP_CACHE_PATH=/tmp/warp-cache
export XDG_CACHE_HOME=/tmp/unitree-cache

RUN_ROOT="${RUN_ROOT:-/home/benjamin/thesis/visualizations/unitree_scan_teacher_fixed_plot_audit/scan_teacher_fixed_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"

for i in $(seq 0 "${NUM_ROLLOUTS:-2}"); do
  OUT_DIR="$RUN_ROOT/rollout_${i}"
  mkdir -p "$OUT_DIR"
  python -u /home/benjamin/thesis/plot_unitree_nav_rollout.py \
    --controller scan_teacher \
    --device "${DEVICE:-cuda:0}" \
    --steps "${STEPS:-320}" \
    --episode-length-s "${EPISODE_LENGTH_S:-16.0}" \
    --output-dir "$OUT_DIR" \
    --require-blocked-corridor \
    --blocked-corridor-radius "${BLOCKED_CORRIDOR_RADIUS:-0.55}" \
    --blocked-corridor-min-cells "${BLOCKED_CORRIDOR_MIN_CELLS:-1}" \
    --blocked-corridor-ignore-end-radius "${BLOCKED_CORRIDOR_IGNORE_END_RADIUS:-0.75}" \
    --debug-goal-through-obstacle \
    --debug-goal-distance "${DEBUG_GOAL_DISTANCE:-3.2}" \
    --debug-obstacle-width-min "${DEBUG_OBSTACLE_WIDTH_MIN:-0.75}" \
    --debug-obstacle-width-max "${DEBUG_OBSTACLE_WIDTH_MAX:-1.25}" \
    --debug-obstacle-height-min "${DEBUG_OBSTACLE_HEIGHT_MIN:-1.0}" \
    --debug-obstacle-height-max "${DEBUG_OBSTACLE_HEIGHT_MAX:-1.0}" \
    --debug-num-obstacles "${DEBUG_NUM_OBSTACLES:-12}" \
    --debug-platform-width "${DEBUG_PLATFORM_WIDTH:-1.8}" \
    --min-start-obstacle-clearance "${MIN_START_OBSTACLE_CLEARANCE:-0.75}" \
    --min-goal-obstacle-clearance "${MIN_GOAL_OBSTACLE_CLEARANCE:-1.00}" \
    --teacher-scan-block-threshold "${TEACHER_SCAN_BLOCK_THRESHOLD:-0.12}" \
    --teacher-scan-block-delta "${TEACHER_SCAN_BLOCK_DELTA:-0.025}" \
    --teacher-sector-half-width "${TEACHER_SECTOR_HALF_WIDTH:-0.35}" \
    --teacher-max-vx "${TEACHER_MAX_VX:-0.72}" \
    --teacher-max-vy "${TEACHER_MAX_VY:-0.45}" \
    --teacher-yaw-gain "${TEACHER_YAW_GAIN:-1.2}" \
    --teacher-align-angle "${TEACHER_ALIGN_ANGLE:-0.75}" \
    --teacher-min-forward-scale "${TEACHER_MIN_FORWARD_SCALE:-0.12}" \
    --teacher-bypass-angle "${TEACHER_BYPASS_ANGLE:-1.15}" \
    --teacher-wall-follow-steps "${TEACHER_WALL_FOLLOW_STEPS:-90}" \
    --teacher-wall-follow-angle "${TEACHER_WALL_FOLLOW_ANGLE:-1.15}" \
    --teacher-wall-follow-clear-risk "${TEACHER_WALL_FOLLOW_CLEAR_RISK:-0.08}" \
    --teacher-clearance-weight "${TEACHER_CLEARANCE_WEIGHT:-5.5}" \
    --teacher-rollout-horizon "${TEACHER_ROLLOUT_HORIZON:-2.4}" \
    --teacher-rollout-clearance "${TEACHER_ROLLOUT_CLEARANCE:-1.00}" \
    --teacher-rollout-clearance-weight "${TEACHER_ROLLOUT_CLEARANCE_WEIGHT:-24.0}" \
    --teacher-goal-stop-dist "${TEACHER_GOAL_STOP_DIST:-0.55}" \
    --teacher-emergency-radius "${TEACHER_EMERGENCY_RADIUS:-0.95}" \
    --teacher-emergency-hard-radius "${TEACHER_EMERGENCY_HARD_RADIUS:-0.90}" \
    --teacher-emergency-speed-scale "${TEACHER_EMERGENCY_SPEED_SCALE:-0.75}" \
    --teacher-emergency-repulsion-weight "${TEACHER_EMERGENCY_REPULSION_WEIGHT:-0.8}" \
    --teacher-emergency-tangent-weight "${TEACHER_EMERGENCY_TANGENT_WEIGHT:-1.2}" \
    --teacher-emergency-goal-weight "${TEACHER_EMERGENCY_GOAL_WEIGHT:-0.4}" \
    > "$OUT_DIR/rollout_${i}.jsonl"
done

python - <<'PY' "$RUN_ROOT"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for p in sorted(root.glob("*.json")):
    try:
        rows.append(json.loads(p.read_text()))
    except Exception:
        pass
print(json.dumps({"run_root": str(root), "rollouts": rows}, indent=2))
PY

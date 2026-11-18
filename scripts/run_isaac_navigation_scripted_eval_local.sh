#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${THESIS_ROOT:-}" ]]; then
  REPO_ROOT="${THESIS_ROOT}"
elif [[ -d "${PWD}/IsaacLab" && -d "${PWD}/safe-locomotion" ]]; then
  REPO_ROOT="${PWD}"
else
  REPO_ROOT="${SCRIPT_ROOT}"
fi
cd "${REPO_ROOT}"

SAFE_LOCO_ROOT="${SAFE_LOCO_ROOT:-${REPO_ROOT}/safe-locomotion}"
TASK="${TASK:-Isaac-Navigation-NoObstacles-Flat-Go2-v0}"
NAV_LOW_LEVEL_CFG="${NAV_LOW_LEVEL_CFG:-official}"
LOW_LEVEL_POLICY_PATH="${LOW_LEVEL_POLICY_PATH:-${REPO_ROOT}/logs/rsl_rl/unitree_go2_flat/2025-11-23_13-46-53_official_task_ppo_12893598/exported/policy.pt}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/logs/isaac_navigation_scripted/local_${CONTROLLER:-goal_straight}_$(date +%Y%m%d_%H%M%S)}"
CONTROLLER="${CONTROLLER:-goal_straight}"
NUM_EPISODES="${NUM_EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-500}"
SEED="${SEED:-0}"
LIN_GAIN="${LIN_GAIN:-0.35}"
YAW_GAIN="${YAW_GAIN:-0.8}"
MAX_LIN_VEL="${MAX_LIN_VEL:-0.35}"
MAX_YAW_VEL="${MAX_YAW_VEL:-0.6}"
FORWARD_HEADING_DEADBAND="${FORWARD_HEADING_DEADBAND:-0.35}"
TEACHER_OBSTACLE_X="${TEACHER_OBSTACLE_X:-1.0}"
TEACHER_OBSTACLE_Y="${TEACHER_OBSTACLE_Y:-0.0}"
TEACHER_OBSTACLE_RADIUS="${TEACHER_OBSTACLE_RADIUS:-0.65}"
TEACHER_OBSTACLE_MARGIN="${TEACHER_OBSTACLE_MARGIN:-0.45}"
TEACHER_OBSTACLE_SIDE="${TEACHER_OBSTACLE_SIDE:-left}"
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-350}"

if [[ ! -f "${LOW_LEVEL_POLICY_PATH}" ]]; then
  echo "Low-level policy not found: ${LOW_LEVEL_POLICY_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
export NAV_LOW_LEVEL_CFG
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_isaac_nav_${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

ISAACLAB_ACTIVATE="${ISAACLAB_ACTIVATE:-${REPO_ROOT}/sc_venv_template_isaaclab/activate.sh}"
if [[ -f "${ISAACLAB_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${ISAACLAB_ACTIVATE}"
fi

VIDEO_ARGS=()
if [[ "${VIDEO}" == "1" ]]; then
  VIDEO_ARGS=(--video --video_length "${VIDEO_LENGTH}")
fi

echo "Isaac scripted navigation eval"
echo "task: ${TASK}"
echo "nav cfg: ${NAV_LOW_LEVEL_CFG}"
echo "policy: ${LOW_LEVEL_POLICY_PATH}"
echo "controller: ${CONTROLLER}"
echo "out: ${OUT_DIR}"

"${REPO_ROOT}/IsaacLab/isaaclab.sh" \
  -p "${SAFE_LOCO_ROOT}/scripts/rsl_rl/eval_navigation_scripted.py" \
  --headless \
  "${VIDEO_ARGS[@]}" \
  --task "${TASK}" \
  --low_level_policy_path "${LOW_LEVEL_POLICY_PATH}" \
  --num_envs 1 \
  --num_episodes "${NUM_EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --output_dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --controller "${CONTROLLER}" \
  --lin_gain "${LIN_GAIN}" \
  --yaw_gain "${YAW_GAIN}" \
  --max_lin_vel "${MAX_LIN_VEL}" \
  --max_yaw_vel "${MAX_YAW_VEL}" \
  --forward_heading_deadband "${FORWARD_HEADING_DEADBAND}" \
  --teacher_obstacle_x "${TEACHER_OBSTACLE_X}" \
  --teacher_obstacle_y "${TEACHER_OBSTACLE_Y}" \
  --teacher_obstacle_radius "${TEACHER_OBSTACLE_RADIUS}" \
  --teacher_obstacle_margin "${TEACHER_OBSTACLE_MARGIN}" \
  --teacher_obstacle_side "${TEACHER_OBSTACLE_SIDE}" \
  --stop_on_success

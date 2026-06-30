#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-codex}}"
mkdir -p "$MPLCONFIGDIR"

for arg in "$@"; do
  case "$arg" in
    *=*)
      export "$arg"
      ;;
    *)
      echo "Unsupported argument: $arg" >&2
      exit 2
      ;;
  esac
done

TEACHER_CHECKPOINT_PATH="${TEACHER_CHECKPOINT_PATH:-}"
if [[ -z "$TEACHER_CHECKPOINT_PATH" ]]; then
  echo "Set TEACHER_CHECKPOINT_PATH=/path/to/best.pt" >&2
  exit 2
fi

python eval_safetygym_imitation_teacher.py \
  --teacher_checkpoint_path "$TEACHER_CHECKPOINT_PATH" \
  --student_checkpoint_path "${STUDENT_CHECKPOINT_PATH:-}" \
  --student_policy "${STUDENT_POLICY:-random}" \
  --env_name "${ENV_NAME:-SafetyCarGoal2-v0}" \
  --seed "${SEED:-1}" \
  --num_episodes "${NUM_EPISODES:-10}" \
  --max_episode_steps "${MAX_EPISODE_STEPS:-0}" \
  --render_mode "${RENDER_MODE:-pygame}" \
  --viewer_fps "${VIEWER_FPS:-20.0}" \
  --fps "${FPS:-30}" \
  --reward_mode "${REWARD_MODE:-dense}" \
  --dense_reward_scale "${DENSE_REWARD_SCALE:-1.0}" \
  --step_penalty "${STEP_PENALTY:--0.001}" \
  --surface_mode "${SURFACE_MODE:-default}" \
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT:-2.0}" \
  --car_force_scale "${CAR_FORCE_SCALE:-2.0}" \
  --car_action_mode "${CAR_ACTION_MODE:-raw_wheels}" \
  --point_action_mode "${POINT_ACTION_MODE:-native}" \
  --intervention_threshold "${INTERVENTION_THRESHOLD:-0.5}" \
  --device "${DEVICE:-cpu}" \
  ${TERMINATE_ON_GOAL:+--terminate_on_goal}

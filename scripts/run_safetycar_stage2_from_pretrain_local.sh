#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-fasttd3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"

VARIANT="${VARIANT:-hilserl}"
ENV_NAME="${ENV_NAME:-SafetyCarGoal2-v0}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-}"
AUTO_RESOLVE_INIT_CHECKPOINT="${AUTO_RESOLVE_INIT_CHECKPOINT:-1}"
PRETRAIN_REPLAY_DATASET_PATH="${PRETRAIN_REPLAY_DATASET_PATH:-}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
LEARNING_STARTS="${LEARNING_STARTS:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_EVERY="${UPDATE_EVERY:-1}"
UPDATES_PER_CYCLE="${UPDATES_PER_CYCLE:-1}"

RENDER_MODE="${RENDER_MODE:-human}"
SURFACE_MODE="${SURFACE_MODE:-default}"
CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-2.0}"
CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-2.0}"
REWARD_MODE="${REWARD_MODE:-sparse}"

USE_INTERVENTION="${USE_INTERVENTION:-1}"
HUMAN_INPUT_DEVICE="${HUMAN_INPUT_DEVICE:-gamepad}"
HUMAN_ACTION_SCALE="${HUMAN_ACTION_SCALE:-1.0}"
INTERVENTION_THRESHOLD="${INTERVENTION_THRESHOLD:-0.1}"
INTERVENTION_HOLD_SECONDS="${INTERVENTION_HOLD_SECONDS:-0.25}"
CONTROLLER_OVERLAY_HZ="${CONTROLLER_OVERLAY_HZ:-20.0}"

GAMEPAD_MODE="${GAMEPAD_MODE:-connect}"
GAMEPAD_HOST="${GAMEPAD_HOST:-127.0.0.1}"
GAMEPAD_PORT="${GAMEPAD_PORT:-8793}"

WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-safetycar_goal2_twostage}"
TS="$(date +%Y%m%d_%H%M%S)"
WANDB_GROUP="${WANDB_GROUP:-${EXPERIMENT_TAG}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT_TAG}_${VARIANT}_stage2_${TS}}"
EXP_NAME="${EXP_NAME:-${WANDB_RUN_NAME}}"
LOG_INTERVAL="${LOG_INTERVAL:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"

case "${VARIANT}" in
  own)
    TRAIN_SCRIPT="train_fast_sac_safetygym.py"
    ;;
  hilserl)
    TRAIN_SCRIPT="train_fastsac_hilserl_safetygym.py"
    ;;
  pvp)
    TRAIN_SCRIPT="train_fastsac_pvp_safetygym.py"
    ;;
  *)
    echo "Unsupported VARIANT=${VARIANT}. Use one of: own, hilserl, pvp." >&2
    exit 2
    ;;
esac

if [[ -z "${INIT_CHECKPOINT_PATH}" && "${AUTO_RESOLVE_INIT_CHECKPOINT}" == "1" ]]; then
  shopt -s nullglob
  matches=(models/safetygym/"${EXPERIMENT_TAG}"_pretrain_dense_*/final.pt)
  shopt -u nullglob
  if [[ ${#matches[@]} -gt 0 ]]; then
    INIT_CHECKPOINT_PATH="${matches[-1]}"
    echo "Auto-resolved latest stage-1 checkpoint: ${INIT_CHECKPOINT_PATH}"
  fi
fi

if [[ -z "${INIT_CHECKPOINT_PATH}" ]]; then
  echo "INIT_CHECKPOINT_PATH must point to a stage-1 final.pt or step_*.pt checkpoint." >&2
  exit 2
fi

echo "Starting SafetyCar stage-2 run from pretrained checkpoint."
echo "Variant: ${VARIANT}"
echo "Checkpoint: ${INIT_CHECKPOINT_PATH}"
echo "Reward mode: ${REWARD_MODE}"

cmd=(
  conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
  --env_name "${ENV_NAME}"
  --exp_name "${EXP_NAME}"
  --render_mode "${RENDER_MODE}"
  --surface_mode "${SURFACE_MODE}"
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}"
  --car_force_scale "${CAR_FORCE_SCALE}"
  --num_envs 1
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --learning_starts "${LEARNING_STARTS}"
  --batch_size "${BATCH_SIZE}"
  --update_every "${UPDATE_EVERY}"
  --updates_per_cycle "${UPDATES_PER_CYCLE}"
  --reward_mode "${REWARD_MODE}"
  --init_checkpoint_path "${INIT_CHECKPOINT_PATH}"
  --human_input_device "${HUMAN_INPUT_DEVICE}"
  --human_action_scale "${HUMAN_ACTION_SCALE}"
  --intervention_threshold "${INTERVENTION_THRESHOLD}"
  --intervention_hold_seconds "${INTERVENTION_HOLD_SECONDS}"
  --controller_overlay_hz "${CONTROLLER_OVERLAY_HZ}"
  --gamepad_mode "${GAMEPAD_MODE}"
  --gamepad_host "${GAMEPAD_HOST}"
  --gamepad_port "${GAMEPAD_PORT}"
  --use_wandb
  --wandb_project "${WANDB_PROJECT}"
  --wandb_mode "${WANDB_MODE_VALUE}"
  --wandb_group "${WANDB_GROUP}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --log_interval "${LOG_INTERVAL}"
  --eval_interval "${EVAL_INTERVAL}"
  --num_eval_episodes "${NUM_EVAL_EPISODES}"
  --save_interval "${SAVE_INTERVAL}"
)

if [[ "${USE_INTERVENTION}" == "1" ]]; then
  cmd+=(--use_intervention)
fi
if [[ -n "${PRETRAIN_REPLAY_DATASET_PATH}" ]]; then
  cmd+=(
    --demo_dataset_path "${PRETRAIN_REPLAY_DATASET_PATH}"
    --demo_dataset_target replay
  )
fi

"${cmd[@]}"

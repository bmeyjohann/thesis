#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-fasttd3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-SafetyCarGoal2-v0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
LEARNING_STARTS="${LEARNING_STARTS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_EVERY="${UPDATE_EVERY:-1}"
UPDATES_PER_CYCLE="${UPDATES_PER_CYCLE:-1}"

RENDER_MODE="${RENDER_MODE:-none}"
SURFACE_MODE="${SURFACE_MODE:-default}"
CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-2.0}"
CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-2.0}"
STEP_PENALTY="${STEP_PENALTY:-0.0}"

ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-512}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-1024}"
NUM_CRITICS="${NUM_CRITICS:-2}"
ALPHA_MIN="${ALPHA_MIN:-5e-4}"
ALPHA_MAX="${ALPHA_MAX:-1.0}"

WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"
LOG_INTERVAL="${LOG_INTERVAL:-2000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-20000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"

BASE_TAG="${BASE_TAG:-safetycar_goal2_dense_native_alpha_floor_compare}"
TS="$(date +%Y%m%d_%H%M%S)"

echo "Starting sequential SafetyCar dense-vs-native comparison."
echo "Run 1: dense reward with alpha floor."
echo "Run 2: native env reward."
echo "Total timesteps per run: ${TOTAL_TIMESTEPS}"
echo "Alpha clamp: [${ALPHA_MIN}, ${ALPHA_MAX}]"
echo "Render mode: ${RENDER_MODE}"

run_train() {
  local reward_mode="$1"
  local exp_name="$2"
  local wandb_run_name="$3"

  conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" train_fast_sac_safetygym.py \
    --env_name "${ENV_NAME}" \
    --exp_name "${exp_name}" \
    --render_mode "${RENDER_MODE}" \
    --surface_mode "${SURFACE_MODE}" \
    --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}" \
    --car_force_scale "${CAR_FORCE_SCALE}" \
    --num_envs 1 \
    --total_timesteps "${TOTAL_TIMESTEPS}" \
    --learning_starts "${LEARNING_STARTS}" \
    --batch_size "${BATCH_SIZE}" \
    --update_every "${UPDATE_EVERY}" \
    --updates_per_cycle "${UPDATES_PER_CYCLE}" \
    --reward_mode "${reward_mode}" \
    --step_penalty "${STEP_PENALTY}" \
    --actor_hidden_dim "${ACTOR_HIDDEN_DIM}" \
    --critic_hidden_dim "${CRITIC_HIDDEN_DIM}" \
    --num_critics "${NUM_CRITICS}" \
    --alpha_min "${ALPHA_MIN}" \
    --alpha_max "${ALPHA_MAX}" \
    --pref_rank_weight 0.0 \
    --demo_sample_ratio 0.0 \
    --log_interval "${LOG_INTERVAL}" \
    --eval_interval "${CHECKPOINT_INTERVAL}" \
    --num_eval_episodes "${NUM_EVAL_EPISODES}" \
    --save_interval "${CHECKPOINT_INTERVAL}" \
    --viz_on_checkpoint \
    --viz_grid_resolution 24 \
    --viz_quiver_stride 4 \
    --viz_device cpu \
    --viz_headings_deg "0,90,180,270" \
    --viz_num_rollouts 3 \
    --eval_save_episode_plots \
    --eval_episode_plot_max_episodes 9 \
    --use_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_mode "${WANDB_MODE_VALUE}" \
    --wandb_run_name "${wandb_run_name}"
}

DENSE_EXP_NAME="${DENSE_EXP_NAME:-${BASE_TAG}_dense_alphafloor_${TS}}"
DENSE_WANDB_RUN_NAME="${DENSE_WANDB_RUN_NAME:-${DENSE_EXP_NAME}}"
NATIVE_EXP_NAME="${NATIVE_EXP_NAME:-${BASE_TAG}_native_alphafloor_${TS}}"
NATIVE_WANDB_RUN_NAME="${NATIVE_WANDB_RUN_NAME:-${NATIVE_EXP_NAME}}"

echo "===== Run 1 / 2: dense reward ====="
echo "exp_name=${DENSE_EXP_NAME}"
run_train "dense" "${DENSE_EXP_NAME}" "${DENSE_WANDB_RUN_NAME}"

echo "===== Run 2 / 2: native reward ====="
echo "exp_name=${NATIVE_EXP_NAME}"
run_train "native" "${NATIVE_EXP_NAME}" "${NATIVE_WANDB_RUN_NAME}"

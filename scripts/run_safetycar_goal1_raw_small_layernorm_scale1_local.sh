#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

for arg in "$@"; do
  case "$arg" in
    *=*) export "$arg" ;;
    *)
      echo "Unexpected argument '$arg'. Use KEY=VALUE overrides only." >&2
      exit 2
      ;;
  esac
done

CONDA_ENV="${CONDA_ENV:-fasttd3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-SafetyCarGoal1-v0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000}"
LEARNING_STARTS="${LEARNING_STARTS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_UPDATES="${NUM_UPDATES:-2}"
POLICY_FREQUENCY="${POLICY_FREQUENCY:-2}"
GAMMA="${GAMMA:-0.99}"
TAU="${TAU:-0.005}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-256}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-512}"
ALPHA_INIT="${ALPHA_INIT:-0.001}"
ALPHA_MIN="${ALPHA_MIN:-0.0}"
ALPHA_MAX="${ALPHA_MAX:-1.0}"
REWARD_MODE="${REWARD_MODE:-dense}"
STEP_PENALTY="${STEP_PENALTY:--0.001}"
SURFACE_MODE="${SURFACE_MODE:-default}"
CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-1.0}"
CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-1.0}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
SEED="${SEED:-1}"

WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"
WANDB_GROUP="${WANDB_GROUP:-safetygym-goal1-scale1}"
TS="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-safetycar_goal1_raw_small_layernorm_scale1_${TS}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}}"

echo "Starting SafetyCar Goal1 small LayerNorm scale=1 baseline."
echo "Env: ${ENV_NAME}"
echo "Total timesteps: ${TOTAL_TIMESTEPS}"
echo "Action scale: wheel_limit=${CAR_WHEEL_COMMAND_LIMIT}, force_scale=${CAR_FORCE_SCALE}"
echo "Reward mode / step penalty: ${REWARD_MODE} / ${STEP_PENALTY}"
echo "Actor/critic dims: ${ACTOR_HIDDEN_DIM}/${CRITIC_HIDDEN_DIM}"
echo "Updates / policy freq: ${NUM_UPDATES}/${POLICY_FREQUENCY}"

exec conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" "${ROOT}/train_fast_sac_safetygym_minimal.py" \
  --env_name "${ENV_NAME}" \
  --exp_name "${EXP_NAME}" \
  --variant plain \
  --seed "${SEED}" \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --learning_starts "${LEARNING_STARTS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_updates "${NUM_UPDATES}" \
  --policy_frequency "${POLICY_FREQUENCY}" \
  --gamma "${GAMMA}" \
  --tau "${TAU}" \
  --actor_learning_rate "${ACTOR_LR}" \
  --critic_learning_rate "${CRITIC_LR}" \
  --actor_hidden_dim "${ACTOR_HIDDEN_DIM}" \
  --critic_hidden_dim "${CRITIC_HIDDEN_DIM}" \
  --module_impl custom \
  --use_layer_norm \
  --alpha_init "${ALPHA_INIT}" \
  --alpha_min "${ALPHA_MIN}" \
  --alpha_max "${ALPHA_MAX}" \
  --critic_loss_reduction sum \
  --obs_normalization \
  --reward_mode "${REWARD_MODE}" \
  --dense_reward_scale 1.0 \
  --step_penalty "${STEP_PENALTY}" \
  --surface_mode "${SURFACE_MODE}" \
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}" \
  --car_force_scale "${CAR_FORCE_SCALE}" \
  --car_action_mode raw_wheels \
  --scale_actor_to_env_bounds \
  --terminate_on_goal \
  --no_reseed_on_episode_reset \
  --pref_sample_ratio 0.0 \
  --pref_rank_weight 0.0 \
  --demo_sample_ratio 0.0 \
  --render_mode none \
  --log_interval "${LOG_INTERVAL}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES}" \
  --save_interval "${SAVE_INTERVAL}" \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_mode "${WANDB_MODE_VALUE}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --wandb_group "${WANDB_GROUP}"

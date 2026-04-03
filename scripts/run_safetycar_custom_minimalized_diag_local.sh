#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

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

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-thesis-safetygym}"

EXP_PREFIX="${EXP_PREFIX:-safetycar_custom_minimalized_diag}"
EXP_NAME="${EXP_NAME:-${EXP_PREFIX}_${TIMESTAMP}}"

GAMMA="${GAMMA:-0.99}"
STEP_PENALTY="${STEP_PENALTY:--0.001}"
LEARNING_STARTS="${LEARNING_STARTS:-5000}"
UPDATES_PER_CYCLE="${UPDATES_PER_CYCLE:-2}"
POLICY_FREQUENCY="${POLICY_FREQUENCY:-2}"
CRITIC_LOSS_REDUCTION="${CRITIC_LOSS_REDUCTION:-sum}"
ALPHA_INIT="${ALPHA_INIT:-1e-3}"
ALPHA_MIN="${ALPHA_MIN:-0.0}"
ALPHA_MAX="${ALPHA_MAX:-1.0}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-512}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-1024}"
CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-2.0}"
CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-2.0}"
PREF_CAPACITY="${PREF_CAPACITY:-100000}"
PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-0.5}"
RESEED_ON_EPISODE_RESET="${RESEED_ON_EPISODE_RESET:-1}"
UNCERTAINTY_LOG_EVERY_STEP="${UNCERTAINTY_LOG_EVERY_STEP:-1}"

OBS_NORM_FLAG="--obs_normalization"
if [[ "${OBS_NORMALIZATION:-1}" == "0" ]]; then
  OBS_NORM_FLAG="--no_obs_normalization"
fi

SCALE_FLAG="--scale_actor_to_env_bounds"
if [[ "${SCALE_ACTOR_TO_ENV_BOUNDS:-1}" == "0" ]]; then
  SCALE_FLAG="--no_scale_actor_to_env_bounds"
fi

RESEED_FLAG="--reseed_on_episode_reset"
if [[ "$RESEED_ON_EPISODE_RESET" == "0" ]]; then
  RESEED_FLAG="--no_reseed_on_episode_reset"
fi

UNCERTAINTY_FLAG="--uncertainty_log_every_step"
if [[ "$UNCERTAINTY_LOG_EVERY_STEP" == "0" ]]; then
  UNCERTAINTY_FLAG="--no_uncertainty_log_every_step"
fi

/home/benjamin/miniconda3/envs/fasttd3/bin/python /home/benjamin/thesis/train_fast_sac_safetygym.py \
  --env_name SafetyCarGoal2-v0 \
  --exp_name "$EXP_NAME" \
  --render_mode none \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --learning_starts "$LEARNING_STARTS" \
  --batch_size 64 \
  --update_every 1 \
  --updates_per_cycle "$UPDATES_PER_CYCLE" \
  --policy_frequency "$POLICY_FREQUENCY" \
  --gamma "$GAMMA" \
  --tau 0.005 \
  --actor_learning_rate 3e-4 \
  --critic_learning_rate 3e-4 \
  --actor_hidden_dim "$ACTOR_HIDDEN_DIM" \
  --critic_hidden_dim "$CRITIC_HIDDEN_DIM" \
  --num_critics 2 \
  --init_scale 0.01 \
  --max_grad_norm 10.0 \
  --alpha_init "$ALPHA_INIT" \
  --alpha_min "$ALPHA_MIN" \
  --alpha_max "$ALPHA_MAX" \
  --critic_loss_reduction "$CRITIC_LOSS_REDUCTION" \
  "$OBS_NORM_FLAG" \
  --reward_mode dense \
  --step_penalty "$STEP_PENALTY" \
  --car_wheel_command_limit "$CAR_WHEEL_COMMAND_LIMIT" \
  --car_force_scale "$CAR_FORCE_SCALE" \
  "$SCALE_FLAG" \
  "$RESEED_FLAG" \
  --pref_capacity "$PREF_CAPACITY" \
  --pref_sample_ratio "$PREF_SAMPLE_RATIO" \
  --pref_rank_weight 0.0 \
  --demo_sample_ratio 0.0 \
  "$UNCERTAINTY_FLAG" \
  --uncertainty_oversight_mode off \
  --log_interval 2000 \
  --eval_interval "$CHECKPOINT_INTERVAL" \
  --num_eval_episodes 10 \
  --save_interval "$CHECKPOINT_INTERVAL" \
  --viz_on_checkpoint \
  --eval_save_episode_plots \
  --eval_episode_plot_max_episodes 9 \
  --use_wandb \
  --wandb_project "$PROJECT" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_run_name "$EXP_NAME"

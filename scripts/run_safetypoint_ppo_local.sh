#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-thesis}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$ROOT/logs/.cache/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$ROOT/logs/.config/wandb}"
mkdir -p "$MPLCONFIGDIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

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
ENV_NAME="${ENV_NAME:-SafetyPointGoal1-v0}"
REWARD_MODE="${REWARD_MODE:-dense_plus_sparse}"
SURFACE_MODE="${SURFACE_MODE:-default}"
EXP_PREFIX="${EXP_PREFIX:-safetypoint_${ENV_NAME//-/_}_${REWARD_MODE}_${SURFACE_MODE}_ppo}"
EXP_NAME="${EXP_NAME:-${EXP_PREFIX}_${TIMESTAMP}}"
WANDB_GROUP="${WANDB_GROUP:-safetypoint_point_diag_${TIMESTAMP}}"

echo "Configured Safety-Gym Point PPO training"
echo "repo:       $ROOT"
echo "run:        $EXP_NAME"
echo "group:      $WANDB_GROUP"
echo "env:        $ENV_NAME"
echo "num envs:   ${NUM_ENVS:-16} (${VEC_ENV:-subproc})"
echo "surface:    $SURFACE_MODE"
echo "car force:  scale=${CAR_FORCE_SCALE:-1.0}, wheel_limit=${CAR_WHEEL_COMMAND_LIMIT:-1.0}"
echo "point mode: ${POINT_ACTION_MODE:-native}"
echo "layout:     ${FIXED_LAYOUT_PRESET:-none}"
echo "curriculum: ${LAYOUT_CURRICULUM:-none} level=${LAYOUT_CURRICULUM_LEVEL:-0}"
echo "reward:     $REWARD_MODE"
echo "success:    ${SUCCESS_REWARD_SCALE:-1.0}"
echo "clearance:  mode=${CLEARANCE_PENALTY_MODE:-softplus}, margin=${CLEARANCE_MARGIN:-0.0}, scale=${CLEARANCE_PENALTY_SCALE:-0.0}, temp=${CLEARANCE_PENALTY_TEMPERATURE:-0.001}"
echo "step pen:   ${STEP_PENALTY:--0.001}"
echo "terminate:  goal=${TERMINATE_ON_GOAL:-1}, cost=${TERMINATE_ON_COST:-0}"

TERMINATE_FLAG=""
if [[ "${TERMINATE_ON_GOAL:-1}" == "1" ]]; then
  TERMINATE_FLAG="--terminate_on_goal"
fi
if [[ "${TERMINATE_ON_COST:-0}" == "1" ]]; then
  TERMINATE_FLAG="${TERMINATE_FLAG} --terminate_on_cost"
fi

NORMALIZE_FLAG="--normalize_obs"
if [[ "${NORMALIZE_OBS:-1}" == "0" ]]; then
  NORMALIZE_FLAG="--no_normalize_obs"
fi
if [[ "${NORMALIZE_REWARD:-0}" == "1" ]]; then
  NORMALIZE_FLAG="${NORMALIZE_FLAG} --normalize_reward"
fi

/home/benjamin/miniconda3/envs/fasttd3/bin/python /home/benjamin/thesis/train_ppo_safetygym_minimal.py \
  --env_name "$ENV_NAME" \
  --exp_name "$EXP_NAME" \
  --seed "${SEED:-1}" \
  --device "${DEVICE:-auto}" \
  --total_timesteps "${TOTAL_TIMESTEPS:-200000}" \
  --num_envs "${NUM_ENVS:-16}" \
  --vec_env "${VEC_ENV:-subproc}" \
  --n_steps "${PPO_N_STEPS:-512}" \
  --batch_size "${PPO_BATCH_SIZE:-1024}" \
  --n_epochs "${PPO_N_EPOCHS:-10}" \
  --gamma "${GAMMA:-0.99}" \
  --gae_lambda "${GAE_LAMBDA:-0.95}" \
  --learning_rate "${PPO_LEARNING_RATE:-3e-4}" \
  --clip_range "${PPO_CLIP_RANGE:-0.2}" \
  --ent_coef "${PPO_ENT_COEF:-0.0}" \
  --vf_coef "${PPO_VF_COEF:-0.5}" \
  --max_grad_norm "${PPO_MAX_GRAD_NORM:-0.5}" \
  --net_arch "${PPO_NET_ARCH:-256,256}" \
  --activation_fn "${PPO_ACTIVATION_FN:-tanh}" \
  --init_ppo_model_path "${INIT_PPO_MODEL_PATH:-}" \
  ${RESET_PPO_OPTIMIZER:+--reset_ppo_optimizer} \
  $NORMALIZE_FLAG \
  --render_mode "${RENDER_MODE:-none}" \
  --surface_mode "$SURFACE_MODE" \
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT:-1.0}" \
  --car_force_scale "${CAR_FORCE_SCALE:-1.0}" \
  --point_action_mode "${POINT_ACTION_MODE:-native}" \
  --point_turn_gain "${POINT_TURN_GAIN:-2.5}" \
  --point_alignment_power "${POINT_ALIGNMENT_POWER:-1.0}" \
  --fixed_layout_preset "${FIXED_LAYOUT_PRESET:-none}" \
  --layout_curriculum "${LAYOUT_CURRICULUM:-none}" \
  --layout_curriculum_level "${LAYOUT_CURRICULUM_LEVEL:-0}" \
  --max_episode_steps "${MAX_EPISODE_STEPS:-0}" \
  $TERMINATE_FLAG \
  --reward_mode "$REWARD_MODE" \
  --dense_reward_scale "${DENSE_REWARD_SCALE:-1.0}" \
  --success_reward_scale "${SUCCESS_REWARD_SCALE:-1.0}" \
  --step_penalty "${STEP_PENALTY:--0.001}" \
  --cost_penalty "${COST_PENALTY:-0.0}" \
  --clearance_penalty_mode "${CLEARANCE_PENALTY_MODE:-softplus}" \
  --clearance_margin "${CLEARANCE_MARGIN:-0.0}" \
  --clearance_penalty_scale "${CLEARANCE_PENALTY_SCALE:-0.0}" \
  --clearance_penalty_temperature "${CLEARANCE_PENALTY_TEMPERATURE:-0.001}" \
  --save_interval "${CHECKPOINT_INTERVAL:-10000}" \
  --log_interval "${LOG_INTERVAL:-2048}" \
  --eval_interval "${EVAL_INTERVAL:-0}" \
  --eval_episodes "${EVAL_EPISODES:-10}" \
  --eval_fixed_layout_preset "${EVAL_FIXED_LAYOUT_PRESET:-train}" \
  --eval_layout_curriculum "${EVAL_LAYOUT_CURRICULUM:-train}" \
  --eval_layout_curriculum_level "${EVAL_LAYOUT_CURRICULUM_LEVEL:--1}" \
  ${EVAL_SAVE_PLOTS:+--eval_save_plots} \
  --eval_plot_max_episodes "${EVAL_PLOT_MAX_EPISODES:-6}" \
  ${EVAL_REWARD_SURFACE:+--eval_reward_surface} \
  --use_wandb \
  --wandb_project "${PROJECT:-thesis-safetygym}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_group "$WANDB_GROUP" \
  --wandb_run_name "$EXP_NAME"

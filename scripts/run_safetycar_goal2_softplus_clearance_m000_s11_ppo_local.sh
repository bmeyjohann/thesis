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
EXP_NAME="${EXP_NAME:-safetycar_goal2_softplus_clearance_m000_s11_ppo_${TIMESTAMP}}"
WANDB_GROUP="${WANDB_GROUP:-safetycar_goal2_softplus_clearance_m000_s11_${TIMESTAMP}}"

echo "Configured Safety-Gym Goal2 PPO softplus-clearance training"
echo "repo:       $ROOT"
echo "run:        $EXP_NAME"
echo "group:      $WANDB_GROUP"
echo "env:        ${ENV_NAME:-SafetyCarGoal2-v0}"
echo "num envs:   ${NUM_ENVS:-8} (${VEC_ENV:-subproc})"
echo "reward:     ${REWARD_MODE:-dense_plus_sparse} + softplus clearance"
echo "clearance:  margin=${CLEARANCE_MARGIN:-0.0}, scale=${CLEARANCE_PENALTY_SCALE:-1.1}, temp=${CLEARANCE_PENALTY_TEMPERATURE:-0.001}"
echo "step pen:   ${STEP_PENALTY:--0.001}"

/home/benjamin/miniconda3/envs/fasttd3/bin/python /home/benjamin/thesis/train_ppo_safetygym_minimal.py \
  --env_name "${ENV_NAME:-SafetyCarGoal2-v0}" \
  --exp_name "$EXP_NAME" \
  --seed "${SEED:-1}" \
  --device "${DEVICE:-auto}" \
  --total_timesteps "${TOTAL_TIMESTEPS:-200000}" \
  --num_envs "${NUM_ENVS:-8}" \
  --vec_env "${VEC_ENV:-subproc}" \
  --n_steps "${PPO_N_STEPS:-512}" \
  --batch_size "${PPO_BATCH_SIZE:-512}" \
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
  --render_mode "${RENDER_MODE:-none}" \
  --surface_mode "${SURFACE_MODE:-default}" \
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT:-1.0}" \
  --car_force_scale "${CAR_FORCE_SCALE:-1.0}" \
  --car_action_mode "${CAR_ACTION_MODE:-raw_wheels}" \
  --obs_mask_mode "${OBS_MASK_MODE:-none}" \
  --max_episode_steps "${MAX_EPISODE_STEPS:-0}" \
  --reward_mode "${REWARD_MODE:-dense_plus_sparse}" \
  --dense_reward_scale "${DENSE_REWARD_SCALE:-1.0}" \
  --step_penalty "${STEP_PENALTY:--0.001}" \
  --cost_penalty "${COST_PENALTY:-0.0}" \
  --clearance_penalty_mode "${CLEARANCE_PENALTY_MODE:-softplus}" \
  --clearance_margin "${CLEARANCE_MARGIN:-0.0}" \
  --clearance_penalty_scale "${CLEARANCE_PENALTY_SCALE:-1.1}" \
  --clearance_penalty_temperature "${CLEARANCE_PENALTY_TEMPERATURE:-0.001}" \
  --save_interval "${CHECKPOINT_INTERVAL:-10000}" \
  --log_interval "${LOG_INTERVAL:-2048}" \
  --use_wandb \
  --wandb_project "${PROJECT:-thesis-safetygym}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_group "$WANDB_GROUP" \
  --wandb_run_name "$EXP_NAME"

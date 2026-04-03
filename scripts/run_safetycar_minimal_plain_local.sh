#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-safetycar_goal2_minimal_plain_${TIMESTAMP}}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
WANDB_MODE="${WANDB_MODE:-online}"

conda run -n fasttd3 python /home/benjamin/thesis/train_fast_sac_safetygym_minimal.py \
  --env_name SafetyCarGoal2-v0 \
  --exp_name "$EXP_NAME" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --learning_starts 5000 \
  --batch_size 64 \
  --num_updates 2 \
  --policy_frequency 2 \
  --gamma 0.99 \
  --tau 0.005 \
  --actor_learning_rate 3e-4 \
  --critic_learning_rate 3e-4 \
  --actor_hidden_dim 512 \
  --critic_hidden_dim 1024 \
  --alpha_init 1e-3 \
  --alpha_min 5e-4 \
  --alpha_max 1.0 \
  --critic_loss_reduction sum \
  --reward_mode dense \
  --dense_reward_scale 1.0 \
  --step_penalty 0.0 \
  --surface_mode default \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --scale_actor_to_env_bounds \
  --obs_normalization \
  --use_wandb \
  --wandb_project thesis-safetygym \
  --wandb_mode "$WANDB_MODE" \
  --wandb_run_name "$EXP_NAME" \
  --wandb_group safetygym-minimal \
  --log_interval 2000 \
  --eval_interval 10000 \
  --num_eval_episodes 10 \
  --save_interval 50000

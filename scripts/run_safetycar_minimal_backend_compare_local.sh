#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-15000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-thesis-safetygym}"

run_case() {
  local name="$1"
  shift
  /home/benjamin/miniconda3/envs/fasttd3/bin/python /home/benjamin/thesis/train_fast_sac_safetygym_minimal.py \
    --env_name SafetyCarGoal2-v0 \
    --exp_name "${name}_${TIMESTAMP}" \
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
    --init_scale 0.01 \
    --max_grad_norm 10.0 \
    --alpha_init 1e-3 \
    --alpha_min 0.0 \
    --alpha_max 1.0 \
    --critic_loss_reduction sum \
    --reward_mode dense \
    --step_penalty -0.001 \
    --car_wheel_command_limit 2.0 \
    --car_force_scale 2.0 \
    --scale_actor_to_env_bounds \
    --log_interval 2000 \
    --eval_interval "$CHECKPOINT_INTERVAL" \
    --num_eval_episodes 10 \
    --save_interval "$CHECKPOINT_INTERVAL" \
    --use_wandb \
    --wandb_project "$PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "${name}_${TIMESTAMP}" \
    "$@"
}

run_case safetycar_min_backend_fastsac_obsnorm \
  --module_impl fastsac \
  --obs_normalization

run_case safetycar_min_backend_custom_obsnorm \
  --module_impl custom \
  --obs_normalization

run_case safetycar_min_backend_custom_noobsnorm \
  --module_impl custom \
  --no_obs_normalization

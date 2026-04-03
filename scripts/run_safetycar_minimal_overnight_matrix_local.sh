#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-thesis-safetygym}"
GROUP="${GROUP:-safetygym-minimal-overnight-${TIMESTAMP}}"

run_case() {
  local name="$1"
  shift
  conda run -n fasttd3 python /home/benjamin/thesis/train_fast_sac_safetygym_minimal.py \
    --env_name SafetyCarGoal2-v0 \
    --exp_name "${name}_${TIMESTAMP}" \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --learning_starts 5000 \
    --batch_size 64 \
    --gamma 0.99 \
    --tau 0.005 \
    --actor_learning_rate 3e-4 \
    --critic_learning_rate 3e-4 \
    --actor_hidden_dim 512 \
    --critic_hidden_dim 1024 \
    --log_interval 2000 \
    --eval_interval 10000 \
    --num_eval_episodes 10 \
    --save_interval 50000 \
    --use_wandb \
    --wandb_project "$PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "${name}_${TIMESTAMP}" \
    --wandb_group "$GROUP" \
    "$@"
}

# 1. Plain dense baseline with old FastSAC-style updates and safe action range.
run_case safetycar_min_dense_pf2_u2_a05e4_w1_f1 \
  --num_updates 2 \
  --policy_frequency 2 \
  --alpha_init 1e-3 \
  --alpha_min 5e-4 \
  --alpha_max 1.0 \
  --critic_loss_reduction sum \
  --reward_mode dense \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --scale_actor_to_env_bounds \
  --obs_normalization

# 2. Same as 1, but car dynamics restored to the stronger 2.0 / 2.0 setting.
run_case safetycar_min_dense_pf2_u2_a05e4_w2_f2 \
  --num_updates 2 \
  --policy_frequency 2 \
  --alpha_init 1e-3 \
  --alpha_min 5e-4 \
  --alpha_max 1.0 \
  --critic_loss_reduction sum \
  --reward_mode dense \
  --car_wheel_command_limit 2.0 \
  --car_force_scale 2.0 \
  --scale_actor_to_env_bounds \
  --obs_normalization

# 3. Dense+Sparse reward with otherwise same minimal setup.
run_case safetycar_min_denseplus_pf2_u2_a05e4_w2_f2 \
  --num_updates 2 \
  --policy_frequency 2 \
  --alpha_init 1e-3 \
  --alpha_min 5e-4 \
  --alpha_max 1.0 \
  --critic_loss_reduction sum \
  --reward_mode dense_plus_sparse \
  --car_wheel_command_limit 2.0 \
  --car_force_scale 2.0 \
  --scale_actor_to_env_bounds \
  --obs_normalization

# 4. Native env reward under the same update regime.
run_case safetycar_min_native_pf2_u2_a05e4_w2_f2 \
  --num_updates 2 \
  --policy_frequency 2 \
  --alpha_init 1e-3 \
  --alpha_min 5e-4 \
  --alpha_max 1.0 \
  --critic_loss_reduction sum \
  --reward_mode native \
  --car_wheel_command_limit 2.0 \
  --car_force_scale 2.0 \
  --scale_actor_to_env_bounds \
  --obs_normalization

# 5. Dense reward, but no alpha floor.
run_case safetycar_min_dense_pf2_u2_a0_w2_f2 \
  --num_updates 2 \
  --policy_frequency 2 \
  --alpha_init 1e-3 \
  --alpha_min 0.0 \
  --alpha_max 1.0 \
  --critic_loss_reduction sum \
  --reward_mode dense \
  --car_wheel_command_limit 2.0 \
  --car_force_scale 2.0 \
  --scale_actor_to_env_bounds \
  --obs_normalization

# 6. Dense reward, actor not rescaled to the widened env bounds.
run_case safetycar_min_dense_pf2_u2_a05e4_w2_f2_noscale \
  --num_updates 2 \
  --policy_frequency 2 \
  --alpha_init 1e-3 \
  --alpha_min 5e-4 \
  --alpha_max 1.0 \
  --critic_loss_reduction sum \
  --reward_mode dense \
  --car_wheel_command_limit 2.0 \
  --car_force_scale 2.0 \
  --no_scale_actor_to_env_bounds \
  --obs_normalization

# 7. Dense reward, mean critic-loss reduction to mimic the current refactor more closely.
run_case safetycar_min_dense_pf2_u2_a05e4_w2_f2_meanloss \
  --num_updates 2 \
  --policy_frequency 2 \
  --alpha_init 1e-3 \
  --alpha_min 5e-4 \
  --alpha_max 1.0 \
  --critic_loss_reduction mean \
  --reward_mode dense \
  --car_wheel_command_limit 2.0 \
  --car_force_scale 2.0 \
  --scale_actor_to_env_bounds \
  --obs_normalization

# 8. Dense reward, current-style actor schedule (no delay, one update).
run_case safetycar_min_dense_pf1_u1_a05e4_w2_f2 \
  --num_updates 1 \
  --policy_frequency 1 \
  --alpha_init 1e-3 \
  --alpha_min 5e-4 \
  --alpha_max 1.0 \
  --critic_loss_reduction sum \
  --reward_mode dense \
  --car_wheel_command_limit 2.0 \
  --car_force_scale 2.0 \
  --scale_actor_to_env_bounds \
  --obs_normalization

# 9. Dense reward, same as 2 but without observation normalization.
run_case safetycar_min_dense_pf2_u2_a05e4_w2_f2_noobsnorm \
  --num_updates 2 \
  --policy_frequency 2 \
  --alpha_init 1e-3 \
  --alpha_min 5e-4 \
  --alpha_max 1.0 \
  --critic_loss_reduction sum \
  --reward_mode dense \
  --car_wheel_command_limit 2.0 \
  --car_force_scale 2.0 \
  --scale_actor_to_env_bounds \
  --no_obs_normalization

# 10. Dense+Sparse with current-style actor schedule, to check whether the reward alone rescues it.
run_case safetycar_min_denseplus_pf1_u1_a05e4_w2_f2 \
  --num_updates 1 \
  --policy_frequency 1 \
  --alpha_init 1e-3 \
  --alpha_min 5e-4 \
  --alpha_max 1.0 \
  --critic_loss_reduction sum \
  --reward_mode dense_plus_sparse \
  --car_wheel_command_limit 2.0 \
  --car_force_scale 2.0 \
  --scale_actor_to_env_bounds \
  --obs_normalization

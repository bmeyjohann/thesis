#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-25000}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-thesis-safetygym}"
GROUP="${GROUP:-safetygym-minimal-gamma-step-${TIMESTAMP}}"

run_case() {
  local name="$1"
  shift
  conda run -n fasttd3 python /home/benjamin/thesis/train_fast_sac_safetygym_minimal.py \
    --env_name SafetyCarGoal2-v0 \
    --exp_name "${name}_${TIMESTAMP}" \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --learning_starts 5000 \
    --batch_size 64 \
    --num_updates 2 \
    --policy_frequency 2 \
    --actor_learning_rate 3e-4 \
    --critic_learning_rate 3e-4 \
    --actor_hidden_dim 512 \
    --critic_hidden_dim 1024 \
    --alpha_init 1e-3 \
    --alpha_min 0.0 \
    --alpha_max 1.0 \
    --critic_loss_reduction sum \
    --reward_mode dense \
    --surface_mode default \
    --car_wheel_command_limit 2.0 \
    --car_force_scale 2.0 \
    --scale_actor_to_env_bounds \
    --obs_normalization \
    --log_interval 2000 \
    --eval_interval 5000 \
    --num_eval_episodes 10 \
    --save_interval 25000 \
    --use_wandb \
    --wandb_project "$PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "${name}_${TIMESTAMP}" \
    --wandb_group "$GROUP" \
    "$@"
}

# 6. Combined shorter horizon plus small step penalty.
run_case safetycar_min_dense_g097_spm001 \
  --gamma 0.97 \
  --step_penalty -0.001

# 2. Slightly shorter-horizon baseline.
run_case safetycar_min_dense_g097_sp0 \
  --gamma 0.97 \
  --step_penalty 0.0

# 4. Small directness penalty, original gamma.
run_case safetycar_min_dense_g099_spm001 \
  --gamma 0.99 \
  --step_penalty -0.001

# 5. Slightly larger directness penalty, original gamma.
run_case safetycar_min_dense_g099_spm002 \
  --gamma 0.99 \
  --step_penalty -0.002

# 3. Aggressively shorter-horizon baseline.
run_case safetycar_min_dense_g095_sp0 \
  --gamma 0.95 \
  --step_penalty 0.0

# 1. Best current dense baseline.
run_case safetycar_min_dense_g099_sp0 \
  --gamma 0.99 \
  --step_penalty 0.0

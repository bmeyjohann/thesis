#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-thesis-safetygym}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"

run_case() {
  local name="$1"
  shift
  /home/benjamin/miniconda3/envs/fasttd3/bin/python /home/benjamin/thesis/train_fast_sac_safetygym.py \
    --env_name SafetyCarGoal2-v0 \
    --exp_name "${name}_${TIMESTAMP}" \
    --render_mode none \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --learning_starts 5000 \
    --batch_size 64 \
    --update_every 1 \
    --updates_per_cycle 1 \
    --gamma 0.99 \
    --tau 0.005 \
    --actor_learning_rate 3e-4 \
    --critic_learning_rate 3e-4 \
    --num_critics 2 \
    --reward_mode dense \
    --car_wheel_command_limit 2.0 \
    --car_force_scale 2.0 \
    --pref_rank_weight 0.0 \
    --demo_sample_ratio 0.0 \
    --log_interval 2000 \
    --eval_interval "$CHECKPOINT_INTERVAL" \
    --num_eval_episodes 10 \
    --save_interval "$CHECKPOINT_INTERVAL" \
    --use_wandb \
    --wandb_project "$PROJECT" \
    --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "${name}_${TIMESTAMP}" \
    --viz_on_checkpoint \
    --eval_save_episode_plots \
    --eval_episode_plot_max_episodes 9 \
    "$@"
}

# 1. Main custom trainer with old-style capacity and observation normalization enabled.
run_case safetycar_custom_dense_oldstyle_obsnorm \
  --actor_hidden_dim 512 \
  --critic_hidden_dim 1024 \
  --obs_normalization \
  --alpha_min 0.0 \
  --alpha_max 1.0

# 2. Same custom trainer/settings, but explicitly without observation normalization.
run_case safetycar_custom_dense_oldstyle_noobsnorm \
  --actor_hidden_dim 512 \
  --critic_hidden_dim 1024 \
  --no_obs_normalization \
  --alpha_min 0.0 \
  --alpha_max 1.0

# 3. Smaller current-capacity custom trainer with observation normalization enabled.
run_case safetycar_custom_dense_small_obsnorm \
  --actor_hidden_dim 256 \
  --critic_hidden_dim 512 \
  --obs_normalization \
  --alpha_min 0.0 \
  --alpha_max 1.0

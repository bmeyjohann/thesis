#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-safetycar_ppo_warm50k_random_safety_stabilize_${TIMESTAMP}}"
INIT="${INIT:-/home/benjamin/thesis/models/safetygym_ppo/safetycar_goal1_ppo_random_safety_ft_normfix_snapshot_20260514/ppo_step_50000_steps.zip}"

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python /home/benjamin/thesis/train_ppo_safetygym_minimal.py \
  --env_name SafetyCarGoal1-v0 \
  --exp_name "${EXP_NAME}" \
  --seed "${SEED:-3}" \
  --device auto \
  --total_timesteps "${TOTAL_TIMESTEPS:-150000}" \
  --num_envs "${NUM_ENVS:-16}" \
  --vec_env subproc \
  --n_steps 512 \
  --batch_size 1024 \
  --n_epochs 5 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --learning_rate 2e-05 \
  --clip_range 0.03 \
  --ent_coef 0.0 \
  --vf_coef 0.5 \
  --max_grad_norm 0.5 \
  --net_arch 256,256 \
  --activation_fn tanh \
  --init_ppo_model_path "${INIT}" \
  --reset_ppo_optimizer \
  --normalize_obs \
  --render_mode none \
  --surface_mode default \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --car_action_mode raw_wheels \
  --obs_mask_mode none \
  --max_episode_steps 0 \
  --terminate_on_goal \
  --terminate_on_cost \
  --reward_mode potential_diff \
  --dense_reward_scale 1.0 \
  --success_reward_scale 12.0 \
  --step_penalty 0.0 \
  --cost_penalty 0.0 \
  --clearance_penalty_mode softplus \
  --clearance_margin 0.10 \
  --clearance_penalty_scale 3.0 \
  --clearance_penalty_temperature 0.001 \
  --save_interval 25000 \
  --log_interval 2048 \
  --eval_interval 25000 \
  --eval_episodes 32 \
  --eval_save_plots \
  --eval_plot_max_episodes 12 \
  --eval_reward_surface \
  --use_wandb \
  --wandb_project thesis-safetygym \
  --wandb_mode online \
  --wandb_group safetycar_curriculum_20260515 \
  --wandb_run_name "${EXP_NAME}"

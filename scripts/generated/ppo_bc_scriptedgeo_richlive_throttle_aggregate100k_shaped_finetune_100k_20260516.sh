#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

export WANDB_MODE="online"
export WANDB_PROJECT="thesis-safetygym"
export MUJOCO_GL="egl"

RUN_NAME="safetycar_ppo_bc_scriptedgeo_richlive_throttle_agg100k_shaped_ft_100k_20260516"
INIT_MODEL="/home/benjamin/thesis/models/safetygym_ppo_bc/safetycar_ppo_bc_scriptedgeo_richlive_throttle_aggregate100k_20260516/best.zip"

/home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/train_ppo_safetygym_minimal.py \
  --env_name SafetyCarGoal1-v0 \
  --exp_name "${RUN_NAME}" \
  --seed 424242 \
  --device cuda \
  --total_timesteps 100000 \
  --num_envs 16 \
  --vec_env subproc \
  --n_steps 512 \
  --batch_size 1024 \
  --n_epochs 8 \
  --gamma 0.995 \
  --gae_lambda 0.95 \
  --learning_rate 1e-4 \
  --clip_range 0.15 \
  --ent_coef 0.005 \
  --vf_coef 0.5 \
  --max_grad_norm 0.5 \
  --net_arch 256,256 \
  --activation_fn tanh \
  --init_ppo_model_path "${INIT_MODEL}" \
  --reset_ppo_optimizer \
  --normalize_obs \
  --render_mode none \
  --surface_mode default \
  --car_action_mode throttle_turn \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --obs_mask_mode privileged_geometry_rich \
  --layout_curriculum car_random_blocked_filter \
  --layout_curriculum_level 0 \
  --terminate_on_goal \
  --terminate_on_cost \
  --reward_mode dense_plus_sparse \
  --dense_reward_scale 1.0 \
  --success_reward_scale 5.0 \
  --step_penalty -0.001 \
  --cost_penalty 0.0 \
  --clearance_penalty_scale 1.1 \
  --clearance_margin 0.0 \
  --clearance_penalty_mode softplus \
  --clearance_penalty_temperature 0.001 \
  --save_interval 25000 \
  --log_interval 4096 \
  --eval_interval 25000 \
  --eval_episodes 64 \
  --eval_save_plots \
  --eval_plot_max_episodes 16 \
  --eval_reward_surface \
  --use_wandb \
  --wandb_project thesis-safetygym \
  --wandb_mode online \
  --wandb_run_name "${RUN_NAME}" \
  --wandb_group "scriptedgeo_bc_rl_20260516"

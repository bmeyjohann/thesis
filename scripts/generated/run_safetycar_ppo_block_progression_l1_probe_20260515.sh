#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-safetycar_ppo_blockprog_l1_goalfirst_probe_${TIMESTAMP}}"

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python /home/benjamin/thesis/train_ppo_safetygym_minimal.py \
  --env_name SafetyCarGoal1-v0 \
  --exp_name "${EXP_NAME}" \
  --seed "${SEED:-1}" \
  --device auto \
  --total_timesteps "${TOTAL_TIMESTEPS:-200000}" \
  --num_envs "${NUM_ENVS:-8}" \
  --vec_env subproc \
  --n_steps 512 \
  --batch_size 512 \
  --n_epochs 10 \
  --gamma 0.995 \
  --gae_lambda 0.95 \
  --learning_rate 3e-4 \
  --clip_range 0.2 \
  --ent_coef 0.0 \
  --vf_coef 0.5 \
  --max_grad_norm 0.5 \
  --net_arch 256,256 \
  --activation_fn tanh \
  --normalize_obs \
  --render_mode none \
  --surface_mode default \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --car_action_mode raw_wheels \
  --obs_mask_mode none \
  --max_episode_steps 0 \
  --terminate_on_goal \
  --layout_curriculum car_block_progression \
  --layout_curriculum_level 1 \
  --eval_layout_curriculum train \
  --eval_layout_curriculum_level -1 \
  --reward_mode dense_plus_sparse \
  --dense_reward_scale 1.0 \
  --success_reward_scale 8.0 \
  --step_penalty -0.001 \
  --cost_penalty 0.0 \
  --clearance_penalty_mode softplus \
  --clearance_margin 0.0 \
  --clearance_penalty_scale 0.3 \
  --clearance_penalty_temperature 0.001 \
  --save_interval 50000 \
  --log_interval 4096 \
  --eval_interval 25000 \
  --eval_episodes 16 \
  --eval_save_plots \
  --eval_plot_max_episodes 8 \
  --use_wandb \
  --wandb_project thesis-safetygym \
  --wandb_mode online \
  --wandb_group safetycar_curriculum_20260515 \
  --wandb_run_name "${EXP_NAME}"

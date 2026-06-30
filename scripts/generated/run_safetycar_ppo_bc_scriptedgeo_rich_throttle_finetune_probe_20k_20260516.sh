#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
CKPT="/home/benjamin/thesis/models/safetygym_ppo_bc/safetycar_ppo_bc_scriptedgeo_rich_throttle_50k_20260516/best.zip"
for _ in $(seq 1 360); do
  if [[ -s "$CKPT" ]]; then break; fi
  echo "waiting for BC checkpoint: $CKPT"
  sleep 5
done
if [[ ! -s "$CKPT" ]]; then
  echo "ERROR: missing BC checkpoint: $CKPT" >&2
  exit 2
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"
export WANDB_MODE="${WANDB_MODE:-online}"
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/train_ppo_safetygym_minimal.py \
  --env_name SafetyCarGoal1-v0 \
  --exp_name safetycar_ppo_bc_scriptedgeo_rich_throttle_finetune_probe_20k_20260516 \
  --seed 888888 \
  --device cpu \
  --total_timesteps 20000 \
  --num_envs 2 \
  --vec_env dummy \
  --n_steps 256 \
  --batch_size 512 \
  --n_epochs 5 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --learning_rate 1e-4 \
  --clip_range 0.1 \
  --ent_coef 0.0 \
  --vf_coef 0.5 \
  --max_grad_norm 0.5 \
  --net_arch 512,512,256 \
  --activation_fn tanh \
  --init_ppo_model_path "$CKPT" \
  --reset_ppo_optimizer \
  --normalize_obs \
  --reward_mode dense_plus_sparse \
  --dense_reward_scale 1.0 \
  --success_reward_scale 5.0 \
  --step_penalty -0.001 \
  --clearance_penalty_mode softplus \
  --clearance_margin 0.0 \
  --clearance_penalty_scale 4.0 \
  --clearance_penalty_temperature 0.001 \
  --terminate_on_goal \
  --terminate_on_cost \
  --layout_curriculum car_random_blocked_filter \
  --layout_curriculum_level 0 \
  --obs_mask_mode privileged_geometry_rich \
  --car_action_mode throttle_turn \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --eval_interval 5000 \
  --eval_episodes 24 \
  --eval_save_plots \
  --eval_plot_max_episodes 12 \
  --eval_reward_surface \
  --save_interval 5000 \
  --use_wandb \
  --wandb_project thesis-safetygym \
  --wandb_mode "$WANDB_MODE"

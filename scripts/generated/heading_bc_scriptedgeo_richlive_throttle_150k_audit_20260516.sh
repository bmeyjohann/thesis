#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
export MPLCONFIGDIR=/tmp/matplotlib-thesis
export WANDB_MODE=offline
PY=/home/benjamin/miniconda3/envs/fasttd3/bin/python
DATA=/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_heading_richlive_throttle_150k_20260516.npz
OUT=/home/benjamin/thesis/models/safetygym_heading_bc/safetycar_heading_bc_scriptedgeo_richlive_throttle_150k_20260516
PLOTS=/home/benjamin/thesis/logs/safetygym_heading_bc/posthoc_audited/heading_bc_scriptedgeo_richlive_throttle_150k_seed424242_64ep_20260516

$PY /home/benjamin/thesis/collect_safetygym_heading_labels.py \
  --dataset_path "$DATA" \
  --env_name SafetyCarGoal1-v0 \
  --seed 424242 \
  --render_mode none \
  --car_action_mode throttle_turn \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --obs_mask_mode privileged_geometry_rich \
  --layout_curriculum car_random_blocked_filter \
  --layout_curriculum_level 0 \
  --reward_mode dense_plus_sparse \
  --success_reward_scale 5.0 \
  --step_penalty -0.001 \
  --clearance_penalty_scale 4.0 \
  --clearance_margin 0.0 \
  --clearance_penalty_mode softplus \
  --clearance_penalty_temperature 0.001 \
  --terminate_on_goal \
  --max_steps 150000 \
  --num_episodes 2000

$PY /home/benjamin/thesis/train_heading_bc_safetygym.py \
  --dataset_path "$DATA" \
  --output_dir "$OUT" \
  --seed 424242 \
  --device cuda \
  --epochs 180 \
  --batch_size 2048 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --hidden_dims 768,768,384 \
  --activation tanh \
  --heading_tolerance 0.20 \
  --forward_throttle 0.8

$PY /home/benjamin/thesis/eval_interactive_safetygym.py \
  --controller policy \
  --policy_format heading_bc \
  --model_path "$OUT/best.pt" \
  --env_name SafetyCarGoal1-v0 \
  --render_mode none \
  --seed 424242 \
  --num_episodes 64 \
  --fps 0 \
  --reward_mode dense_plus_sparse \
  --success_reward_scale 5.0 \
  --step_penalty -0.001 \
  --terminate_on_goal \
  --terminate_on_cost \
  --layout_curriculum car_random_blocked_filter \
  --layout_curriculum_level 0 \
  --obs_mask_mode privileged_geometry_rich \
  --car_action_mode throttle_turn \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --save_episode_plots \
  --episode_plot_dir "$PLOTS" \
  --episode_plot_max_episodes 64 \
  --episode_plot_reward_surface

#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
export MUJOCO_GL="egl"

/home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/eval_interactive_safetygym.py \
  --controller policy \
  --policy_format heading_bc \
  --model_path /home/benjamin/thesis/models/safetygym_heading_bc/safetycar_heading_bc_scriptedgeo_richlive_throttle_dagger_weighted450k_20260516/best.pt \
  --env_name SafetyCarGoal1-v0 \
  --render_mode none \
  --seed 424242 \
  --num_episodes 64 \
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
  --episode_plot_dir /home/benjamin/thesis/logs/safetygym_heading_bc/posthoc_audited/heading_bc_scriptedgeo_richlive_throttle_dagger_weighted450k_seed424242_64ep_20260516 \
  --episode_plot_max_episodes 16 \
  --episode_plot_reward_surface

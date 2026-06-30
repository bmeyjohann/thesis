#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/scripts/collect_safetygym_demos.py \
  --env_name SafetyCarGoal1-v0 \
  --seed 868686 \
  --render_mode none \
  --surface_mode default \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --car_action_mode throttle_turn \
  --obs_mask_mode privileged_geometry_rich \
  --reward_mode dense_plus_sparse \
  --dense_reward_scale 1.0 \
  --success_reward_scale 5.0 \
  --step_penalty -0.001 \
  --clearance_penalty_mode softplus \
  --clearance_margin 0.0 \
  --clearance_penalty_scale 4.0 \
  --clearance_penalty_temperature 0.001 \
  --terminate_on_goal \
  --layout_curriculum car_random_blocked_filter \
  --layout_curriculum_level 0 \
  --human_input_device scripted_geo \
  --num_episodes 100000 \
  --max_steps 50000 \
  --fps 0 \
  --dataset_path /home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_randomblocked_rich_throttle_50k_20260516.npz

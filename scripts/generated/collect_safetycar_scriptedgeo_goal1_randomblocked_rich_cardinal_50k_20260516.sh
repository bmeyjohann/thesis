#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
OUT="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_randomblocked_rich_cardinal_50k_20260516.npz"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/scripts/collect_safetygym_demos.py \
  --dataset_path "$OUT" \
  --human_input_device scripted_geo \
  --env_name SafetyCarGoal1-v0 \
  --seed 515151 \
  --render_mode none \
  --num_episodes 2000 \
  --max_steps 50000 \
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
  --obs_mask_mode privileged_geometry_rich \
  --car_action_mode cardinal \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0

#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

OUT="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_randomblocked_privgeom_70k_20260515.npz"
mkdir -p "$(dirname "$OUT")" /tmp/mpl

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/scripts/collect_safetygym_demos.py \
  --env_name SafetyCarGoal1-v0 \
  --seed 626262 \
  --render_mode none \
  --num_episodes 400 \
  --fps 0 \
  --human_input_device scripted_geo \
  --obs_mask_mode privileged_geometry \
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
  --car_action_mode raw_wheels \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --dataset_path "$OUT"

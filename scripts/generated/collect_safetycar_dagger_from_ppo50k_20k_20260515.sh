#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

MODEL_PATH="/home/benjamin/thesis/models/safetygym_ppo/safetycar_goal1_ppo_random_safety_ft_normfix_snapshot_20260514/ppo_step_50000_steps.zip"
DATASET_PATH="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_dagger_from_ppo50k_scriptedgeo_blocked_20k_20260515.npz"

if [[ ! -s "$MODEL_PATH" ]]; then
  echo "ERROR: PPO checkpoint not found: $MODEL_PATH" >&2
  exit 2
fi

/home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/scripts/collect_safetygym_dagger_labels.py \
  --model_path "$MODEL_PATH" \
  --student_policy ppo \
  --dataset_path "$DATASET_PATH" \
  --env_name SafetyCarGoal1-v0 \
  --seed 424242 \
  --device cpu \
  --render_mode none \
  --surface_mode default \
  --car_action_mode raw_wheels \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --obs_mask_mode none \
  --layout_curriculum car_random_blocked_filter \
  --layout_curriculum_level 0 \
  --fixed_layout_preset none \
  --reward_mode dense_plus_sparse \
  --dense_reward_scale 1.0 \
  --success_reward_scale 5.0 \
  --step_penalty -0.001 \
  --clearance_penalty_mode softplus \
  --clearance_margin 0.0 \
  --clearance_penalty_scale 4.0 \
  --clearance_penalty_temperature 0.001 \
  --terminate_on_goal \
  --num_episodes 1000 \
  --max_steps 20000

#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
DATASET_PATH="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_randomblocked_rich_throttle_50k_20260516.npz"
for _ in $(seq 1 360); do
  if [[ -s "$DATASET_PATH" ]]; then break; fi
  echo "waiting for dataset: $DATASET_PATH"
  sleep 5
done
if [[ ! -s "$DATASET_PATH" ]]; then
  echo "ERROR: dataset not found: $DATASET_PATH" >&2
  exit 2
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/train_ppo_bc_safetygym.py \
  --dataset_path "$DATASET_PATH" \
  --env_name SafetyCarGoal1-v0 \
  --exp_name safetycar_ppo_bc_scriptedgeo_rich_throttle_50k_20260516 \
  --seed 878787 \
  --device cpu \
  --epochs 80 \
  --batch_size 2048 \
  --learning_rate 3e-4 \
  --weight_decay 1e-5 \
  --net_arch 512,512,256 \
  --activation_fn tanh \
  --initial_log_std -2.5 \
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
  --car_force_scale 1.0

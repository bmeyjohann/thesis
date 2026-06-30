#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
EXPERT="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_randomblocked_richlive_throttle_50k_20260516.npz"
DAGGER="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_dagger_from_ppo_bc_scriptedgeo_blocked_richlive_throttle_50k_20260516.npz"
MERGED="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_plus_dagger_bc_blocked_richlive_throttle_100k_20260516.npz"
for _ in $(seq 1 720); do
  if [[ -s "$EXPERT" && -s "$DAGGER" ]]; then break; fi
  echo "waiting for datasets expert=$EXPERT dagger=$DAGGER"
  sleep 5
done
if [[ ! -s "$EXPERT" || ! -s "$DAGGER" ]]; then
  echo "ERROR: missing input datasets" >&2
  exit 2
fi
/home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/scripts/merge_safetygym_transition_datasets.py \
  --output "$MERGED" \
  --inputs "$EXPERT" "$DAGGER"
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/train_ppo_bc_safetygym.py \
  --dataset_path "$MERGED" \
  --env_name SafetyCarGoal1-v0 \
  --exp_name safetycar_ppo_bc_scriptedgeo_richlive_throttle_aggregate100k_20260516 \
  --seed 636363 \
  --device cpu \
  --epochs 90 \
  --batch_size 2048 \
  --learning_rate 2e-4 \
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

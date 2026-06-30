#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
CKPT="/home/benjamin/thesis/models/safetygym_ppo_bc/safetycar_ppo_bc_scriptedgeo_rich_throttle_50k_20260516/best.zip"
DATASET_PATH="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_dagger_from_ppo_bc_scriptedgeo_blocked_rich_throttle_50k_20260516.npz"
for _ in $(seq 1 360); do
  if [[ -s "$CKPT" && -s "$(dirname "$CKPT")/vecnormalize.pkl" ]]; then break; fi
  echo "waiting for PPO-BC checkpoint: $CKPT"
  sleep 5
done
if [[ ! -s "$CKPT" ]]; then
  echo "ERROR: missing checkpoint: $CKPT" >&2
  exit 2
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/scripts/collect_safetygym_dagger_labels.py \
  --model_path "$CKPT" \
  --student_policy ppo \
  --dataset_path "$DATASET_PATH" \
  --env_name SafetyCarGoal1-v0 \
  --seed 989898 \
  --device cpu \
  --render_mode none \
  --max_steps 50000 \
  --num_episodes 2000 \
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
  --car_action_mode throttle_turn \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0

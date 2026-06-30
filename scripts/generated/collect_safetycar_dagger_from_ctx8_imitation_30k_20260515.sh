#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

OUT="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_dagger_from_ctx8_imitation_privgeom_throttleturn_30k_20260515.npz"
MODEL="/home/benjamin/thesis/models/safetygym_imitation/safetycar_imitation_bc_throttleturn_privgeom_ctx8_70kdata_20260515/best.pt"
mkdir -p "$(dirname "$OUT")" /tmp/mpl

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/scripts/collect_safetygym_dagger_labels.py \
  --model_path "$MODEL" \
  --student_policy imitation \
  --student_intervention_threshold 0.0 \
  --dataset_path "$OUT" \
  --env_name SafetyCarGoal1-v0 \
  --seed 747474 \
  --device auto \
  --render_mode none \
  --max_steps 30000 \
  --num_episodes 1000 \
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
  --obs_mask_mode privileged_geometry \
  --car_action_mode throttle_turn \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0

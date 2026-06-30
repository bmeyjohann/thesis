#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

EPISODES="${EPISODES:-20}"
SEED="${SEED:-10065}"
PYTHON="${PYTHON:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/benjamin/thesis/models/safetygym_minimal/safetycar_goal1_goalonly_small_ln_pretrain_20260511/step_25000.pt}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" \
"${PYTHON}" /home/benjamin/thesis/eval_interactive_safetygym.py \
  --model_path "${MODEL_PATH}" \
  --controller policy \
  --policy_format fastsac \
  --intervention_mode human \
  --human_input_device scripted_geo \
  --render_mode none \
  --env_name SafetyCarGoal1-v0 \
  --seed "${SEED}" \
  --layout_curriculum car_random_blocked_filter \
  --terminate_on_goal \
  --reward_mode dense \
  --dense_reward_scale 1.0 \
  --success_reward_scale 0.0 \
  --step_penalty 0.0 \
  --clearance_penalty_scale 0.0 \
  --footprint_cost \
  --footprint_cost_mode visual \
  --footprint_cost_margin 0.0 \
  --footprint_cost_value 1.0 \
  --teacher_clearance_source visual_footprint \
  --teacher_override_mode clearance_projected_release \
  --teacher_override_clearance_threshold 0.50 \
  --teacher_override_clearance_exit_threshold 0.20 \
  --scripted_geo_lookahead 1.6 \
  --scripted_geo_safety_margin 0.30 \
  --scripted_geo_grid_resolution 0.05 \
  --scripted_geo_emergency_clearance 0.14 \
  --scripted_geo_action_shield_steps 0 \
  --car_action_mode raw_wheels \
  --scale_actor_to_env_bounds \
  --use_layer_norm \
  --actor_hidden_dim 256 \
  --load_checkpoint_args \
  --num_episodes "${EPISODES}" \
  --fps 0

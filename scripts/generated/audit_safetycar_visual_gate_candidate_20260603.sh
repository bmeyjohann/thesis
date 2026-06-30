#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

EPISODES="${EPISODES:-100}"
SEED="${SEED:-10065}"
PYTHON="${PYTHON:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/benjamin/thesis/models/safetygym_minimal/safetycar_goal1_goalonly_small_ln_pretrain_20260511/step_25000.pt}"
TEACHER_OVERRIDE_CLEARANCE_THRESHOLD="${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD:-0.45}"
TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD="${TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD:-0.50}"
TEACHER_MODE_OVERRIDE="${TEACHER_MODE_OVERRIDE:-clearance}"
HUMAN_INPUT_DEVICE="${HUMAN_INPUT_DEVICE:-scripted_geo}"
SCRIPTED_GEO_SAFETY_MARGIN="${SCRIPTED_GEO_SAFETY_MARGIN:-0.28}"
SCRIPTED_GEO_EMERGENCY_CLEARANCE="${SCRIPTED_GEO_EMERGENCY_CLEARANCE:-0.12}"
SCRIPTED_GEO_ACTION_SHIELD_STEPS="${SCRIPTED_GEO_ACTION_SHIELD_STEPS:-0}"
OUT_DIR="${OUT_DIR:-/home/benjamin/thesis/logs/safetygym_eval_audits/visual_gate_candidate_t${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD//./}_exit${TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD//./}_teacher_m${SCRIPTED_GEO_SAFETY_MARGIN//./}_directional_e${SCRIPTED_GEO_EMERGENCY_CLEARANCE//./}_shield${SCRIPTED_GEO_ACTION_SHIELD_STEPS}_${EPISODES}ep_20260604}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" \
"${PYTHON}" /home/benjamin/thesis/eval_interactive_safetygym.py \
  --model_path "${MODEL_PATH}" \
  --controller policy \
  --policy_format fastsac \
  --intervention_mode human \
  --human_input_device "${HUMAN_INPUT_DEVICE}" \
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
  --teacher_override_mode "${TEACHER_MODE_OVERRIDE}" \
  --teacher_override_clearance_threshold "${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD}" \
  --teacher_override_clearance_exit_threshold "${TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD}" \
  --scripted_geo_lookahead 1.6 \
  --scripted_geo_safety_margin "${SCRIPTED_GEO_SAFETY_MARGIN}" \
  --scripted_geo_grid_resolution 0.05 \
  --scripted_geo_emergency_clearance "${SCRIPTED_GEO_EMERGENCY_CLEARANCE}" \
  --scripted_geo_action_shield_steps "${SCRIPTED_GEO_ACTION_SHIELD_STEPS}" \
  --car_action_mode raw_wheels \
  --scale_actor_to_env_bounds \
  --use_layer_norm \
  --actor_hidden_dim 256 \
  --load_checkpoint_args \
  --num_episodes "${EPISODES}" \
  --fps 0 \
  --save_episode_plots \
  --episode_plot_dir "${OUT_DIR}" \
  --episode_plot_max_episodes "${EPISODES}" \
  --episode_plot_reward_surface

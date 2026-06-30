#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:?label required}"
MODEL_PATH="${2:?model checkpoint path required}"
ROOT="/home/benjamin/thesis"
PY="${PY:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/logs/safetygym_eval_audits/flowsec_${LABEL}_100ep_20260603}"
SEED="${SEED:-1}"
EPISODES="${EPISODES:-100}"

cd "$ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

"$PY" "$ROOT/eval_interactive_safetygym.py" \
  --model_path "$MODEL_PATH" \
  --controller policy \
  --policy_format fastsac \
  --render_mode none \
  --env_name SafetyCarGoal1-v0 \
  --seed "$SEED" \
  --layout_curriculum car_random_blocked_filter \
  --terminate_on_goal \
  --reward_mode dense \
  --dense_reward_scale 1.0 \
  --success_reward_scale 0.0 \
  --step_penalty 0.0 \
  --clearance_penalty_scale 0.0 \
  --car_action_mode raw_wheels \
  --scale_actor_to_env_bounds \
  --use_layer_norm \
  --actor_hidden_dim 256 \
  --device cpu \
  --num_episodes "$EPISODES" \
  --fps 30 \
  --save_episode_plots \
  --episode_plot_dir "$OUT_DIR" \
  --episode_plot_max_episodes 25 \
  --episode_plot_reward_surface

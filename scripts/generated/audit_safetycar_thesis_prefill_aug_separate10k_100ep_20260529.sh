#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
PY="${PY:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
MODEL="$ROOT/models/safetygym_minimal/safetycar_goal1_thesis_scriptedgeo_reward_120000_seed23_20260529_051701_clearance_or_progress_prefill_aug_separate/step_10000.pt"
OUT="$ROOT/logs/safetygym_eval_audits/thesis_prefill_aug_separate10k_100ep_20260529"

cd "$ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"

exec "$PY" "$ROOT/eval_interactive_safetygym.py" \
  --model_path "$MODEL" \
  --controller policy \
  --render_mode none \
  --env_name SafetyCarGoal1-v0 \
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
  --load_checkpoint_args \
  --num_episodes 100 \
  --save_episode_plots \
  --episode_plot_dir "$OUT" \
  --episode_plot_max_episodes 25 \
  --episode_plot_reward_surface

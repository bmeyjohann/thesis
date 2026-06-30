#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

OUT_DIR="/home/benjamin/thesis/logs/safetygym_minimal/posthoc_audited/scriptedgeo_goal1_randomblocked_seed424242_100ep_allplots"
mkdir -p "$OUT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/eval_interactive_safetygym.py \
  --controller scripted_geo \
  --env_name SafetyCarGoal1-v0 \
  --render_mode none \
  --seed 424242 \
  --num_episodes 100 \
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
  --scale_actor_to_env_bounds \
  --save_episode_plots \
  --episode_plot_dir "$OUT_DIR" \
  --episode_plot_max_episodes 100 \
  2>&1 | tee "$OUT_DIR/eval_stdout.log"

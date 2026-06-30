#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

CKPT="/home/benjamin/thesis/models/safetygym_imitation/safetycar_imitation_bc_scriptedgeo_rich_raw_70kdata_20260515/best.pt"
OUT_DIR="/home/benjamin/thesis/logs/safetygym_minimal/posthoc_audited/imitation_bc_scriptedgeo_rich_raw_seed424242_100ep_20260515"
for _ in $(seq 1 180); do
  if [[ -s "$CKPT" ]]; then
    break
  fi
  echo "waiting for checkpoint: $CKPT"
  sleep 5
done
if [[ ! -s "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/eval_interactive_safetygym.py \
  --controller imitation \
  --imitation_checkpoint_path "$CKPT" \
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
  --terminate_on_cost \
  --layout_curriculum car_random_blocked_filter \
  --layout_curriculum_level 0 \
  --obs_mask_mode privileged_geometry_rich \
  --car_action_mode raw_wheels \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --scale_actor_to_env_bounds \
  --intervention_threshold 0.0 \
  --save_episode_plots \
  --episode_plot_reward_surface \
  --episode_plot_surface_resolution 140 \
  --episode_plot_dir "$OUT_DIR" \
  --episode_plot_max_episodes 100 \
  2>&1 | tee "$OUT_DIR/eval_stdout.log"

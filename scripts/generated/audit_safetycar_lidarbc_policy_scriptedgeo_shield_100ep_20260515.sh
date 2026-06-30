#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

OUT_DIR="/home/benjamin/thesis/logs/safetygym_minimal/posthoc_audited/lidarbc_policy_scriptedgeo_studentfwd_shield_t0_seed626262_100ep_allplots"
MODEL="/home/benjamin/thesis/models/safetygym_minimal/safetycar_offline_astar_bc_h256_b512_lidarbc_60kdata_20260515/step_60000.pt"
mkdir -p "$OUT_DIR"
if [[ ! -s "$MODEL" ]]; then
  echo "ERROR: model not found: $MODEL" >&2
  exit 3
fi

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/eval_interactive_safetygym.py \
  --model_path "$MODEL" \
  --controller policy \
  --intervention_mode human \
  --human_input_device scripted_geo \
  --teacher_override_clearance_threshold 0.0 \
  --teacher_override_clearance_exit_threshold 0.15 \
  --teacher_override_mode student_forward_clearance \
  --teacher_goal_progress_steps 5 \
  --teacher_goal_progress_epsilon 0.001 \
  --render_mode none \
  --seed 626262 \
  --num_episodes 100 \
  --save_episode_plots \
  --episode_plot_dir "$OUT_DIR" \
  --episode_plot_max_episodes 100 \
  2>&1 | tee "$OUT_DIR/eval_stdout.log"

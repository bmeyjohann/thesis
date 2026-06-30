#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

OUT_DIR="/home/benjamin/thesis/logs/safetygym_minimal/posthoc_audited/lidarbc_step60000_policy_seed424242_100ep_allplots"
mkdir -p "$OUT_DIR"

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/eval_interactive_safetygym.py \
  --model_path /home/benjamin/thesis/models/safetygym_minimal/safetycar_offline_astar_bc_h256_b512_lidarbc_60kdata_20260515/step_60000.pt \
  --controller policy \
  --render_mode none \
  --seed 424242 \
  --num_episodes 100 \
  --save_episode_plots \
  --episode_plot_dir "$OUT_DIR" \
  --episode_plot_max_episodes 100 \
  2>&1 | tee "$OUT_DIR/eval_stdout.log"

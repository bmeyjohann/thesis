#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

OUT_DIR="/home/benjamin/thesis/logs/safetygym_minimal/posthoc_audited/lidar60k_mixedbc15_step10000_seed525151_100ep_allplots"
MODEL="/home/benjamin/thesis/models/safetygym_minimal/safetycar_lidar60k_mixedbc15_safety_termcost_finetune_15k_20260515/step_10000.pt"
mkdir -p "$OUT_DIR"
if [[ ! -s "$MODEL" ]]; then
  echo "ERROR: model not found: $MODEL" >&2
  exit 3
fi

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/eval_interactive_safetygym.py \
  --model_path "$MODEL" \
  --controller policy \
  --render_mode none \
  --seed 525151 \
  --num_episodes 100 \
  --save_episode_plots \
  --episode_plot_dir "$OUT_DIR" \
  --episode_plot_max_episodes 100 \
  2>&1 | tee "$OUT_DIR/eval_stdout.log"

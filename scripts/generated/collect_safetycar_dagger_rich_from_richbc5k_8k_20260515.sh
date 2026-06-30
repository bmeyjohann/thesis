#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

OUT="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_dagger_rich_from_richbc5k_8k_20260515.npz"
MODEL="/home/benjamin/thesis/models/safetygym_minimal/safetycar_offline_scriptedgeo_bc_throttleturn_rich_scratch_8kdata_20260515/step_5000.pt"
mkdir -p "$(dirname "$OUT")" /tmp/mpl

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/scripts/collect_safetygym_dagger_labels.py \
  --model_path "$MODEL" \
  --dataset_path "$OUT" \
  --env_name SafetyCarGoal1-v0 \
  --seed 737373 \
  --device auto \
  --render_mode none \
  --max_steps 8000 \
  --num_episodes 1000

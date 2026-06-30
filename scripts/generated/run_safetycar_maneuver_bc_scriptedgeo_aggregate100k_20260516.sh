#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
DATASET_PATH="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_plus_dagger_bc_blocked_rich_throttle_100k_20260516.npz"
if [[ ! -s "$DATASET_PATH" ]]; then
  echo "ERROR: dataset missing: $DATASET_PATH" >&2
  exit 2
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/train_maneuver_bc_safetygym.py \
  --dataset_path "$DATASET_PATH" \
  --exp_name safetycar_maneuver_bc_scriptedgeo_rich_throttle_aggregate100k_20260516 \
  --seed 929292 \
  --device cpu \
  --epochs 120 \
  --batch_size 2048 \
  --learning_rate 3e-4 \
  --weight_decay 1e-5 \
  --hidden_dims 512,512,256 \
  --activation tanh \
  --forward_turn_weight 0.25 \
  --forward_throttle 0.8

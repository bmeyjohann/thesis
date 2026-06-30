#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis
DATASET_PATH="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_plus_dagger_bc_blocked_rich_throttle_100k_20260516.npz"
if [[ ! -s "$DATASET_PATH" ]]; then
  echo "ERROR: dataset missing: $DATASET_PATH" >&2
  exit 2
fi
exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/build_knn_bc_safetygym.py \
  --dataset_path "$DATASET_PATH" \
  --exp_name safetycar_knn_bc_scriptedgeo_rich_throttle_aggregate100k_k3_20260516 \
  --k 3

#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

DATASET_PATH="/home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_randomblocked_rich_raw_70k_20260515.npz"
for _ in $(seq 1 360); do
  if [[ -s "$DATASET_PATH" ]]; then
    break
  fi
  echo "waiting for dataset: $DATASET_PATH"
  sleep 5
done
if [[ ! -s "$DATASET_PATH" ]]; then
  echo "ERROR: dataset not found: $DATASET_PATH" >&2
  exit 2
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/train_safetygym_human_imitation.py \
  --dataset_path "$DATASET_PATH" \
  --env_name SafetyCarGoal1-v0 \
  --output_dir /home/benjamin/thesis/models/safetygym_imitation \
  --exp_name safetycar_imitation_bc_scriptedgeo_rich_raw_70kdata_20260515 \
  --seed 737373 \
  --device cpu \
  --epochs 100 \
  --batch_size 2048 \
  --learning_rate 1e-4 \
  --weight_decay 1e-5 \
  --hidden_dim 768 \
  --num_layers 5 \
  --dropout 0.03 \
  --context_len 8 \
  --action_loss_weight 1.0 \
  --decision_loss_weight 0.0 \
  --non_intervention_action_weight 0.0 \
  --val_fraction 0.1 \
  --intervention_threshold 0.0 \
  --surface_mode default \
  --car_action_mode raw_wheels \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0

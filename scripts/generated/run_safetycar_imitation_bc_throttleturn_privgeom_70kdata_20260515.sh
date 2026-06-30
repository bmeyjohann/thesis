#!/usr/bin/env bash
set -euo pipefail
cd /home/benjamin/thesis

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"

exec /home/benjamin/miniconda3/envs/fasttd3/bin/python \
  /home/benjamin/thesis/train_safetygym_human_imitation.py \
  --dataset_path /home/benjamin/thesis/data/safetygym_datasets/safetycar_goal1_scriptedgeo_randomblocked_privgeom_throttleturn_70k_20260515.npz \
  --env_name SafetyCarGoal1-v0 \
  --output_dir /home/benjamin/thesis/models/safetygym_imitation \
  --exp_name safetycar_imitation_bc_throttleturn_privgeom_70kdata_20260515 \
  --seed 916 \
  --device auto \
  --epochs 120 \
  --batch_size 1024 \
  --learning_rate 3e-4 \
  --weight_decay 1e-5 \
  --hidden_dim 512 \
  --num_layers 4 \
  --dropout 0.0 \
  --context_len 1 \
  --action_loss_weight 1.0 \
  --decision_loss_weight 0.0 \
  --non_intervention_action_weight 0.0 \
  --val_fraction 0.1 \
  --intervention_threshold 0.0 \
  --surface_mode default \
  --car_action_mode throttle_turn \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0

#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-codex}}"
mkdir -p "$MPLCONFIGDIR"

for arg in "$@"; do
  case "$arg" in
    *=*)
      export "$arg"
      ;;
    *)
      echo "Unsupported argument: $arg" >&2
      exit 2
      ;;
  esac
done

DATASET_PATH="${DATASET_PATH:-}"
if [[ -z "$DATASET_PATH" ]]; then
  echo "Set DATASET_PATH=/path/to/exported_replay_dataset.npz" >&2
  exit 2
fi

python train_safetygym_human_imitation.py \
  --dataset_path "$DATASET_PATH" \
  --env_name "${ENV_NAME:-SafetyCarGoal2-v0}" \
  --output_dir "${OUTPUT_DIR:-models/safetygym_imitation}" \
  --exp_name "${EXP_NAME:-}" \
  --seed "${SEED:-1}" \
  --device "${DEVICE:-auto}" \
  --epochs "${EPOCHS:-50}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --learning_rate "${LEARNING_RATE:-3e-4}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --hidden_dim "${HIDDEN_DIM:-256}" \
  --num_layers "${NUM_LAYERS:-3}" \
  --dropout "${DROPOUT:-0.0}" \
  --architecture "${ARCHITECTURE:-mlp}" \
  --num_heads "${NUM_HEADS:-4}" \
  --context_len "${CONTEXT_LEN:-8}" \
  --action_loss_weight "${ACTION_LOSS_WEIGHT:-1.0}" \
  --decision_loss_weight "${DECISION_LOSS_WEIGHT:-1.0}" \
  --non_intervention_action_weight "${NON_INTERVENTION_ACTION_WEIGHT:-0.0}" \
  --val_fraction "${VAL_FRACTION:-0.1}" \
  --intervention_threshold "${INTERVENTION_THRESHOLD:-0.5}" \
  --surface_mode "${SURFACE_MODE:-default}" \
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT:-2.0}" \
  --car_force_scale "${CAR_FORCE_SCALE:-2.0}" \
  --car_action_mode "${CAR_ACTION_MODE:-raw_wheels}" \
  --point_action_mode "${POINT_ACTION_MODE:-native}" \
  --max_rows "${MAX_ROWS:-0}"

#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
LOOP_DIR="$ROOT/external/LOOP"
cd "$LOOP_DIR"

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

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
ENV_ID="${ENV_ID:-SafetyCarGoal1-v0}"
POLICY="${POLICY:-safeLOOP_ARC}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-50000}"
START_TIMESTEPS="${START_TIMESTEPS:-10000}"
EVAL_FREQ="${EVAL_FREQ:-3000}"
DYNAMICS_FREQ="${DYNAMICS_FREQ:-250}"
SEED="${SEED:-0}"
METRIC_WINDOW_EPISODES="${METRIC_WINDOW_EPISODES:-20}"
LIDAR_COST_THRESHOLD="${LIDAR_COST_THRESHOLD:-0.5}"
LIDAR_COST_SCALE="${LIDAR_COST_SCALE:-10.0}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym-loop}"
EXP_NAME="${EXP_NAME:-loop_${ENV_ID}_${POLICY}_${TIMESTAMP}}"
WANDB_GROUP="${WANDB_GROUP:-$EXP_NAME}"

export PYTHONPATH="$ROOT/safety-gymnasium:$LOOP_DIR:${PYTHONPATH:-}"

echo "Starting LOOP/SafeLOOP Safety-Gymnasium run"
echo "repo:       $LOOP_DIR"
echo "env:        $ENV_ID"
echo "policy:     $POLICY"
echo "steps:      $MAX_TIMESTEPS"
echo "wandb:      $WANDB_PROJECT / $WANDB_MODE"

/home/benjamin/miniconda3/envs/fasttd3/bin/python "$LOOP_DIR/train_loop_safety.py" \
  --policy "$POLICY" \
  --env "$ENV_ID" \
  --seed "$SEED" \
  --start_timesteps "$START_TIMESTEPS" \
  --eval_freq "$EVAL_FREQ" \
  --max_timesteps "$MAX_TIMESTEPS" \
  --dynamics_freq "$DYNAMICS_FREQ" \
  --exp_name "$EXP_NAME" \
  --config "${CONFIG_PATH:-configs/safety_config_gym.yml}" \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_group "$WANDB_GROUP" \
  --wandb_run_name "$EXP_NAME" \
  --metric_window_episodes "$METRIC_WINDOW_EPISODES" \
  --lidar_cost_threshold "$LIDAR_COST_THRESHOLD" \
  --lidar_cost_scale "$LIDAR_COST_SCALE"

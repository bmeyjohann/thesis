#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

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
ENV_NAME="${ENV_NAME:-SafetyPointGoal1-v0}"
EXP_PREFIX="${EXP_PREFIX:-safetygym_flow_${ENV_NAME//-/_}}"
EXP_NAME="${EXP_NAME:-${EXP_PREFIX}_${TIMESTAMP}}"
WANDB_GROUP="${WANDB_GROUP:-safetygym_flow_${TIMESTAMP}}"

echo "Configured Safety-Gym flow-matching chunk training"
echo "repo:       $ROOT"
echo "run:        $EXP_NAME"
echo "group:      $WANDB_GROUP"
echo "env:        $ENV_NAME"
echo "chunk len:  ${CHUNK_LEN:-8}"
echo "dataset:    ${DATASET_PATH:-${DATASET_STEPS:-20000} teacher steps}"
echo "train:      ${TRAIN_STEPS:-20000} updates"
echo "wandb:      ${WANDB_MODE:-online}"

export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$ROOT/logs/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$ROOT/logs/wandb_config}"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-safetygym-flow}" \
/home/benjamin/miniconda3/envs/fasttd3/bin/python /home/benjamin/thesis/train_flow_matching_safetygym.py \
  --env_name "$ENV_NAME" \
  --exp_name "$EXP_NAME" \
  --seed "${SEED:-1}" \
  --device "${DEVICE:-auto}" \
  --dataset_steps "${DATASET_STEPS:-20000}" \
  --dataset_path "${DATASET_PATH:-}" \
  --dataset_max_rows "${DATASET_MAX_ROWS:-0}" \
  --dataset_log_episodes "${DATASET_LOG_EPISODES:-10}" \
  --teacher_mode "${TEACHER_MODE:-goal}" \
  --teacher_clearance_scale "${TEACHER_CLEARANCE_SCALE:-1.5}" \
  --train_steps "${TRAIN_STEPS:-20000}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --chunk_len "${CHUNK_LEN:-8}" \
  --hidden_dim "${HIDDEN_DIM:-256}" \
  --depth "${DEPTH:-3}" \
  --learning_rate "${LEARNING_RATE:-3e-4}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --sample_steps "${SAMPLE_STEPS:-8}" \
  --eval_num_samples "${EVAL_NUM_SAMPLES:-8}" \
  --eval_chunk_selector "${EVAL_CHUNK_SELECTOR:-best_goal}" \
  --reward_mode "${REWARD_MODE:-dense_plus_sparse}" \
  --dense_reward_scale "${DENSE_REWARD_SCALE:-1.0}" \
  --success_reward_scale "${SUCCESS_REWARD_SCALE:-1.0}" \
  --step_penalty "${STEP_PENALTY:-0.0}" \
  --cost_penalty "${COST_PENALTY:-0.0}" \
  --clearance_penalty_scale "${CLEARANCE_PENALTY_SCALE:-0.0}" \
  --clearance_margin "${CLEARANCE_MARGIN:-0.0}" \
  --clearance_penalty_mode "${CLEARANCE_PENALTY_MODE:-hinge_power}" \
  --clearance_penalty_temperature "${CLEARANCE_PENALTY_TEMPERATURE:-0.08}" \
  --forward_reward_scale "${FORWARD_REWARD_SCALE:-0.0}" \
  --backward_penalty_scale "${BACKWARD_PENALTY_SCALE:-0.0}" \
  --heading_reward_scale "${HEADING_REWARD_SCALE:-0.0}" \
  --surface_mode "${SURFACE_MODE:-default}" \
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT:-2.0}" \
  --car_force_scale "${CAR_FORCE_SCALE:-2.0}" \
  --car_action_mode "${CAR_ACTION_MODE:-throttle_turn}" \
  --point_action_mode "${POINT_ACTION_MODE:-world_velocity}" \
  --point_turn_gain "${POINT_TURN_GAIN:-2.5}" \
  --point_alignment_power "${POINT_ALIGNMENT_POWER:-1.0}" \
  --obs_mask_mode "${OBS_MASK_MODE:-goal_only_lidar}" \
  --max_episode_steps "${MAX_EPISODE_STEPS:-250}" \
  --fixed_layout_preset "${FIXED_LAYOUT_PRESET:-none}" \
  --layout_curriculum "${LAYOUT_CURRICULUM:-none}" \
  --layout_curriculum_level "${LAYOUT_CURRICULUM_LEVEL:-0}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES:-10}" \
  --eval_save_episode_plots \
  --eval_episode_plot_max_episodes "${EVAL_EPISODE_PLOT_MAX_EPISODES:-9}" \
  --log_interval "${LOG_INTERVAL:-500}" \
  --eval_interval "${EVAL_INTERVAL:-2000}" \
  --save_interval "${SAVE_INTERVAL:-10000}" \
  --use_wandb \
  --wandb_project "${PROJECT:-thesis-safetygym}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_group "$WANDB_GROUP" \
  --wandb_run_name "$EXP_NAME"

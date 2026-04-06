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
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-25000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
WANDB_MODE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-thesis-safetygym}"
SEED="${SEED:-1}"
ENV_NAME="${ENV_NAME:-SafetyCarGoal2-v0}"

EXP_PREFIX="${EXP_PREFIX:-safetycar_min_human_ready}"
EXP_NAME="${EXP_NAME:-${EXP_PREFIX}_${TIMESTAMP}}"
VARIANT="${VARIANT:-plain}"

REWARD_MODE="${REWARD_MODE:-dense}"
STEP_PENALTY="${STEP_PENALTY:--0.001}"
COST_PENALTY="${COST_PENALTY:-0.0}"
COST_PENALTY_WARMUP_STEPS="${COST_PENALTY_WARMUP_STEPS:-0}"
COST_PENALTY_RAMP_STEPS="${COST_PENALTY_RAMP_STEPS:-0}"
CLEARANCE_PENALTY_SCALE="${CLEARANCE_PENALTY_SCALE:-0.0}"
CLEARANCE_MARGIN="${CLEARANCE_MARGIN:-0.0}"
CLEARANCE_PENALTY_POWER="${CLEARANCE_PENALTY_POWER:-1.0}"
CLEARANCE_PENALTY_WARMUP_STEPS="${CLEARANCE_PENALTY_WARMUP_STEPS:-0}"
CLEARANCE_PENALTY_RAMP_STEPS="${CLEARANCE_PENALTY_RAMP_STEPS:-0}"
FORWARD_REWARD_SCALE="${FORWARD_REWARD_SCALE:-0.0}"
BACKWARD_PENALTY_SCALE="${BACKWARD_PENALTY_SCALE:-0.0}"
HEADING_REWARD_SCALE="${HEADING_REWARD_SCALE:-0.0}"
LEARNING_STARTS="${LEARNING_STARTS:-5000}"
NUM_UPDATES="${NUM_UPDATES:-2}"
POLICY_FREQUENCY="${POLICY_FREQUENCY:-2}"
GAMMA="${GAMMA:-0.99}"
ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE:-3e-4}"
CRITIC_LEARNING_RATE="${CRITIC_LEARNING_RATE:-3e-4}"
ALPHA_INIT="${ALPHA_INIT:-1e-3}"
ALPHA_MIN="${ALPHA_MIN:-0.0}"
ALPHA_MAX="${ALPHA_MAX:-1.0}"
CRITIC_LOSS_REDUCTION="${CRITIC_LOSS_REDUCTION:-sum}"
ACTOR_HIDDEN_DIM="${ACTOR_HIDDEN_DIM:-512}"
CRITIC_HIDDEN_DIM="${CRITIC_HIDDEN_DIM:-1024}"
CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-2.0}"
CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-2.0}"
CAR_ACTION_MODE="${CAR_ACTION_MODE:-raw_wheels}"
OBS_MASK_MODE="${OBS_MASK_MODE:-none}"
MODULE_IMPL="${MODULE_IMPL:-custom}"
RENDER_MODE="${RENDER_MODE:-none}"
LOG_INTERVAL="${LOG_INTERVAL:-2000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"

OBS_NORM_FLAG="--obs_normalization"
if [[ "${OBS_NORMALIZATION:-1}" == "0" ]]; then
  OBS_NORM_FLAG="--no_obs_normalization"
fi

SCALE_FLAG="--scale_actor_to_env_bounds"
if [[ "${SCALE_ACTOR_TO_ENV_BOUNDS:-1}" == "0" ]]; then
  SCALE_FLAG="--no_scale_actor_to_env_bounds"
fi

INTERVENTION_FLAG=""
if [[ "${USE_INTERVENTION:-0}" == "1" ]]; then
  INTERVENTION_FLAG="--use_intervention"
fi

RESEED_FLAG="--no_reseed_on_episode_reset"
if [[ "${RESEED_ON_EPISODE_RESET:-0}" == "1" ]]; then
  RESEED_FLAG="--reseed_on_episode_reset"
fi

TERMINATE_ON_GOAL_FLAG=""
if [[ "${TERMINATE_ON_GOAL:-0}" == "1" ]]; then
  TERMINATE_ON_GOAL_FLAG="--terminate_on_goal"
fi

USE_LAYER_NORM_FLAG=""
if [[ "${USE_LAYER_NORM:-0}" == "1" ]]; then
  USE_LAYER_NORM_FLAG="--use_layer_norm"
fi

DEMO_DATASET_AUTO_LOAD_FLAG=""
if [[ "${DEMO_DATASET_AUTO_LOAD:-0}" == "1" ]]; then
  DEMO_DATASET_AUTO_LOAD_FLAG="--demo_dataset_auto_load"
fi

STORE_INTERVENED_FLAG=""
if [[ "${STORE_INTERVENED_IN_DEMO_BUFFER:-0}" == "1" ]]; then
  STORE_INTERVENED_FLAG="--store_intervened_in_demo_buffer"
fi

CRITIC_RESET_FLAG=""
if [[ "${CRITIC_RESET_AFTER_PRETRAIN:-0}" == "1" ]]; then
  CRITIC_RESET_FLAG="--critic_reset_after_pretrain"
fi

LOAD_ACTOR_FLAG="--load_actor_from_checkpoint"
if [[ "${LOAD_ACTOR_FROM_CHECKPOINT:-1}" == "0" ]]; then
  LOAD_ACTOR_FLAG="--no_load_actor_from_checkpoint"
fi

LOAD_CRITIC_FLAG="--load_critic_from_checkpoint"
if [[ "${LOAD_CRITIC_FROM_CHECKPOINT:-1}" == "0" ]]; then
  LOAD_CRITIC_FLAG="--no_load_critic_from_checkpoint"
fi

LOAD_TARGET_FLAG="--load_critic_target_from_checkpoint"
if [[ "${LOAD_CRITIC_TARGET_FROM_CHECKPOINT:-1}" == "0" ]]; then
  LOAD_TARGET_FLAG="--no_load_critic_target_from_checkpoint"
fi

LOAD_ALPHA_FLAG="--load_alpha_from_checkpoint"
if [[ "${LOAD_ALPHA_FROM_CHECKPOINT:-1}" == "0" ]]; then
  LOAD_ALPHA_FLAG="--no_load_alpha_from_checkpoint"
fi

LOAD_OPT_FLAG=""
if [[ "${LOAD_OPTIMIZER_STATE_FROM_CHECKPOINT:-0}" == "1" ]]; then
  LOAD_OPT_FLAG="--load_optimizer_state_from_checkpoint"
fi

SAVE_OPT_FLAG="--save_optimizer_state_in_checkpoints"
if [[ "${SAVE_OPTIMIZER_STATE_IN_CHECKPOINTS:-1}" == "0" ]]; then
  SAVE_OPT_FLAG="--no_save_optimizer_state_in_checkpoints"
fi

VIZ_FLAG=""
if [[ "${VIZ_ON_CHECKPOINT:-0}" == "1" ]]; then
  VIZ_FLAG="--viz_on_checkpoint"
fi

EVAL_PLOTS_FLAG=""
if [[ "${EVAL_SAVE_EPISODE_PLOTS:-0}" == "1" ]]; then
  EVAL_PLOTS_FLAG="--eval_save_episode_plots"
fi

EXPORT_REPLAY_FLAG=""
if [[ "${EXPORT_FINAL_REPLAY_DATASET:-0}" == "1" ]]; then
  EXPORT_REPLAY_FLAG="--export_final_replay_dataset"
fi

EXPORT_DEMO_FLAG=""
if [[ "${EXPORT_FINAL_DEMO_DATASET:-0}" == "1" ]]; then
  EXPORT_DEMO_FLAG="--export_final_demo_dataset"
fi

OFFLINE_ONLY_FLAG=""
if [[ "${OFFLINE_ONLY:-0}" == "1" ]]; then
  OFFLINE_ONLY_FLAG="--offline_only"
fi

PREF_STOPGRAD_FLAG=""
if [[ "${PREF_STOPGRAD_POSITIVE:-0}" == "1" ]]; then
  PREF_STOPGRAD_FLAG="--pref_stopgrad_positive"
fi

HEADING_POSITIVE_ONLY_FLAG="--heading_positive_only"
if [[ "${HEADING_POSITIVE_ONLY:-1}" == "0" ]]; then
  HEADING_POSITIVE_ONLY_FLAG="--no_heading_positive_only"
fi

/home/benjamin/miniconda3/envs/fasttd3/bin/python /home/benjamin/thesis/train_fast_sac_safetygym_minimal.py \
  --env_name "$ENV_NAME" \
  --exp_name "$EXP_NAME" \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --render_mode "$RENDER_MODE" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --learning_starts "$LEARNING_STARTS" \
  --batch_size 64 \
  --num_updates "$NUM_UPDATES" \
  --policy_frequency "$POLICY_FREQUENCY" \
  --gamma "$GAMMA" \
  --tau 0.005 \
  --actor_learning_rate "$ACTOR_LEARNING_RATE" \
  --critic_learning_rate "$CRITIC_LEARNING_RATE" \
  --actor_hidden_dim "$ACTOR_HIDDEN_DIM" \
  --critic_hidden_dim "$CRITIC_HIDDEN_DIM" \
  --module_impl "$MODULE_IMPL" \
  $USE_LAYER_NORM_FLAG \
  --init_scale 0.01 \
  --max_grad_norm 10.0 \
  --alpha_init "$ALPHA_INIT" \
  --alpha_min "$ALPHA_MIN" \
  --alpha_max "$ALPHA_MAX" \
  --critic_loss_reduction "$CRITIC_LOSS_REDUCTION" \
  "$OBS_NORM_FLAG" \
  --reward_mode "$REWARD_MODE" \
  --step_penalty "$STEP_PENALTY" \
  --cost_penalty "$COST_PENALTY" \
  --cost_penalty_warmup_steps "$COST_PENALTY_WARMUP_STEPS" \
  --cost_penalty_ramp_steps "$COST_PENALTY_RAMP_STEPS" \
  --clearance_penalty_scale "$CLEARANCE_PENALTY_SCALE" \
  --clearance_margin "$CLEARANCE_MARGIN" \
  --clearance_penalty_power "$CLEARANCE_PENALTY_POWER" \
  --clearance_penalty_warmup_steps "$CLEARANCE_PENALTY_WARMUP_STEPS" \
  --clearance_penalty_ramp_steps "$CLEARANCE_PENALTY_RAMP_STEPS" \
  --forward_reward_scale "$FORWARD_REWARD_SCALE" \
  --backward_penalty_scale "$BACKWARD_PENALTY_SCALE" \
  --heading_reward_scale "$HEADING_REWARD_SCALE" \
  "$HEADING_POSITIVE_ONLY_FLAG" \
  --car_wheel_command_limit "$CAR_WHEEL_COMMAND_LIMIT" \
  --car_force_scale "$CAR_FORCE_SCALE" \
  --car_action_mode "$CAR_ACTION_MODE" \
  --obs_mask_mode "$OBS_MASK_MODE" \
  --max_episode_steps "${MAX_EPISODE_STEPS:-0}" \
  $TERMINATE_ON_GOAL_FLAG \
  "$SCALE_FLAG" \
  "$RESEED_FLAG" \
  $INTERVENTION_FLAG \
  --human_input_device "${HUMAN_INPUT_DEVICE:-keyboard}" \
  --human_action_scale "${HUMAN_ACTION_SCALE:-1.0}" \
  --intervention_threshold "${INTERVENTION_THRESHOLD:-0.1}" \
  --intervention_hold_seconds "${INTERVENTION_HOLD_SECONDS:-0.25}" \
  --teacher_override_clearance_threshold "${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD:--1.0}" \
  --controller_fps_limit "${CONTROLLER_FPS_LIMIT:-0}" \
  --controller_overlay_hz "${CONTROLLER_OVERLAY_HZ:-20.0}" \
  --gamepad_mode "${GAMEPAD_MODE:-local}" \
  --gamepad_host "${GAMEPAD_HOST:-}" \
  --gamepad_port "${GAMEPAD_PORT:-0}" \
  --gamepad_cache_path "${GAMEPAD_CACHE_PATH:-}" \
  --gamepad_reconnect_seconds "${GAMEPAD_RECONNECT_SECONDS:-2.0}" \
  --gamepad_config_path "${GAMEPAD_CONFIG_PATH:-}" \
  --gamepad_device_index "${GAMEPAD_DEVICE_INDEX:-0}" \
  --expert_checkpoint_path "${EXPERT_CHECKPOINT_PATH:-}" \
  --expert_safe_checkpoint_path "${EXPERT_SAFE_CHECKPOINT_PATH:-}" \
  --expert_switch_clearance_threshold "${EXPERT_SWITCH_CLEARANCE_THRESHOLD:-0.08}" \
  --expert_device "${EXPERT_DEVICE:-cpu}" \
  --pref_capacity "${PREF_CAPACITY:-100000}" \
  --pref_sampling_mode "${PREF_SAMPLING_MODE:-linked}" \
  --pref_sample_ratio "${PREF_SAMPLE_RATIO:-0.0}" \
  --pref_rank_weight "${PREF_RANK_WEIGHT:-0.0}" \
  --pref_rank_margin "${PREF_RANK_MARGIN:-0.1}" \
  --pref_loss_type "${PREF_LOSS_TYPE:-margin}" \
  --pref_lambda_init "${PREF_LAMBDA_INIT:-1.0}" \
  --pref_lambda_lr "${PREF_LAMBDA_LR:-1e-3}" \
  --pref_lambda_max "${PREF_LAMBDA_MAX:-10.0}" \
  --pref_lambda_ema "${PREF_LAMBDA_EMA:-0.9}" \
  --pref_violation_clip "${PREF_VIOLATION_CLIP:-10.0}" \
  --pref_violation_target "${PREF_VIOLATION_TARGET:-0.0}" \
  --pref_lagrangian_violation_type "${PREF_LAGRANGIAN_VIOLATION_TYPE:-hinge}" \
  $PREF_STOPGRAD_FLAG \
  --demo_sample_ratio "${DEMO_SAMPLE_RATIO:-0.0}" \
  --prefill_demo_episodes "${PREFILL_DEMO_EPISODES:-0}" \
  --prefill_max_steps_per_episode "${PREFILL_MAX_STEPS_PER_EPISODE:-0}" \
  --prefill_policy "${PREFILL_POLICY:-student}" \
  $STORE_INTERVENED_FLAG \
  --demo_dataset_path "${DEMO_DATASET_PATH:-}" \
  --demo_dataset_dir "${DEMO_DATASET_DIR:-}" \
  $DEMO_DATASET_AUTO_LOAD_FLAG \
  --demo_dataset_target "${DEMO_DATASET_TARGET:-variant}" \
  --demo_dataset_max_rows "${DEMO_DATASET_MAX_ROWS:-0}" \
  --demo_pretrain_updates "${DEMO_PRETRAIN_UPDATES:-0}" \
  --demo_pretrain_batch_size "${DEMO_PRETRAIN_BATCH_SIZE:-0}" \
  $CRITIC_RESET_FLAG \
  --init_checkpoint_path "${INIT_CHECKPOINT_PATH:-}" \
  $LOAD_ACTOR_FLAG \
  $LOAD_CRITIC_FLAG \
  $LOAD_TARGET_FLAG \
  $LOAD_ALPHA_FLAG \
  $LOAD_OPT_FLAG \
  "$SAVE_OPT_FLAG" \
  --export_dataset_dir "${EXPORT_DATASET_DIR:-}" \
  --export_dataset_max_rows "${EXPORT_DATASET_MAX_ROWS:-0}" \
  $OFFLINE_ONLY_FLAG \
  --export_replay_dataset_interval "${EXPORT_REPLAY_DATASET_INTERVAL:-0}" \
  --export_replay_dataset_path "${EXPORT_REPLAY_DATASET_PATH:-}" \
  --export_replay_dataset_dir "${EXPORT_REPLAY_DATASET_DIR:-}" \
  --export_replay_dataset_label "${EXPORT_REPLAY_DATASET_LABEL:-online_replay}" \
  --export_replay_dataset_max_rows "${EXPORT_REPLAY_DATASET_MAX_ROWS:-0}" \
  $EXPORT_REPLAY_FLAG \
  --export_final_replay_dataset_path "${EXPORT_FINAL_REPLAY_DATASET_PATH:-}" \
  $EXPORT_DEMO_FLAG \
  --export_final_demo_dataset_path "${EXPORT_FINAL_DEMO_DATASET_PATH:-}" \
  --log_interval "$LOG_INTERVAL" \
  --eval_interval "$CHECKPOINT_INTERVAL" \
  --num_eval_episodes "$NUM_EVAL_EPISODES" \
  --save_interval "$CHECKPOINT_INTERVAL" \
  $VIZ_FLAG \
  --viz_grid_resolution "${VIZ_GRID_RESOLUTION:-48}" \
  --viz_quiver_stride "${VIZ_QUIVER_STRIDE:-4}" \
  --viz_device "${VIZ_DEVICE:-cpu}" \
  --viz_seed "${VIZ_SEED:-0}" \
  --viz_first_step "${VIZ_FIRST_STEP:-0}" \
  --viz_headings_deg "${VIZ_HEADINGS_DEG:-0,90,180,270}" \
  --viz_num_rollouts "${VIZ_NUM_ROLLOUTS:-4}" \
  $EVAL_PLOTS_FLAG \
  --eval_episode_plot_max_episodes "${EVAL_EPISODE_PLOT_MAX_EPISODES:-9}" \
  --use_wandb \
  --wandb_project "$PROJECT" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_mode "$WANDB_MODE" \
  --wandb_group "${WANDB_GROUP:-}" \
  --wandb_run_name "$EXP_NAME"

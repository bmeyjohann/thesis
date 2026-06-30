#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?mode required: scriptedgeo_reward|heading_reward|pcpo_reward|pcpo_value|pcpo_cost}"
STEPS="${STEPS:-60000}"
SEED="${SEED:-1}"
TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
ROOT="/home/benjamin/thesis"
cd "$ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"

if [[ "${NO_INIT_CHECKPOINT:-0}" == "1" ]]; then
  INIT_CKPT=""
else
  INIT_CKPT="${INIT_CKPT:-$ROOT/models/safetygym_minimal/safetycar_goal1_goalonly_small_ln_pretrain_20260511/step_25000.pt}"
fi
HEADING_CKPT="${HEADING_CKPT:-$ROOT/models/safetygym_heading_bc/safetycar_heading_bc_scriptedgeo_richlive_throttle_dagger200k_20260516/best.pt}"
PCPO_CKPT="${PCPO_CKPT:-$ROOT/models/safetygym_teachers/safe_rl/pcpo_model_599.pt}"
PCPO_CFG="${PCPO_CFG:-$ROOT/models/safetygym_teachers/safe_rl/pcpo_eval.yaml}"

HUMAN_INPUT_DEVICE="scripted_geo"
TEACHER_MODE="reward_progress"
EXPERT_FLAGS=()
case "$MODE" in
  scriptedgeo_reward)
    HUMAN_INPUT_DEVICE="scripted_geo"
    TEACHER_MODE="reward_progress"
    ;;
  heading_reward)
    HUMAN_INPUT_DEVICE="heading_bc"
    TEACHER_MODE="reward_progress"
    EXPERT_FLAGS=(--expert_checkpoint_path "$HEADING_CKPT")
    ;;
  pcpo_reward)
    HUMAN_INPUT_DEVICE="safe_rl"
    TEACHER_MODE="reward_progress"
    EXPERT_FLAGS=(--expert_checkpoint_path "$PCPO_CKPT" --expert_config_path "$PCPO_CFG")
    ;;
  pcpo_value)
    HUMAN_INPUT_DEVICE="safe_rl"
    TEACHER_MODE="pcpo_value_progress"
    EXPERT_FLAGS=(--expert_checkpoint_path "$PCPO_CKPT" --expert_config_path "$PCPO_CFG")
    ;;
  pcpo_cost)
    HUMAN_INPUT_DEVICE="safe_rl"
    TEACHER_MODE="pcpo_cost_value_progress"
    EXPERT_FLAGS=(--expert_checkpoint_path "$PCPO_CKPT" --expert_config_path "$PCPO_CFG")
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
HUMAN_INPUT_DEVICE="${HUMAN_INPUT_DEVICE_OVERRIDE:-$HUMAN_INPUT_DEVICE}"
TEACHER_MODE="${TEACHER_MODE_OVERRIDE:-$TEACHER_MODE}"

VIZ_FLAGS=()
if [[ "${DISABLE_POLICY_VIZ:-0}" != "1" ]]; then
  VIZ_FLAGS=(
    --viz_on_checkpoint
    --viz_grid_resolution 48
    --viz_quiver_stride 4
    --viz_device cpu
    --viz_num_rollouts 4
  )
fi

PREF_STOPGRAD_FLAGS=()
if [[ "${PREF_STOPGRAD_POSITIVE:-1}" != "0" ]]; then
  PREF_STOPGRAD_FLAGS=(--pref_stopgrad_positive)
fi

CHECKPOINT_LOAD_FLAGS=()
if [[ "${LOAD_CRITIC_FROM_CHECKPOINT:-0}" != "1" ]]; then
  CHECKPOINT_LOAD_FLAGS+=(--no_load_critic_from_checkpoint)
fi
if [[ "${LOAD_CRITIC_TARGET_FROM_CHECKPOINT:-0}" != "1" ]]; then
  CHECKPOINT_LOAD_FLAGS+=(--no_load_critic_target_from_checkpoint)
fi
if [[ "${LOAD_ALPHA_FROM_CHECKPOINT:-0}" != "1" ]]; then
  CHECKPOINT_LOAD_FLAGS+=(--no_load_alpha_from_checkpoint)
fi
if [[ "${LOAD_OPTIMIZER_STATE_FROM_CHECKPOINT:-0}" == "1" ]]; then
  CHECKPOINT_LOAD_FLAGS+=(--load_optimizer_state_from_checkpoint)
fi

FOOTPRINT_FLAGS=()
if [[ "${FOOTPRINT_COST:-0}" == "1" ]]; then
  FOOTPRINT_FLAGS=(
    --footprint_cost
    --footprint_cost_mode "${FOOTPRINT_COST_MODE:-visual}"
    --footprint_cost_margin "${FOOTPRINT_COST_MARGIN:-0.0}"
    --footprint_cost_value "${FOOTPRINT_COST_VALUE:-1.0}"
  )
fi

EXP_NAME="safetycar_goal1_thesis_${MODE}_${STEPS}_seed${SEED}_${TS}"

WANDB_FLAGS=()
if [[ "${DISABLE_WANDB:-0}" != "1" ]]; then
  WANDB_FLAGS=(
    --use_wandb
    --wandb_mode "${WANDB_MODE}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_group "${WANDB_GROUP:-safetycar_thesis_method_intervention_20260518}"
    --wandb_run_name "$EXP_NAME"
  )
fi

echo "[run] $EXP_NAME"
echo "[run] init=$INIT_CKPT teacher=$HUMAN_INPUT_DEVICE gate=$TEACHER_MODE steps=$STEPS seed=$SEED"

"$PY" train_fast_sac_safetygym_minimal.py \
  --env_name SafetyCarGoal1-v0 \
  --exp_name "$EXP_NAME" \
  --variant own \
  --seed "$SEED" \
  --device auto \
  --torch_num_threads 1 \
  --torch_num_interop_threads 1 \
  --total_timesteps "$STEPS" \
  --learning_starts 1000 \
  --batch_size 64 \
  --num_updates "${NUM_UPDATES:-2}" \
  --policy_frequency "${POLICY_FREQUENCY:-2}" \
  --actor_update_start_step "${ACTOR_UPDATE_START_STEP:-0}" \
  --buffer_size 1000000 \
  --gamma 0.99 \
  --tau 0.005 \
  --actor_learning_rate "${ACTOR_LEARNING_RATE:-0.0003}" \
  --critic_learning_rate "${CRITIC_LEARNING_RATE:-0.0003}" \
  --max_grad_norm "${MAX_GRAD_NORM:-10.0}" \
  --alpha_init "${ALPHA_INIT:-0.001}" \
  --alpha_min "${ALPHA_MIN:-0.0}" \
  --alpha_max "${ALPHA_MAX:-1.0}" \
  --module_impl custom \
  --actor_hidden_dim 256 \
  --critic_hidden_dim 512 \
  --use_layer_norm \
  --temporal_encoder "${TEMPORAL_ENCODER:-none}" \
  --obs_normalization \
  ${FREEZE_OBS_NORMALIZER_AFTER_LOAD:+--freeze_obs_normalizer_after_load} \
  --reward_mode "${REWARD_MODE:-potential_diff}" \
  --dense_reward_scale "${DENSE_REWARD_SCALE:-1.0}" \
  --success_reward_scale "${SUCCESS_REWARD_SCALE:-1.0}" \
  --step_penalty "${STEP_PENALTY:--0.001}" \
  --clearance_penalty_scale "${CLEARANCE_PENALTY_SCALE:-1.1}" \
  --clearance_margin "${CLEARANCE_MARGIN:-0.0}" \
  --clearance_penalty_mode softplus \
  --clearance_penalty_temperature 0.001 \
  "${FOOTPRINT_FLAGS[@]}" \
  --car_wheel_command_limit 2.0 \
  --car_force_scale 2.0 \
  --car_action_mode raw_wheels \
  --obs_mask_mode "${OBS_MASK_MODE:-none}" \
  --obs_frame_stack "${OBS_FRAME_STACK:-1}" \
  --layout_curriculum car_random_blocked_filter \
  --layout_seed_replay "${LAYOUT_SEED_REPLAY:-}" \
  --layout_seed_replay_prob "${LAYOUT_SEED_REPLAY_PROB:-0.0}" \
  --layout_seed_replay_mode "${LAYOUT_SEED_REPLAY_MODE:-cycle}" \
  --terminate_on_goal \
  --no_reseed_on_episode_reset \
  --scale_actor_to_env_bounds \
  --use_intervention \
  --human_input_device "$HUMAN_INPUT_DEVICE" \
  --scripted_geo_heading_tolerance "${SCRIPTED_GEO_HEADING_TOLERANCE:-0.20}" \
  --scripted_geo_lookahead "${SCRIPTED_GEO_LOOKAHEAD:-1.0}" \
  --scripted_geo_safety_margin "${SCRIPTED_GEO_SAFETY_MARGIN:-0.18}" \
  --scripted_geo_grid_resolution "${SCRIPTED_GEO_GRID_RESOLUTION:-0.08}" \
  --scripted_geo_emergency_clearance "${SCRIPTED_GEO_EMERGENCY_CLEARANCE:-0.08}" \
  --scripted_geo_action_shield_steps "${SCRIPTED_GEO_ACTION_SHIELD_STEPS:-1}" \
  "${EXPERT_FLAGS[@]}" \
  --expert_device cpu \
  --teacher_override_mode "$TEACHER_MODE" \
  --teacher_override_clearance_threshold "${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD:--1.0}" \
  --teacher_override_clearance_exit_threshold "${TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD:--1.0}" \
  --teacher_clearance_source "${TEACHER_CLEARANCE_SOURCE:-keepout}" \
  --teacher_progress_bad_steps "${TEACHER_PROGRESS_BAD_STEPS:-3}" \
  --teacher_progress_good_steps "${TEACHER_PROGRESS_GOOD_STEPS:-5}" \
  --teacher_progress_epsilon "${TEACHER_PROGRESS_EPSILON:-0.0001}" \
  --teacher_progress_trigger_mode "${TEACHER_PROGRESS_TRIGGER_MODE:-worse}" \
  --teacher_progress_release_mode "${TEACHER_PROGRESS_RELEASE_MODE:-improve}" \
  --teacher_progress_score_mode "${TEACHER_PROGRESS_SCORE_MODE:-reward_wrapper}" \
  --teacher_progress_dense_scale "${TEACHER_PROGRESS_DENSE_SCALE:-1.0}" \
  --teacher_progress_clearance_scale "${TEACHER_PROGRESS_CLEARANCE_SCALE:--1.0}" \
  --teacher_progress_clearance_margin "${TEACHER_PROGRESS_CLEARANCE_MARGIN:-0.0}" \
  --teacher_progress_clearance_mode "${TEACHER_PROGRESS_CLEARANCE_MODE:-softplus}" \
  --teacher_progress_clearance_temperature "${TEACHER_PROGRESS_CLEARANCE_TEMPERATURE:-0.001}" \
  --pref_capacity 100000 \
  --pref_sampling_mode "${PREF_SAMPLING_MODE:-linked}" \
  --pref_sample_ratio "${PREF_SAMPLE_RATIO:-0.5}" \
  --pref_rank_weight "${PREF_RANK_WEIGHT:-1.0}" \
  --pref_rank_margin "${PREF_RANK_MARGIN:-0.1}" \
  --pref_loss_type lagrangian \
  --pref_lambda_init "${PREF_LAMBDA_INIT:-1.0}" \
  --pref_lambda_lr "${PREF_LAMBDA_LR:-0.001}" \
  --pref_lambda_max "${PREF_LAMBDA_MAX:-10.0}" \
  --pref_lambda_ema 0.9 \
  --pref_violation_clip 10.0 \
  --pref_violation_target 0.0 \
  "${PREF_STOPGRAD_FLAGS[@]}" \
  --pref_obs_noise_std "${PREF_OBS_NOISE_STD:-0.0}" \
  --pref_action_noise_std "${PREF_ACTION_NOISE_STD:-0.0}" \
  --pref_action_noise_copies "${PREF_ACTION_NOISE_COPIES:-1}" \
  --pref_action_delta_min "${PREF_ACTION_DELTA_MIN:-0.0}" \
  --pref_action_delta_weight_scale "${PREF_ACTION_DELTA_WEIGHT_SCALE:-0.0}" \
  --pref_action_delta_weight_max "${PREF_ACTION_DELTA_WEIGHT_MAX:-10.0}" \
  --actor_bc_weight "${ACTOR_BC_WEIGHT:-0.0}" \
  --actor_reference_distill_weight "${ACTOR_REFERENCE_DISTILL_WEIGHT:-0.0}" \
  --actor_bc_only_until_step "${ACTOR_BC_ONLY_UNTIL_STEP:-0}" \
  --actor_bc_obstacle_lidar_weight_scale "${ACTOR_BC_OBSTACLE_LIDAR_WEIGHT_SCALE:-0.0}" \
  --actor_bc_goal_block_weight_scale "${ACTOR_BC_GOAL_BLOCK_WEIGHT_SCALE:-0.0}" \
  ${INTERVENTION_AUX_HEAD:+--intervention_aux_head} \
  --intervention_aux_weight "${INTERVENTION_AUX_WEIGHT:-0.0}" \
  --intervention_aux_pos_weight "${INTERVENTION_AUX_POS_WEIGHT:-0.0}" \
  --intervention_aux_pretrain_updates "${INTERVENTION_AUX_PRETRAIN_UPDATES:-0}" \
  --intervention_aux_pretrain_batch_size "${INTERVENTION_AUX_PRETRAIN_BATCH_SIZE:-0}" \
  --intervention_aux_pretrain_distill_weight "${INTERVENTION_AUX_PRETRAIN_DISTILL_WEIGHT:-1.0}" \
  --intervention_aux_pretrain_lr "${INTERVENTION_AUX_PRETRAIN_LR:-0.0}" \
  --demo_sample_ratio "${DEMO_SAMPLE_RATIO:-0.0}" \
  --prefill_demo_episodes "${PREFILL_DEMO_EPISODES:-0}" \
  --prefill_max_steps_per_episode "${PREFILL_MAX_STEPS_PER_EPISODE:-0}" \
  --prefill_policy "${PREFILL_POLICY:-student}" \
  ${STORE_INTERVENED_IN_DEMO_BUFFER:+--store_intervened_in_demo_buffer} \
  --demo_pretrain_updates "${DEMO_PRETRAIN_UPDATES:-0}" \
  --init_checkpoint_path "$INIT_CKPT" \
  "${CHECKPOINT_LOAD_FLAGS[@]}" \
  ${LOAD_OBS_NORMALIZER_FROM_CHECKPOINT:+--load_obs_normalizer_from_checkpoint} \
  ${NO_LOAD_OBS_NORMALIZER_FROM_CHECKPOINT:+--no_load_obs_normalizer_from_checkpoint} \
  "${WANDB_FLAGS[@]}" \
  --log_interval 1000 \
  --eval_interval "${EVAL_INTERVAL:-10000}" \
  --num_eval_episodes 20 \
  --eval_layout_seed_replay_prob "${EVAL_LAYOUT_SEED_REPLAY_PROB:--1.0}" \
  --save_interval "${SAVE_INTERVAL:-10000}" \
  "${VIZ_FLAGS[@]}" \
  --eval_save_episode_plots \
  --eval_episode_plot_max_episodes 9

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/benjamin/thesis}"
PY="${PY:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
METHOD="${METHOD:-pvp}"
INIT_MODE="${INIT_MODE:-goal_policy}" # goal_policy|scratch
STEPS="${STEPS:-30000}"
SEED="${SEED:-211}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym-competitors}"
WANDB_GROUP="${WANDB_GROUP:-safetycar_oldenv_competitors_20260611}"
if [[ "$INIT_MODE" == "scratch" || "${NO_INIT_CHECKPOINT:-0}" == "1" ]]; then
  INIT_CKPT=""
else
  INIT_CKPT="${INIT_CKPT:-$ROOT/models/safetygym_minimal/safetycar_goal1_goalonly_small_ln_pretrain_20260511/step_25000.pt}"
fi

cd "$ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig-safetygym-competitors}"
mkdir -p "$MPLCONFIGDIR"

case "$METHOD" in
  pvp)
    VARIANT="pvp"
    EXTRA=(
      --prefill_demo_episodes "${PREFILL_DEMO_EPISODES:-40}"
      --prefill_max_steps_per_episode "${PREFILL_MAX_STEPS_PER_EPISODE:-300}"
      --prefill_policy "${PREFILL_POLICY:-student}"
      --pvp_proxy_value_bound "${PVP_PROXY_VALUE_BOUND:-1.0}"
    )
    if [[ "${PVP_INCLUDE_ENV_REWARD_IN_TD:-0}" == "1" ]]; then
      EXTRA+=(--pvp_include_env_reward_in_td)
    fi
    ;;
  eil)
    VARIANT="eil"
    EXTRA=(
      --eil_threshold "${EIL_THRESHOLD:-0.0}"
      --eil_good_margin "${EIL_GOOD_MARGIN:-0.0}"
      --eil_bad_margin "${EIL_BAD_MARGIN:-0.01}"
      --eil_pair_margin "${EIL_PAIR_MARGIN:-0.01}"
      --eil_bad_pre_steps "${EIL_BAD_PRE_STEPS:-8}"
    )
    ;;
  hilserl|hil-serl)
    VARIANT="hilserl"
    EXTRA=(
      --demo_sample_ratio "${DEMO_SAMPLE_RATIO:-0.5}"
      --prefill_demo_episodes "${PREFILL_DEMO_EPISODES:-40}"
      --prefill_max_steps_per_episode "${PREFILL_MAX_STEPS_PER_EPISODE:-300}"
      --prefill_policy "${PREFILL_POLICY:-student}"
      --store_intervened_in_demo_buffer
      --actor_bc_weight "${ACTOR_BC_WEIGHT:-0.0}"
    )
    ;;
  bc|behavior_cloning)
    VARIANT="own"
    EXTRA=(
      --pref_sample_ratio 0.0
      --pref_rank_weight 0.0
      --pref_capacity 0
      --demo_sample_ratio "${DEMO_SAMPLE_RATIO:-0.5}"
      --prefill_demo_episodes "${PREFILL_DEMO_EPISODES:-40}"
      --prefill_max_steps_per_episode "${PREFILL_MAX_STEPS_PER_EPISODE:-300}"
      --prefill_policy "${PREFILL_POLICY:-student}"
      --store_intervened_in_demo_buffer
      --actor_bc_weight "${ACTOR_BC_WEIGHT:-1.0}"
      --actor_bc_only_until_step "${ACTOR_BC_ONLY_UNTIL_STEP:-$STEPS}"
    )
    ;;
  own)
    VARIANT="own"
    EXTRA=(
      --pref_sample_ratio "${PREF_SAMPLE_RATIO:-1.0}"
      --pref_rank_weight "${PREF_RANK_WEIGHT:-1.0}"
      --pref_rank_margin "${PREF_RANK_MARGIN:-0.1}"
      --pref_loss_type "${PREF_LOSS_TYPE:-lagrangian}"
      --pref_lambda_init "${PREF_LAMBDA_INIT:-1.0}"
      --pref_lambda_lr "${PREF_LAMBDA_LR:-0.00025}"
      --pref_lambda_max "${PREF_LAMBDA_MAX:-3.0}"
      --pref_action_delta_min "${PREF_ACTION_DELTA_MIN:-0.25}"
      --pref_action_delta_weight_scale "${PREF_ACTION_DELTA_WEIGHT_SCALE:-1.0}"
      --pref_action_delta_weight_max "${PREF_ACTION_DELTA_WEIGHT_MAX:-4.0}"
      --actor_bc_weight "${ACTOR_BC_WEIGHT:-1.0}"
      --prefill_demo_episodes "${PREFILL_DEMO_EPISODES:-40}"
      --prefill_max_steps_per_episode "${PREFILL_MAX_STEPS_PER_EPISODE:-300}"
      --prefill_policy "${PREFILL_POLICY:-student}"
    )
    ;;
  *)
    echo "Unknown METHOD=$METHOD. Use pvp, eil, hilserl, bc, or own." >&2
    exit 2
    ;;
esac

EXP_NAME="${EXP_NAME:-safetycar_goal1_oldenv_${METHOD}_${INIT_MODE}_${STEPS}_seed${SEED}_${RUN_TS}}"
echo "[oldenv-competitor] method=$METHOD variant=$VARIANT init=$INIT_MODE exp=$EXP_NAME steps=$STEPS seed=$SEED"

"$PY" train_fast_sac_safetygym_minimal.py \
  --env_name SafetyCarGoal1-v0 \
  --exp_name "$EXP_NAME" \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --device auto \
  --torch_num_threads 1 \
  --torch_num_interop_threads 1 \
  --total_timesteps "$STEPS" \
  --learning_starts "${LEARNING_STARTS:-1000}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --num_updates "${NUM_UPDATES:-2}" \
  --policy_frequency "${POLICY_FREQUENCY:-2}" \
  --buffer_size 1000000 \
  --gamma "${GAMMA:-0.99}" \
  --tau 0.005 \
  --actor_learning_rate "${ACTOR_LEARNING_RATE:-0.0003}" \
  --critic_learning_rate "${CRITIC_LEARNING_RATE:-0.0003}" \
  --max_grad_norm "${MAX_GRAD_NORM:-10.0}" \
  --alpha_init "${ALPHA_INIT:-0.001}" \
  --alpha_min "${ALPHA_MIN:-0.001}" \
  --alpha_max "${ALPHA_MAX:-0.001}" \
  --module_impl custom \
  --actor_hidden_dim 256 \
  --critic_hidden_dim 512 \
  --use_layer_norm \
  --obs_normalization \
  --obs_frame_stack "${OBS_FRAME_STACK:-1}" \
  --reward_mode "${REWARD_MODE:-dense}" \
  --dense_reward_scale "${DENSE_REWARD_SCALE:-1.0}" \
  --success_reward_scale "${SUCCESS_REWARD_SCALE:-0.0}" \
  --step_penalty "${STEP_PENALTY:-0.0}" \
  --clearance_penalty_scale "${CLEARANCE_PENALTY_SCALE:-0.0}" \
  --car_wheel_command_limit 2.0 \
  --car_force_scale 2.0 \
  --car_action_mode raw_wheels \
  --layout_curriculum car_random_blocked_filter \
  --terminate_on_goal \
  --no_reseed_on_episode_reset \
  --scale_actor_to_env_bounds \
  --use_intervention \
  --human_input_device scripted_geo \
  --scripted_geo_heading_tolerance "${SCRIPTED_GEO_HEADING_TOLERANCE:-0.20}" \
  --scripted_geo_lookahead "${SCRIPTED_GEO_LOOKAHEAD:-1.0}" \
  --scripted_geo_safety_margin "${SCRIPTED_GEO_SAFETY_MARGIN:-0.18}" \
  --scripted_geo_grid_resolution "${SCRIPTED_GEO_GRID_RESOLUTION:-0.08}" \
  --scripted_geo_emergency_clearance "${SCRIPTED_GEO_EMERGENCY_CLEARANCE:-0.08}" \
  --scripted_geo_action_shield_steps "${SCRIPTED_GEO_ACTION_SHIELD_STEPS:-1}" \
  --teacher_override_mode "${TEACHER_MODE_OVERRIDE:-clearance_or_progress}" \
  --teacher_clearance_source "${TEACHER_CLEARANCE_SOURCE:-keepout}" \
  --teacher_override_clearance_threshold "${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD:-0.08}" \
  --teacher_override_clearance_exit_threshold "${TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD:-0.14}" \
  --teacher_progress_score_mode "${TEACHER_PROGRESS_SCORE_MODE:-euclidean}" \
  --teacher_progress_dense_scale "${TEACHER_PROGRESS_DENSE_SCALE:-1.0}" \
  --teacher_progress_trigger_mode "${TEACHER_PROGRESS_TRIGGER_MODE:-not_improving}" \
  --teacher_progress_release_mode "${TEACHER_PROGRESS_RELEASE_MODE:-improve}" \
  --teacher_progress_bad_steps "${TEACHER_PROGRESS_BAD_STEPS:-3}" \
  --teacher_progress_good_steps "${TEACHER_PROGRESS_GOOD_STEPS:-5}" \
  --teacher_progress_epsilon "${TEACHER_PROGRESS_EPSILON:-0.0005}" \
  --init_checkpoint_path "$INIT_CKPT" \
  --no_load_critic_from_checkpoint \
  --no_load_critic_target_from_checkpoint \
  --no_load_alpha_from_checkpoint \
  --use_wandb \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_group "$WANDB_GROUP" \
  --wandb_run_name "$EXP_NAME" \
  --log_interval "${LOG_INTERVAL:-1000}" \
  --eval_interval "${EVAL_INTERVAL:-10000}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES:-20}" \
  --save_interval "${SAVE_INTERVAL:-10000}" \
  --eval_save_episode_plots \
  --eval_episode_plot_max_episodes "${EVAL_EPISODE_PLOT_MAX_EPISODES:-9}" \
  "${EXTRA[@]}"

#!/usr/bin/env bash
set -euo pipefail

# Reproduce the two May native-center-cost recipes from their saved args.json.
# Checkpoints are audited separately because the embedded evaluator can hang.
for arg in "$@"; do
  if [[ "$arg" != *=* ]]; then
    echo "expected KEY=VALUE override, got: $arg" >&2
    exit 2
  fi
  export "$arg"
done

ROOT="${ROOT:-/home/benjamin/thesis}"
PY="${PY:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
RECIPE="${RECIPE:-own_hybrid_scratch}"
STEPS="${STEPS:-10000}"
SEED="${SEED:-43}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym-competitors}"
WANDB_GROUP="${WANDB_GROUP:-safetycar_native_historical_20260623}"
HUMAN_INPUT_DEVICE="${HUMAN_INPUT_DEVICE:-scripted_geo_legacy}"
GOAL_CKPT="${GOAL_CKPT:-$ROOT/models/safetygym_minimal/safetycar_goal1_goalonly_small_ln_pretrain_20260511/step_25000.pt}"

case "$RECIPE" in
  reference_bc_goal)
    VARIANT="own"
    INIT_CKPT="$GOAL_CKPT"
    METHOD_FLAGS=(
      --pref_capacity 100000 --pref_sample_ratio 0.0 --pref_rank_weight 0.0
      --pref_stopgrad_positive
      --prefill_demo_episodes 40 --prefill_max_steps_per_episode 300 --prefill_policy student
      --actor_bc_weight 1.0 --actor_bc_teacher_only
      --actor_bc_obstacle_lidar_weight_scale 2.0 --actor_bc_goal_block_weight_scale 2.0
    )
    ;;
  own_hybrid_scratch)
    VARIANT="own"
    INIT_CKPT=""
    METHOD_FLAGS=(
      --pref_capacity 100000 --pref_sampling_mode linked --pref_sample_ratio 0.5
      --pref_rank_weight 1.0 --pref_rank_margin 0.1 --pref_loss_type lagrangian
      --pref_lambda_init 1.0 --pref_lambda_lr 0.00025 --pref_lambda_max 3.0 --pref_lambda_ema 0.9
      --pref_violation_clip 10.0 --pref_violation_target 0.0
      --pref_action_noise_std 0.05 --pref_action_noise_copies 2
      --pref_action_delta_min 0.25 --pref_action_delta_weight_scale 1.0 --pref_action_delta_weight_max 4.0
      --actor_bc_weight 0.25 --actor_bc_teacher_only
      --actor_bc_obstacle_lidar_weight_scale 2.0 --actor_bc_goal_block_weight_scale 2.0
      --prefill_demo_episodes 40 --prefill_max_steps_per_episode 300 --prefill_policy student
    )
    ;;
  pvp_goal)
    VARIANT="pvp"
    INIT_CKPT="$GOAL_CKPT"
    METHOD_FLAGS=(
      --pvp_proxy_value_bound "${PVP_PROXY_VALUE_BOUND:-1.0}"
      --prefill_demo_episodes 40 --prefill_max_steps_per_episode 300 --prefill_policy student
    )
    ;;
  hilserl_goal)
    # HIL-SERL-style online intervention duplication: intervened rows enter a
    # separate demo buffer and every update mixes replay with that buffer.
    VARIANT="hilserl"
    INIT_CKPT="$GOAL_CKPT"
    METHOD_FLAGS=(
      --demo_sample_ratio 0.5 --store_intervened_in_demo_buffer
      --prefill_demo_episodes 40 --prefill_max_steps_per_episode 300 --prefill_policy student
    )
    ;;
  eil_goal)
    VARIANT="eil"
    INIT_CKPT="$GOAL_CKPT"
    METHOD_FLAGS=(
      --eil_threshold 0.0 --eil_good_margin 0.0 --eil_bad_margin 0.01 --eil_pair_margin 0.01 --eil_bad_pre_steps 8
    )
    ;;
  bc_dagger_goal)
    # A DAgger-style online BC proxy.  It is not a dedicated ensemble HG/HD-Dagger implementation.
    VARIANT="own"
    INIT_CKPT="$GOAL_CKPT"
    METHOD_FLAGS=(
      --pref_capacity 0 --pref_sample_ratio 0.0 --pref_rank_weight 0.0
      --demo_sample_ratio 0.5 --store_intervened_in_demo_buffer
      --prefill_demo_episodes 40 --prefill_max_steps_per_episode 300 --prefill_policy student
      --actor_bc_weight 1.0 --actor_bc_teacher_only --actor_bc_only_until_step "$STEPS"
    )
    ;;
  *)
    echo "unknown RECIPE=$RECIPE" >&2
    exit 2
    ;;
esac

EXP_NAME="${EXP_NAME:-safetycar_goal1_native_historical_${RECIPE}_${STEPS}_seed${SEED}_${RUN_TS}}"
cd "$ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig-safetygym-native-historical}"
mkdir -p "$MPLCONFIGDIR"

echo "[native-historical] recipe=$RECIPE variant=$VARIANT seed=$SEED steps=$STEPS"
"$PY" train_fast_sac_safetygym_minimal.py \
  --env_name SafetyCarGoal1-v0 --exp_name "$EXP_NAME" --variant "$VARIANT" --seed "$SEED" \
  --device auto --torch_num_threads 1 --torch_num_interop_threads 1 \
  --total_timesteps "$STEPS" --learning_starts 1000 --batch_size 64 --num_updates 2 --policy_frequency 2 \
  --buffer_size 1000000 --gamma 0.99 --tau 0.005 --actor_learning_rate 0.0003 --critic_learning_rate 0.0003 \
  --max_grad_norm 10.0 --alpha_init 0.001 --alpha_min 0.0 --alpha_max 1.0 --critic_loss_reduction sum \
  --module_impl custom --actor_hidden_dim 256 --critic_hidden_dim 512 --use_layer_norm --obs_normalization --obs_frame_stack 1 \
  --reward_mode dense --dense_reward_scale 1.0 --success_reward_scale 0.0 --step_penalty 0.0 --clearance_penalty_scale 0.0 \
  --car_wheel_command_limit 2.0 --car_force_scale 2.0 --car_action_mode raw_wheels --layout_curriculum car_random_blocked_filter \
  --terminate_on_goal --no_reseed_on_episode_reset --scale_actor_to_env_bounds \
  --use_intervention --human_input_device "$HUMAN_INPUT_DEVICE" \
  --teacher_override_mode clearance_or_progress --teacher_clearance_source keepout \
  --teacher_override_clearance_threshold 0.08 --teacher_override_clearance_exit_threshold 0.14 \
  --teacher_progress_score_mode euclidean --teacher_progress_dense_scale 1.0 \
  --teacher_progress_trigger_mode not_improving --teacher_progress_release_mode improve \
  --teacher_progress_bad_steps 3 --teacher_progress_good_steps 5 --teacher_progress_epsilon 0.0005 \
  --init_checkpoint_path "$INIT_CKPT" --no_load_critic_from_checkpoint --no_load_critic_target_from_checkpoint --no_load_alpha_from_checkpoint \
  --use_wandb --wandb_mode "$WANDB_MODE" --wandb_project "$WANDB_PROJECT" --wandb_group "$WANDB_GROUP" --wandb_run_name "$EXP_NAME" \
  --log_interval "${LOG_INTERVAL:-1000}" --eval_interval 0 --save_interval "${SAVE_INTERVAL:-5000}" \
  "${METHOD_FLAGS[@]}"

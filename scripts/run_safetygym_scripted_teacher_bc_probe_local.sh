#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/benjamin/thesis"
cd "$ROOT"

PY="${PY:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000}"
BC_EVAL_INTERVAL="${BC_EVAL_INTERVAL:-10000}"
EXPORT_REPLAY_DATASET_INTERVAL="${EXPORT_REPLAY_DATASET_INTERVAL:-10000}"

INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-$ROOT/models/safetygym_minimal/safetycar_scriptedgeo_bc70k_shapedrl_finetune_30k_20260515/step_10000.pt}"
EXP_NAME="${EXP_NAME:-safetycar_scripted_teacher_bc_probe_${TOTAL_TIMESTEPS}_seed${SEED}_${RUN_TS}}"
DATASET_DIR="${DATASET_DIR:-$ROOT/datasets/safetygym_human_interventions}"
DATASET_LABEL="${DATASET_LABEL:-scripted_geo_clearance_or_progress_bc_probe_seed${SEED}_${RUN_TS}}"
FINAL_DATASET_PATH="${FINAL_DATASET_PATH:-$DATASET_DIR/SafetyCarGoal1-v0__${DATASET_LABEL}__final.npz}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"

WANDB_FLAGS=()
if [[ "${USE_WANDB:-1}" == "1" ]]; then
  WANDB_FLAGS=(
    --use_wandb
    --wandb_mode "$WANDB_MODE"
    --wandb_project "$WANDB_PROJECT"
    --wandb_group "safetygym_scripted_teacher_bc_probe"
    --wandb_run_name "$EXP_NAME"
  )
fi

echo "[run] $EXP_NAME"
echo "[run] fixed_student=$INIT_CHECKPOINT_PATH teacher=scripted_geo gate=clearance_or_progress steps=$TOTAL_TIMESTEPS bc_interval=$BC_EVAL_INTERVAL"

"$PY" train_fast_sac_safetygym_minimal.py \
  --env_name SafetyCarGoal1-v0 \
  --exp_name "$EXP_NAME" \
  --variant own \
  --seed "$SEED" \
  --device auto \
  --torch_num_threads 1 \
  --torch_num_interop_threads 1 \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --learning_starts 1000000000 \
  --batch_size 64 \
  --num_updates 0 \
  --policy_frequency 2 \
  --buffer_size 1000000 \
  --gamma 0.99 \
  --tau 0.005 \
  --actor_learning_rate 0.0003 \
  --critic_learning_rate 0.0003 \
  --module_impl custom \
  --actor_hidden_dim 256 \
  --critic_hidden_dim 512 \
  --use_layer_norm \
  --obs_normalization \
  --reward_mode dense \
  --dense_reward_scale 1.0 \
  --success_reward_scale 0.0 \
  --step_penalty 0.0 \
  --clearance_penalty_scale 0.0 \
  --clearance_margin 0.0 \
  --car_wheel_command_limit 1.0 \
  --car_force_scale 1.0 \
  --car_action_mode raw_wheels \
  --obs_mask_mode privileged_geometry \
  --layout_curriculum car_random_blocked_filter \
  --terminate_on_goal \
  --no_reseed_on_episode_reset \
  --scale_actor_to_env_bounds \
  --use_intervention \
  --human_input_device scripted_geo \
  --expert_device cpu \
  --teacher_override_mode clearance_or_progress \
  --teacher_override_clearance_threshold "${TEACHER_OVERRIDE_CLEARANCE_THRESHOLD:-0.08}" \
  --teacher_override_clearance_exit_threshold "${TEACHER_OVERRIDE_CLEARANCE_EXIT_THRESHOLD:-0.14}" \
  --teacher_progress_bad_steps "${TEACHER_PROGRESS_BAD_STEPS:-3}" \
  --teacher_progress_good_steps "${TEACHER_PROGRESS_GOOD_STEPS:-5}" \
  --teacher_progress_epsilon "${TEACHER_PROGRESS_EPSILON:-0.0005}" \
  --teacher_progress_trigger_mode not_improving \
  --teacher_progress_release_mode improve \
  --teacher_progress_score_mode euclidean \
  --teacher_progress_dense_scale 1.0 \
  --teacher_progress_clearance_scale -1.0 \
  --pref_capacity 0 \
  --pref_sample_ratio 0.0 \
  --pref_rank_weight 0.0 \
  --demo_sample_ratio 0.0 \
  --prefill_demo_episodes 0 \
  --demo_pretrain_updates 0 \
  --init_checkpoint_path "$INIT_CHECKPOINT_PATH" \
  --no_load_critic_from_checkpoint \
  --no_load_critic_target_from_checkpoint \
  --no_load_alpha_from_checkpoint \
  "${WANDB_FLAGS[@]}" \
  --log_interval 1000 \
  --eval_interval 10000 \
  --num_eval_episodes 20 \
  --save_interval 10000 \
  --export_replay_dataset_interval "$EXPORT_REPLAY_DATASET_INTERVAL" \
  --export_replay_dataset_dir "$DATASET_DIR" \
  --export_replay_dataset_label "$DATASET_LABEL" \
  --export_final_replay_dataset \
  --export_final_replay_dataset_path "$FINAL_DATASET_PATH" \
  --bc_eval_interval "$BC_EVAL_INTERVAL" \
  --bc_eval_epochs "${BC_EVAL_EPOCHS:-20}" \
  --bc_eval_batch_size "${BC_EVAL_BATCH_SIZE:-256}" \
  --bc_eval_context_len "${BC_EVAL_CONTEXT_LEN:-8}" \
  --bc_eval_hidden_dim "${BC_EVAL_HIDDEN_DIM:-256}" \
  --bc_eval_num_layers "${BC_EVAL_NUM_LAYERS:-3}" \
  --bc_eval_intervention_threshold "${BC_EVAL_INTERVENTION_THRESHOLD:-0.5}" \
  --bc_eval_num_episodes "${BC_EVAL_NUM_EPISODES:-20}" \
  --bc_eval_render_mode none \
  --bc_eval_fps 0 \
  --bc_eval_device "${BC_EVAL_DEVICE:-cpu}" \
  --bc_eval_student_policy checkpoint \
  --bc_eval_student_checkpoint_path "$INIT_CHECKPOINT_PATH" \
  --bc_eval_dataset_label "$DATASET_LABEL"

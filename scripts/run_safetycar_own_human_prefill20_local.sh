#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-fasttd3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_MODE_VALUE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-SafetyCarGoal2-v0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
LEARNING_STARTS="${LEARNING_STARTS:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
UPDATE_EVERY="${UPDATE_EVERY:-1}"
UPDATES_PER_CYCLE="${UPDATES_PER_CYCLE:-1}"

RENDER_MODE="${RENDER_MODE:-human}"
SURFACE_MODE="${SURFACE_MODE:-default}"
CAR_WHEEL_COMMAND_LIMIT="${CAR_WHEEL_COMMAND_LIMIT:-2.0}"
CAR_FORCE_SCALE="${CAR_FORCE_SCALE:-2.0}"
HUMAN_ACTION_SCALE="${HUMAN_ACTION_SCALE:-1.0}"
INTERVENTION_THRESHOLD="${INTERVENTION_THRESHOLD:-0.1}"
INTERVENTION_HOLD_SECONDS="${INTERVENTION_HOLD_SECONDS:-0.25}"
CONTROLLER_OVERLAY_HZ="${CONTROLLER_OVERLAY_HZ:-20.0}"

PREFILL_DEMO_EPISODES="${PREFILL_DEMO_EPISODES:-2}"
PREFILL_MAX_STEPS_PER_EPISODE="${PREFILL_MAX_STEPS_PER_EPISODE:-0}"
PREFILL_POLICY="${PREFILL_POLICY:-zero}"

WANDB_PROJECT="${WANDB_PROJECT:-thesis-safetygym}"
TS="$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-safetycar_goal2_own_human_prefill20_${TS}}"
LOG_INTERVAL="${LOG_INTERVAL:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"

echo "Starting SafetyCar own-method run with human-only demo prefill."
echo "You will collect ${PREFILL_DEMO_EPISODES} demo episodes first, then online learning begins."
echo "Focus the keyboard control window for intervention input."

conda run -n "${CONDA_ENV}" "${PYTHON_BIN}" train_fast_sac_safetygym.py \
  --env_name "${ENV_NAME}" \
  --render_mode "${RENDER_MODE}" \
  --surface_mode "${SURFACE_MODE}" \
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}" \
  --car_force_scale "${CAR_FORCE_SCALE}" \
  --num_envs 1 \
  --total_timesteps "${TOTAL_TIMESTEPS}" \
  --learning_starts "${LEARNING_STARTS}" \
  --batch_size "${BATCH_SIZE}" \
  --update_every "${UPDATE_EVERY}" \
  --updates_per_cycle "${UPDATES_PER_CYCLE}" \
  --reward_mode dense \
  --use_intervention \
  --intervention_threshold "${INTERVENTION_THRESHOLD}" \
  --intervention_hold_seconds "${INTERVENTION_HOLD_SECONDS}" \
  --human_action_scale "${HUMAN_ACTION_SCALE}" \
  --controller_overlay_hz "${CONTROLLER_OVERLAY_HZ}" \
  --prefill_demo_episodes "${PREFILL_DEMO_EPISODES}" \
  --prefill_max_steps_per_episode "${PREFILL_MAX_STEPS_PER_EPISODE}" \
  --prefill_policy "${PREFILL_POLICY}" \
  --pref_rank_weight 1.0 \
  --pref_rank_margin 0.1 \
  --pref_loss_type lagrangian \
  --pref_lambda_init 1.0 \
  --pref_lambda_lr 1e-3 \
  --pref_lambda_max 10.0 \
  --pref_lambda_ema 0.9 \
  --pref_violation_clip 10.0 \
  --pref_violation_target 0.0 \
  --pref_lagrangian_violation_type hinge \
  --pref_stopgrad_positive \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_mode "${WANDB_MODE_VALUE}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --log_interval "${LOG_INTERVAL}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --num_eval_episodes "${NUM_EVAL_EPISODES}" \
  --save_interval "${SAVE_INTERVAL}"

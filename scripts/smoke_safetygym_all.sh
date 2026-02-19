#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-fasttd3}
ENV_NAME=${ENV_NAME:-SafetyCarGoal2-v0}
TIMESTEPS=${TIMESTEPS:-64}
LEARNING_STARTS=${LEARNING_STARTS:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
LOG_INTERVAL=${LOG_INTERVAL:-32}
USE_INTERVENTION=${USE_INTERVENTION:-0}
RENDER_MODE=${RENDER_MODE:-none}
SURFACE_MODE=${SURFACE_MODE:-default}
CAR_WHEEL_COMMAND_LIMIT=${CAR_WHEEL_COMMAND_LIMIT:-2.0}
CAR_FORCE_SCALE=${CAR_FORCE_SCALE:-2.0}
PREFILL_DEMO_EPISODES=${PREFILL_DEMO_EPISODES:-0}
PREFILL_MAX_STEPS_PER_EPISODE=${PREFILL_MAX_STEPS_PER_EPISODE:-0}
PREFILL_POLICY=${PREFILL_POLICY:-student}
DEMO_PRETRAIN_UPDATES=${DEMO_PRETRAIN_UPDATES:-0}
DEMO_PRETRAIN_BATCH_SIZE=${DEMO_PRETRAIN_BATCH_SIZE:-0}
CRITIC_RESET_AFTER_PRETRAIN=${CRITIC_RESET_AFTER_PRETRAIN:-0}
UNCERTAINTY_PRE_INTERVENTION_WINDOW=${UNCERTAINTY_PRE_INTERVENTION_WINDOW:-25}
UNCERTAINTY_OVERSIGHT_THRESHOLD=${UNCERTAINTY_OVERSIGHT_THRESHOLD:-0.0}

INTERVENTION_ARGS=()
if [[ "${USE_INTERVENTION}" == "1" ]]; then
  INTERVENTION_ARGS+=(--use_intervention)
  echo "Running interactive smoke with keyboard intervention enabled."
  echo "Focus the 'SafetyGym Controls' window for WASD/arrow overrides."
  echo "Set RENDER_MODE=human if you also want the env viewer window."
else
  echo "Running non-interactive smoke (no human intervention)."
fi
if [[ "${PREFILL_DEMO_EPISODES}" != "0" && "${USE_INTERVENTION}" != "1" ]]; then
  echo "PREFILL_DEMO_EPISODES>0 requires USE_INTERVENTION=1"
  exit 2
fi

COMMON_ARGS=(
  --env_name "${ENV_NAME}"
  --render_mode "${RENDER_MODE}"
  --surface_mode "${SURFACE_MODE}"
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}"
  --car_force_scale "${CAR_FORCE_SCALE}"
  --total_timesteps "${TIMESTEPS}"
  --learning_starts "${LEARNING_STARTS}"
  --batch_size "${BATCH_SIZE}"
  --log_interval "${LOG_INTERVAL}"
  --eval_interval 0
  --save_interval 0
  --reward_mode sparse
  --prefill_demo_episodes "${PREFILL_DEMO_EPISODES}"
  --prefill_max_steps_per_episode "${PREFILL_MAX_STEPS_PER_EPISODE}"
  --prefill_policy "${PREFILL_POLICY}"
  --demo_pretrain_updates "${DEMO_PRETRAIN_UPDATES}"
  --demo_pretrain_batch_size "${DEMO_PRETRAIN_BATCH_SIZE}"
  --uncertainty_pre_intervention_window "${UNCERTAINTY_PRE_INTERVENTION_WINDOW}"
  --uncertainty_oversight_mode signal_only
  --uncertainty_oversight_threshold "${UNCERTAINTY_OVERSIGHT_THRESHOLD}"
)
if [[ "${CRITIC_RESET_AFTER_PRETRAIN}" == "1" ]]; then
  COMMON_ARGS+=(--critic_reset_after_pretrain)
fi

conda run -n "${CONDA_ENV}" python train_fast_sac_safetygym.py \
  "${COMMON_ARGS[@]}" \
  "${INTERVENTION_ARGS[@]}"

conda run -n "${CONDA_ENV}" python train_fastsac_pvp_safetygym.py \
  "${COMMON_ARGS[@]}" \
  "${INTERVENTION_ARGS[@]}"

conda run -n "${CONDA_ENV}" python train_fastsac_hilserl_safetygym.py \
  "${COMMON_ARGS[@]}" \
  "${INTERVENTION_ARGS[@]}"

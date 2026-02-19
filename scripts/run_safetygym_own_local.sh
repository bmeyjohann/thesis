#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-fasttd3}
SURFACE_MODE=${SURFACE_MODE:-default}
CAR_WHEEL_COMMAND_LIMIT=${CAR_WHEEL_COMMAND_LIMIT:-2.0}
CAR_FORCE_SCALE=${CAR_FORCE_SCALE:-2.0}

conda run -n "${CONDA_ENV}" python train_fast_sac_safetygym.py \
  --env_name SafetyCarGoal2-v0 \
  --surface_mode "${SURFACE_MODE}" \
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}" \
  --car_force_scale "${CAR_FORCE_SCALE}" \
  --num_envs 1 \
  --use_intervention \
  --reward_mode sparse \
  --total_timesteps 500000

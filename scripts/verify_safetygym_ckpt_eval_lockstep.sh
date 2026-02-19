#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <checkpoint.pt> [env_name]"
  exit 2
fi

CKPT="$1"
ENV_NAME="${2:-SafetyCarGoal2-v0}"
CONDA_ENV=${CONDA_ENV:-fasttd3}
SURFACE_MODE=${SURFACE_MODE:-default}
CAR_WHEEL_COMMAND_LIMIT=${CAR_WHEEL_COMMAND_LIMIT:-2.0}
CAR_FORCE_SCALE=${CAR_FORCE_SCALE:-2.0}

conda run -n "${CONDA_ENV}" python eval_interactive_safetygym.py \
  --model_path "${CKPT}" \
  --env_name "${ENV_NAME}" \
  --surface_mode "${SURFACE_MODE}" \
  --car_wheel_command_limit "${CAR_WHEEL_COMMAND_LIMIT}" \
  --car_force_scale "${CAR_FORCE_SCALE}" \
  --controller policy \
  --intervention_mode none \
  --render_mode none \
  --num_episodes 1 \
  --fps 0

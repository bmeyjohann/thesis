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
GROUP="${WANDB_GROUP:-safetycar_goal2_softplus_clearance_m000_s11_compare_${TIMESTAMP}}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"

case "${ALGO_VARIANT:-both}" in
  fastsac)
    ./scripts/run_safetycar_goal2_softplus_clearance_m000_s11_local.sh \
      "EXP_NAME=safetycar_goal2_softplus_clearance_m000_s11_fastsac_${TIMESTAMP}" \
      "WANDB_GROUP=$GROUP" \
      "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS" \
      "CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL" \
      "$@"
    ;;
  ppo)
    ./scripts/run_safetycar_goal2_softplus_clearance_m000_s11_ppo_local.sh \
      "EXP_NAME=safetycar_goal2_softplus_clearance_m000_s11_ppo_${TIMESTAMP}" \
      "WANDB_GROUP=$GROUP" \
      "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS" \
      "CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL" \
      "$@"
    ;;
  both)
    echo "Starting FastSAC first, then PPO. Use ALGO_VARIANT=fastsac or ALGO_VARIANT=ppo for only one."
    ./scripts/run_safetycar_goal2_softplus_clearance_m000_s11_local.sh \
      "EXP_NAME=safetycar_goal2_softplus_clearance_m000_s11_fastsac_${TIMESTAMP}" \
      "WANDB_GROUP=$GROUP" \
      "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS" \
      "CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL" \
      "$@"
    ./scripts/run_safetycar_goal2_softplus_clearance_m000_s11_ppo_local.sh \
      "EXP_NAME=safetycar_goal2_softplus_clearance_m000_s11_ppo_${TIMESTAMP}" \
      "WANDB_GROUP=$GROUP" \
      "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS" \
      "CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL" \
      "$@"
    ;;
  *)
    echo "Unsupported ALGO_VARIANT=${ALGO_VARIANT}. Use fastsac, ppo, or both." >&2
    exit 2
    ;;
esac

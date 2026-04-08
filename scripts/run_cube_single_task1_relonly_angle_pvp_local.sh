#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "unexpected positional argument: ${arg}" >&2
    exit 2
  fi
  export "${arg}"
done

TARGET_SCRIPT="${PWD}/scripts/run_cube_single_task1_relonly_angle_baseline_local.sh"
if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  TARGET_SCRIPT="/home/benjamin/thesis/scripts/run_cube_single_task1_relonly_angle_baseline_local.sh"
fi
if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "could not locate run_cube_single_task1_relonly_angle_baseline_local.sh" >&2
  exit 1
fi

export ALGO_VARIANT="${ALGO_VARIANT:-pvp}"
export ENV_NAME="${ENV_NAME:-cube-single-singletask-task1-v0}"
export DISABLE_ROTATION="${DISABLE_ROTATION:-1}"
export NUM_UPDATES="${NUM_UPDATES:-7}"
export CTA_RATIO="${CTA_RATIO:-2}"
export GAMMA="${GAMMA:-0.97}"
export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}"
export DEMO_PREFILL_EPISODES="${DEMO_PREFILL_EPISODES:-0}"
export DEMO_PREFILL_NUM_ENVS="${DEMO_PREFILL_NUM_ENVS:-0}"
export DEMO_SAMPLE_RATIO="${DEMO_SAMPLE_RATIO:-0.5}"
export PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-0.0}"
export PREF_STOPGRAD_POSITIVE="${PREF_STOPGRAD_POSITIVE:-0}"
export STORE_INTERVENED_IN_DEMO_BUFFER="${STORE_INTERVENED_IN_DEMO_BUFFER:-0}"
export FIXED_ALPHA="${FIXED_ALPHA:-0.0}"
export ALPHA_INIT="${ALPHA_INIT:-0.0}"
export ALPHA_MIN="${ALPHA_MIN:-0.0}"
export ALPHA_MAX="${ALPHA_MAX:-0.0}"
export ALPHA_FREEZE_STEPS="${ALPHA_FREEZE_STEPS:-0}"
export PVP_PROXY_VALUE_BOUND="${PVP_PROXY_VALUE_BOUND:-1.0}"
export NAME_SUFFIX="${NAME_SUFFIX:-pvp}"

exec bash "${TARGET_SCRIPT}"

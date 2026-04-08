#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/run_cube_single_task1_relonly_angle_baseline_local.sh"
if [[ ! -x "${TARGET_SCRIPT}" ]]; then
  TARGET_SCRIPT="${PWD}/scripts/run_cube_single_task1_relonly_angle_baseline_local.sh"
fi

exec bash "${TARGET_SCRIPT}" \
  ENV_NAME="${ENV_NAME:-cube-single-singletask-task1-v0}" \
  DISABLE_ROTATION="${DISABLE_ROTATION:-1}" \
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000}" \
  CTA_RATIO="${CTA_RATIO:-2}" \
  NUM_UPDATES="${NUM_UPDATES:-7}" \
  FIXED_ALPHA="${FIXED_ALPHA:--1}" \
  PREF_CRITIC_SCOPE="${PREF_CRITIC_SCOPE:-min}" \
  PREF_LOSS_TYPE="${PREF_LOSS_TYPE:-bradley_terry}" \
  PREF_STOPGRAD_POSITIVE="${PREF_STOPGRAD_POSITIVE:-1}" \
  STORE_INTERVENED_IN_DEMO_BUFFER="${STORE_INTERVENED_IN_DEMO_BUFFER:-1}" \
  NAME_SUFFIX="${NAME_SUFFIX:-autores_bt_minscope_storeinterv}"

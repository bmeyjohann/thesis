#!/usr/bin/env bash
set -euo pipefail

# Forward KEY=VALUE overrides to the shared angle-baseline launcher.
EXTRA_ARGS=("$@")

REPO_ROOT="${REPO_ROOT:-$PWD}"
TARGET_SCRIPT="${REPO_ROOT}/scripts/run_cube_single_task1_relonly_angle_baseline_local.sh"
if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  TARGET_SCRIPT="/home/benjamin/thesis/scripts/run_cube_single_task1_relonly_angle_baseline_local.sh"
fi
if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Could not locate run_cube_single_task1_relonly_angle_baseline_local.sh" >&2
  exit 2
fi

bash "${TARGET_SCRIPT}" \
  DISABLE_ROTATION="${DISABLE_ROTATION:-1}" \
  CTA_RATIO="${CTA_RATIO:-2}" \
  NUM_UPDATES="${NUM_UPDATES:-7}" \
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}" \
  GAMMA="${GAMMA:-0.97}" \
  FIXED_ALPHA="${FIXED_ALPHA:-0.001}" \
  ALPHA_INIT="${ALPHA_INIT:-0.001}" \
  ALPHA_MIN="${ALPHA_MIN:-0.001}" \
  ALPHA_MAX="${ALPHA_MAX:-0.001}" \
  ALPHA_FREEZE_STEPS="${ALPHA_FREEZE_STEPS:-0}" \
  PREF_RANK_WEIGHT="${PREF_RANK_WEIGHT:-0.0}" \
  PREF_SAMPLE_RATIO="${PREF_SAMPLE_RATIO:-0.0}" \
  STORE_INTERVENED_IN_DEMO_BUFFER="${STORE_INTERVENED_IN_DEMO_BUFFER:-1}" \
  NAME_SUFFIX="${NAME_SUFFIX:-hilserl_anglebaseline_matched}" \
  "${EXTRA_ARGS[@]}"

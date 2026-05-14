#!/usr/bin/env bash
set -euo pipefail

export ALGO_VARIANT="${ALGO_VARIANT:-pvp}"
export NUM_ENVS="${NUM_ENVS:-1}"
export EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
BASELINE_SCRIPT="${PWD}/scripts/run_pointmaze_arena_state_baseline_local.sh"
if [[ ! -f "${BASELINE_SCRIPT}" ]]; then
  BASELINE_SCRIPT="$(dirname "$0")/run_pointmaze_arena_state_baseline_local.sh"
fi
exec bash "${BASELINE_SCRIPT}" "$@"

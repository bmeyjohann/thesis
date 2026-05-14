#!/usr/bin/env bash
set -euo pipefail

export ALGO_VARIANT="${ALGO_VARIANT:-own}"
BASELINE_SCRIPT="${PWD}/scripts/run_pointmaze_arena_state_baseline_local.sh"
if [[ ! -f "${BASELINE_SCRIPT}" ]]; then
  BASELINE_SCRIPT="$(dirname "$0")/run_pointmaze_arena_state_baseline_local.sh"
fi
exec bash "${BASELINE_SCRIPT}" "$@"

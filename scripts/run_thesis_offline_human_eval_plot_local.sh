#!/usr/bin/env bash
set -euo pipefail

POSITIONAL_ARGS=()
for arg in "$@"; do
  if [[ "${arg}" == *=* ]]; then
    export "${arg}"
  else
    POSITIONAL_ARGS+=("${arg}")
  fi
done

INPUT_ROOT="${INPUT_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"

if [[ -z "${INPUT_ROOT}" && ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
  INPUT_ROOT="${POSITIONAL_ARGS[0]}"
fi
if [[ -z "${INPUT_ROOT}" ]]; then
  echo "INPUT_ROOT is required" >&2
  exit 2
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/home/benjamin/thesis/local/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

exec "${PYTHON_BIN}" "/home/benjamin/thesis/scripts/thesis_offline_human_eval.py" \
  plot \
  --input-root "${INPUT_ROOT}"

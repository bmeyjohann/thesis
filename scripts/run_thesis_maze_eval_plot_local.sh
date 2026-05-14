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

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "/home/benjamin/miniconda3/envs/fasttd3/bin/python" ]]; then
  PYTHON_BIN="/home/benjamin/miniconda3/envs/fasttd3/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
UNCERTAINTY="${UNCERTAINTY:-ci95}"
MPLCONFIGDIR_VALUE="${MPLCONFIGDIR_VALUE:-/home/benjamin/thesis/local/mplconfig}"

if [[ -z "${INPUT_DIR}" && ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
  INPUT_DIR="${POSITIONAL_ARGS[0]}"
fi

if [[ -z "${OUTPUT_DIR}" && ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
  OUTPUT_DIR="${POSITIONAL_ARGS[1]}"
fi

if [[ "${UNCERTAINTY}" == "ci95" && ${#POSITIONAL_ARGS[@]} -ge 3 ]]; then
  UNCERTAINTY="${POSITIONAL_ARGS[2]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 3 ]]; then
  echo "too many positional arguments: ${POSITIONAL_ARGS[*]}" >&2
  exit 2
fi

if [[ -z "${INPUT_DIR}" ]]; then
  echo "INPUT_DIR must be set" >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "OUTPUT_DIR must be set" >&2
  exit 2
fi

mkdir -p "${MPLCONFIGDIR_VALUE}"

cd /home/benjamin/thesis
MPLCONFIGDIR="${MPLCONFIGDIR_VALUE}" \
"${PYTHON_BIN}" /home/benjamin/thesis/scripts/thesis_maze_eval.py plot \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --uncertainty "${UNCERTAINTY}"

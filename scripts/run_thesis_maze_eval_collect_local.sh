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

MANIFEST="${MANIFEST:-}"
SUITE="${SUITE:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ -z "${MANIFEST}" && ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
  MANIFEST="${POSITIONAL_ARGS[0]}"
fi

if [[ -z "${OUTPUT_DIR}" && ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
  OUTPUT_DIR="${POSITIONAL_ARGS[1]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 2 ]]; then
  echo "too many positional arguments: ${POSITIONAL_ARGS[*]}" >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "OUTPUT_DIR must be set" >&2
  exit 2
fi

ARGS=(/home/benjamin/thesis/scripts/thesis_maze_eval.py collect --output-dir "${OUTPUT_DIR}")
if [[ -n "${MANIFEST}" ]]; then
  ARGS+=(--manifest "${MANIFEST}")
else
  ARGS+=(--suite "${SUITE}")
fi

cd /home/benjamin/thesis
"${PYTHON_BIN}" "${ARGS[@]}"

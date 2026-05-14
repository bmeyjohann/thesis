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

MANIFEST="${MANIFEST:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/benjamin/miniconda3/envs/fasttd3/bin/python}"

if [[ -z "${MANIFEST}" && ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
  MANIFEST="${POSITIONAL_ARGS[0]}"
fi
if [[ -z "${OUTPUT_ROOT}" && ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
  OUTPUT_ROOT="${POSITIONAL_ARGS[1]}"
fi

if [[ -z "${MANIFEST}" ]]; then
  echo "MANIFEST is required" >&2
  exit 2
fi

CMD=(
  "${PYTHON_BIN}" "/home/benjamin/thesis/scripts/thesis_offline_human_eval.py"
  collect
  --manifest "${MANIFEST}"
)
if [[ -n "${OUTPUT_ROOT}" ]]; then
  CMD+=(--output-root "${OUTPUT_ROOT}")
fi

exec "${CMD[@]}"

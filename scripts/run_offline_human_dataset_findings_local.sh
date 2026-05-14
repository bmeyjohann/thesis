#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="/home/benjamin/miniconda3/envs/fasttd3/bin/python"

if [[ -x "${PYTHON_BIN:-}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON="$DEFAULT_PYTHON"
else
  PYTHON="python3"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/local/offline_human_dataset_findings}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/local/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

exec "$PYTHON" "$REPO_ROOT/scripts/plot_offline_human_dataset_findings.py" \
  --output-root "$OUTPUT_ROOT"

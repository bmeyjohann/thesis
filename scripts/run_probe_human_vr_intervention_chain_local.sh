#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "${REPO_ROOT}/scripts/probe_human_vr_intervention_chain.py" && -f "${PWD}/scripts/probe_human_vr_intervention_chain.py" ]]; then
  REPO_ROOT="${PWD}"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
STEPS="${STEPS:-256}"
PORT="${PORT:-18765}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/models/fast_sac/cube_single_task1_human_collect_norot_fixedalpha1e3_20260408_143217/cube_single_singletask_task1_v0_step25000.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/local/reports/human_vr_intervention_probe.json}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/probe_human_vr_intervention_chain.py" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --steps "${STEPS}" \
  --port "${PORT}" \
  --output_json "${OUTPUT_JSON}" \
  "$@"

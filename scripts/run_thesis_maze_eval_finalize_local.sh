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
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
TIMEOUT_SEC="${TIMEOUT_SEC:-21600}"
POLL_SEC="${POLL_SEC:-120}"

if [[ -z "${MANIFEST}" && ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
  MANIFEST="${POSITIONAL_ARGS[0]}"
fi

if [[ -z "${OUTPUT_ROOT}" && ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
  OUTPUT_ROOT="${POSITIONAL_ARGS[1]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 2 ]]; then
  echo "too many positional arguments: ${POSITIONAL_ARGS[*]}" >&2
  exit 2
fi

if [[ -z "${MANIFEST}" ]]; then
  echo "MANIFEST must be set" >&2
  exit 2
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "OUTPUT_ROOT must be set" >&2
  exit 2
fi

cd /home/benjamin/thesis

"${PYTHON_BIN}" - "$MANIFEST" "$TIMEOUT_SEC" "$POLL_SEC" <<'PY'
import json
import re
import sys
import time
from pathlib import Path

manifest_path = Path(sys.argv[1]).expanduser().resolve()
timeout_sec = int(sys.argv[2])
poll_sec = int(sys.argv[3])

if not manifest_path.exists():
    raise SystemExit(f"manifest not found: {manifest_path}")

payload = json.loads(manifest_path.read_text())
target_step = int(payload.get("launch_args", {}).get("total_timesteps", 0))
jobs = payload.get("jobs", [])

eval_step_re = re.compile(r"\[Eval\].*?steps=(\d+)")
deadline = time.time() + timeout_sec

while True:
    completed = 0
    missing = 0
    latest = []
    for job in jobs:
        training_log = Path(job["training_log"])
        if not training_log.exists():
            missing += 1
            latest.append((job["exp_name"], -1))
            continue
        max_step = -1
        try:
            with training_log.open("r", errors="ignore") as handle:
                for line in handle:
                    match = eval_step_re.search(line)
                    if match:
                        max_step = max(max_step, int(match.group(1)))
        except Exception:
            max_step = -1
        latest.append((job["exp_name"], max_step))
        if max_step >= target_step:
            completed += 1
    print(
        f"[Finalize] target_step={target_step} completed={completed}/{len(jobs)} "
        f"missing_logs={missing} latest_min={min(step for _, step in latest)} "
        f"latest_max={max(step for _, step in latest)}",
        flush=True,
    )
    if completed >= len(jobs):
        break
    if time.time() >= deadline:
        print("[Finalize] timeout reached; collecting whatever is available", flush=True)
        break
    time.sleep(poll_sec)
PY

COLLECT_DIR="${OUTPUT_ROOT}/collection"
PLOTS_DIR="${OUTPUT_ROOT}/plots"

bash /home/benjamin/thesis/scripts/run_thesis_maze_eval_collect_local.sh \
  MANIFEST="${MANIFEST}" \
  OUTPUT_DIR="${COLLECT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}"

bash /home/benjamin/thesis/scripts/run_thesis_maze_eval_plot_local.sh \
  INPUT_DIR="${COLLECT_DIR}" \
  OUTPUT_DIR="${PLOTS_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}"

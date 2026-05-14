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
JOB_EXP_NAME="${JOB_EXP_NAME:-}"

if [[ -z "${MANIFEST}" && ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
  MANIFEST="${POSITIONAL_ARGS[0]}"
fi
if [[ -z "${JOB_EXP_NAME}" && ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
  JOB_EXP_NAME="${POSITIONAL_ARGS[1]}"
fi

if [[ -z "${MANIFEST}" || -z "${JOB_EXP_NAME}" ]]; then
  echo "MANIFEST and JOB_EXP_NAME are required" >&2
  exit 2
fi

python3 - "$MANIFEST" "$JOB_EXP_NAME" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).expanduser().resolve()
job_exp_name = sys.argv[2]
payload = json.loads(manifest_path.read_text())
job = next((item for item in payload.get("jobs", []) if item.get("exp_name") == job_exp_name), None)
if job is None:
    raise SystemExit(f"job not found in manifest: {job_exp_name}")
cmd = job.get("command")
if not isinstance(cmd, list) or not cmd:
    raise SystemExit(f"invalid command for job: {job_exp_name}")
os.chdir(str(Path("/home/benjamin/thesis")))
os.execvp(cmd[0], cmd)
PY

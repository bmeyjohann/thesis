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
METHOD="${METHOD:-}"

if [[ -z "${MANIFEST}" && ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
  MANIFEST="${POSITIONAL_ARGS[0]}"
fi

if [[ -z "${METHOD}" && ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
  METHOD="${POSITIONAL_ARGS[1]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 2 ]]; then
  echo "too many positional arguments: ${POSITIONAL_ARGS[*]}" >&2
  exit 2
fi

if [[ -z "${MANIFEST}" ]]; then
  echo "MANIFEST must be set" >&2
  exit 2
fi

if [[ -z "${METHOD}" ]]; then
  echo "METHOD must be set" >&2
  exit 2
fi

BATCH_SCRIPT="$(python3 - "$MANIFEST" "$METHOD" <<'PY'
import json
import shlex
import sys
import tempfile
from pathlib import Path

manifest_path = Path(sys.argv[1]).expanduser().resolve()
method = sys.argv[2]
if not manifest_path.exists():
    raise SystemExit(f"manifest not found: {manifest_path}")

payload = json.loads(manifest_path.read_text())
jobs = [job for job in payload.get("jobs", []) if str(job.get("method", "")) == method]
if not jobs:
    raise SystemExit(f"no jobs found in manifest for method={method!r}")

lines = [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    "cd /home/benjamin/thesis",
]
for job in jobs:
    exp_name = str(job.get("exp_name", ""))
    cmd = job.get("command")
    if not isinstance(cmd, list) or not cmd:
        raise SystemExit(f"invalid command for manifest job: {exp_name}")
    quoted = " ".join(shlex.quote(str(part)) for part in cmd)
    lines.append(f"echo '=== Batch launching {exp_name} ({method}) ==='")
    lines.append(quoted)

handle = tempfile.NamedTemporaryFile("w", delete=False, suffix=f"_{method}_maze_manifest_batch.sh")
with handle:
    handle.write("\n".join(lines) + "\n")
print(handle.name)
PY
)"

trap 'rm -f "${BATCH_SCRIPT}"' EXIT
chmod +x "${BATCH_SCRIPT}"
exec bash "${BATCH_SCRIPT}"

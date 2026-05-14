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
MODULUS="${MODULUS:-1}"
REMAINDER="${REMAINDER:-0}"
SUMMARY_PATH="${SUMMARY_PATH:-}"
FINALIZE_AFTER="${FINALIZE_AFTER:-0}"
WAIT_FOR_SUMMARY="${WAIT_FOR_SUMMARY:-}"
COLLECT_OUTPUT_ROOT="${COLLECT_OUTPUT_ROOT:-}"
PLOT_INPUT_ROOT="${PLOT_INPUT_ROOT:-}"

if [[ -z "${MANIFEST}" && ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
  MANIFEST="${POSITIONAL_ARGS[0]}"
fi
if [[ -z "${MANIFEST}" ]]; then
  echo "MANIFEST is required" >&2
  exit 2
fi

python3 - "$MANIFEST" "$MODULUS" "$REMAINDER" "$SUMMARY_PATH" "$FINALIZE_AFTER" "$WAIT_FOR_SUMMARY" "$COLLECT_OUTPUT_ROOT" "$PLOT_INPUT_ROOT" <<'PY'
import json
import os
import subprocess
import sys
import time
from pathlib import Path

manifest_path = Path(sys.argv[1]).expanduser().resolve()
modulus = max(1, int(sys.argv[2]))
remainder = int(sys.argv[3])
summary_override = sys.argv[4].strip()
finalize_after = bool(int(sys.argv[5]))
wait_for_summary = sys.argv[6].strip()
collect_output_root = sys.argv[7].strip()
plot_input_root = sys.argv[8].strip()

payload = json.loads(manifest_path.read_text())
jobs = payload.get("jobs", [])
selected = [job for idx, job in enumerate(jobs) if idx % modulus == remainder]
if not selected:
    raise SystemExit(f"no jobs selected from {manifest_path} for modulus={modulus} remainder={remainder}")

summary_dir = manifest_path.parents[1] / "batch_summaries"
summary_dir.mkdir(parents=True, exist_ok=True)
summary_path = Path(summary_override).expanduser().resolve() if summary_override else (
    summary_dir / f"{manifest_path.stem}_mod{modulus}_rem{remainder}.json"
)

results = []
start = time.time()
for index, job in enumerate(selected, start=1):
    exp_name = str(job.get("exp_name", f"job{index}"))
    command = job.get("command")
    if not isinstance(command, list) or not command:
        results.append({"exp_name": exp_name, "exit_code": -1, "status": "invalid_command"})
        continue
    print(f"[Batch {remainder}/{modulus}] ({index}/{len(selected)}) starting {exp_name}", flush=True)
    launched = time.time()
    completed = subprocess.run(command, cwd="/home/benjamin/thesis")
    duration = time.time() - launched
    results.append(
        {
            "exp_name": exp_name,
            "exit_code": int(completed.returncode),
            "status": "ok" if completed.returncode == 0 else "failed",
            "duration_sec": duration,
            "metrics_path": job.get("metrics_path", ""),
            "output_dir": job.get("output_dir", ""),
        }
    )
    print(
        f"[Batch {remainder}/{modulus}] finished {exp_name} "
        f"exit_code={completed.returncode} duration_sec={duration:.1f}",
        flush=True,
    )

summary = {
    "manifest": str(manifest_path),
    "modulus": modulus,
    "remainder": remainder,
    "num_selected_jobs": len(selected),
    "total_wall_sec": time.time() - start,
    "results": results,
}
summary_path.write_text(json.dumps(summary, indent=2))
failed = [item for item in results if int(item.get("exit_code", 1)) != 0]
if failed:
    print(f"[Batch {remainder}/{modulus}] failures={len(failed)} summary={summary_path}", flush=True)
    raise SystemExit(1)

if finalize_after:
    wait_path = Path(wait_for_summary).expanduser().resolve() if wait_for_summary else None
    if wait_path is not None:
        print(f"[Batch {remainder}/{modulus}] waiting for peer summary {wait_path}", flush=True)
        while not wait_path.exists():
            time.sleep(30)
    if collect_output_root:
        collect_cmd = [
            "/home/benjamin/thesis/scripts/run_thesis_offline_human_eval_collect_local.sh",
            f"MANIFEST={manifest_path}",
            f"OUTPUT_ROOT={collect_output_root}",
        ]
        print(f"[Batch {remainder}/{modulus}] running collect", flush=True)
        subprocess.run(collect_cmd, cwd="/home/benjamin/thesis", check=True)
    if plot_input_root:
        plot_cmd = [
            "/home/benjamin/thesis/scripts/run_thesis_offline_human_eval_plot_local.sh",
            f"INPUT_ROOT={plot_input_root}",
        ]
        print(f"[Batch {remainder}/{modulus}] running plot", flush=True)
        subprocess.run(plot_cmd, cwd="/home/benjamin/thesis", check=True)

print(f"[Batch {remainder}/{modulus}] all jobs passed summary={summary_path}", flush=True)
PY

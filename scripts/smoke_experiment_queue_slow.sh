#!/usr/bin/env bash
set -euo pipefail

echo "smoke queue slow: start"
python - <<'PY'
import time

for step in range(10):
    print(f"slow-step={step}", flush=True)
    time.sleep(0.5)
print("smoke queue slow: done", flush=True)
PY

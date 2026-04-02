#!/usr/bin/env bash
set -euo pipefail

echo "smoke queue success: start"
python - <<'PY'
import os
import time

print("cwd=", os.getcwd())
print("conda_env=", os.environ.get("CONDA_DEFAULT_ENV"))
for step in range(3):
    print(f"success-step={step}")
    time.sleep(0.2)
print("smoke queue success: done")
PY

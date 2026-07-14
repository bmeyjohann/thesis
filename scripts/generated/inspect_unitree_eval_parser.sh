#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

python - <<'PY'
from pathlib import Path
lines = Path("eval_unitree_nav_baselines.py").read_text(errors="replace").splitlines()
for i, line in enumerate(lines):
    if "def parse_args" in line:
        for j in range(i, min(len(lines), i + 180)):
            print(f"{j+1:04d}: {lines[j]}")
        break
PY

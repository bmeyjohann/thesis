#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

python - <<'PY'
from pathlib import Path
for rel in ["eval_unitree_nav_baselines.py"]:
    lines = Path(rel).read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if "def _apply_debug_obstacle_overrides" in line:
            start = max(0, i - 10)
            end = min(len(lines), i + 160)
            print(f"===== {rel}:{start+1}-{end} =====")
            for j in range(start, end):
                print(f"{j+1:04d}: {lines[j]}")
PY

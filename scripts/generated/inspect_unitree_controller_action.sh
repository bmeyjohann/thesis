#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
python - <<'PY'
from pathlib import Path
lines = Path("eval_unitree_nav_baselines.py").read_text(errors="replace").splitlines()
for needle in ["def controller_action", "def _direct_goal_action", "def _current_goal_distance"]:
    for i, line in enumerate(lines):
        if needle in line:
            print(f"\n===== {needle} at line {i+1} =====")
            for j in range(max(0, i - 10), min(len(lines), i + 110)):
                print(f"{j+1:04d}: {lines[j]}")
            break
PY

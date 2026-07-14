#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

python - <<'PY'
from pathlib import Path

paths = [
    "external/unitree_rl_mjlab/src/tasks/navigation/mdp/obstacles.py",
    "external/unitree_rl_mjlab/src/tasks/navigation/mdp/observations.py",
    "external/unitree_rl_mjlab/src/tasks/navigation/config/g1/env_cfgs.py",
    "external/unitree_rl_mjlab/src/tasks/navigation/config/g1/safe_env_cfgs.py",
    "external/unitree_rl_mjlab/src/tasks/navigation/navigation_env_cfg.py",
]

needles = [
    "obstacle", "height", "scan", "scanner", "ray", "lidar",
    "box", "cube", "sphere", "cylinder", "geom", "terrain",
    "radius", "size", "spawn", "random",
]

for rel in paths:
    path = Path(rel)
    print(f"\n===== {rel} =====")
    if not path.exists():
        print("MISSING")
        continue
    lines = path.read_text(errors="replace").splitlines()
    selected = set()
    for i, line in enumerate(lines):
        low = line.lower()
        if any(n in low for n in needles):
            for j in range(max(0, i - 4), min(len(lines), i + 8)):
                selected.add(j)
    last = -10
    for i in sorted(selected):
        if i > last + 1:
            print("...")
        print(f"{i+1:04d}: {lines[i]}")
        last = i
PY

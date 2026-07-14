#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

python - <<'PY'
from pathlib import Path

paths = [
    "eval_unitree_nav_baselines.py",
    "train_unitree_nav_thesis.py",
    "plot_unitree_nav_rollout.py",
]
needles = [
    "debug_obstacle", "fixed obstacle", "goal_through", "blocked",
    "terrain_generator", "HfDiscrete", "height_scan", "GridPatternCfg",
    "scan_range", "scan", "obstacle_width", "obstacle_height",
]

for rel in paths:
    p = Path(rel)
    print(f"\n===== {rel} =====")
    lines = p.read_text(errors="replace").splitlines()
    selected = set()
    for i, line in enumerate(lines):
        low = line.lower()
        if any(n.lower() in low for n in needles):
            for j in range(max(0, i - 6), min(len(lines), i + 14)):
                selected.add(j)
    last = -10
    for i in sorted(selected):
        if i > last + 1:
            print("...")
        print(f"{i+1:04d}: {lines[i]}")
        last = i
PY

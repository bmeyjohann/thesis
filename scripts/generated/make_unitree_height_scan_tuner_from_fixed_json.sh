#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
export MPLBACKEND=Agg

python - <<'PY'
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

fixed_candidates = sorted(Path("visualizations/unitree_height_scan_tuner").glob("*/rotation_diagnostic/height_scan_tuner_data_rotation_fixed.json"))
if not fixed_candidates:
    raise SystemExit("No fixed rotation JSON found")
src = fixed_candidates[-1]
data = json.loads(src.read_text())
out_dir = src.parent
mode = data.get("mapping_diagnostic", {}).get("best_mode", "unknown")

fig, axes = plt.subplots(1, len(data["snapshots"]), figsize=(6.4 * len(data["snapshots"]), 6.2), dpi=150, squeeze=False)
obstacles = np.asarray(data["obstacle_cells"], dtype=np.float32)
for idx, snap in enumerate(data["snapshots"]):
    ax = axes[0][idx]
    xy = np.asarray(snap["xy"], dtype=np.float32)
    goal = np.asarray(snap["goal"], dtype=np.float32)
    pts = np.asarray(snap["scan_points_world"], dtype=np.float32)
    vals = np.asarray(snap["scan_values"], dtype=np.float32)
    blocked = vals < data["defaults"]["threshold"]
    if obstacles.size:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], s=8, c="black", alpha=0.18, marker="s", label="obstacles")
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=vals,
        s=np.where(blocked, 95, 45),
        cmap="viridis",
        edgecolors=np.where(blocked, "red", "white"),
        linewidths=np.where(blocked, 1.6, 0.5),
        zorder=4,
    )
    ax.scatter(xy[0], xy[1], c="white", edgecolors="black", s=110, zorder=5, label="robot")
    ax.scatter(goal[0], goal[1], marker="*", s=240, c="limegreen", edgecolors="black", zorder=5, label="goal")
    ax.arrow(xy[0], xy[1], math.cos(snap["heading"]) * 0.45, math.sin(snap["heading"]) * 0.45, color="deepskyblue", width=0.012, head_width=0.08, zorder=6, label="heading")
    # Current teacher vector was computed before visual remapping. Keep it visible,
    # but label it as action output, not as a scan cell mapping.
    vec = np.asarray(snap["scan_teacher_world_vec"], dtype=np.float32)
    ax.arrow(xy[0], xy[1], vec[0], vec[1], color="orange", width=0.022, head_width=0.12, zorder=7, label="current teacher action")
    ax.set_title(f"step {snap['step']} | fixed scan mapping: {mode}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=7)
fig.suptitle(f"Rotation-fixed height-scan overlay: {mode}")
fig.tight_layout()
png_path = out_dir / "height_scan_tuner_static_preview_rotation_fixed.png"
fig.savefig(png_path)
plt.close(fig)

html_template_candidates = sorted(Path("visualizations/unitree_height_scan_tuner").glob("*/height_scan_teacher_tuner.html"))
if not html_template_candidates:
    raise SystemExit("No tuner HTML template found")
template = html_template_candidates[-1].read_text()
start = template.index("const DATA = ") + len("const DATA = ")
end = template.index(";\nconst $", start)
fixed_html = template[:start] + json.dumps(data) + template[end:]
html_path = out_dir / "height_scan_teacher_tuner_rotation_fixed.html"
fixed_html = fixed_html.replace("Height Scan Teacher Tuner", f"Height Scan Teacher Tuner - Rotation Fixed ({mode})", 1)
html_path.write_text(fixed_html, encoding="utf-8")

print(json.dumps({"source": str(src), "best_mode": mode, "png": str(png_path), "html": str(html_path)}), flush=True)
PY

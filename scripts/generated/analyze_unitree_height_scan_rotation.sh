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


def latest_data() -> Path:
    roots = sorted(Path("visualizations/unitree_height_scan_tuner").glob("*/height_scan_tuner_data.json"))
    if not roots:
        roots = sorted(Path("visualizations/unitree_height_scan_snapshots").glob("*/height_scan_teacher_snapshots.json"))
    if not roots:
        raise SystemExit("No height scan tuner/snapshot data found")
    return roots[-1]


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def transform_points(local: np.ndarray, xy: np.ndarray, heading: float, mode: str) -> np.ndarray:
    # local columns are the stored grid coordinates from the previous assumption.
    f = local[:, 0]
    l = local[:, 1]
    if mode == "current_fwd_lat":
        body = np.stack([f, l], axis=1)
        theta = heading
    elif mode == "swap_lat_fwd":
        body = np.stack([l, f], axis=1)
        theta = heading
    elif mode == "fwd_neglat":
        body = np.stack([f, -l], axis=1)
        theta = heading
    elif mode == "negfwd_lat":
        body = np.stack([-f, l], axis=1)
        theta = heading
    elif mode == "plus90":
        body = np.stack([f, l], axis=1)
        theta = heading + math.pi / 2
    elif mode == "minus90":
        body = np.stack([f, l], axis=1)
        theta = heading - math.pi / 2
    elif mode == "swap_plus90":
        body = np.stack([l, f], axis=1)
        theta = heading + math.pi / 2
    elif mode == "swap_minus90":
        body = np.stack([l, f], axis=1)
        theta = heading - math.pi / 2
    else:
        raise ValueError(mode)
    return xy[None, :] + body @ rot(theta).T


def min_dist_to_obstacles(points: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    if len(points) == 0 or len(obstacles) == 0:
        return np.full((len(points),), np.inf)
    # chunk not needed: 49 x cells is small enough
    return np.linalg.norm(points[:, None, :] - obstacles[None, :, :], axis=-1).min(axis=1)


src = latest_data()
data = json.loads(src.read_text())
obstacles = np.asarray(data.get("obstacle_cells", []), dtype=np.float32)
snapshots = data["snapshots"]
modes = [
    "current_fwd_lat",
    "swap_lat_fwd",
    "fwd_neglat",
    "negfwd_lat",
    "plus90",
    "minus90",
    "swap_plus90",
    "swap_minus90",
]

rows = []
for mode in modes:
    blocked_dists = []
    free_dists = []
    blocked_counts = []
    for snap in snapshots:
        vals = np.asarray(snap["scan_values"], dtype=np.float32)
        local = np.asarray(snap["scan_points_local"], dtype=np.float32)
        xy = np.asarray(snap["xy"], dtype=np.float32)
        heading = float(snap["heading"])
        # Use the same default absolute threshold for "blocked" as the current teacher snapshot.
        blocked = vals < 0.12
        world = transform_points(local, xy, heading, mode)
        d = min_dist_to_obstacles(world, obstacles)
        if blocked.any():
            blocked_dists.extend(d[blocked].tolist())
        if (~blocked).any():
            free_dists.extend(d[~blocked].tolist())
        blocked_counts.append(int(blocked.sum()))
    mean_blocked = float(np.mean(blocked_dists)) if blocked_dists else float("inf")
    median_blocked = float(np.median(blocked_dists)) if blocked_dists else float("inf")
    mean_free = float(np.mean(free_dists)) if free_dists else float("inf")
    # Better mapping: blocked points close to obstacles, free points farther away.
    score = median_blocked - 0.15 * mean_free
    rows.append(
        {
            "mode": mode,
            "score": score,
            "mean_blocked_dist": mean_blocked,
            "median_blocked_dist": median_blocked,
            "mean_free_dist": mean_free,
            "blocked_counts": blocked_counts,
        }
    )
rows.sort(key=lambda r: r["score"])
best = rows[0]

out_dir = src.parent / "rotation_diagnostic"
out_dir.mkdir(parents=True, exist_ok=True)

cols = 4
fig, axes = plt.subplots(2, cols, figsize=(5.1 * cols, 9.5), dpi=150, squeeze=False)
snap = snapshots[0]
vals = np.asarray(snap["scan_values"], dtype=np.float32)
local = np.asarray(snap["scan_points_local"], dtype=np.float32)
xy = np.asarray(snap["xy"], dtype=np.float32)
goal = np.asarray(snap["goal"], dtype=np.float32)
heading = float(snap["heading"])
blocked = vals < 0.12
for idx, mode in enumerate(modes):
    ax = axes[idx // cols][idx % cols]
    world = transform_points(local, xy, heading, mode)
    if len(obstacles):
        ax.scatter(obstacles[:, 0], obstacles[:, 1], s=7, c="black", alpha=0.18, marker="s")
    ax.scatter(
        world[:, 0],
        world[:, 1],
        c=vals,
        s=np.where(blocked, 95, 45),
        cmap="viridis",
        edgecolors=np.where(blocked, "red", "white"),
        linewidths=np.where(blocked, 1.6, 0.5),
        zorder=4,
    )
    ax.scatter(xy[0], xy[1], c="white", edgecolors="black", s=95, zorder=5)
    ax.scatter(goal[0], goal[1], marker="*", c="limegreen", edgecolors="black", s=190, zorder=5)
    ax.arrow(xy[0], xy[1], math.cos(heading) * 0.45, math.sin(heading) * 0.45, color="deepskyblue", width=0.01, head_width=0.07)
    row = next(r for r in rows if r["mode"] == mode)
    ax.set_title(f"{mode}\nmed_block={row['median_blocked_dist']:.2f}, free={row['mean_free_dist']:.2f}", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
fig.suptitle(f"Height scan mapping diagnostic. Best by score: {best['mode']}")
fig.tight_layout()
plot_path = out_dir / "height_scan_rotation_diagnostic.png"
fig.savefig(plot_path)
plt.close(fig)

fixed = dict(data)
fixed["mapping_diagnostic"] = {"source": str(src), "best_mode": best["mode"], "rows": rows}
for snap in fixed["snapshots"]:
    local = np.asarray(snap["scan_points_local"], dtype=np.float32)
    xy = np.asarray(snap["xy"], dtype=np.float32)
    heading = float(snap["heading"])
    snap["scan_points_world_original"] = snap["scan_points_world"]
    snap["scan_points_world"] = transform_points(local, xy, heading, best["mode"]).tolist()
fixed_json = out_dir / "height_scan_tuner_data_rotation_fixed.json"
fixed_json.write_text(json.dumps(fixed), encoding="utf-8")
summary = {"source": str(src), "best_mode": best["mode"], "rows": rows, "plot": str(plot_path), "fixed_json": str(fixed_json)}
(out_dir / "height_scan_rotation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary), flush=True)
PY

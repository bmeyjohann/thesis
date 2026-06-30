#!/usr/bin/env python3
"""Create a compact contact sheet from Isaac navigation trajectory JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_episodes", type=int, default=12)
    args = parser.parse_args()

    data = json.loads(Path(args.trajectory_json).read_text(encoding="utf-8"))
    trajectories = data.get("trajectories", [])
    goals = data.get("goals", [])
    successes = data.get("successes", [])
    obstacles = data.get("obstacles", [])
    n = min(args.max_episodes, len(successes), len(trajectories))
    if n <= 0:
        raise SystemExit("No completed trajectories in JSON")

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)

    all_x = []
    all_y = []
    for traj in trajectories[:n]:
        all_x.extend([p[0] for p in traj])
        all_y.extend([p[1] for p in traj])
    all_x.extend([g[0] for g in goals[:n]])
    all_y.extend([g[1] for g in goals[:n]])
    for obs in obstacles:
        all_x.append(obs["x"])
        all_y.append(obs["y"])
    pad = 0.4
    xlim = (min(all_x) - pad, max(all_x) + pad) if all_x else (-1, 1)
    ylim = (min(all_y) - pad, max(all_y) + pad) if all_y else (-1, 1)

    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            continue
        traj = trajectories[idx]
        goal = goals[idx] if idx < len(goals) else None
        ok = bool(successes[idx])
        color = "tab:green" if ok else "tab:orange"
        if len(traj) >= 2:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, color=color, linewidth=2.0)
            ax.scatter(xs[0], ys[0], color="black", marker="o", s=35, label="start")
            ax.scatter(xs[-1], ys[-1], color=color, marker="x", s=50, label="end")
        if goal is not None:
            ax.scatter([goal[0]], [goal[1]], color="tab:blue", marker="*", s=90, label="goal")
        for obs in obstacles:
            circle = plt.Circle((obs["x"], obs["y"]), obs["radius"], color="tab:red", fill=False, linewidth=2)
            ax.add_patch(circle)
        ax.set_title(f"episode {idx}: {'success' if ok else 'fail'}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

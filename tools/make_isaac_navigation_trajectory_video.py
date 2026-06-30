#!/usr/bin/env python3
"""Create a trajectory animation from Isaac navigation trajectory JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def _limits(trajectories, goals, obstacles, pad: float = 0.5):
    xs, ys = [], []
    for traj in trajectories:
        xs.extend(p[0] for p in traj)
        ys.extend(p[1] for p in traj)
    xs.extend(g[0] for g in goals)
    ys.extend(g[1] for g in goals)
    for obs in obstacles:
        r = float(obs.get("radius", 0.0))
        xs.extend([float(obs["x"]) - r, float(obs["x"]) + r])
        ys.extend([float(obs["y"]) - r, float(obs["y"]) + r])
    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0)
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_episodes", type=int, default=10)
    args = parser.parse_args()

    data = json.loads(Path(args.trajectory_json).read_text(encoding="utf-8"))
    trajectories = data.get("trajectories", [])[: args.max_episodes]
    goals = data.get("goals", [])[: args.max_episodes]
    successes = data.get("successes", [])[: args.max_episodes]
    obstacles = data.get("obstacles", [])
    if not trajectories:
        raise SystemExit("No trajectories found")

    max_len = max(len(t) for t in trajectories)
    xlim, ylim = _limits(trajectories, goals, obstacles)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(Path(args.trajectory_json).parent.name)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)

    for obs in obstacles:
        circle = plt.Circle((obs["x"], obs["y"]), obs["radius"], color="tab:red", fill=False, linewidth=2.0)
        ax.add_patch(circle)
    if goals:
        ax.scatter([g[0] for g in goals], [g[1] for g in goals], color="tab:blue", marker="*", s=80, label="goal")

    lines = []
    heads = []
    for idx, traj in enumerate(trajectories):
        ok = bool(successes[idx]) if idx < len(successes) else False
        color = "tab:green" if ok else "tab:orange"
        (line,) = ax.plot([], [], color=color, alpha=0.85, linewidth=1.8)
        head = ax.scatter([], [], color=color, s=28, marker="o")
        if traj:
            ax.scatter([traj[0][0]], [traj[0][1]], color="black", s=16, marker="o")
        lines.append(line)
        heads.append(head)
    ax.legend(loc="best")

    def update(frame):
        artists = []
        for line, head, traj in zip(lines, heads, trajectories):
            if not traj:
                continue
            end = min(frame + 1, len(traj))
            xs = [p[0] for p in traj[:end]]
            ys = [p[1] for p in traj[:end]]
            line.set_data(xs, ys)
            head.set_offsets([[xs[-1], ys[-1]]])
            artists.extend([line, head])
        return artists

    anim = FuncAnimation(fig, update, frames=max_len, interval=1000 / max(1, args.fps), blit=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        anim.save(output, writer=PillowWriter(fps=args.fps))
    else:
        anim.save(output, writer=FFMpegWriter(fps=args.fps, bitrate=1800))
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

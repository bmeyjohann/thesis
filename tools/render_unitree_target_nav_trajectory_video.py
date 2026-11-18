from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_unitree_target_terrain_policy import _draw_layout
from unitree_target_terrain import generate_target_terrain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terrain-seed", type=int, default=3)
    parser.add_argument("--preset", default="balanced")
    parser.add_argument("--arena-size", type=float, default=24.0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    data = json.loads(args.metrics.read_text())
    episodes = data["episodes"][: args.episodes]
    layout = generate_target_terrain(args.terrain_seed, preset=args.preset, arena_size=args.arena_size)
    frames: list[tuple[int, int]] = []
    for episode_index, episode in enumerate(episodes):
        trajectory = episode["trajectory_xy"]
        frames.extend((episode_index, i) for i in range(1, len(trajectory) + 1, max(1, args.stride)))
        if frames[-1] != (episode_index, len(trajectory)):
            frames.append((episode_index, len(trajectory)))

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    def draw(frame_index: int):
        episode_index, end = frames[frame_index]
        episode = episodes[episode_index]
        trajectory = np.asarray(episode["trajectory_xy"][:end], dtype=float)
        ax.clear()
        _draw_layout(ax, layout)
        if len(trajectory):
            ax.plot(trajectory[:, 0], trajectory[:, 1], color="#12263a", lw=3, zorder=8)
            ax.scatter(*trajectory[-1], color="#f7f7f2", edgecolor="black", s=90, zorder=10)
        ax.scatter(*episode["goal_xy"], marker="*", color="#59cd90", edgecolor="black", s=180, zorder=10)
        ax.set_xlim(-args.arena_size / 2, args.arena_size / 2)
        ax.set_ylim(-args.arena_size / 2, args.arena_size / 2)
        ax.set_aspect("equal")
        ax.set_title(
            f"Posthoc trajectory diagnostic (not simulator camera) | episode {episode_index + 1} | "
            f"t={end * 0.05:.2f}s"
        )
        return ()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(fig, draw, frames=len(frames), interval=50, blit=False)
    movie.save(args.output, writer=animation.FFMpegWriter(fps=20, bitrate=2400))
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

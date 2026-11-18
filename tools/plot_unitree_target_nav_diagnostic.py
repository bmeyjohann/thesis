from __future__ import annotations

import argparse
import json
import re
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_unitree_target_terrain_policy import _draw_layout
from unitree_target_navigation import _segment_feature_labels
from unitree_target_terrain import _material_at, generate_target_terrain


def _inside_obstacle(point: np.ndarray, obstacle) -> bool:
    delta = point - np.asarray(obstacle.center)
    c, s = math.cos(-obstacle.yaw), math.sin(-obstacle.yaw)
    local = np.asarray([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]])
    return bool(np.all(np.abs(local) <= 0.5 * np.asarray(obstacle.size[:2])))


def classify_point(point: np.ndarray, layout) -> str:
    for obstacle in layout.obstacles:
        if _inside_obstacle(point, obstacle):
            return "moving obstacle" if obstacle.moving else "static obstacle"
    for brush in layout.geometry:
        if np.linalg.norm(point - np.asarray(brush.center)) <= brush.radius:
            return brush.kind
    return _material_at(layout, float(point[0]), float(point[1]))


def _episode_local_coordinates(episode: dict, layout) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return trajectory/start/goal/cost points in the terrain-local frame."""
    if episode.get("trajectory_local_xy") is not None:
        return (
            np.asarray(episode["trajectory_local_xy"], dtype=float),
            np.asarray(episode["start_local_xy"], dtype=float),
            np.asarray(episode["goal_local_xy"], dtype=float),
            np.asarray(episode.get("cost_points_local_xy") or [], dtype=float).reshape(-1, 2),
        )
    start_world = np.asarray(episode["start_xy"], dtype=float)
    if episode.get("env_origin_xy") is not None:
        origin = np.asarray(episode["env_origin_xy"], dtype=float)
    else:
        # Backward compatibility for the original broken artifacts. The old
        # reset was anchored at layout.start_xy with small jitter, so rounding
        # the inferred offset by arena width recovers its environment origin.
        origin = np.round((start_world - np.asarray(layout.start_xy)) / layout.arena_size) * layout.arena_size
    trajectory = np.asarray(episode["trajectory_xy"], dtype=float) - origin
    start = start_world - origin
    goal = np.asarray(episode["goal_xy"], dtype=float) - origin
    costs = np.asarray(episode.get("cost_points_xy") or [], dtype=float).reshape(-1, 2) - origin
    return trajectory, start, goal, costs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--terrain-seed", type=int, default=3)
    parser.add_argument("--preset", default="balanced")
    parser.add_argument("--arena-size", type=float, default=24.0)
    args = parser.parse_args()

    layout = generate_target_terrain(args.terrain_seed, preset=args.preset, arena_size=args.arena_size)
    def _checkpoint_step(path: Path, fallback: int) -> int:
        name = path.parent.name.lower()
        match = re.search(r"(?:step[_-]?|final[_-]?)(\d+)(k)?", name)
        if match:
            value = int(match.group(1))
            return value * 1000 if match.group(2) else value
        return fallback

    metric_paths = list(args.root.glob("step_*/policy_metrics.json"))
    metric_paths.extend(args.root.glob("final_*/policy_metrics.json"))
    metric_paths.sort(key=lambda path: _checkpoint_step(path, 10**12))
    if not metric_paths:
        metric_paths = sorted(args.root.glob("*/**/*_metrics.json"))
    summaries = []
    failure_counts: dict[str, int] = {}
    for path in metric_paths:
        data = json.loads(path.read_text())
        step = _checkpoint_step(path, len(summaries))
        episodes = data["episodes"]
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
        for index, (ax, episode) in enumerate(zip(axes.flat, episodes)):
            _draw_layout(ax, layout)
            trajectory, start, goal, cost_points = _episode_local_coordinates(episode, layout)
            if len(trajectory):
                ax.plot(trajectory[:, 0], trajectory[:, 1], color="#12263a", lw=2.0, zorder=8)
            ax.scatter(*start, marker="o", s=60, color="#f7f7f2", edgecolor="black", zorder=10)
            ax.scatter(*goal, marker="*", s=130, color="#59cd90", edgecolor="black", zorder=10)
            outcome_label = "success" if bool(episode["success"]) else "timeout"
            if len(cost_points):
                terminal = cost_points[-1]
                outcome_label = classify_point(terminal, layout)
                failure_counts[outcome_label] = failure_counts.get(outcome_label, 0) + 1
                ax.scatter(*terminal, marker="X", s=120, color="#d7263d", edgecolor="white", zorder=12)
            ax.set_title(
                f"ep {index + 1}: {'success' if episode['success'] else 'fail'} | "
                f"{episode['episode_length_s']:.2f}s | {outcome_label}",
                fontsize=9,
            )
            ax.set_xlim(-args.arena_size / 2, args.arena_size / 2)
            ax.set_ylim(-args.arena_size / 2, args.arena_size / 2)
            ax.set_aspect("equal")
        controller = str(data.get("controller", "policy"))
        fig.suptitle(f"Target-terrain {controller} at step {step}: 5 Hz, unit-norm command", fontsize=16)
        output = args.root / f"trajectories_{path.parent.name}.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)
        summaries.append(
            (
                step,
                float(data["success_rate"]),
                float(data["costful_episode_rate"]),
                float(data["mean_episode_length_s"]),
                float(data["mean_min_goal_distance"]),
                float(
                    np.mean(
                        [
                            bool(ep.get("nontrivial_route", _segment_feature_labels(
                                _episode_local_coordinates(ep, layout)[1],
                                _episode_local_coordinates(ep, layout)[2],
                                layout,
                            )))
                            for ep in episodes
                        ]
                    )
                ),
            )
        )

    values = np.asarray(summaries, dtype=float)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    axes[0].plot(values[:, 0], values[:, 1], "o-", label="success")
    axes[0].plot(values[:, 0], values[:, 2], "s-", label="fall/cost episode")
    axes[0].set_ylabel("episode fraction")
    axes[0].legend()
    axes[1].plot(values[:, 0], values[:, 3], "o-", color="#f46036")
    axes[1].set_ylabel("mean survival (s)")
    axes[2].plot(values[:, 0], values[:, 4], "o-", color="#2e86ab")
    axes[2].set_ylabel("mean closest goal distance (m)")
    axes[3].plot(values[:, 0], values[:, 5], "o-", color="#7b2cbf")
    axes[3].set_ylabel("non-trivial route fraction")
    for ax in axes:
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.25)
    fig.suptitle("Target-terrain SAC checkpoint evaluation")
    fig.savefig(args.root / "checkpoint_summary.png", dpi=170)
    plt.close(fig)
    (args.root / "failure_surface_counts.json").write_text(json.dumps(failure_counts, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "failure_surfaces": failure_counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_unitree_target_terrain_policy import _draw_layout
from tools.plot_unitree_target_nav_diagnostic import classify_point
from unitree_target_navigation import _segment_feature_labels
from unitree_target_terrain import generate_target_terrain


def _obstacle_clear(point: np.ndarray, layout, clearance: float, half: float) -> bool:
    if np.any(np.abs(point) > half):
        return False
    for obstacle in layout.obstacles:
        radius = 0.5 * math.hypot(float(obstacle.size[0]), float(obstacle.size[1]))
        if np.linalg.norm(point - np.asarray(obstacle.center)) < radius + clearance:
            return False
    return True


def _sample_scenario(seed: int, arena_size: float, preset: str) -> dict[str, object]:
    layout = generate_target_terrain(seed, preset=preset, arena_size=arena_size)
    rng = np.random.default_rng(10_000 + seed)
    half = arena_size / 2.0 - 1.0
    if seed < 5:
        distance_range = (1.5, 3.0)
        stage = "close"
    elif seed < 10:
        distance_range = (3.0, 6.0)
        stage = "medium"
    else:
        distance_range = (6.0, 10.0)
        stage = "far"

    for _ in range(5000):
        start = rng.uniform(-half, half, size=2)
        if not _obstacle_clear(start, layout, clearance=0.55, half=half):
            continue
        angle = rng.uniform(-math.pi, math.pi)
        distance = rng.uniform(*distance_range)
        goal = start + distance * np.asarray([math.cos(angle), math.sin(angle)])
        if not _obstacle_clear(goal, layout, clearance=0.55, half=half):
            continue
        features = _segment_feature_labels(start, goal, layout)
        if not features:
            continue
        return {
            "seed": seed,
            "stage": stage,
            "start_xy": start.tolist(),
            "goal_xy": goal.tolist(),
            "distance_m": float(np.linalg.norm(goal - start)),
            "start_surface": classify_point(start, layout),
            "goal_surface": classify_point(goal, layout),
            "route_features": list(features),
        }
    raise RuntimeError(f"Could not sample a feature-crossing scenario for seed {seed}")


def _plot_page(scenarios: list[dict[str, object]], output: Path, arena_size: float, preset: str) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.8), constrained_layout=True)
    for ax, scenario in zip(axes.flat, scenarios):
        seed = int(scenario["seed"])
        layout = generate_target_terrain(seed, preset=preset, arena_size=arena_size)
        start = np.asarray(scenario["start_xy"], dtype=float)
        goal = np.asarray(scenario["goal_xy"], dtype=float)
        _draw_layout(ax, layout)
        ax.annotate(
            "",
            xy=goal,
            xytext=start,
            arrowprops={"arrowstyle": "->", "lw": 2.4, "color": "#14213d"},
            zorder=12,
        )
        ax.scatter(*start, s=75, marker="o", color="white", edgecolor="black", zorder=14)
        ax.scatter(*goal, s=150, marker="*", color="#56c596", edgecolor="black", zorder=14)
        labels = ", ".join(scenario["route_features"])
        ax.set_title(
            f"seed {seed} | {scenario['stage']} | {scenario['distance_m']:.1f} m\n"
            f"route crosses: {labels}",
            fontsize=9,
        )
        ax.set_xlim(-arena_size / 2, arena_size / 2)
        ax.set_ylim(-arena_size / 2, arena_size / 2)
        ax.set_aspect("equal")
    fig.suptitle(
        "Arena-wide start/goal proposals: obstacle-clear endpoints, feature-crossing routes\n"
        "white circle = start, green star = goal; starts may lie on traversable material/geometry",
        fontsize=15,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--arena-size", type=float, default=24.0)
    parser.add_argument("--preset", default="balanced")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [_sample_scenario(seed, args.arena_size, args.preset) for seed in range(args.count)]
    for page, offset in enumerate(range(0, len(scenarios), 10), start=1):
        _plot_page(
            scenarios[offset : offset + 10],
            args.output_dir / f"scenarios_{offset:02d}_{min(offset + 9, len(scenarios) - 1):02d}.png",
            args.arena_size,
            args.preset,
        )
    (args.output_dir / "scenario_manifest.json").write_text(json.dumps(scenarios, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "count": len(scenarios)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

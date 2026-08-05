#!/usr/bin/env python3
"""Plot matched blocked-layout evaluation of Unitree goal-only baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = (
    ("direct_goal", "Direct goal\n(scan-blind)", "direct_goal_metrics.json"),
    ("old_obstacle_free", "Old actor\n(obstacle-free train)", "policy_metrics.json"),
    ("obstacle_20k", "Obstacle train\n20k transitions", "policy_metrics.json"),
    ("obstacle_40k", "Obstacle train\n40k transitions", "policy_metrics.json"),
)


def load_row(root: Path, key: str, label: str, filename: str) -> dict:
    summary = json.loads((root / key / filename).read_text(encoding="utf-8"))
    episodes = summary["episodes"]
    safe_success = sum(
        bool(ep["success"]) and float(ep["cost_sum"]) <= 0.0 for ep in episodes
    ) / len(episodes)
    return {
        "key": key,
        "label": label,
        "episodes": len(episodes),
        "success_rate": float(summary["success_rate"]),
        "safe_success_rate": safe_success,
        "costful_episode_rate": float(summary["costful_episode_rate"]),
        "mean_cost_sum": float(summary["mean_cost_sum"]),
        "mean_collision_steps": float(summary["mean_collision_steps"]),
        "mean_time_to_success_s": float(summary["mean_time_to_success_s_success_only"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [load_row(args.root, *spec) for spec in RUNS]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "matched_goalonly_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    colors = ["#747474", "#c47a35", "#2f8293", "#246447"]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), facecolor="#f3efe5")
    for axis in axes.ravel():
        axis.set_facecolor("#fffdf7")
        axis.grid(axis="y", alpha=0.25, color="#786f62")
        axis.set_xticks(x, labels)

    width = 0.36
    axes[0, 0].bar(x - width / 2, [r["success_rate"] for r in rows], width, label="Success", color="#28789a")
    axes[0, 0].bar(x + width / 2, [r["safe_success_rate"] for r in rows], width, label="Safe success", color="#2f8f5b")
    axes[0, 0].set(title="Task performance", ylabel="Episode rate", ylim=(0, 1.02))
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(x, [r["costful_episode_rate"] for r in rows], color=colors)
    axes[0, 1].set(title="Episodes with any cost", ylabel="Costful episode rate", ylim=(0, 1.02))

    axes[1, 0].bar(x, [r["mean_cost_sum"] for r in rows], color=colors)
    axes[1, 0].set(title="Collision severity", ylabel="Mean episode cost", yscale="symlog")

    axes[1, 1].bar(x, [r["mean_time_to_success_s"] for r in rows], color=colors)
    axes[1, 1].set(title="Successful episodes only", ylabel="Mean time to success (s)")

    fig.suptitle("Unitree goal-only policies on the same 100 blocked layouts", fontsize=16, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.947,
        "One deterministic policy/controller per column; identical serialized terrain/start/goal cohort, 90 s horizon, and cost semantics.",
        ha="center",
        fontsize=9.5,
        color="#4c473e",
    )
    fig.tight_layout(rect=(0.025, 0.025, 0.98, 0.91))
    output = args.output_dir / "matched_goalonly_comparison.png"
    fig.savefig(output, dpi=190, facecolor=fig.get_facecolor())
    print(output.resolve())
    print(csv_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

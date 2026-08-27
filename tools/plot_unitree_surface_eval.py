#!/usr/bin/env python3
"""Summarize homogeneous Unitree surface evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SURFACES = ("rigid", "ice", "sand")
COLORS = {"rigid": "#70757a", "ice": "#31a8e0", "sand": "#e3ad27"}
DISPLAY_NAMES = {"rigid": "Rigid", "ice": "Ice", "sand": "Sand proxy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--policy-hz", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    trajectories = {}
    for surface in SURFACES:
        summary = json.loads((args.input_dir / surface / "summary.json").read_text())
        rows = summary["paths"]
        summaries.append(summary)
        trajectories[surface] = rows

    horizon_steps = max(int(row["steps_survived"]) for row in trajectories["rigid"])
    horizon_s = horizon_steps / args.policy_hz
    csv_rows = []
    for surface, summary in zip(SURFACES, summaries):
        rows = trajectories[surface]
        csv_rows.append(
            {
                "surface": surface,
                "trials": len(rows),
                "horizon_s": horizon_s,
                "survival_rate": summary["survival_rate"],
                "functional_traversal_success_rate": summary["functional_traversal_success_rate"],
                "mean_forward_speed_mps": summary["mean_forward_speed_mps"],
                "mean_command_progress_m": summary["mean_command_progress_m"],
                "mean_survival_time_s": np.mean([row["steps_survived"] for row in rows]) / args.policy_hz,
                "sliding_friction": summary["surface_friction"][0],
                "torsional_friction": summary["surface_friction"][1],
                "rolling_friction": summary["surface_friction"][2],
            }
        )

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    x = np.arange(len(SURFACES))
    colors = [COLORS[surface] for surface in SURFACES]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(
        "Flat locomotion policy on homogeneous surfaces\n"
        f"One policy; {len(trajectories['rigid'])} headings/surface; "
        f"{horizon_s:.0f} s at {args.policy_hz:.0f} Hz; deterministic actions; sand is friction-only",
        fontsize=15,
        fontweight="bold",
    )

    metrics = (
        ("mean_forward_speed_mps", "Forward speed", "m/s"),
        ("survival_rate", "Survival over 20 s", "fraction"),
        ("mean_survival_time_s", "Time upright", "seconds"),
        ("mean_command_progress_m", "Forward progress", "meters"),
    )
    for axis, (key, title, ylabel) in zip(axes.flat, metrics):
        values = [row[key] for row in csv_rows]
        axis.bar(x, values, color=colors, edgecolor="#222222", linewidth=0.8, alpha=0.9)
        axis.set_xticks(x, [DISPLAY_NAMES[surface] for surface in SURFACES])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value + max(values + [1.0]) * 0.025, f"{value:.3f}", ha="center", fontsize=9)

    axes[0, 0].axhline(csv_rows[0]["mean_forward_speed_mps"], color="#333333", linestyle="--", linewidth=1)
    axes[1, 0].set_ylim(0, horizon_s * 1.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()

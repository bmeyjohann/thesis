#!/usr/bin/env python3
"""Aggregate and plot the homogeneous sand calibration sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = (
    "rigid_reference",
    "friction_1p25",
    "friction_2p0",
    "friction_4p0",
    "soft_mild",
    "soft_deep",
    "margin_2cm",
    "damping_1p8",
    "damping_3p0",
    "drag_25",
    "drag_50",
    "drag_80",
    "soft_drag_35",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for name in ORDER:
        path = args.input_dir / name / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        paths = summary["paths"]
        rows.append(
            {
                "variant": name,
                "num_paths": summary["num_paths"],
                "survival_rate": summary["survival_rate"],
                "functional_success_rate": summary["functional_traversal_success_rate"],
                "mean_forward_speed_mps": summary["mean_forward_speed_mps"],
                "mean_progress_ratio": float(np.mean([row["command_progress_ratio"] for row in paths])),
                "mean_lateral_drift_m": float(np.mean([row["lateral_drift_m"] for row in paths])),
                "mean_upright_s": summary["mean_steps_survived"] * summary["step_dt"],
            }
        )
    if not rows:
        raise RuntimeError(f"No completed summaries under {args.input_dir}")

    rigid_speed = rows[0]["mean_forward_speed_mps"]
    for row in rows:
        row["speed_vs_rigid"] = row["mean_forward_speed_mps"] / rigid_speed

    csv_path = args.input_dir / "sand_calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["variant"].replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    colors = ["#73777b" if row["variant"] == "rigid_reference" else "#d7a51f" for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("Sand physics calibration with one flat-terrain locomotion policy", fontsize=16, fontweight="bold")
    panels = (
        ("speed_vs_rigid", "Speed relative to rigid", "ratio"),
        ("survival_rate", "Survival", "fraction"),
        ("mean_progress_ratio", "Command-direction progress", "ratio"),
        ("mean_lateral_drift_m", "Lateral drift", "meters"),
    )
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        values = [row[key] for row in rows]
        axis.bar(x, values, color=colors, edgecolor="#262626", linewidth=0.7)
        axis.set_xticks(x, labels, rotation=0, fontsize=8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        if key == "speed_vs_rigid":
            axis.axhspan(0.55, 0.80, color="#4e9f6b", alpha=0.14, label="desired slow range")
            axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
            axis.legend(loc="best")
    fig.savefig(args.input_dir / "sand_calibration.png", dpi=180, bbox_inches="tight")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

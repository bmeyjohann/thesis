#!/usr/bin/env python3
"""Plot speed, survival, and progress for a locomotion stress matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.input_dir.glob("*/summary.json")):
        data = json.loads(path.read_text())
        rows.append({
            "condition": path.parent.name,
            "speed_mps": float(data["mean_forward_speed_mps"]),
            "survival_rate": float(data["survival_rate"]),
            "functional_success_rate": float(data["functional_traversal_success_rate"]),
            "progress_m": float(data["mean_command_progress_m"]),
            "mean_steps_survived": float(data["mean_steps_survived"]),
        })
    if not rows:
        raise RuntimeError(f"No summaries found below {args.input_dir}")

    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["condition"].replace("_", "\n") for row in rows]
    colors = ["#64748b" if row["condition"] == "rigid" else "#d99a22" if "sand" in row["condition"] else "#2d9cdb" if "ice" in row["condition"] else "#608255" for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    metrics = (
        ("speed_mps", "Mean forward speed", "m/s"),
        ("survival_rate", "Full-horizon survival", "fraction"),
        ("functional_success_rate", "Functional traversal", "fraction"),
        ("progress_m", "Command-direction progress", "m"),
    )
    for axis, (key, title, unit) in zip(axes.flat, metrics, strict=True):
        values = [row[key] for row in rows]
        axis.bar(range(len(rows)), values, color=colors, edgecolor="#20252b", linewidth=0.6)
        axis.set_xticks(range(len(rows)), labels, fontsize=8)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Gentle-locomotion checkpoint: matched material and geometry stress tests", fontsize=16, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()

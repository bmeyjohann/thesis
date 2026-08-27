#!/usr/bin/env python3
"""Plot the long-horizon localized sand-drag refinement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        drag = 0 if path.parent.name == "rigid_reference" else int(path.parent.name.split("_")[1])
        trajectories = summary["paths"]
        rows.append(
            {
                "variant": path.parent.name,
                "drag_n_per_mps": drag,
                "survival_rate": summary["survival_rate"],
                "functional_success_rate": summary["functional_traversal_success_rate"],
                "mean_forward_speed_mps": summary["mean_forward_speed_mps"],
                "mean_progress_ratio": float(np.mean([row["command_progress_ratio"] for row in trajectories])),
                "mean_lateral_drift_m": float(np.mean([row["lateral_drift_m"] for row in trajectories])),
            }
        )
    rows.sort(key=lambda row: row["drag_n_per_mps"])
    rigid_speed = rows[0]["mean_forward_speed_mps"]
    for row in rows:
        row["speed_vs_rigid"] = row["mean_forward_speed_mps"] / rigid_speed
    with (args.input_dir / "sand_drag_refinement.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    x = [row["drag_n_per_mps"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(x, [row["speed_vs_rigid"] for row in rows], marker="o", color="#d19b18")
    axes[0].axhspan(0.6, 0.8, color="#4e9f6b", alpha=0.15, label="desired slow range")
    axes[0].set(xlabel="linear drag (N per m/s)", ylabel="speed / rigid speed", title="Slowdown")
    axes[0].legend()
    axes[1].plot(x, [row["survival_rate"] for row in rows], marker="s", label="survival")
    axes[1].plot(x, [row["functional_success_rate"] for row in rows], marker="^", label="functional traversal")
    axes[1].set(xlabel="linear drag (N per m/s)", ylabel="fraction", title="Robustness", ylim=(-0.05, 1.05))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Localized sand resistance, 20 s and 8 headings", fontsize=15, fontweight="bold")
    fig.savefig(args.input_dir / "sand_drag_refinement.png", dpi=180, bbox_inches="tight")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

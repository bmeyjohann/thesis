#!/usr/bin/env python3
"""Summarize speed, robustness, and root-height effects of sand proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "rigid_reference": "rigid",
    "drag_15": "drag 15",
    "soft_mild": "soft",
    "soft_drag_10": "soft + drag 10",
    "soft_drag_15": "soft + drag 15",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("visualizations/unitree_sand_sink_refinement"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = [name for name in LABELS if (args.input_dir / name / "summary.json").exists()]
    summaries = [json.loads((args.input_dir / name / "summary.json").read_text()) for name in names]
    reference_speed = summaries[0]["mean_forward_speed_mps"]
    reference_height = np.mean([row["mean_root_height_m"] for row in summaries[0]["paths"]])

    speed_ratio = [summary["mean_forward_speed_mps"] / reference_speed for summary in summaries]
    survival = [summary["survival_rate"] for summary in summaries]
    functional = [summary["functional_traversal_success_rate"] for summary in summaries]
    height_delta_mm = [
        1000.0
        * (np.mean([row["mean_root_height_m"] for row in summary["paths"]]) - reference_height)
        for summary in summaries
    ]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle("Sand proxy refinement: 20 s, 8 headings", fontsize=18, fontweight="bold")
    axes[0].bar(x, speed_ratio, color="#d29a16")
    axes[0].axhspan(0.7, 0.8, color="#3f8f68", alpha=0.15, label="target slowdown")
    axes[0].set_ylabel("speed / rigid speed")
    axes[0].set_title("Slowdown")
    axes[0].legend()

    width = 0.36
    axes[1].bar(x - width / 2, survival, width, label="survival", color="#2475a8")
    axes[1].bar(x + width / 2, functional, width, label="functional traversal", color="#e46b20")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("fraction")
    axes[1].set_title("Robustness")
    axes[1].legend()

    colors = ["#3f8f68" if value >= -2.0 else "#b64c3e" for value in height_delta_mm]
    axes[2].bar(x, height_delta_mm, color=colors)
    axes[2].axhline(0.0, color="#333333", linewidth=1)
    axes[2].set_ylabel("mean root-height change (mm)")
    axes[2].set_title("Compliance / sink proxy")

    for axis in axes:
        axis.set_xticks(x, [LABELS[name] for name in names], rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.22)

    output = args.output or args.input_dir / "sand_sink_refinement.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

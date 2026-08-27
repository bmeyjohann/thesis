#!/usr/bin/env python3
"""Plot survival-first results for stronger sand drag/compliance variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("visualizations/unitree_sand_survival_sweep")
    )
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*/summary.json"))
    rows = []
    for path in paths:
        summary = json.loads(path.read_text())
        heights = [item["mean_root_height_m"] for item in summary["paths"]]
        rows.append(
            {
                "name": path.parent.name,
                "survival": summary["survival_rate"],
                "speed": summary["mean_forward_speed_mps"],
                "height": float(np.mean(heights)),
            }
        )
    reference = next(row for row in rows if row["name"] == "rigid_reference")
    def sort_key(row: dict[str, float | str]) -> tuple[int, int]:
        name = str(row["name"])
        if name == "rigid_reference":
            return (0, 0)
        drag = int(name.rsplit("_", 1)[-1])
        family = 1 if name.startswith("drag_") else 2 if "soft_mild" in name else 3
        return (family, drag)

    rows.sort(key=sort_key)

    labels = [row["name"].replace("rigid_reference", "rigid").replace("_", " ") for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    fig.suptitle("Stronger sand proxies: survival-first sweep", fontsize=18, fontweight="bold")
    axes[0].bar(x, [row["survival"] for row in rows], color="#2878a8")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("survival fraction")
    axes[0].set_title("Primary criterion")
    axes[1].bar(x, [row["speed"] / reference["speed"] for row in rows], color="#d59a12")
    axes[1].set_ylabel("speed / rigid speed")
    axes[1].set_title("Resulting slowdown")
    axes[2].bar(
        x,
        [1000.0 * (row["height"] - reference["height"]) for row in rows],
        color="#aa4c40",
    )
    axes[2].axhline(0.0, color="#333333", linewidth=1)
    axes[2].set_ylabel("mean root-height change (mm)")
    axes[2].set_title("Compliance proxy")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=55, ha="right")
        axis.grid(axis="y", alpha=0.2)
    output = args.input_dir / "sand_survival_sweep.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

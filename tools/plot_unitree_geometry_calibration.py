#!/usr/bin/env python3
"""Plot controlled ramp and stair traversal limits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for summary_path in sorted(args.input_dir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        row = summary["paths"][0]
        name = summary_path.parent.name
        kind = summary["geometry_kind"]
        difficulty = int(name.split("_")[1].replace("deg", "").replace("cm", ""))
        structure_exit_m = 2.0 + 2.0 * summary["geometry_side_length"] + summary["geometry_plateau_length"]
        structure_progress_fraction = row["command_direction_progress_m"] / structure_exit_m
        rows.append(
            {
                "variant": name,
                "kind": kind,
                "difficulty": difficulty,
                "difficulty_unit": "degrees" if kind == "ramp" else "cm step height",
                "survived": int(not row["fell"]),
                "functional_success": int(row["functional_traversal_success"]),
                "duration_s": row["duration_s"],
                "progress_m": row["command_direction_progress_m"],
                "progress_ratio": row["command_progress_ratio"],
                "structure_progress_fraction": structure_progress_fraction,
                "crossed_structure": int(structure_progress_fraction >= 1.0),
                "mean_speed_mps": row["mean_forward_speed_mps"],
            }
        )
    rows.sort(key=lambda row: (row["kind"], row["difficulty"]))
    if not rows:
        raise RuntimeError(f"No geometry summaries under {args.input_dir}")
    with (args.input_dir / "geometry_calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, kind, xlabel in zip(axes, ("ramp", "stairs"), ("slope (degrees)", "step height (cm)")):
        subset = [row for row in rows if row["kind"] == kind]
        x = [row["difficulty"] for row in subset]
        axis.plot(
            x,
            [row["structure_progress_fraction"] for row in subset],
            marker="o",
            label="fraction of path to structure exit",
        )
        axis.plot(x, [row["survived"] for row in subset], marker="s", label="survived")
        axis.plot(x, [row["crossed_structure"] for row in subset], marker="^", label="crossed structure")
        axis.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
        axis.set_title(kind.capitalize())
        axis.set_xlabel(xlabel)
        axis.set_ylabel("ratio / binary outcome")
        axis.set_ylim(-0.05, 1.08)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Flat-policy bidirectional geometry capability", fontsize=15, fontweight="bold")
    fig.savefig(args.input_dir / "geometry_calibration.png", dpi=180, bbox_inches="tight")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

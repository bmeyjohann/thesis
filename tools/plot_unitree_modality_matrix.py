#!/usr/bin/env python3
"""Aggregate and plot the no-memory modality transfer matrix."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODALITIES = ("height_scan", "depth", "mono_rgb", "stereo_rgb")
GEOMETRIES = ("flat", "random_rough", "cobblestone", "stairs", "stepping_stones")
MATERIALS = ("rigid", "slippery", "sand_drag")
METRICS = (
    ("mean_velocity_tracking_error", "Velocity tracking error", "m/s", "magma_r"),
    ("fall_events_per_1000_env_steps", "Falls per 1,000 env steps", "events", "magma_r"),
    ("mean_achieved_speed", "Achieved planar speed", "m/s", "viridis"),
    ("mean_step_reward", "Mean step reward", "reward", "viridis"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["modality"], row["geometry"], row["material"])].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_dir / "modality_transfer_matrix_aggregate.csv"
    aggregate_rows = []
    for key, values in sorted(grouped.items()):
        result = {"modality": key[0], "geometry": key[1], "material": key[2], "num_seeds": len(values)}
        for metric, *_ in METRICS:
            data = np.array([float(value[metric]) for value in values])
            result[metric + "_mean"] = float(data.mean())
            result[metric + "_std"] = float(data.std(ddof=1)) if len(data) > 1 else 0.0
        aggregate_rows.append(result)
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    labels = [f"{geometry}\n{material}" for geometry in GEOMETRIES for material in MATERIALS]
    fig, axes = plt.subplots(2, 2, figsize=(22, 8.5), constrained_layout=True)
    for axis, (metric, title, unit, cmap) in zip(axes.flat, METRICS):
        matrix = np.zeros((len(MODALITIES), len(labels)))
        for row_idx, modality in enumerate(MODALITIES):
            for col_idx, (geometry, material) in enumerate(
                (pair for geometry in GEOMETRIES for pair in ((geometry, value) for value in MATERIALS))
            ):
                values = grouped[(modality, geometry, material)]
                matrix[row_idx, col_idx] = np.mean([float(value[metric]) for value in values])
        image = axis.imshow(matrix, aspect="auto", cmap=cmap)
        axis.set_title(title)
        axis.set_yticks(range(len(MODALITIES)), MODALITIES)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                axis.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, label=unit)
    fig.suptitle("Unitree G1 no-memory modality transfer matrix (mean over fixed seeds)", fontsize=16)
    figure_path = args.output_dir / "modality_transfer_matrix.png"
    fig.savefig(figure_path, dpi=180)
    print(figure_path.resolve())
    print(aggregate_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

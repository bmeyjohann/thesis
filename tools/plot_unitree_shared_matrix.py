#!/usr/bin/env python3
"""Plot modality transfer for the three shared Unitree students."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARCHITECTURES = ("nomemory", "gru", "gru_reconstruction")
MODALITIES = ("height_scan", "depth", "mono_rgb", "stereo_rgb")
GEOMETRIES = ("flat", "random_rough", "cobblestone", "stairs", "stepping_stones")
MATERIALS = ("rigid", "slippery", "sand_drag")
METRICS = (
    ("mean_velocity_tracking_error", "Velocity tracking error", "m/s", "magma_r"),
    ("fall_events_per_1000_env_steps", "Falls per 1,000 env steps", "events", "magma_r"),
    ("mean_achieved_speed", "Achieved planar speed", "m/s", "viridis"),
    ("mean_abs_action_delta", "Action jitter", "mean absolute delta", "magma_r"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.results.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["architecture"], row["modality"], row["geometry"], row["material"])].append(row)
    policies = [(architecture, modality) for architecture in ARCHITECTURES for modality in MODALITIES]
    cells = [(geometry, material) for geometry in GEOMETRIES for material in MATERIALS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(24, 15), constrained_layout=True)
    for axis, (metric, title, unit, cmap) in zip(axes.flat, METRICS):
        matrix = np.full((len(policies), len(cells)), np.nan)
        for row_index, policy in enumerate(policies):
            for column_index, cell in enumerate(cells):
                values = grouped[(*policy, *cell)]
                if values:
                    matrix[row_index, column_index] = np.mean([float(value[metric]) for value in values])
        image = axis.imshow(matrix, aspect="auto", cmap=cmap)
        axis.set_title(title)
        axis.set_yticks(range(len(policies)), [f"{a} / {m}" for a, m in policies])
        axis.set_xticks(range(len(cells)), [f"{g}\n{m}" for g, m in cells], rotation=45, ha="right")
        fig.colorbar(image, ax=axis, label=unit)
    fig.suptitle("Shared Unitree multimodal students: modality x terrain transfer", fontsize=17)
    output = args.output_dir / "shared_multimodal_transfer_matrix.png"
    fig.savefig(output, dpi=180)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

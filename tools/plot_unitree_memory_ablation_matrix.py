#!/usr/bin/env python3
"""Compare no-memory, GRU, and GRU-reconstruction transfer matrices."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GEOMETRIES = ("flat", "random_rough", "cobblestone", "stairs", "stepping_stones")
MATERIALS = ("rigid", "slippery", "sand_drag")
METRICS = (
    ("mean_velocity_tracking_error", "Velocity tracking error", "m/s", "magma_r"),
    ("fall_events_per_1000_env_steps", "Falls per 1,000 env steps", "events", "magma_r"),
    ("mean_achieved_speed", "Achieved planar speed", "m/s", "viridis"),
    ("mean_abs_action_delta", "Action jitter", "mean absolute delta", "magma_r"),
)


def _read(path: Path, architecture: str | None = None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if architecture is not None:
        for row in rows:
            row["architecture"] = architecture
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-memory", type=Path, required=True)
    parser.add_argument("--recurrent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = _read(args.no_memory, "no_memory") + _read(args.recurrent)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["architecture"], row["modality"], row["geometry"], row["material"])].append(row)
    policies = sorted({(row["architecture"], row["modality"]) for row in rows})
    cells = [(geometry, material) for geometry in GEOMETRIES for material in MATERIALS]
    labels = [f"{geometry}\n{material}" for geometry, material in cells]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(23, 15), constrained_layout=True)
    for axis, (metric, title, unit, cmap) in zip(axes.flat, METRICS):
        matrix = np.zeros((len(policies), len(cells)))
        for row_index, (architecture, modality) in enumerate(policies):
            for column_index, (geometry, material) in enumerate(cells):
                values = grouped[(architecture, modality, geometry, material)]
                matrix[row_index, column_index] = np.mean([float(value[metric]) for value in values])
        image = axis.imshow(matrix, aspect="auto", cmap=cmap)
        axis.set_title(title)
        axis.set_yticks(range(len(policies)), [f"{a} / {m}" for a, m in policies])
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=axis, label=unit)
    fig.suptitle("Unitree G1 memory and reconstruction ablation (fixed-seed means)", fontsize=17)
    output = args.output_dir / "memory_reconstruction_transfer_matrix.png"
    fig.savefig(output, dpi=180)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

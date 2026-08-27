#!/usr/bin/env python3
"""Build the fixed modality-by-physical-cell evaluation plan."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

from unitree_multimodal_locomotion import GEOMETRIES, MATERIALS, MODALITIES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 23, 37, 51, 73))
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "modality", "geometry", "material", "seed", "episodes",
            "student_checkpoint", "status", "summary_file",
        ))
        writer.writeheader()
        for modality, geometry, material, seed in product(
            MODALITIES, GEOMETRIES, MATERIALS, args.seeds
        ):
            writer.writerow({
                "modality": modality, "geometry": geometry, "material": material,
                "seed": seed, "episodes": args.episodes,
                "student_checkpoint": "", "status": "blocked_on_student_checkpoint",
                "summary_file": "",
            })
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

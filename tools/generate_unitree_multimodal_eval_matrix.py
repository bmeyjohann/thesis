#!/usr/bin/env python3
"""Generate the fixed terrain x material x modality evaluation manifest."""

from __future__ import annotations

import argparse
import csv
import sys

from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unitree_multimodal_locomotion import GEOMETRIES, MATERIALS, MODALITIES

DEFAULT_SEEDS = (11, 23, 37, 51, 73)


def rows(episodes_per_cell: int):
    for geometry, material, modality, seed, episode in product(
        GEOMETRIES, MATERIALS, MODALITIES, DEFAULT_SEEDS, range(episodes_per_cell)
    ):
        yield {
            "geometry": geometry,
            "material": material,
            "modality": modality,
            "seed": seed,
            "episode": episode,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-cell", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes_per_cell < 1:
        raise ValueError("episodes-per-cell must be positive")
    manifest = list(rows(args.episodes_per_cell))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print(f"wrote {len(manifest)} episodes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

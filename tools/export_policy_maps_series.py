#!/usr/bin/env python3
"""Generate policy/critic maps for every checkpoint in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from visualize_policy_map import generate_policy_map  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True,
                        help="Directory containing checkpoint .pt files")
    parser.add_argument("--pattern", type=str, default="*.pt",
                        help="Glob pattern of checkpoints to include (default: *.pt)")
    parser.add_argument("--output_dir", type=Path, default=Path("visualizations/series"),
                        help="Directory to store generated maps")
    parser.add_argument("--env_name", type=str, default=None,
                        help="Optional override of env name (otherwise read from checkpoint)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for evaluation (cpu or cuda)")
    parser.add_argument("--grid_resolution", type=int, default=64)
    parser.add_argument("--quiver_stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for goal sampling (used if cache missing)")
    parser.add_argument("--goal_cache", type=Path, default=None,
                        help="Optional cache file to pin goal/range across checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = sorted(args.model_dir.glob(args.pattern))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matching {args.pattern} in {args.model_dir}")

    cache_path = args.goal_cache
    if cache_path is None:
        cache_path = args.output_dir / "policy_goal.json"

    for ckpt in checkpoints:
        tag = ckpt.stem
        print(f"Generating map for {ckpt}...")
        generate_policy_map(
            model_path=ckpt,
            output_dir=args.output_dir,
            tag=tag,
            env_name=args.env_name,
            device=args.device,
            grid_resolution=args.grid_resolution,
            quiver_stride=args.quiver_stride,
            seed=args.seed,
            cache_path=cache_path,
        )


if __name__ == "__main__":
    main()

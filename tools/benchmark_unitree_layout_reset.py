#!/usr/bin/env python3
"""Benchmark Unitree terrain build cost versus persistent-tile reset cost."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_unitree_nav_baselines import make_env
from unitree_nav_layout import set_terrain_tile_indices, terrain_tile_shape


def _args(base: argparse.Namespace, rows: int, cols: int) -> argparse.Namespace:
    values = vars(base).copy()
    values.update(
        debug_terrain_rows=rows,
        debug_terrain_cols=cols,
        resample_terrain_tiles=False,
    )
    return argparse.Namespace(**values)


def _benchmark(base: argparse.Namespace, rows: int, cols: int) -> dict[str, object]:
    args = _args(base, rows, cols)
    started = time.perf_counter()
    env = make_env(args, num_envs=1, render=False)
    build_s = time.perf_counter() - started
    tile_rows, tile_cols = terrain_tile_shape(env)
    unwrapped = env.env.unwrapped
    env_ids = torch.zeros(1, dtype=torch.long, device=unwrapped.device)
    reset_times = []
    origins = []
    for tile in range(min(int(args.num_resets), tile_rows * tile_cols)):
        set_terrain_tile_indices(
            env,
            env_ids,
            torch.as_tensor([tile], device=unwrapped.device),
        )
        started = time.perf_counter()
        env.reset()
        reset_times.append(time.perf_counter() - started)
        origins.append(unwrapped.scene.env_origins[0, :2].detach().cpu().tolist())
    env.close()
    return {
        "terrain_rows": tile_rows,
        "terrain_cols": tile_cols,
        "tile_count": tile_rows * tile_cols,
        "build_s": build_s,
        "mean_tile_reset_s": sum(reset_times) / max(1, len(reset_times)),
        "max_tile_reset_s": max(reset_times, default=0.0),
        "unique_origins": len({tuple(round(v, 4) for v in origin) for origin in origins}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default="Unitree-G1-Nav-Obstacles-Safe-Collision")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--episode-length-s", type=float, default=60.0)
    parser.add_argument("--num-resets", type=int, default=12)
    parser.add_argument("--persistent-rows", type=int, default=10)
    parser.add_argument("--persistent-cols", type=int, default=20)
    parser.add_argument("--low-level-policy-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug-obstacle-width-min", type=float, default=1.15)
    parser.add_argument("--debug-obstacle-width-max", type=float, default=1.55)
    parser.add_argument("--debug-obstacle-height-min", type=float, default=1.0)
    parser.add_argument("--debug-obstacle-height-max", type=float, default=1.0)
    parser.add_argument("--debug-num-obstacles", type=int, default=3)
    parser.add_argument("--debug-platform-width", type=float, default=2.6)
    parser.add_argument("--debug-obstacle-border-width", type=float, default=0.0)
    args = parser.parse_args()
    results = {
        "persistent_large": _benchmark(args, args.persistent_rows, args.persistent_cols),
        "small_bank": _benchmark(args, 2, 2),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

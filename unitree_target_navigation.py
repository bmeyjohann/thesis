"""Shared target-terrain configuration for Unitree navigation train/eval."""

from __future__ import annotations

import argparse
import math


def configure_target_navigation_terrain(env_cfg, args, *, num_envs: int) -> bool:
    """Replace the legacy obstacle tiles with deterministic target arenas."""
    if not bool(getattr(args, "target_terrain", False)):
        return False

    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
    from unitree_target_terrain import TargetArenaTerrainCfg

    arena_size = float(getattr(args, "target_terrain_arena_size", 24.0))
    configured_seed = int(getattr(args, "target_terrain_seed", -1))
    seed = configured_seed if configured_seed >= 0 else int(getattr(args, "seed", 0))
    rows = max(1, int(math.floor(math.sqrt(num_envs))))
    cols = int(math.ceil(num_envs / rows))
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=seed,
        curriculum=False,
        size=(arena_size, arena_size),
        border_width=1.0,
        num_rows=rows,
        num_cols=cols,
        color_scheme="height",
        sub_terrains={
            "target": TargetArenaTerrainCfg(
                proportion=1.0,
                size=(arena_size, arena_size),
                seed=seed,
                preset=str(getattr(args, "target_terrain_preset", "balanced")),
                material_resolution=float(getattr(args, "target_terrain_material_resolution", 1.0)),
            )
        },
        add_lights=True,
    )
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.curriculum.pop("terrain_levels", None)
    env_cfg.events.pop("randomize_terrain", None)
    # The legacy obstacle task treats any non-foot leg contact with the entire
    # terrain body as an obstacle collision. Direct-geom ramps/material tiles
    # therefore produce false collision cost during ordinary traversal. Keep
    # fall cost/termination, but remove that legacy heightfield-specific term.
    if getattr(env_cfg, "costs", None) is not None:
        env_cfg.costs.pop("collision", None)
    env_cfg.scene.sensors = tuple(
        sensor
        for sensor in tuple(env_cfg.scene.sensors or ())
        if getattr(sensor, "name", "") != "obstacle_collision"
    )
    env_cfg.sim.nconmax = max(int(getattr(env_cfg.sim, "nconmax", 0) or 0), 512)
    return True


def add_target_terrain_args(parser) -> None:
    parser.add_argument("--target-terrain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-terrain-preset", choices=["balanced", "traversal", "navigation"], default="balanced")
    parser.add_argument("--target-terrain-seed", type=int, default=-1)
    parser.add_argument("--target-terrain-arena-size", type=float, default=24.0)
    parser.add_argument("--target-terrain-material-resolution", type=float, default=1.0)

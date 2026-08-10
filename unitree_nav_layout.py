"""Fast layout randomization over a pre-generated Unitree terrain bank."""

from __future__ import annotations

import torch


def terrain_tile_shape(env) -> tuple[int, int]:
    terrain = env.env.unwrapped.scene.terrain if hasattr(env, "env") else env.scene.terrain
    origins = terrain.terrain_origins
    if origins is None or origins.ndim != 3:
        return 0, 0
    return int(origins.shape[0]), int(origins.shape[1])


def set_terrain_tile_indices(env, env_ids: torch.Tensor, tile_indices: torch.Tensor) -> None:
    """Move environments to selected pre-generated terrain tiles.

    This only changes the reset origin. The terrain mesh/heightfields are not
    regenerated, so the following normal reset places robot and goal on the new
    tile without paying Isaac/MuJoCo build cost.
    """
    unwrapped = env.env.unwrapped if hasattr(env, "env") else env
    terrain = unwrapped.scene.terrain
    origins = terrain.terrain_origins
    if origins is None or origins.ndim != 3:
        return
    env_ids = env_ids.to(device=origins.device, dtype=torch.long).reshape(-1)
    tile_indices = tile_indices.to(device=origins.device, dtype=torch.long).reshape(-1)
    rows, cols = int(origins.shape[0]), int(origins.shape[1])
    if tile_indices.numel() != env_ids.numel():
        raise ValueError("tile_indices must have one value per env id")
    tile_indices = torch.remainder(tile_indices, rows * cols)
    levels = torch.div(tile_indices, cols, rounding_mode="floor")
    types = torch.remainder(tile_indices, cols)
    terrain.terrain_levels[env_ids] = levels
    terrain.terrain_types[env_ids] = types
    terrain.env_origins[env_ids] = origins[levels, types]


def reset_terrain_tiles(env, env_ids: torch.Tensor, *, unique: bool = True) -> None:
    """Reset event that cycles through a shuffled terrain bank without replacement."""
    rows, cols = terrain_tile_shape(env)
    if rows <= 0 or cols <= 0:
        return
    env_ids = env_ids.reshape(-1)
    count = rows * cols
    terrain = env.scene.terrain
    if unique and env_ids.numel() <= count:
        order = getattr(terrain, "_unitree_nav_tile_order", None)
        cursor = int(getattr(terrain, "_unitree_nav_tile_cursor", 0))
        if order is None or int(order.numel()) != count:
            order = torch.randperm(count, device=env_ids.device)
            cursor = 0
        selected = []
        remaining = int(env_ids.numel())
        while remaining > 0:
            available = count - cursor
            take = min(remaining, available)
            selected.append(order[cursor : cursor + take])
            cursor += take
            remaining -= take
            if cursor >= count:
                order = torch.randperm(count, device=env_ids.device)
                cursor = 0
        tile_indices = torch.cat(selected, dim=0)
        terrain._unitree_nav_tile_order = order
        terrain._unitree_nav_tile_cursor = cursor
    else:
        tile_indices = torch.randint(count, (env_ids.numel(),), device=env_ids.device)
    set_terrain_tile_indices(env, env_ids, tile_indices)


def configure_start_position_range(env_cfg, *, half_width: float) -> None:
    """Override root-reset x/y jitter while preserving all other reset settings."""
    if float(half_width) <= 0.0:
        return
    reset_cfg = env_cfg.events.get("reset_base")
    if reset_cfg is None:
        raise ValueError("start-position randomization requires a reset_base event")
    pose_range = dict(reset_cfg.params.get("pose_range", {}))
    pose_range["x"] = (-float(half_width), float(half_width))
    pose_range["y"] = (-float(half_width), float(half_width))
    reset_cfg.params = {**reset_cfg.params, "pose_range": pose_range}


def configure_terrain_tile_resets(env_cfg, *, enabled: bool) -> None:
    """Prepend tile-origin randomization before robot reset events."""
    if not enabled:
        return
    from mjlab.managers.event_manager import EventTermCfg

    env_cfg.events = {
        "resample_terrain_tile": EventTermCfg(
            func=reset_terrain_tiles,
            mode="reset",
            params={"unique": True},
        ),
        **env_cfg.events,
    }

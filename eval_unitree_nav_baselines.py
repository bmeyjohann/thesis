#!/usr/bin/env python3
"""Evaluate/video Unitree G1 navigation baselines.

Baselines:
- direct_goal: turn/drive directly toward pose_command, ignores height_scan.
- scan_teacher: observation-only expert using pose_command + height_scan.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_unitree_nav_thesis import (
    DEFAULT_LOW_LEVEL,
    ROOT,
    ScanTeacherState,
    _extract_actor_obs,
    _extract_cost,
    _setup_unitree_imports,
    scan_teacher_action,
)
from unitree_nav_geom_teacher import GeomTeacherState, local_geometry_scan_teacher_action
from unitree_nav_eval_manifest import (
    apply_layout_entries,
    capture_layout_entries,
    load_layout_manifest,
    save_layout_manifest,
)
from unitree_nav_observation import current_unitree_scan_obs, prepare_unitree_actor_obs, unitree_goal_distance


def _frontal_obstacle_mask(obs: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Detect obstacle evidence in the policy-visible forward scan corridor."""
    current = current_unitree_scan_obs(
        obs,
        scan_history=int(getattr(args, "scan_history", 1)),
        action_history=int(getattr(args, "action_history", 0)),
    )
    scan_values = current[:, 9:]
    pattern = str(getattr(args, "height_scan_pattern", "grid"))
    if pattern == "forward_frustum":
        from unitree_nav_scan import scan_lateral_forward_coordinates

        side = int(getattr(args, "height_scan_frustum_side", 17))
        if scan_values.shape[1] != side * side:
            raise ValueError(
                f"Expected {side}x{side} forward-frustum scan, got {scan_values.shape[1]} values"
            )
        scan = torch.nan_to_num(scan_values.reshape(-1, side, side), nan=1.0)
        lateral, forward = scan_lateral_forward_coordinates(
            pattern=pattern,
            side=side,
            forward_size=float(getattr(args, "height_scan_forward_size", 3.0) or 3.0),
            lateral_size=float(getattr(args, "height_scan_lateral_size", 3.0) or 3.0),
            frustum_near=float(getattr(args, "height_scan_frustum_near", 0.25)),
            frustum_far=float(getattr(args, "height_scan_frustum_far", 4.0)),
            frustum_fov_deg=float(getattr(args, "height_scan_frustum_fov_deg", 70.0)),
            device=scan.device,
        )
    else:
        resolution = float(args.height_scan_resolution)
        rows = int(round(float(args.height_scan_lateral_size) / resolution)) + 1
        cols = int(round(float(args.height_scan_forward_size) / resolution)) + 1
        if scan_values.shape[1] != rows * cols:
            raise ValueError(
                f"Expected {rows}x{cols} scan from configured footprint, got {scan_values.shape[1]} values"
            )
        scan = torch.nan_to_num(scan_values.reshape(-1, rows, cols), nan=1.0)
        lateral_axis = torch.linspace(
            -float(args.height_scan_lateral_size) / 2.0,
            float(args.height_scan_lateral_size) / 2.0,
            rows,
            device=scan.device,
        )
        forward_axis = torch.linspace(
            -float(args.height_scan_forward_size) / 2.0,
            float(args.height_scan_forward_size) / 2.0,
            cols,
            device=scan.device,
        )
        lateral, forward = torch.meshgrid(lateral_axis, forward_axis, indexing="ij")

    corridor = (
        (lateral.abs() <= float(args.decisiveness_front_half_width))
        & (forward >= float(args.decisiveness_front_min_distance))
        & (forward <= float(args.decisiveness_front_max_distance))
    )
    blocked = scan < float(args.teacher_scan_block_threshold)
    if float(args.teacher_scan_block_delta) > 0.0:
        reference = torch.quantile(scan.flatten(start_dim=1), 0.9, dim=1)[:, None, None]
        blocked |= scan < (reference - float(args.teacher_scan_block_delta))
    return (blocked & corridor[None, :, :]).flatten(start_dim=1).any(dim=1)


def _prepare_policy_obs(raw_obs: Any, args: argparse.Namespace) -> torch.Tensor:
    obs = _extract_actor_obs(raw_obs).to(args.device, dtype=torch.float32)
    return prepare_unitree_actor_obs(
        obs,
        mask_proprioception=bool(getattr(args, "mask_proprioception", False)),
        mask_goal_heading=bool(args.mask_goal_heading),
        mask_height_scan=bool(getattr(args, "mask_height_scan", False)),
        goal_encoding=str(getattr(args, "goal_encoding", "cartesian")),
        goal_distance_scale=float(getattr(args, "goal_distance_scale", 14.0)),
        velocity_scale=float(getattr(args, "velocity_scale", 1.0)),
    )


def _apply_debug_obstacle_overrides(args: argparse.Namespace, env_cfg) -> None:
    """Optionally make the heightfield obstacle distribution easier to inspect."""
    width_min = float(getattr(args, "debug_obstacle_width_min", 0.0))
    width_max = float(getattr(args, "debug_obstacle_width_max", 0.0))
    height_min = float(getattr(args, "debug_obstacle_height_min", 0.0))
    height_max = float(getattr(args, "debug_obstacle_height_max", 0.0))
    num_obstacles = int(getattr(args, "debug_num_obstacles", 0))
    platform_width = float(getattr(args, "debug_platform_width", 0.0))
    border_width = float(getattr(args, "debug_obstacle_border_width", 0.0))
    terrain_rows = int(getattr(args, "debug_terrain_rows", 0))
    terrain_cols = int(getattr(args, "debug_terrain_cols", 0))
    disable_obstacles = bool(getattr(args, "disable_obstacles", False))
    if max(
        width_min,
        width_max,
        height_min,
        height_max,
        num_obstacles,
        platform_width,
        border_width,
        terrain_rows,
        terrain_cols,
    ) <= 0 and not disable_obstacles:
        return
    terrain = getattr(env_cfg.scene, "terrain", None)
    generator = getattr(terrain, "terrain_generator", None) if terrain is not None else None
    if generator is None or "discrete_obstacles" not in getattr(generator, "sub_terrains", {}):
        raise ValueError("--debug-obstacle-* overrides require a discrete_obstacles heightfield terrain")
    # Avoid mutating the globally registered task config object.
    env_cfg.scene.terrain.terrain_generator = copy.deepcopy(generator)
    obstacle_cfg = env_cfg.scene.terrain.terrain_generator.sub_terrains["discrete_obstacles"]
    if bool(getattr(args, "strict_min_size_obstacles", False)):
        from unitree_nav_terrain import enable_strict_min_size_obstacles

        enable_strict_min_size_obstacles(obstacle_cfg)
    if width_min > 0.0 or width_max > 0.0:
        current = tuple(getattr(obstacle_cfg, "obstacle_width_range"))
        obstacle_cfg.obstacle_width_range = (width_min or current[0], width_max or current[1])
    if height_min > 0.0 or height_max > 0.0:
        current = tuple(getattr(obstacle_cfg, "obstacle_height_range"))
        obstacle_cfg.obstacle_height_range = (height_min or current[0], height_max or current[1])
    if disable_obstacles:
        obstacle_cfg.num_obstacles = 0
    elif num_obstacles > 0:
        obstacle_cfg.num_obstacles = num_obstacles
    if platform_width > 0.0:
        obstacle_cfg.platform_width = platform_width
    if border_width > 0.0:
        obstacle_cfg.border_width = border_width
    if terrain_rows > 0:
        env_cfg.scene.terrain.terrain_generator.num_rows = terrain_rows
    if terrain_cols > 0:
        env_cfg.scene.terrain.terrain_generator.num_cols = terrain_cols


def _goal_reached_termination(env, command_name: str, threshold: float):
    command = env.command_manager.get_command(command_name)
    return torch.linalg.norm(command[:, :2], dim=-1) <= float(threshold)


def _goal_termination_mask(env, num_envs: int, device: torch.device) -> torch.Tensor:
    manager = getattr(env.env.unwrapped, "termination_manager", None)
    if manager is not None and "goal_reached" in manager.active_terms:
        return manager.get_term("goal_reached").to(device=device).reshape(num_envs).bool().clone()
    return torch.zeros(num_envs, dtype=torch.bool, device=device)


def _apply_scan_and_goal_overrides(args: argparse.Namespace, env_cfg) -> None:
    """Keep scanner density and goal semantics identical across all entrypoints."""
    scan_pattern = str(getattr(args, "height_scan_pattern", "grid"))
    scan_resolution = float(getattr(args, "height_scan_resolution", 0.0))
    scan_forward_size = float(getattr(args, "height_scan_forward_size", 0.0))
    scan_lateral_size = float(getattr(args, "height_scan_lateral_size", 0.0))
    if scan_pattern == "forward_frustum" or scan_resolution > 0.0:
        sensors = copy.deepcopy(tuple(env_cfg.scene.sensors or ()))
        found = False
        for sensor in sensors:
            if getattr(sensor, "name", "") != "terrain_scan":
                continue
            if scan_pattern == "forward_frustum":
                from unitree_nav_scan import ForwardFrustumGridPatternCfg

                sensor.pattern = ForwardFrustumGridPatternCfg(
                    near_distance=float(getattr(args, "height_scan_frustum_near", 0.25)),
                    far_distance=float(getattr(args, "height_scan_frustum_far", 4.0)),
                    horizontal_fov_deg=float(getattr(args, "height_scan_frustum_fov_deg", 70.0)),
                    side=int(getattr(args, "height_scan_frustum_side", 17)),
                )
            else:
                sensor.pattern.resolution = scan_resolution
                current_size = tuple(sensor.pattern.size)
                sensor.pattern.size = (
                    scan_forward_size or float(current_size[0]),
                    scan_lateral_size or float(current_size[1]),
                )
            found = True
            break
        if not found:
            raise ValueError("Configured height scan requires a terrain_scan sensor")
        env_cfg.scene.sensors = sensors

    success_dist = float(getattr(args, "success_dist", 0.5))
    if "pose" in env_cfg.commands:
        env_cfg.commands["pose"].success_threshold = success_dist
    episodic_goals = str(getattr(args, "navigation_episode_mode", "episodic")) == "episodic"
    if bool(getattr(args, "terminate_on_goal", True)) and episodic_goals:
        from mjlab.managers.termination_manager import TerminationTermCfg

        env_cfg.terminations["goal_reached"] = TerminationTermCfg(
            func=_goal_reached_termination,
            params={"command_name": "pose", "threshold": success_dist},
            time_out=True,
        )
    if not episodic_goals:
        env_cfg.terminations.pop("time_out", None)


def make_env(args: argparse.Namespace, *, num_envs: int, render: bool):
    _setup_unitree_imports(Path(args.low_level_policy_path).resolve())
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    from src.envs import build_env

    env_cfg = load_env_cfg(args.task, play=render)
    if render:
        env_cfg.viewer.distance = float(getattr(args, "video_camera_distance", env_cfg.viewer.distance))
        env_cfg.viewer.elevation = float(getattr(args, "video_camera_elevation", env_cfg.viewer.elevation))
        env_cfg.viewer.azimuth = float(getattr(args, "video_camera_azimuth", env_cfg.viewer.azimuth))
        env_cfg.viewer.width = int(getattr(args, "video_width", env_cfg.viewer.width))
        env_cfg.viewer.height = int(getattr(args, "video_height", env_cfg.viewer.height))
        env_cfg.viewer.max_extra_envs = int(
            getattr(args, "video_max_extra_envs", env_cfg.viewer.max_extra_envs)
        )
        env_cfg.viewer.enable_shadows = bool(
            getattr(args, "video_enable_shadows", env_cfg.viewer.enable_shadows)
        )
        env_cfg.viewer.enable_reflections = bool(
            getattr(args, "video_enable_reflections", env_cfg.viewer.enable_reflections)
        )
    from unitree_nav_layout import configure_start_position_range, configure_terrain_tile_resets

    if hasattr(args, "seed"):
        env_cfg.seed = int(args.seed)
    from unitree_target_navigation import configure_target_navigation_terrain

    target_terrain = configure_target_navigation_terrain(env_cfg, args, num_envs=int(num_envs))
    if not target_terrain:
        _apply_debug_obstacle_overrides(args, env_cfg)
    _apply_scan_and_goal_overrides(args, env_cfg)
    terrain_generator = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
    if terrain_generator is not None and hasattr(terrain_generator, "seed"):
        # Environment reset seeding does not necessarily seed procedural terrain generation.
        terrain_generator.seed = int(args.seed)
    if not target_terrain:
        configure_terrain_tile_resets(
            env_cfg,
            enabled=bool(getattr(args, "resample_terrain_tiles", False)),
        )
    configure_start_position_range(
        env_cfg,
        half_width=float(getattr(args, "start_position_range", 0.0)),
    )
    env_cfg.scene.num_envs = int(num_envs)
    continuous_goals = str(getattr(args, "navigation_episode_mode", "episodic")) == "continuous_goals"
    environment_horizon = (
        float(getattr(args, "continuous_environment_horizon_s", 3600.0))
        if continuous_goals
        else float(args.episode_length_s)
    )
    env_cfg.episode_length_s = environment_horizon
    if "pose" in env_cfg.commands:
        env_cfg.commands["pose"].resampling_time_range = (environment_horizon, environment_horizon)
        goal_min = float(getattr(args, "goal_distance_min", 0.0))
        goal_max = float(getattr(args, "goal_distance_max", 0.0))
        if goal_min > 0.0 or goal_max > 0.0:
            if goal_min <= 0.0 or goal_max < goal_min:
                raise ValueError("goal_distance_min/max must satisfy 0 < min <= max")
            env_cfg.commands["pose"].distance_range = (goal_min, goal_max)

    env = build_env(env_cfg, device=args.device, render_mode=("rgb_array" if render else None))
    return RslRlVecEnvWrapper(env, clip_actions=1.0)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "data") and hasattr(value.data, "detach"):
        return value.data.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _terrain_obstacle_cells_by_env(env) -> list[np.ndarray]:
    """Return raised heightfield cell centers for each active env.

    This privileged geometry is only for evaluating/resetting feasible goals.
    It is not used by the observation-only scan teacher.
    """
    unwrapped = env.env.unwrapped
    env_origins = _to_numpy(unwrapped.scene.env_origins)
    if unwrapped.scene.terrain.terrain_origins is None:
        return [np.zeros((0, 2), dtype=np.float32) for _ in range(len(env_origins))]
    terrain_origins = _to_numpy(unwrapped.scene.terrain.terrain_origins).reshape(-1, 3)
    model = unwrapped.sim.model
    nrow = _to_numpy(model.hfield_nrow).astype(int)
    ncol = _to_numpy(model.hfield_ncol).astype(int)
    # Target arenas are assembled from direct MuJoCo geoms and intentionally
    # have no hfield. Their geometry is still visible to the policy scanner;
    # legacy heightfield-only feasibility diagnostics simply have no cells.
    if nrow.size == 0 or ncol.size == 0:
        return [np.zeros((0, 2), dtype=np.float32) for _ in range(len(env_origins))]
    sizes = _to_numpy(model.hfield_size)
    data = _to_numpy(model.hfield_data)
    offsets = np.concatenate([[0], np.cumsum(nrow * ncol)])
    result: list[np.ndarray] = []
    for env_origin in env_origins:
        tile_idx = int(np.argmin(np.linalg.norm(terrain_origins[:, :2] - env_origin[:2], axis=1)))
        rows = int(nrow[tile_idx])
        cols = int(ncol[tile_idx])
        height = data[offsets[tile_idx] : offsets[tile_idx + 1]].reshape(rows, cols)
        obstacle_mask = height > max(0.05, float(np.nanmax(height)) * 0.1)
        if np.any(obstacle_mask):
            sx, sy = float(sizes[tile_idx, 0]), float(sizes[tile_idx, 1])
            origin = terrain_origins[tile_idx]
            # MuJoCo heightfield data is shaped (nrow, ncol), with ncol
            # samples in x and nrow samples in y. This must match the renderer
            # conversion; otherwise privileged obstacle overlays and feasibility
            # filters are spatially wrong.
            xs = np.linspace(origin[0] - sx, origin[0] + sx, cols)
            ys = np.linspace(origin[1] - sy, origin[1] + sy, rows)
            grid_x, grid_y = np.meshgrid(xs, ys)
            result.append(np.stack([grid_x[obstacle_mask], grid_y[obstacle_mask]], axis=-1))
        else:
            result.append(np.zeros((0, 2), dtype=np.float32))
    return result


def _terrain_obstacle_cells_global(env) -> np.ndarray:
    """Return raised cell centers for the complete persistent terrain bank."""
    unwrapped = env.env.unwrapped
    terrain = unwrapped.scene.terrain
    if terrain.terrain_origins is None:
        return np.zeros((0, 2), dtype=np.float32)
    terrain_origins = _to_numpy(terrain.terrain_origins).reshape(-1, 3)
    model = unwrapped.sim.model
    nrow = _to_numpy(model.hfield_nrow).astype(int)
    ncol = _to_numpy(model.hfield_ncol).astype(int)
    sizes = _to_numpy(model.hfield_size)
    data = _to_numpy(model.hfield_data)
    offsets = np.concatenate([[0], np.cumsum(nrow * ncol)])
    cells_by_tile: list[np.ndarray] = []
    for tile_idx in range(min(len(terrain_origins), len(nrow))):
        rows = int(nrow[tile_idx])
        cols = int(ncol[tile_idx])
        height = data[offsets[tile_idx] : offsets[tile_idx + 1]].reshape(rows, cols)
        obstacle_mask = height > max(0.05, float(np.nanmax(height)) * 0.1)
        if not np.any(obstacle_mask):
            continue
        sx, sy = float(sizes[tile_idx, 0]), float(sizes[tile_idx, 1])
        origin = terrain_origins[tile_idx]
        xs = np.linspace(origin[0] - sx, origin[0] + sx, cols)
        ys = np.linspace(origin[1] - sy, origin[1] + sy, rows)
        grid_x, grid_y = np.meshgrid(xs, ys)
        cells_by_tile.append(np.stack([grid_x[obstacle_mask], grid_y[obstacle_mask]], axis=-1))
    if not cells_by_tile:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(cells_by_tile, axis=0).astype(np.float32, copy=False)


def _goal_clearances(env, obstacle_cells: list[np.ndarray]) -> torch.Tensor:
    device = torch.device(env.env.unwrapped.device)
    term = env.env.unwrapped.command_manager._terms["pose"]
    goals = _to_numpy(term._goal_pos_w)
    vals = []
    for i, cells in enumerate(obstacle_cells):
        if cells.size == 0:
            vals.append(float("inf"))
        else:
            vals.append(float(np.min(np.linalg.norm(cells - goals[i, :2], axis=1))))
    return torch.as_tensor(vals, device=device, dtype=torch.float32)


def _robot_clearances(env, obstacle_cells: list[np.ndarray]) -> torch.Tensor:
    device = torch.device(env.env.unwrapped.device)
    robot = env.env.unwrapped.scene["robot"]
    roots = _to_numpy(robot.data.root_link_pos_w)
    vals = []
    for i, cells in enumerate(obstacle_cells):
        if cells.size == 0:
            vals.append(float("inf"))
        else:
            vals.append(float(np.min(np.linalg.norm(cells - roots[i, :2], axis=1))))
    return torch.as_tensor(vals, device=device, dtype=torch.float32)


def _goal_positions_xy(env) -> np.ndarray:
    term = env.env.unwrapped.command_manager._terms["pose"]
    return _to_numpy(term._goal_pos_w)[:, :2]


def _robot_positions_xy(env) -> np.ndarray:
    robot = env.env.unwrapped.scene["robot"]
    return _to_numpy(robot.data.root_link_pos_w)[:, :2]


def _resample_continuous_goals(
    args: argparse.Namespace,
    env,
    obstacle_cells: list[np.ndarray],
    env_ids: torch.Tensor | np.ndarray,
):
    """Sample new collision-free goals without resetting simulator state."""
    term = env.env.unwrapped.command_manager._terms["pose"]
    starts = _robot_positions_xy(env)
    goals = _to_numpy(term._goal_pos_w).copy()
    headings = _to_numpy(term._goal_heading_w).copy()
    terrain = env.env.unwrapped.scene.terrain
    origins = _to_numpy(terrain.env_origins)[:, :2]
    terrain_origins = _to_numpy(terrain.terrain_origins).reshape(-1, 3)[:, :2]
    generator = getattr(getattr(env.env.unwrapped.cfg.scene, "terrain", None), "terrain_generator", None)
    tile_size = tuple(getattr(generator, "size", (8.0, 8.0))) if generator is not None else (8.0, 8.0)
    boundary_margin = max(float(getattr(args, "continuous_goal_boundary_margin", 0.5)), 0.0)
    region_mode = str(getattr(args, "continuous_goal_region_mode", "assigned_tile"))
    if region_mode == "terrain_bank":
        global_cells = _terrain_obstacle_cells_global(env)
        effective_obstacle_cells = [global_cells for _ in range(len(starts))]
        bank_lower = np.min(terrain_origins, axis=0) - np.asarray(tile_size[:2], dtype=np.float32) / 2.0
        bank_upper = np.max(terrain_origins, axis=0) + np.asarray(tile_size[:2], dtype=np.float32) / 2.0
    else:
        effective_obstacle_cells = obstacle_cells
        bank_lower = bank_upper = None
    min_distance = float(getattr(args, "continuous_goal_distance_min", 0.0))
    max_distance = float(getattr(args, "continuous_goal_distance_max", 0.0))
    if min_distance <= 0.0:
        min_distance = max(float(getattr(args, "goal_distance_min", 0.0)), 1.0)
    if max_distance < min_distance:
        max_distance = max(float(getattr(args, "goal_distance_max", 0.0)), min_distance)
    clearance_required = max(float(getattr(args, "min_goal_obstacle_clearance", 0.0)), 0.0)
    require_blocked = bool(getattr(args, "continuous_goal_require_blocked_corridor", False))
    blocked_probability = float(getattr(args, "continuous_goal_blocked_probability", 1.0 if require_blocked else 0.0))
    blocked_probability = min(1.0, max(0.0, blocked_probability))
    attempts = max(1, int(getattr(args, "continuous_goal_resample_attempts", 256)))
    selected = [int(i) for i in _to_numpy(env_ids).reshape(-1)]
    stats_by_env: dict[int, dict[str, float | int | bool]] = {}

    for i in selected:
        chosen = None
        chosen_stats = None
        cells = effective_obstacle_cells[i]
        if region_mode == "terrain_bank" and cells.size:
            planning_radius = max_distance + max(clearance_required, float(getattr(args, "blocked_corridor_radius", 0.45))) + 1.0
            cells = cells[np.linalg.norm(cells - starts[i].reshape(1, 2), axis=1) <= planning_radius]
        require_blocked_this_goal = require_blocked and float(np.random.random()) < blocked_probability
        if region_mode == "terrain_bank":
            lower = bank_lower + boundary_margin
            upper = bank_upper - boundary_margin
        else:
            lower = origins[i] - np.asarray(tile_size[:2], dtype=np.float32) / 2.0 + boundary_margin
            upper = origins[i] + np.asarray(tile_size[:2], dtype=np.float32) / 2.0 - boundary_margin
        def sample_candidate(enforce_blocked: bool):
            for _ in range(attempts):
                angle = float(np.random.uniform(-np.pi, np.pi))
                distance = float(np.random.uniform(min_distance, max_distance))
                candidate = starts[i] + distance * np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
                if np.any(candidate < lower) or np.any(candidate > upper):
                    continue
                clearance = (
                    float("inf")
                    if cells.size == 0
                    else float(np.min(np.linalg.norm(cells - candidate.reshape(1, 2), axis=1)))
                )
                if clearance < clearance_required:
                    continue
                stats = _line_obstacle_stats(
                    starts[i],
                    candidate,
                    cells,
                    corridor_radius=float(getattr(args, "blocked_corridor_radius", 0.45)),
                    ignore_end_radius=float(getattr(args, "blocked_corridor_ignore_end_radius", 0.75)),
                )
                if enforce_blocked and (
                    not bool(stats["blocked"])
                    or int(stats["blocked_cell_count"]) < int(getattr(args, "blocked_corridor_min_cells", 1))
                ):
                    continue
                return candidate, stats
            return None, None

        chosen, chosen_stats = sample_candidate(require_blocked_this_goal)
        if chosen is None and require_blocked_this_goal:
            # A local region can contain no feasible blocked route. Preserve the
            # distance and clearance constraints, but fall back to a clear route
            # rather than failing an otherwise valid continuous rollout.
            chosen, chosen_stats = sample_candidate(False)
        if chosen is None:
            raise RuntimeError(
                f"Could not sample a continuous Unitree goal for env {i} after {attempts} attempts "
                f"(distance={min_distance}-{max_distance}, clearance={clearance_required}, "
                f"region={region_mode}, tile_size={tile_size})"
            )
        goals[i, :2] = chosen
        headings[i] = 0.0
        stats_by_env[i] = chosen_stats

    term._goal_pos_w[:] = torch.as_tensor(goals, device=term._goal_pos_w.device, dtype=term._goal_pos_w.dtype)
    term._goal_heading_w[:] = torch.as_tensor(headings, device=term._goal_heading_w.device, dtype=term._goal_heading_w.dtype)
    term._update_command()
    obs_raw = _recompute_observations(env)
    clearances = _goal_clearances(env, effective_obstacle_cells)
    if bool(torch.any(clearances[torch.as_tensor(selected, device=clearances.device)] < clearance_required).item()):
        raise RuntimeError("Continuous Unitree goal clearance postcondition failed")
    return obs_raw, clearances, stats_by_env


def _terrain_tile_ids(env) -> np.ndarray:
    terrain = env.env.unwrapped.scene.terrain
    levels = _to_numpy(terrain.terrain_levels).astype(np.int64).reshape(-1)
    types = _to_numpy(terrain.terrain_types).astype(np.int64).reshape(-1)
    origins = terrain.terrain_origins
    cols = int(origins.shape[1]) if origins is not None and origins.ndim == 3 else 1
    return levels * cols + types


def _obstacle_components(cells: np.ndarray) -> list[np.ndarray]:
    """Split rasterized obstacle cells into 8-connected physical objects."""
    if cells.size == 0:
        return []
    xs = np.unique(np.round(cells[:, 0], decimals=5))
    ys = np.unique(np.round(cells[:, 1], decimals=5))
    dx = float(np.min(np.diff(xs))) if xs.size > 1 else 1.0
    dy = float(np.min(np.diff(ys))) if ys.size > 1 else 1.0
    origin = np.min(cells, axis=0)
    ij = np.rint((cells - origin.reshape(1, 2)) / np.asarray([dx, dy])).astype(np.int64)
    lookup = {tuple(index): row for row, index in enumerate(ij)}
    unseen = set(lookup)
    components: list[np.ndarray] = []
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        rows: list[int] = []
        while stack:
            current = stack.pop()
            rows.append(lookup[current])
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    if ox == 0 and oy == 0:
                        continue
                    neighbor = (current[0] + ox, current[1] + oy)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
        components.append(cells[np.asarray(rows, dtype=np.int64)])
    return components


def _set_goal_through_obstacle(
    args: argparse.Namespace,
    env,
    obstacle_cells: list[np.ndarray],
    env_ids: torch.Tensor | np.ndarray | None = None,
):
    """Place the goal behind an obstacle relative to the current start.

    This is privileged diagnostic layout setup only. It intentionally creates
    hard visual cases without giving the controller any privileged geometry.
    """
    probability = float(getattr(args, "goal_through_obstacle_prob", 0.0))
    if not bool(getattr(args, "debug_goal_through_obstacle", False)) and probability <= 0.0:
        return None
    starts = _robot_positions_xy(env)
    term = env.env.unwrapped.command_manager._terms["pose"]
    goals = _to_numpy(term._goal_pos_w).copy()
    headings = _to_numpy(term._goal_heading_w).copy()
    goal_min = float(getattr(args, "goal_distance_min", 0.0))
    goal_max = float(getattr(args, "goal_distance_max", 0.0))
    fallback_distance = float(getattr(args, "debug_goal_distance", 3.2))
    if goal_min <= 0.0 or goal_max < goal_min:
        goal_min = goal_max = fallback_distance
    blocked_goal_max = float(getattr(args, "blocked_goal_max_distance", 0.0))
    if blocked_goal_max > 0.0:
        goal_max = max(goal_max, blocked_goal_max)
    min_obstacle_dist = float(getattr(args, "debug_goal_obstacle_min_dist", 0.8))
    max_obstacle_dist = float(getattr(args, "debug_goal_obstacle_max_dist", 2.2))
    min_goal_clearance = float(getattr(args, "min_goal_obstacle_clearance", 0.0))
    corridor_radius = float(getattr(args, "blocked_corridor_radius", 0.45))
    ignore_end_radius = float(getattr(args, "blocked_corridor_ignore_end_radius", 0.75))
    min_blocked_cells = int(getattr(args, "blocked_corridor_min_cells", 1))
    distance_sampling = str(getattr(args, "blocked_goal_distance_sampling", "nearest"))
    placement_mode = str(getattr(args, "blocked_goal_placement_mode", "obstacle_multiplier"))
    multiplier_min = max(1.0, float(getattr(args, "blocked_goal_distance_multiplier_min", 1.0)))
    multiplier_max = max(multiplier_min, float(getattr(args, "blocked_goal_distance_multiplier_max", 2.0)))
    candidate_attempts = max(1, int(getattr(args, "blocked_goal_candidate_attempts", 64)))
    selected = set(range(len(obstacle_cells))) if env_ids is None else set(int(i) for i in _to_numpy(env_ids).reshape(-1))
    for i, cells in enumerate(obstacle_cells):
        if i not in selected:
            continue
        if not bool(getattr(args, "debug_goal_through_obstacle", False)) and np.random.random() > probability:
            continue
        if cells.size == 0:
            continue
        start = starts[i]
        component_info = []
        for component in _obstacle_components(cells):
            center = np.mean(component, axis=0)
            center_dist = float(np.linalg.norm(center - start))
            nearest_dist = float(np.min(np.linalg.norm(component - start.reshape(1, 2), axis=1)))
            if nearest_dist <= goal_max - ignore_end_radius and center_dist >= min_obstacle_dist:
                component_info.append((component, center, center_dist, nearest_dist))
        preferred = [item for item in component_info if item[3] <= max_obstacle_dist]
        candidates = preferred or component_info
        np.random.shuffle(candidates)
        targets = [(center, center_dist) for _, center, center_dist, _ in candidates]
        cell_targets = cells[
            (np.linalg.norm(cells - start.reshape(1, 2), axis=1) >= min_obstacle_dist)
            & (np.linalg.norm(cells - start.reshape(1, 2), axis=1) <= goal_max - ignore_end_radius)
        ].copy()
        np.random.shuffle(cell_targets)
        targets.extend(
            (target, float(np.linalg.norm(target - start)))
            for target in cell_targets[: min(128, len(cell_targets))]
        )
        chosen_goal = None
        for target, obstacle_distance in targets:
            direction = target - start
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            if placement_mode == "obstacle_multiplier":
                lower = max(goal_min, multiplier_min * obstacle_distance)
                upper = min(goal_max, multiplier_max * obstacle_distance)
                if upper < lower:
                    continue
                distances = np.random.uniform(lower, upper, size=candidate_attempts)
            else:
                count = max(2, int(np.ceil((goal_max - goal_min) / 0.1)) + 1)
                distances = np.linspace(goal_min, goal_max, num=count)
                if distance_sampling == "uniform":
                    np.random.shuffle(distances)
                elif distance_sampling == "farthest":
                    distances = distances[::-1]
            for goal_distance in distances:
                goal = start + direction * float(goal_distance)
                clearance = float(np.min(np.linalg.norm(cells - goal.reshape(1, 2), axis=1)))
                stats = _line_obstacle_stats(
                    start,
                    goal,
                    cells,
                    corridor_radius=corridor_radius,
                    ignore_end_radius=ignore_end_radius,
                )
                if clearance >= min_goal_clearance + 0.02 and bool(stats["blocked"]) and int(stats["blocked_cell_count"]) >= min_blocked_cells:
                    chosen_goal = goal
                    break
            if chosen_goal is not None:
                break
        if chosen_goal is None:
            continue
        goals[i, :2] = chosen_goal[:2]
        headings[i] = 0.0
    term._goal_pos_w[:] = torch.as_tensor(goals, device=term._goal_pos_w.device, dtype=term._goal_pos_w.dtype)
    term._goal_heading_w[:] = torch.as_tensor(headings, device=term._goal_heading_w.device, dtype=term._goal_heading_w.dtype)
    term._update_command()
    return _recompute_observations(env)


def _line_obstacle_stats(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacle_xy: np.ndarray,
    *,
    corridor_radius: float,
    ignore_end_radius: float,
) -> dict[str, float | int | bool]:
    """Measure obstacle cells intersecting the start-goal corridor.

    This is privileged diagnostic/reset logic only. It is not used by the
    observation-only teacher, which receives only pose_command + height_scan.
    """
    if obstacle_xy.size == 0:
        return {
            "blocked": False,
            "blocked_cell_count": 0,
            "blocking_component_count": 0,
            "nearest_corridor_obstacle_dist": float("inf"),
            "path_length": float(np.linalg.norm(goal_xy - start_xy)),
        }
    vec = goal_xy - start_xy
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return {
            "blocked": False,
            "blocked_cell_count": 0,
            "blocking_component_count": 0,
            "nearest_corridor_obstacle_dist": float("inf"),
            "path_length": length,
        }
    rel = obstacle_xy - start_xy.reshape(1, 2)
    t = np.clip((rel @ vec) / (length * length), 0.0, 1.0)
    closest = start_xy.reshape(1, 2) + t.reshape(-1, 1) * vec.reshape(1, 2)
    dist_to_segment = np.linalg.norm(obstacle_xy - closest, axis=1)
    dist_to_start = np.linalg.norm(obstacle_xy - start_xy.reshape(1, 2), axis=1)
    dist_to_goal = np.linalg.norm(obstacle_xy - goal_xy.reshape(1, 2), axis=1)
    between = (t > 0.0) & (t < 1.0)
    away_from_endpoints = (dist_to_start >= float(ignore_end_radius)) & (dist_to_goal >= float(ignore_end_radius))
    corridor = between & away_from_endpoints
    if not np.any(corridor):
        nearest = float("inf")
        count = 0
    else:
        nearest = float(np.min(dist_to_segment[corridor]))
        count = int(np.sum(dist_to_segment[corridor] <= float(corridor_radius)))
    blocking_components = 0
    for component in _obstacle_components(obstacle_xy):
        component_rel = component - start_xy.reshape(1, 2)
        component_t = np.clip((component_rel @ vec) / (length * length), 0.0, 1.0)
        component_closest = start_xy.reshape(1, 2) + component_t.reshape(-1, 1) * vec.reshape(1, 2)
        component_dist = np.linalg.norm(component - component_closest, axis=1)
        component_start_dist = np.linalg.norm(component - start_xy.reshape(1, 2), axis=1)
        component_goal_dist = np.linalg.norm(component - goal_xy.reshape(1, 2), axis=1)
        component_corridor = (
            (component_t > 0.0)
            & (component_t < 1.0)
            & (component_start_dist >= float(ignore_end_radius))
            & (component_goal_dist >= float(ignore_end_radius))
            & (component_dist <= float(corridor_radius))
        )
        blocking_components += int(np.any(component_corridor))
    return {
        "blocked": bool(count > 0),
        "blocked_cell_count": count,
        "blocking_component_count": blocking_components,
        "nearest_corridor_obstacle_dist": nearest,
        "path_length": length,
    }


def _layout_blocked_corridor_stats(args: argparse.Namespace, env, obstacle_cells: list[np.ndarray]) -> list[dict[str, float | int | bool]]:
    starts = _robot_positions_xy(env)
    goals = _goal_positions_xy(env)
    return [
        _line_obstacle_stats(
            starts[i],
            goals[i],
            obstacle_cells[i],
            corridor_radius=float(getattr(args, "blocked_corridor_radius", 0.45)),
            ignore_end_radius=float(getattr(args, "blocked_corridor_ignore_end_radius", 0.75)),
        )
        for i in range(len(obstacle_cells))
    ]


def _validate_required_blocked_corridors(
    args: argparse.Namespace,
    env,
    obstacle_cells: list[np.ndarray],
    env_ids: torch.Tensor | np.ndarray | None = None,
) -> list[dict[str, float | int | bool]]:
    stats = _layout_blocked_corridor_stats(args, env, obstacle_cells)
    selected = range(len(stats)) if env_ids is None else [int(i) for i in _to_numpy(env_ids).reshape(-1)]
    min_goal_clearance = float(getattr(args, "min_goal_obstacle_clearance", 0.0))
    if min_goal_clearance > 0.0:
        clearances = _goal_clearances(env, obstacle_cells).detach().cpu().numpy()
        invalid_clearance = [i for i in selected if float(clearances[i]) < min_goal_clearance]
        if invalid_clearance:
            raise RuntimeError(
                f"Goal-clearance invariant failed for Unitree env ids {invalid_clearance}: "
                f"clearances={[float(clearances[i]) for i in invalid_clearance]} required={min_goal_clearance}"
            )
    if not bool(getattr(args, "require_blocked_corridor", False)):
        return stats
    minimum = int(getattr(args, "blocked_corridor_min_cells", 1))
    invalid = [i for i in selected if not bool(stats[i]["blocked"]) or int(stats[i]["blocked_cell_count"]) < minimum]
    if invalid:
        raise RuntimeError(
            f"Blocked-corridor invariant failed for Unitree env ids {invalid}: "
            f"{[stats[i] for i in invalid]}"
        )
    return stats


def _recompute_observations(env):
    env.env.unwrapped.command_manager.compute(dt=0.0)
    manager = env.env.unwrapped.observation_manager
    if hasattr(manager, "_obs_buffer"):
        manager._obs_buffer = None
    return manager.compute(update_history=True)


def _current_goal_distance(env, num_envs: int, device: torch.device) -> torch.Tensor:
    try:
        term = env.env.unwrapped.command_manager._terms["pose"]
        cmd = term._pose_command_b[:, :2].to(device=device, dtype=torch.float32)
        return torch.linalg.norm(cmd, dim=-1).reshape(num_envs)
    except Exception:
        return torch.full((num_envs,), float("inf"), device=device)


def _resample_close_goals(
    args: argparse.Namespace,
    env,
    obstacle_cells: list[np.ndarray],
    env_ids: torch.Tensor | np.ndarray | None = None,
):
    min_clearance = float(getattr(args, "min_goal_obstacle_clearance", 0.0))
    unwrapped = env.env.unwrapped
    term = unwrapped.command_manager._terms["pose"]
    num_envs = int(term._pose_command_b.shape[0])
    selected = None
    if env_ids is not None:
        selected = torch.zeros(num_envs, dtype=torch.bool, device=term._pose_command_b.device)
        selected[torch.as_tensor(env_ids, device=selected.device, dtype=torch.long).reshape(-1)] = True

    # The first environment reset can precede the first pose-command sample,
    # leaving a zero vector that is incorrectly counted as immediate success.
    goal_distance = torch.linalg.norm(term._pose_command_b[:, :2], dim=-1)
    invalid_goal = goal_distance <= 1e-4
    if selected is not None:
        invalid_goal &= selected
    invalid_ids = torch.nonzero(invalid_goal, as_tuple=False).flatten()
    changed = invalid_ids.numel() > 0
    if changed:
        term._resample_command(invalid_ids)
        term._update_command()
    if min_clearance <= 0.0:
        return (_recompute_observations(env) if changed else None), None

    attempts = int(getattr(args, "goal_clearance_resample_attempts", 50))
    clearances = _goal_clearances(env, obstacle_cells)
    forced_blocked = bool(getattr(args, "debug_goal_through_obstacle", False)) or float(
        getattr(args, "goal_through_obstacle_prob", 0.0)
    ) > 0.0
    if forced_blocked:
        # Unconstrained command resampling would destroy the intentionally
        # blocked layout. The outer feasible-reset loop must retry instead.
        return (_recompute_observations(env) if changed else None), clearances
    for _ in range(max(0, attempts)):
        bad_mask = clearances < min_clearance
        if selected is not None:
            bad_mask &= selected
        bad = torch.nonzero(bad_mask, as_tuple=False).flatten()
        if bad.numel() == 0:
            break
        term._resample_command(bad)
        term._update_command()
        changed = True
        clearances = _goal_clearances(env, obstacle_cells)
    return (_recompute_observations(env) if changed else None), clearances


def _reset_until_feasible(args: argparse.Namespace, env):
    """Reset whole env until the privileged start/goal clearance constraints hold.

    This is an eval/diagnostic feasibility filter only. It is intentionally not
    available to the teacher policy, which remains height-map observation-only.
    """
    min_start = float(getattr(args, "min_start_obstacle_clearance", 0.0))
    min_goal = float(getattr(args, "min_goal_obstacle_clearance", 0.0))
    attempts = max(1, int(getattr(args, "start_clearance_resample_attempts", 20)))
    require_blocked = bool(getattr(args, "require_blocked_corridor", False))
    blocked_attempts = max(attempts, int(getattr(args, "blocked_corridor_resample_attempts", attempts)))
    min_blocked_cells = int(getattr(args, "blocked_corridor_min_cells", 1))
    obs_raw = None
    start_clearances = None
    goal_clearances = None
    layout_stats = None
    obstacle_cells = _terrain_obstacle_cells_by_env(env)
    found_feasible = False
    max_attempts = blocked_attempts if require_blocked else attempts
    accepted_attempt = 0
    for attempt_index in range(max_attempts):
        obs_raw, _ = env.reset()
        obstacle_cells = _terrain_obstacle_cells_by_env(env)
        debug_obs = _set_goal_through_obstacle(args, env, obstacle_cells)
        if debug_obs is not None:
            obs_raw = debug_obs
        # Forced blocked goals still need the same clearance repair as training.
        filtered_obs, goal_clearances = _resample_close_goals(args, env, obstacle_cells)
        if filtered_obs is not None:
            obs_raw = filtered_obs
        if goal_clearances is None:
            goal_clearances = _goal_clearances(env, obstacle_cells)
        start_clearances = _robot_clearances(env, obstacle_cells)
        start_ok = min_start <= 0.0 or bool(torch.all(start_clearances >= min_start).detach().cpu().item())
        goal_ok = min_goal <= 0.0 or bool(torch.all(goal_clearances >= min_goal).detach().cpu().item())
        layout_stats = _layout_blocked_corridor_stats(args, env, obstacle_cells)
        blocked_ok = (not require_blocked) or all(
            bool(stats["blocked"]) and int(stats["blocked_cell_count"]) >= min_blocked_cells
            for stats in layout_stats
        )
        if start_ok and goal_ok and blocked_ok:
            found_feasible = True
            accepted_attempt = attempt_index + 1
            break
    if obs_raw is None:
        obs_raw, _ = env.reset()
        obstacle_cells = _terrain_obstacle_cells_by_env(env)
    if layout_stats is None:
        layout_stats = _layout_blocked_corridor_stats(args, env, obstacle_cells)
    if not found_feasible and (min_start > 0.0 or min_goal > 0.0 or require_blocked):
        details = {
            "min_start_obstacle_clearance": min_start,
            "start_clearances": (
                start_clearances.detach().cpu().tolist() if start_clearances is not None else None
            ),
            "min_goal_obstacle_clearance": min_goal,
            "goal_clearances": (
                goal_clearances.detach().cpu().tolist() if goal_clearances is not None else None
            ),
            "require_blocked_corridor": require_blocked,
            "blocked_corridor_min_cells": min_blocked_cells,
            "layout_stats": layout_stats,
            "attempts": blocked_attempts if require_blocked else attempts,
        }
        raise RuntimeError(f"Could not sample a feasible Unitree navigation reset: {details}")
    if accepted_attempt > 1:
        print(
            f"[layout-resample] seed={getattr(args, 'seed', '?')} accepted deterministic "
            f"feasible reset attempt {accepted_attempt}/{max_attempts}",
            flush=True,
        )
    return obs_raw, start_clearances, goal_clearances, layout_stats, obstacle_cells


def _apply_manifest_layouts(args: argparse.Namespace, env, env_ids, entries):
    """Apply paired layouts and refresh all geometry-derived evaluation metadata."""
    apply_layout_entries(env, env_ids, entries)
    obs_raw = _recompute_observations(env)
    obstacle_cells = _terrain_obstacle_cells_by_env(env)
    start_clearances = _robot_clearances(env, obstacle_cells)
    goal_clearances = _goal_clearances(env, obstacle_cells)
    layout_stats = _validate_required_blocked_corridors(args, env, obstacle_cells, env_ids)
    return obs_raw, start_clearances, goal_clearances, layout_stats, obstacle_cells


def generate_layout_manifest(args: argparse.Namespace) -> Path:
    """Sample a fixed feasible cohort without running any controller."""
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    env = make_env(args, num_envs=args.num_envs, render=False)
    entries: list[dict[str, Any]] = []
    try:
        while len(entries) < int(args.num_episodes):
            _, _, _, _, _ = _reset_until_feasible(args, env)
            entries.extend(capture_layout_entries(env))
    finally:
        env.close()
    entries = entries[: int(args.num_episodes)]
    metadata = {
        "seed": int(args.seed),
        "task": str(args.task),
        "num_episodes": len(entries),
        "generator_num_envs": int(args.num_envs),
        "goal_distance_min": float(args.goal_distance_min),
        "goal_distance_max": float(args.goal_distance_max),
        "success_dist": float(args.success_dist),
        "require_blocked_corridor": bool(args.require_blocked_corridor),
        "blocked_corridor_min_cells": int(args.blocked_corridor_min_cells),
        "min_start_obstacle_clearance": float(args.min_start_obstacle_clearance),
        "min_goal_obstacle_clearance": float(args.min_goal_obstacle_clearance),
    }
    path = Path(args.generate_layout_manifest).resolve()
    save_layout_manifest(path, entries=entries, metadata=metadata)
    print(f"[layout-manifest] wrote {len(entries)} paired episodes to {path}", flush=True)
    return path


def _cost_terms(extras: Any, num_envs: int, device: torch.device, term_names: list[str]) -> dict[str, torch.Tensor]:
    if not isinstance(extras, dict) or "costs" not in extras or not torch.is_tensor(extras["costs"]):
        return {name: torch.zeros(num_envs, device=device) for name in term_names}
    costs = extras["costs"].to(device=device, dtype=torch.float32)
    if costs.ndim == 1:
        costs = costs.reshape(num_envs, -1)
    elif costs.shape[0] != num_envs:
        costs = costs.reshape(num_envs, -1)
    out = {}
    for idx, name in enumerate(term_names):
        if idx < costs.shape[1]:
            out[name] = costs[:, idx]
        else:
            out[name] = torch.zeros(num_envs, device=device)
    return out


def direct_goal_action(
    obs: torch.Tensor,
    *,
    max_vx: float,
    max_vy: float,
    yaw_gain: float,
    align_angle: float,
) -> torch.Tensor:
    goal_xy = obs[:, 6:8]
    goal_dist = torch.linalg.norm(goal_xy, dim=-1)
    angle = torch.atan2(goal_xy[:, 1], goal_xy[:, 0]).clamp(-math.pi / 2.0, math.pi / 2.0)
    yaw = torch.clamp(float(yaw_gain) * angle, -1.0, 1.0)
    speed_gate = (angle.abs() <= float(align_angle)).float()
    vx = float(max_vx) * torch.clamp(torch.cos(angle), min=0.0) * (0.25 + 0.75 * speed_gate)
    vy = torch.clamp(float(max_vy) * torch.sin(angle), -1.0, 1.0)
    action = torch.stack([vx, vy, yaw], dim=-1).clamp(-1.0, 1.0)
    return torch.where((goal_dist > 1e-6).unsqueeze(-1), action, torch.zeros_like(action))


def _load_policy_actor(args: argparse.Namespace, *, obs_dim: int, act_dim: int):
    from safetygym_utils.sac import build_sac

    if not getattr(args, "model_path", ""):
        raise ValueError("--controller policy requires --model-path")
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    ckpt_args = argparse.Namespace(**checkpoint.get("args", {}))
    if str(checkpoint.get("policy_family", "sac_actor")) == "hg_dagger":
        from unitree_nav_competitors import HGDaggerActorEnsemble

        actor = HGDaggerActorEnsemble(
            obs_dim=int(checkpoint.get("obs_dim", obs_dim)),
            act_dim=int(checkpoint.get("act_dim", act_dim)),
            num_envs=int(getattr(args, "num_envs", 1)),
            hidden_dim=int(getattr(ckpt_args, "hidden_dim", getattr(args, "hidden_dim", 256))),
            ensemble_size=int(getattr(ckpt_args, "hg_ensemble_size", 5)),
            use_layer_norm=bool(getattr(ckpt_args, "use_layer_norm", getattr(args, "use_layer_norm", False))),
            policy_encoder=str(getattr(ckpt_args, "policy_encoder", "mlp")),
            scan_history=int(getattr(ckpt_args, "scan_history", 1)),
            action_history=int(getattr(ckpt_args, "action_history", 0)),
            device=torch.device(args.device),
        )
        actor.load_state_dict(checkpoint["actor_state_dict"])
        actor.eval()
        return actor
    sac = build_sac(
        obs_dim=int(checkpoint.get("obs_dim", obs_dim)),
        act_dim=int(checkpoint.get("act_dim", act_dim)),
        hidden_actor=int(getattr(ckpt_args, "hidden_dim", getattr(args, "hidden_dim", 256))),
        hidden_critic=int(getattr(ckpt_args, "hidden_dim", getattr(args, "hidden_dim", 256))),
        num_critics=2,
        use_layer_norm=bool(getattr(ckpt_args, "use_layer_norm", getattr(args, "use_layer_norm", False))),
        layer_norm_eps=1e-5,
        init_scale=0.01,
        lr_actor=float(getattr(ckpt_args, "lr_actor", 3e-4)),
        lr_critic=float(getattr(ckpt_args, "lr_critic", 3e-4)),
        weight_decay=0.0,
        num_envs=int(getattr(args, "num_envs", 1)),
        device=torch.device(args.device),
        alpha_init=float(getattr(ckpt_args, "alpha_init", 0.001)),
        temporal_encoder=(
            "unitree_scan_cnn" if getattr(ckpt_args, "policy_encoder", "mlp") == "scan_cnn" else "none"
        ),
        obs_frame_stack=int(getattr(ckpt_args, "scan_history", 1)),
        unitree_action_history=int(getattr(ckpt_args, "action_history", 0)),
    )
    sac.actor.load_state_dict(checkpoint["actor_state_dict"])
    sac.actor.eval()
    return sac.actor


def controller_action(
    obs: torch.Tensor,
    args: argparse.Namespace,
    policy_actor=None,
    teacher_state: ScanTeacherState | None = None,
    env=None,
    obstacle_cells: list[np.ndarray] | None = None,
) -> torch.Tensor:
    from unitree_nav_observation import current_unitree_scan_obs

    teacher_obs = current_unitree_scan_obs(
        obs,
        scan_history=getattr(args, "scan_history", 1),
        action_history=getattr(args, "action_history", 0),
    )
    if args.controller == "direct_goal":
        return direct_goal_action(
            teacher_obs,
            max_vx=args.teacher_max_vx,
            max_vy=args.teacher_max_vy,
            yaw_gain=args.teacher_yaw_gain,
            align_angle=args.teacher_align_angle,
        )
    if args.controller == "scan_teacher":
        action, _, _ = scan_teacher_action(
            teacher_obs,
            scan_block_threshold=args.teacher_scan_block_threshold,
            scan_block_delta=args.teacher_scan_block_delta,
            goal_sector_half_width=args.teacher_sector_half_width,
            align_angle=args.teacher_align_angle,
            max_vx=args.teacher_max_vx,
            max_vy=args.teacher_max_vy,
            yaw_gain=args.teacher_yaw_gain,
            intervention_delta=0.0,
            intervene_on_blocked_goal=True,
            clearance_weight=args.teacher_clearance_weight,
            clearance_power=args.teacher_clearance_power,
            speed_clearance_scale=args.teacher_speed_clearance_scale,
            num_sectors=args.teacher_num_sectors,
            min_forward_scale=args.teacher_min_forward_scale,
            escape_risk_threshold=args.teacher_escape_risk_threshold,
            escape_forward_scale=args.teacher_escape_forward_scale,
            escape_lateral_scale=args.teacher_escape_lateral_scale,
            escape_radius=args.teacher_escape_radius,
            escape_all_directions=args.teacher_escape_all_directions,
            bypass_angle=args.teacher_bypass_angle,
            goal_stop_dist=args.teacher_goal_stop_dist,
            state=teacher_state,
            wall_follow_steps=args.teacher_wall_follow_steps,
            wall_follow_angle=args.teacher_wall_follow_angle,
            wall_follow_clear_risk=args.teacher_wall_follow_clear_risk,
            rollout_horizon=args.teacher_rollout_horizon,
            rollout_clearance=args.teacher_rollout_clearance,
            rollout_samples=args.teacher_rollout_samples,
            rollout_clearance_weight=args.teacher_rollout_clearance_weight,
            rollout_forward_bias=args.teacher_rollout_forward_bias,
            emergency_radius=args.teacher_emergency_radius,
            emergency_hard_radius=args.teacher_emergency_hard_radius,
            emergency_speed_scale=args.teacher_emergency_speed_scale,
            emergency_repulsion_weight=args.teacher_emergency_repulsion_weight,
            emergency_tangent_weight=args.teacher_emergency_tangent_weight,
            emergency_goal_weight=args.teacher_emergency_goal_weight,
            planner=getattr(args, "teacher_scan_planner", "heuristic"),
            astar_clearance=getattr(args, "teacher_scan_astar_clearance", 0.6),
            astar_cell_padding=getattr(args, "teacher_scan_astar_cell_padding", 0.35),
            astar_resolution=getattr(args, "teacher_scan_astar_resolution", 0.25),
            astar_waypoint_index=getattr(args, "teacher_scan_astar_waypoint_index", 3),
            astar_side_penalty=getattr(args, "teacher_scan_astar_side_penalty", 8.0),
            astar_commit_steps=getattr(args, "teacher_scan_astar_commit_steps", 30),
        )
        return action
    if args.controller == "geom_scan_teacher":
        if env is None or obstacle_cells is None:
            raise ValueError("geom_scan_teacher requires env and obstacle_cells")
        geom_state = teacher_state if isinstance(teacher_state, GeomTeacherState) else None
        return local_geometry_scan_teacher_action(teacher_obs, env=env, obstacle_cells=obstacle_cells, args=args, state=geom_state)
    if args.controller == "policy":
        if policy_actor is None:
            raise ValueError("policy_actor is required for --controller policy")
        with torch.no_grad():
            _, _, mean = policy_actor(obs)
        return mean.clamp(-1.0, 1.0)
    raise ValueError(f"Unsupported controller: {args.controller}")


def smooth_policy_action(
    action: torch.Tensor,
    previous: torch.Tensor,
    *,
    controller: str,
    smoothing: float,
) -> torch.Tensor:
    keep = min(0.99, max(0.0, float(smoothing)))
    if controller != "policy" or keep <= 0.0:
        return action
    return keep * previous + (1.0 - keep) * action


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    continuous_goals = str(getattr(args, "navigation_episode_mode", "episodic")) == "continuous_goals"
    if continuous_goals and str(getattr(args, "layout_manifest", "")):
        print(
            "[layout-manifest] overriding checkpoint navigation_episode_mode=continuous_goals "
            "with episodic replay for the fixed paired cohort",
            flush=True,
        )
        args.navigation_episode_mode = "episodic"
        continuous_goals = False
    manifest = (
        load_layout_manifest(Path(args.layout_manifest).resolve(), minimum_episodes=int(args.num_episodes))
        if str(getattr(args, "layout_manifest", ""))
        else None
    )
    if manifest is not None:
        # Tile indices only identify geometry within one deterministic terrain bank.
        # The manifest seed is therefore part of the layout and must override the
        # caller/checkpoint seed before constructing the environment.
        args.seed = int(manifest["metadata"]["seed"])
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    env = make_env(args, num_envs=args.num_envs, render=False)
    cost_term_names = list(getattr(env.env.unwrapped.cost_manager, "active_terms", []))
    if manifest is not None:
        if int(args.num_episodes) < int(args.num_envs):
            raise ValueError("Paired manifest evaluation requires num_episodes >= num_envs")
        obs_raw, _ = env.reset()
        initial_ids = list(range(int(args.num_envs)))
        obs_raw, start_clearances, goal_clearances, layout_stats, obstacle_cells = _apply_manifest_layouts(
            args, env, initial_ids, manifest["episodes"][: int(args.num_envs)]
        )
        print(f"[layout-manifest] replaying {args.num_episodes} episodes from {Path(args.layout_manifest).resolve()}", flush=True)
    else:
        obs_raw, start_clearances, goal_clearances, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
    from unitree_nav_observation import UnitreeScanHistory

    scan_history = UnitreeScanHistory(
        getattr(args, "scan_history", 1),
        getattr(args, "action_history", 0),
        int(env.action_space.shape[-1]),
        history_stride=getattr(args, "scan_history_stride", 1)
    )
    obs = scan_history.reset(_prepare_policy_obs(obs_raw, args))
    num_envs = int(obs.shape[0])
    target_episodes = int(args.num_episodes)
    quota_base, quota_remainder = divmod(target_episodes, num_envs)
    episode_quota = [quota_base + int(i < quota_remainder) for i in range(num_envs)]
    completed_per_env = [0 for _ in range(num_envs)]
    manifest_index_by_env = list(range(num_envs)) if manifest is not None else [-1 for _ in range(num_envs)]
    policy_actor = (
        _load_policy_actor(args, obs_dim=int(obs.shape[1]), act_dim=int(env.action_space.shape[-1]))
        if args.controller == "policy"
        else None
    )
    teacher_state = ScanTeacherState(num_envs, torch.device(args.device)) if args.controller == "scan_teacher" else None
    if args.controller == "geom_scan_teacher":
        teacher_state = GeomTeacherState.create(num_envs)
    step_dt = 0.05

    ep_return = torch.zeros(num_envs, device=args.device)
    ep_cost = torch.zeros(num_envs, device=args.device)
    ep_cost_terms = {name: torch.zeros(num_envs, device=args.device) for name in cost_term_names}
    ep_costful_steps = torch.zeros(num_envs, device=args.device)
    ep_collision_steps = torch.zeros(num_envs, device=args.device)
    ep_success = torch.zeros(num_envs, dtype=torch.bool, device=args.device)
    ep_first_success_step = torch.full((num_envs,), -1, dtype=torch.long, device=args.device)
    ep_steps = torch.zeros(num_envs, dtype=torch.long, device=args.device)
    ep_action_delta = torch.zeros(num_envs, device=args.device)
    ep_action_flips = torch.zeros(num_envs, device=args.device)
    ep_action_sum = torch.zeros(num_envs, int(env.action_space.shape[-1]), device=args.device)
    ep_action_abs_sum = torch.zeros_like(ep_action_sum)
    ep_small_action_steps = torch.zeros(num_envs, device=args.device)
    ep_front_obstacle_steps = torch.zeros(num_envs, device=args.device)
    ep_front_steering_sum = torch.zeros(num_envs, device=args.device)
    ep_front_abs_steering_sum = torch.zeros(num_envs, device=args.device)
    ep_front_small_steering_steps = torch.zeros(num_envs, device=args.device)
    ep_front_action_delta_sum = torch.zeros(num_envs, device=args.device)
    ep_min_goal_distance = torch.full((num_envs,), float("inf"), device=args.device)
    previous_action = torch.zeros(num_envs, int(env.action_space.shape[-1]), device=args.device)
    episode_start_xy = _robot_positions_xy(env).copy()
    episode_goal_xy = _goal_positions_xy(env).copy()
    episode_tile_ids = _terrain_tile_ids(env).copy()
    record_trajectories = bool(getattr(args, "record_trajectories", False))
    episode_trajectories = [[episode_start_xy[i].tolist()] for i in range(num_envs)]
    episode_cost_points = [[] for _ in range(num_envs)]
    episode_obstacle_cells = [cells.tolist() for cells in obstacle_cells]

    episodes: list[dict[str, float | int | bool | None]] = []
    start = time.time()
    while len(episodes) < int(args.num_episodes):
        # The command cache can be stale immediately after an auto-reset; use
        # the same body-relative goal that the policy receives.
        pre_dist = unitree_goal_distance(obs, goal_encoding=args.goal_encoding, goal_distance_scale=args.goal_distance_scale)
        ep_min_goal_distance = torch.minimum(ep_min_goal_distance, pre_dist)
        pre_success = (pre_dist <= float(args.success_dist)) & (~ep_success)
        ep_first_success_step[pre_success] = ep_steps[pre_success]
        ep_success |= pre_dist <= float(args.success_dist)

        with torch.no_grad():
            action = controller_action(obs, args, policy_actor, teacher_state, env=env, obstacle_cells=obstacle_cells)
            action = smooth_policy_action(
                action,
                previous_action,
                controller=args.controller,
                smoothing=args.policy_action_smoothing,
            )
        front_obstacle = _frontal_obstacle_mask(obs, args)
        steering = 0.5 * (action[:, 1] / 0.65 + action[:, 2] / 0.85)
        action_delta = torch.linalg.norm(action - previous_action, dim=-1)
        ep_action_delta += torch.linalg.norm(action - previous_action, dim=-1)
        ep_action_sum += action
        ep_action_abs_sum += action.abs()
        ep_small_action_steps += (torch.linalg.norm(action, dim=-1) < 0.15).float()
        ep_front_obstacle_steps += front_obstacle.float()
        ep_front_steering_sum += torch.where(front_obstacle, steering, 0.0)
        ep_front_abs_steering_sum += torch.where(front_obstacle, steering.abs(), 0.0)
        ep_front_small_steering_steps += (front_obstacle & (steering.abs() < 0.15)).float()
        ep_front_action_delta_sum += torch.where(front_obstacle, action_delta, 0.0)
        active = (action[:, 1:].abs() > 0.1) & (previous_action[:, 1:].abs() > 0.1)
        ep_action_flips += ((action[:, 1:] * previous_action[:, 1:] < 0.0) & active).any(dim=-1).float()
        previous_action.copy_(action)
        next_raw, reward, done, extras = env.step(action)
        terminal_success = _goal_termination_mask(env, num_envs, torch.device(args.device))
        next_current_obs = _prepare_policy_obs(next_raw, args)
        reward = reward.to(args.device, dtype=torch.float32).reshape(num_envs)
        done = done.to(args.device).reshape(num_envs).bool()
        next_obs = scan_history.step(next_current_obs, done, action=action)
        cost_vec = _extract_cost(extras, num_envs, torch.device(args.device))
        cost = cost_vec.reshape(num_envs, -1).sum(dim=1) if cost_vec.ndim > 1 else cost_vec.reshape(num_envs)
        cost_terms = _cost_terms(extras, num_envs, torch.device(args.device), cost_term_names)
        if record_trajectories:
            current_robot_xy = _robot_positions_xy(env)
            for i in range(num_envs):
                # Isaac auto-resets before env.step() returns on terminal steps.
                # Do not connect that next-episode spawn to this trajectory.
                terminal = bool(done[i].detach().cpu().item())
                point = current_robot_xy[i].tolist()
                if not terminal:
                    episode_trajectories[i].append(point)
                if float(cost[i].detach().cpu().item()) > 0.0:
                    episode_cost_points[i].append(
                        episode_trajectories[i][-1] if terminal else point
                    )

        ep_steps += 1
        ep_return += reward
        ep_cost += cost
        for name, value in cost_terms.items():
            ep_cost_terms[name] += value
        ep_costful_steps += (cost > 0.0).float()
        collision_cost = cost_terms.get("collision")
        if collision_cost is not None:
            ep_collision_steps += (collision_cost > 0.0).float()
        dist = unitree_goal_distance(next_obs, goal_encoding=args.goal_encoding, goal_distance_scale=args.goal_distance_scale)
        just_success = ((dist <= float(args.success_dist)) | terminal_success) & (~ep_success)
        ep_first_success_step[just_success] = ep_steps[just_success]
        ep_success |= (dist <= float(args.success_dist)) | terminal_success

        goal_timeout = (
            continuous_goals
            & (int(args.evaluation_task_max_steps) > 0)
            & (ep_steps >= int(args.evaluation_task_max_steps))
        )
        goal_boundary = (ep_success | goal_timeout) if continuous_goals else torch.zeros_like(done)
        task_boundary = done | goal_boundary
        if task_boundary.any():
            boundary_idx = torch.nonzero(task_boundary, as_tuple=False).flatten()
            for i in boundary_idx.tolist():
                if len(episodes) >= int(args.num_episodes):
                    break
                if completed_per_env[i] >= episode_quota[i]:
                    continue
                first_step = int(ep_first_success_step[i].detach().cpu().item())
                episodes.append(
                    {
                        "success": bool(ep_success[i].detach().cpu().item()),
                        "time_to_success_s": (first_step * step_dt if first_step >= 0 else None),
                        "episode_length_s": float(ep_steps[i].detach().cpu().item() * step_dt),
                        "cost_sum": float(ep_cost[i].detach().cpu().item()),
                        "start_obstacle_clearance": (
                            float(start_clearances[i].detach().cpu().item()) if start_clearances is not None else None
                        ),
                        "goal_obstacle_clearance": (
                            float(goal_clearances[i].detach().cpu().item()) if goal_clearances is not None else None
                        ),
                        "blocked_corridor": bool(layout_stats[i]["blocked"]) if layout_stats is not None else None,
                        "blocked_corridor_cell_count": (
                            int(layout_stats[i]["blocked_cell_count"]) if layout_stats is not None else None
                        ),
                        "blocking_component_count": (
                            int(layout_stats[i]["blocking_component_count"]) if layout_stats is not None else None
                        ),
                        "nearest_corridor_obstacle_dist": (
                            float(layout_stats[i]["nearest_corridor_obstacle_dist"]) if layout_stats is not None else None
                        ),
                        "straight_path_length": float(layout_stats[i]["path_length"]) if layout_stats is not None else None,
                        "terrain_tile_id": int(episode_tile_ids[i]),
                        "start_xy": episode_start_xy[i].tolist(),
                        "goal_xy": episode_goal_xy[i].tolist(),
                        "trajectory_xy": episode_trajectories[i] if record_trajectories else None,
                        "cost_points_xy": episode_cost_points[i] if record_trajectories else None,
                        "obstacle_cells_xy": episode_obstacle_cells[i] if record_trajectories else None,
                        **{
                            f"{name}_cost_sum": float(values[i].detach().cpu().item())
                            for name, values in ep_cost_terms.items()
                        },
                        "costful_steps": int(ep_costful_steps[i].detach().cpu().item()),
                        "collision_steps": int(ep_collision_steps[i].detach().cpu().item()),
                        "mean_action_delta": float(
                            ep_action_delta[i].detach().cpu().item() / max(1, int(ep_steps[i].item()))
                        ),
                        "action_sign_flips": int(ep_action_flips[i].detach().cpu().item()),
                        "min_goal_distance": float(ep_min_goal_distance[i].detach().cpu().item()),
                        "mean_action": (
                            ep_action_sum[i].detach().cpu().numpy() / max(1, int(ep_steps[i].item()))
                        ).tolist(),
                        "mean_abs_action": (
                            ep_action_abs_sum[i].detach().cpu().numpy() / max(1, int(ep_steps[i].item()))
                        ).tolist(),
                        "small_action_fraction": float(
                            ep_small_action_steps[i].detach().cpu().item() / max(1, int(ep_steps[i].item()))
                        ),
                        "front_obstacle_steps": int(ep_front_obstacle_steps[i].detach().cpu().item()),
                        "front_obstacle_abs_steering": float(
                            ep_front_abs_steering_sum[i].detach().cpu().item()
                            / max(1, int(ep_front_obstacle_steps[i].item()))
                        ),
                        "front_obstacle_small_steering_fraction": float(
                            ep_front_small_steering_steps[i].detach().cpu().item()
                            / max(1, int(ep_front_obstacle_steps[i].item()))
                        ),
                        "front_obstacle_action_delta": float(
                            ep_front_action_delta_sum[i].detach().cpu().item()
                            / max(1, int(ep_front_obstacle_steps[i].item()))
                        ),
                        "front_obstacle_directional_consistency": float(
                            abs(ep_front_steering_sum[i].detach().cpu().item())
                            / max(ep_front_abs_steering_sum[i].detach().cpu().item(), 1e-6)
                        ),
                        "return": float(ep_return[i].detach().cpu().item()),
                        "env_index": int(i),
                        "manifest_episode_index": (
                            int(manifest_index_by_env[i]) if manifest is not None else None
                        ),
                    }
                )
                completed_per_env[i] += 1
                print(
                    "[eval-episode] "
                    f"controller={args.controller} episode={len(episodes)} "
                    f"success={bool(ep_success[i].detach().cpu().item())} "
                    f"cost={float(ep_cost[i].detach().cpu().item()):.1f} "
                    f"costful_steps={int(ep_costful_steps[i].detach().cpu().item())} "
                    f"collision_steps={int(ep_collision_steps[i].detach().cpu().item())} "
                    f"steps={int(ep_steps[i].detach().cpu().item())}",
                    flush=True,
                )
            ep_return[boundary_idx] = 0.0
            ep_cost[boundary_idx] = 0.0
            for values in ep_cost_terms.values():
                values[boundary_idx] = 0.0
            ep_costful_steps[boundary_idx] = 0.0
            ep_collision_steps[boundary_idx] = 0.0
            ep_success[boundary_idx] = False
            ep_first_success_step[boundary_idx] = -1
            ep_steps[boundary_idx] = 0
            ep_action_delta[boundary_idx] = 0.0
            ep_action_flips[boundary_idx] = 0.0
            ep_action_sum[boundary_idx] = 0.0
            ep_action_abs_sum[boundary_idx] = 0.0
            ep_small_action_steps[boundary_idx] = 0.0
            ep_front_obstacle_steps[boundary_idx] = 0.0
            ep_front_steering_sum[boundary_idx] = 0.0
            ep_front_abs_steering_sum[boundary_idx] = 0.0
            ep_front_small_steering_steps[boundary_idx] = 0.0
            ep_front_action_delta_sum[boundary_idx] = 0.0
            ep_min_goal_distance[boundary_idx] = float("inf")
            previous_action[boundary_idx] = 0.0
            if teacher_state is not None:
                teacher_state.reset(task_boundary)

            if len(episodes) >= int(args.num_episodes):
                break

            continuation_idx = torch.nonzero(task_boundary & (~done), as_tuple=False).flatten()
            if continuation_idx.numel() > 0:
                next_raw, next_goal_clearances, next_layout_stats = _resample_continuous_goals(
                    args,
                    env,
                    obstacle_cells,
                    continuation_idx,
                )
                next_obs = scan_history.reset(
                    _prepare_policy_obs(next_raw, args),
                    continuation_idx,
                )
                current_start_clearances = _robot_clearances(env, obstacle_cells)
                start_clearances[continuation_idx] = current_start_clearances[continuation_idx]
                goal_clearances[continuation_idx] = next_goal_clearances[continuation_idx]
                for i in continuation_idx.tolist():
                    layout_stats[i] = next_layout_stats[i]

        if done.any():
            done_idx = torch.nonzero(done, as_tuple=False).flatten()
            if manifest is not None:
                replay_ids = [
                    i for i in done_idx.tolist()
                    if completed_per_env[i] < episode_quota[i]
                ]
                if replay_ids:
                    replay_indices = [i + completed_per_env[i] * num_envs for i in replay_ids]
                    replay_entries = [manifest["episodes"][index] for index in replay_indices]
                    manifest_index_by_env = list(manifest_index_by_env)
                    for env_id, manifest_index in zip(replay_ids, replay_indices):
                        manifest_index_by_env[env_id] = manifest_index
                    next_raw, replay_start_clearances, replay_goal_clearances, replay_layout_stats, obstacle_cells = _apply_manifest_layouts(
                        args, env, replay_ids, replay_entries
                    )
                    replay_ids_tensor = torch.as_tensor(
                        replay_ids, device=done_idx.device, dtype=torch.long
                    )
                    start_clearances[replay_ids_tensor] = replay_start_clearances[replay_ids_tensor]
                    goal_clearances[replay_ids_tensor] = replay_goal_clearances[replay_ids_tensor]
                    for env_id in replay_ids:
                        layout_stats[env_id] = replay_layout_stats[env_id]
                    next_obs = scan_history.reset(
                        _prepare_policy_obs(next_raw, args),
                        replay_ids_tensor,
                    )
            elif (
                num_envs == 1
                and (
                    float(getattr(args, "min_start_obstacle_clearance", 0.0)) > 0.0
                    or bool(getattr(args, "require_blocked_corridor", False))
                )
            ):
                next_raw, start_clearances, goal_clearances, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
                next_obs = scan_history.reset(_prepare_policy_obs(next_raw, args))
            else:
                if bool(getattr(args, "resample_terrain_tiles", False)):
                    obstacle_cells = _terrain_obstacle_cells_by_env(env)
                next_start_clearances = _robot_clearances(env, obstacle_cells)
                if start_clearances is None:
                    start_clearances = next_start_clearances
                else:
                    start_clearances[done_idx] = next_start_clearances[done_idx]
                adjusted_obs = _set_goal_through_obstacle(args, env, obstacle_cells, env_ids=done_idx)
                if adjusted_obs is not None:
                    next_raw = adjusted_obs
                    next_obs = scan_history.reset(
                        _prepare_policy_obs(next_raw, args), done_idx
                    )
                # Repair only freshly reset envs. Re-sampling every command here
                # silently changes goals in still-active vectorized episodes.
                filtered_obs, next_goal_clearances = _resample_close_goals(
                    args,
                    env,
                    obstacle_cells,
                    env_ids=done_idx,
                )
                if filtered_obs is not None:
                    next_raw = filtered_obs
                    next_obs = scan_history.reset(
                        _prepare_policy_obs(next_raw, args), done_idx
                    )
                if next_goal_clearances is None:
                    next_goal_clearances = _goal_clearances(env, obstacle_cells)
                if goal_clearances is None:
                    goal_clearances = next_goal_clearances
                else:
                    goal_clearances[done_idx] = next_goal_clearances[done_idx]
                next_layout_stats = _validate_required_blocked_corridors(args, env, obstacle_cells, done_idx)
                if layout_stats is None:
                    layout_stats = next_layout_stats
                else:
                    for i in done_idx.tolist():
                        layout_stats[i] = next_layout_stats[i]
            current_start_xy = _robot_positions_xy(env)
            current_goal_xy = _goal_positions_xy(env)
            current_tile_ids = _terrain_tile_ids(env)
            for i in done_idx.tolist():
                episode_start_xy[i] = current_start_xy[i]
                episode_goal_xy[i] = current_goal_xy[i]
                episode_tile_ids[i] = current_tile_ids[i]
                if record_trajectories:
                    episode_trajectories[i] = [current_start_xy[i].tolist()]
                    episode_cost_points[i] = []
                    episode_obstacle_cells[i] = obstacle_cells[i].tolist()
        if task_boundary.any():
            current_start_xy = _robot_positions_xy(env)
            current_goal_xy = _goal_positions_xy(env)
            current_tile_ids = _terrain_tile_ids(env)
            for i in boundary_idx.tolist():
                episode_start_xy[i] = current_start_xy[i]
                episode_goal_xy[i] = current_goal_xy[i]
                episode_tile_ids[i] = current_tile_ids[i]
                if record_trajectories:
                    episode_trajectories[i] = [current_start_xy[i].tolist()]
                    episode_cost_points[i] = []
                    episode_obstacle_cells[i] = obstacle_cells[i].tolist()
        obs = next_obs

    env.close()
    if manifest is not None:
        episodes.sort(key=lambda episode: int(episode["manifest_episode_index"]))
    successes = [float(ep["success"]) for ep in episodes]
    costs = [float(ep["cost_sum"]) for ep in episodes]
    costful_steps = [float(ep["costful_steps"]) for ep in episodes]
    collision_steps = [float(ep["collision_steps"]) for ep in episodes]
    lengths = [float(ep["episode_length_s"]) for ep in episodes]
    tts = [float(ep["time_to_success_s"]) for ep in episodes if ep["time_to_success_s"] is not None]
    returns = [float(ep["return"]) for ep in episodes]
    tile_ids = [int(ep["terrain_tile_id"]) for ep in episodes]
    rounded_layouts = {
        (
            int(ep["terrain_tile_id"]),
            tuple(round(float(value), 2) for value in ep["start_xy"]),
            tuple(round(float(value), 2) for value in ep["goal_xy"]),
        )
        for ep in episodes
    }
    action_deltas = [float(ep["mean_action_delta"]) for ep in episodes]
    action_flips = [float(ep["action_sign_flips"]) for ep in episodes]
    min_goal_distances = [float(ep["min_goal_distance"]) for ep in episodes]
    action_means = [list(ep["mean_action"]) for ep in episodes]
    abs_action_means = [list(ep["mean_abs_action"]) for ep in episodes]
    small_action_fractions = [float(ep["small_action_fraction"]) for ep in episodes]
    front_steps = [int(ep["front_obstacle_steps"]) for ep in episodes]
    front_step_total = max(1, sum(front_steps))
    blocked_counts = [float(ep.get("blocked_corridor_cell_count") or 0.0) for ep in episodes]
    blocking_component_counts = [float(ep.get("blocking_component_count") or 0.0) for ep in episodes]
    path_lengths = [float(ep["straight_path_length"]) for ep in episodes if ep.get("straight_path_length") is not None]
    corridor_dists = [
        float(ep["nearest_corridor_obstacle_dist"])
        for ep in episodes
        if ep.get("nearest_corridor_obstacle_dist") is not None
        and math.isfinite(float(ep["nearest_corridor_obstacle_dist"]))
    ]
    term_means = {
        f"mean_{name}_cost_sum": sum(float(ep.get(f"{name}_cost_sum", 0.0)) for ep in episodes) / max(1, len(episodes))
        for name in cost_term_names
    }
    summary = {
        "controller": args.controller,
        "task": args.task,
        "navigation_episode_mode": str(args.navigation_episode_mode),
        "num_episodes": len(episodes),
        "success_rate": sum(successes) / max(1, len(successes)),
        "mean_time_to_success_s_success_only": sum(tts) / max(1, len(tts)),
        "mean_episode_length_s": sum(lengths) / max(1, len(lengths)),
        "mean_cost_sum": sum(costs) / max(1, len(costs)),
        "costful_episode_rate": sum(1.0 for c in costs if c > 0.0) / max(1, len(costs)),
        "mean_costful_steps": sum(costful_steps) / max(1, len(costful_steps)),
        "mean_collision_steps": sum(collision_steps) / max(1, len(collision_steps)),
        "mean_return": sum(returns) / max(1, len(returns)),
        "unique_terrain_tiles": int(len(set(tile_ids))),
        "unique_layout_triplets_2cm": int(len(rounded_layouts)),
        "duplicate_layout_triplet_rate_2cm": float(1.0 - len(rounded_layouts) / max(1, len(episodes))),
        "mean_action_delta": sum(action_deltas) / max(1, len(action_deltas)),
        "mean_action_sign_flips": sum(action_flips) / max(1, len(action_flips)),
        "mean_min_goal_distance": sum(min_goal_distances) / max(1, len(min_goal_distances)),
        "mean_action": [
            sum(float(action[axis]) for action in action_means) / max(1, len(action_means))
            for axis in range(int(env.action_space.shape[-1]))
        ],
        "mean_abs_action": [
            sum(float(action[axis]) for action in abs_action_means) / max(1, len(abs_action_means))
            for axis in range(int(env.action_space.shape[-1]))
        ],
        "mean_small_action_fraction": sum(small_action_fractions) / max(1, len(small_action_fractions)),
        "front_obstacle_step_fraction": sum(front_steps) / max(1, sum(float(ep["episode_length_s"]) / 0.05 for ep in episodes)),
        "front_obstacle_abs_steering": sum(
            float(ep["front_obstacle_abs_steering"]) * int(ep["front_obstacle_steps"]) for ep in episodes
        ) / front_step_total,
        "front_obstacle_small_steering_fraction": sum(
            float(ep["front_obstacle_small_steering_fraction"]) * int(ep["front_obstacle_steps"]) for ep in episodes
        ) / front_step_total,
        "front_obstacle_action_delta": sum(
            float(ep["front_obstacle_action_delta"]) * int(ep["front_obstacle_steps"]) for ep in episodes
        ) / front_step_total,
        "mean_front_obstacle_directional_consistency": sum(
            float(ep["front_obstacle_directional_consistency"]) for ep in episodes if int(ep["front_obstacle_steps"]) > 0
        ) / max(1, sum(int(ep["front_obstacle_steps"]) > 0 for ep in episodes)),
        "policy_action_smoothing": float(args.policy_action_smoothing),
        "layout_manifest": str(Path(args.layout_manifest).resolve()) if manifest is not None else None,
        "layout_manifest_version": int(manifest["version"]) if manifest is not None else None,
        "require_blocked_corridor": bool(args.require_blocked_corridor),
        "blocked_corridor_radius": float(args.blocked_corridor_radius),
        "blocked_corridor_ignore_end_radius": float(args.blocked_corridor_ignore_end_radius),
        "blocked_corridor_min_cells": int(args.blocked_corridor_min_cells),
        "mean_blocked_corridor_cell_count": sum(blocked_counts) / max(1, len(blocked_counts)),
        "mean_blocking_component_count": sum(blocking_component_counts) / max(1, len(blocking_component_counts)),
        "mean_straight_path_length": sum(path_lengths) / max(1, len(path_lengths)),
        "mean_nearest_corridor_obstacle_dist": sum(corridor_dists) / max(1, len(corridor_dists)),
        **term_means,
        "min_start_obstacle_clearance": float(args.min_start_obstacle_clearance),
        "min_goal_obstacle_clearance": float(args.min_goal_obstacle_clearance),
        "wall_time_s": time.time() - start,
        "episodes": episodes,
    }
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.controller}_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, sort_keys=True), flush=True)
    print(f"[metrics] wrote {out_path}", flush=True)
    return summary


def record_video(args: argparse.Namespace) -> None:
    import imageio.v2 as imageio

    manifest = None
    manifest_index = int(getattr(args, "video_manifest_index", -1))
    if str(getattr(args, "layout_manifest", "")):
        minimum_episodes = manifest_index + 1 if manifest_index >= 0 else 1
        manifest = load_layout_manifest(Path(args.layout_manifest).resolve(), minimum_episodes=minimum_episodes)
        args.seed = int(manifest["metadata"]["seed"])
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    env = make_env(args, num_envs=1, render=True)
    if manifest is not None and manifest_index >= 0:
        obs_raw, _ = env.reset()
        obs_raw, _, _, _, obstacle_cells = _apply_manifest_layouts(
            args, env, [0], [manifest["episodes"][manifest_index]]
        )
        print(f"[video] replaying manifest episode {manifest_index}", flush=True)
    else:
        obs_raw, _, _, _, obstacle_cells = _reset_until_feasible(args, env)
    from unitree_nav_observation import UnitreeScanHistory

    video_history = UnitreeScanHistory(
        getattr(args, "scan_history", 1),
        getattr(args, "action_history", 0),
        int(env.action_space.shape[-1]),
        history_stride=getattr(args, "scan_history_stride", 1),
    )
    obs = video_history.reset(_prepare_policy_obs(obs_raw, args))
    policy_actor = (
        _load_policy_actor(args, obs_dim=int(obs.shape[1]), act_dim=int(env.action_space.shape[-1]))
        if args.controller == "policy"
        else None
    )
    teacher_state = ScanTeacherState(1, torch.device(args.device)) if args.controller == "scan_teacher" else None
    if args.controller == "geom_scan_teacher":
        teacher_state = GeomTeacherState.create(1)
    frames = []
    for _ in range(int(args.video_length)):
        with torch.no_grad():
            action = controller_action(obs, args, policy_actor, teacher_state, env=env, obstacle_cells=obstacle_cells)
        obs_raw, _, done, _ = env.step(action)
        terminal_success = _goal_termination_mask(env, 1, torch.device(args.device))
        obs = video_history.step(_prepare_policy_obs(obs_raw, args), done, action=action)
        goal_distance = unitree_goal_distance(
            obs,
            goal_encoding=args.goal_encoding,
            goal_distance_scale=args.goal_distance_scale,
        )
        reached_goal = bool(
            terminal_success.reshape(-1)[0].item()
            or goal_distance.reshape(-1)[0].item() <= float(args.success_dist)
        )
        frame = env.env.render()
        if frame is not None:
            frames.append(frame)
        if bool(done.reshape(-1)[0].item()) or reached_goal:
            if bool(getattr(args, "video_stop_on_done", False)):
                break
            if (
                reached_goal
                and not bool(done.reshape(-1)[0].item())
                and str(getattr(args, "navigation_episode_mode", "episodic")) == "continuous_goals"
            ):
                obs_raw, _, _ = _resample_continuous_goals(
                    args,
                    env,
                    obstacle_cells,
                    torch.zeros(1, dtype=torch.long, device=args.device),
                )
                obs = video_history.reset(_prepare_policy_obs(obs_raw, args))
            else:
                obs_raw, _, _, _, obstacle_cells = _reset_until_feasible(args, env)
                obs = video_history.reset(_prepare_policy_obs(obs_raw, args))
            if teacher_state is not None:
                teacher_state.reset()
    env.close()
    video_dir = Path(args.video_dir) / args.controller
    video_dir.mkdir(parents=True, exist_ok=True)
    out_path = video_dir / f"{args.controller}.mp4"
    if not frames:
        raise RuntimeError("No RGB frames were captured from env.render()")
    imageio.mimsave(out_path, frames, fps=int(args.video_fps))
    print(f"[video] wrote {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    from unitree_target_navigation import add_target_terrain_args

    add_target_terrain_args(parser)
    parser.add_argument("--controller", choices=["direct_goal", "scan_teacher", "geom_scan_teacher", "policy"], required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="Unitree-G1-Nav-Obstacles-Safe-Collision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--episode-length-s", type=float, default=16.0)
    parser.add_argument(
        "--navigation-episode-mode",
        choices=["episodic", "continuous_goals"],
        default="episodic",
    )
    parser.add_argument("--continuous-environment-horizon-s", type=float, default=3600.0)
    parser.add_argument("--evaluation-task-max-steps", type=int, default=640)
    parser.add_argument("--continuous-goal-distance-min", type=float, default=0.0)
    parser.add_argument("--continuous-goal-distance-max", type=float, default=0.0)
    parser.add_argument(
        "--continuous-goal-region-mode",
        choices=["assigned_tile", "terrain_bank"],
        default="assigned_tile",
)
    parser.add_argument("--continuous-goal-resample-attempts", type=int, default=256)
    parser.add_argument("--continuous-goal-boundary-margin", type=float, default=0.5)
    parser.add_argument("--continuous-goal-require-blocked-corridor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--continuous-goal-blocked-probability", type=float, default=1.0)
    parser.add_argument("--resample-terrain-tiles", action="store_true")
    parser.add_argument("--start-position-range", type=float, default=0.0)
    parser.add_argument("--success-dist", type=float, default=0.5)
    parser.add_argument("--terminate-on-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--height-scan-resolution", type=float, default=0.5)
    parser.add_argument("--height-scan-pattern", choices=["grid", "forward_frustum"], default="grid")
    parser.add_argument("--height-scan-frustum-near", type=float, default=0.25)
    parser.add_argument("--height-scan-frustum-far", type=float, default=4.0)
    parser.add_argument("--height-scan-frustum-fov-deg", type=float, default=70.0)
    parser.add_argument("--height-scan-frustum-side", type=int, default=17)
    parser.add_argument("--height-scan-forward-size", type=float, default=3.0)
    parser.add_argument("--height-scan-lateral-size", type=float, default=3.0)
    parser.add_argument("--scan-history", type=int, default=1)
    parser.add_argument("--scan-history-stride", type=int, default=1)
    parser.add_argument("--action-history", type=int, default=0)
    parser.add_argument("--mask-height-scan", action="store_true")
    parser.add_argument("--mask-proprioception", action="store_true")
    parser.add_argument("--mask-goal-heading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--goal-encoding", choices=["cartesian", "distance_bearing"], default="cartesian")
    parser.add_argument("--goal-distance-scale", type=float, default=14.0)
    parser.add_argument("--velocity-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-env-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-action-smoothing", type=float, default=0.0)
    parser.add_argument("--goal-distance-min", type=float, default=0.0)
    parser.add_argument("--goal-distance-max", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--use-layer-norm", action="store_true")
    parser.add_argument("--low-level-policy-path", default=str(DEFAULT_LOW_LEVEL))
    parser.add_argument("--output-dir", default=str(ROOT / "logs" / "unitree_mjlab" / "baseline_eval"))
    parser.add_argument("--run-name", default=f"unitree_baselines_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--teacher-scan-block-threshold", type=float, default=0.12)
    parser.add_argument("--teacher-scan-block-delta", type=float, default=0.0)
    parser.add_argument("--decisiveness-front-half-width", type=float, default=0.75)
    parser.add_argument("--decisiveness-front-min-distance", type=float, default=0.25)
    parser.add_argument("--decisiveness-front-max-distance", type=float, default=2.0)
    parser.add_argument("--teacher-scan-planner", choices=["heuristic", "astar"], default="heuristic")
    parser.add_argument("--teacher-scan-astar-clearance", type=float, default=0.6)
    parser.add_argument("--teacher-scan-astar-cell-padding", type=float, default=0.35)
    parser.add_argument("--teacher-scan-astar-resolution", type=float, default=0.25)
    parser.add_argument("--teacher-scan-astar-waypoint-index", type=int, default=3)
    parser.add_argument("--teacher-scan-astar-side-penalty", type=float, default=8.0)
    parser.add_argument("--teacher-scan-astar-commit-steps", type=int, default=30)
    parser.add_argument("--teacher-sector-half-width", type=float, default=0.35)
    parser.add_argument("--teacher-align-angle", type=float, default=0.55)
    parser.add_argument("--teacher-max-vx", type=float, default=0.95)
    parser.add_argument("--teacher-max-vy", type=float, default=0.45)
    parser.add_argument("--teacher-yaw-gain", type=float, default=1.2)
    parser.add_argument("--teacher-clearance-weight", type=float, default=0.0)
    parser.add_argument("--teacher-clearance-power", type=float, default=2.0)
    parser.add_argument("--teacher-speed-clearance-scale", type=float, default=0.0)
    parser.add_argument("--teacher-num-sectors", type=int, default=13)
    parser.add_argument("--teacher-min-forward-scale", type=float, default=0.12)
    parser.add_argument("--teacher-escape-risk-threshold", type=float, default=0.0)
    parser.add_argument("--teacher-escape-forward-scale", type=float, default=0.0)
    parser.add_argument("--teacher-escape-lateral-scale", type=float, default=1.0)
    parser.add_argument("--teacher-escape-radius", type=float, default=1.0)
    parser.add_argument("--teacher-escape-all-directions", action="store_true")
    parser.add_argument("--teacher-bypass-angle", type=float, default=0.0)
    parser.add_argument("--teacher-goal-stop-dist", type=float, default=0.0)
    parser.add_argument("--teacher-wall-follow-steps", type=int, default=0)
    parser.add_argument("--teacher-wall-follow-angle", type=float, default=0.9)
    parser.add_argument("--teacher-wall-follow-clear-risk", type=float, default=0.15)
    parser.add_argument("--teacher-rollout-horizon", type=float, default=0.0)
    parser.add_argument("--teacher-rollout-clearance", type=float, default=0.65)
    parser.add_argument("--teacher-rollout-samples", type=int, default=8)
    parser.add_argument("--teacher-rollout-clearance-weight", type=float, default=20.0)
    parser.add_argument("--teacher-rollout-forward-bias", type=float, default=0.05)
    parser.add_argument("--teacher-emergency-radius", type=float, default=0.0)
    parser.add_argument("--teacher-emergency-hard-radius", type=float, default=0.0)
    parser.add_argument("--teacher-emergency-speed-scale", type=float, default=0.75)
    parser.add_argument("--teacher-emergency-repulsion-weight", type=float, default=0.8)
    parser.add_argument("--teacher-emergency-tangent-weight", type=float, default=1.2)
    parser.add_argument("--teacher-emergency-goal-weight", type=float, default=0.4)
    parser.add_argument("--teacher-geom-planner", choices=["astar", "sector"], default="astar")
    parser.add_argument("--teacher-geom-scan-margin", type=float, default=0.15)
    parser.add_argument("--teacher-geom-back-margin", type=float, default=0.35)
    parser.add_argument("--teacher-geom-max-forward", type=float, default=1.75)
    parser.add_argument("--teacher-geom-max-lateral", type=float, default=1.75)
    parser.add_argument("--teacher-geom-lookahead", type=float, default=1.6)
    parser.add_argument("--teacher-geom-clearance", type=float, default=0.62)
    parser.add_argument("--teacher-geom-grid-resolution", type=float, default=0.18)
    parser.add_argument("--teacher-geom-waypoint-index", type=int, default=3)
    parser.add_argument("--teacher-geom-side-penalty", type=float, default=4.0)
    parser.add_argument("--teacher-geom-side-frame", choices=["body", "goal"], default="body")
    parser.add_argument("--teacher-geom-disengage-clear-steps", type=int, default=12)
    parser.add_argument("--teacher-geom-command-smoothing", type=float, default=0.0)
    parser.add_argument("--teacher-geom-memory-radius", type=float, default=0.0)
    parser.add_argument("--teacher-geom-waypoint-commit-distance", type=float, default=0.0)
    parser.add_argument("--teacher-geom-waypoint-reach-dist", type=float, default=0.25)
    parser.add_argument("--teacher-geom-stall-window", type=int, default=0)
    parser.add_argument("--teacher-geom-stall-progress-epsilon", type=float, default=0.08)
    parser.add_argument("--teacher-geom-stall-recovery-steps", type=int, default=60)
    parser.add_argument("--teacher-geom-stall-recovery-angle", type=float, default=0.9)
    parser.add_argument("--teacher-geom-stall-flip-side", action="store_true")
    parser.add_argument("--teacher-geom-obstacle-speed-radius", type=float, default=0.0)
    parser.add_argument("--teacher-geom-near-obstacle-vx-scale", type=float, default=0.7)
    parser.add_argument("--teacher-geom-max-angle", type=float, default=1.5707963267948966)
    parser.add_argument("--teacher-geom-candidate-count", type=int, default=31)
    parser.add_argument("--teacher-geom-soft-width", type=float, default=0.9)
    parser.add_argument("--teacher-geom-hard-penalty", type=float, default=35.0)
    parser.add_argument("--teacher-geom-risk-weight", type=float, default=9.0)
    parser.add_argument("--teacher-geom-angle-weight", type=float, default=0.15)
    parser.add_argument("--teacher-geom-forward-bias", type=float, default=0.2)
    parser.add_argument("--teacher-geom-emergency-radius", type=float, default=0.8)
    parser.add_argument("--teacher-geom-emergency-repulsion-weight", type=float, default=1.2)
    parser.add_argument("--teacher-geom-emergency-tangent-weight", type=float, default=1.4)
    parser.add_argument("--teacher-geom-emergency-goal-weight", type=float, default=0.25)
    parser.add_argument("--min-goal-obstacle-clearance", type=float, default=0.0)
    parser.add_argument("--goal-clearance-resample-attempts", type=int, default=50)
    parser.add_argument("--min-start-obstacle-clearance", type=float, default=0.0)
    parser.add_argument("--start-clearance-resample-attempts", type=int, default=20)
    parser.add_argument("--require-blocked-corridor", action="store_true")
    parser.add_argument("--blocked-corridor-radius", type=float, default=0.45)
    parser.add_argument("--blocked-corridor-ignore-end-radius", type=float, default=0.75)
    parser.add_argument("--blocked-corridor-min-cells", type=int, default=1)
    parser.add_argument("--blocked-corridor-resample-attempts", type=int, default=100)
    parser.add_argument("--blocked-goal-max-distance", type=float, default=0.0)
    parser.add_argument(
        "--blocked-goal-distance-sampling",
        choices=["nearest", "uniform", "farthest"],
        default="nearest",
    )
    parser.add_argument("--blocked-goal-placement-mode", choices=["obstacle_multiplier", "distance_grid"], default="obstacle_multiplier")
    parser.add_argument("--blocked-goal-distance-multiplier-min", type=float, default=1.0)
    parser.add_argument("--blocked-goal-distance-multiplier-max", type=float, default=2.0)
    parser.add_argument("--blocked-goal-candidate-attempts", type=int, default=64)
    parser.add_argument("--debug-obstacle-width-min", type=float, default=1.0)
    parser.add_argument("--debug-obstacle-width-max", type=float, default=1.4)
    parser.add_argument("--strict-min-size-obstacles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-obstacle-height-min", type=float, default=1.0)
    parser.add_argument("--debug-obstacle-height-max", type=float, default=1.0)
    parser.add_argument("--debug-num-obstacles", type=int, default=6)
    parser.add_argument("--disable-obstacles", action="store_true")
    parser.add_argument("--debug-platform-width", type=float, default=2.0)
    parser.add_argument("--debug-obstacle-border-width", type=float, default=0.0)
    parser.add_argument("--debug-terrain-rows", type=int, default=0)
    parser.add_argument("--debug-terrain-cols", type=int, default=0)
    parser.add_argument("--debug-goal-through-obstacle", action="store_true")
    parser.add_argument("--goal-through-obstacle-prob", type=float, default=0.0)
    parser.add_argument("--debug-goal-distance", type=float, default=3.2)
    parser.add_argument("--debug-goal-obstacle-min-dist", type=float, default=0.8)
    parser.add_argument("--debug-goal-obstacle-max-dist", type=float, default=2.2)
    parser.add_argument("--record-trajectories", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", default=str(ROOT / "logs" / "unitree_mjlab" / "baseline_videos"))
    parser.add_argument("--video-length", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-manifest-index", type=int, default=-1)
    parser.add_argument("--video-stop-on-done", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-camera-distance", type=float, default=5.0)
    parser.add_argument("--video-camera-elevation", type=float, default=-10.0)
    parser.add_argument("--video-camera-azimuth", type=float, default=90.0)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=640)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--layout-manifest", default="")
    parser.add_argument("--generate-layout-manifest", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from unitree_nav_checkpoint import apply_unitree_checkpoint_config

    apply_unitree_checkpoint_config(args)
    if args.generate_layout_manifest:
        generate_layout_manifest(args)
        return 0
    if not args.skip_metrics:
        evaluate(args)
    if args.record_video:
        record_video(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

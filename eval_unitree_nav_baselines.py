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
    ) <= 0:
        return
    terrain = getattr(env_cfg.scene, "terrain", None)
    generator = getattr(terrain, "terrain_generator", None) if terrain is not None else None
    if generator is None or "discrete_obstacles" not in getattr(generator, "sub_terrains", {}):
        raise ValueError("--debug-obstacle-* overrides require a discrete_obstacles heightfield terrain")
    # Avoid mutating the globally registered task config object.
    env_cfg.scene.terrain.terrain_generator = copy.deepcopy(generator)
    obstacle_cfg = env_cfg.scene.terrain.terrain_generator.sub_terrains["discrete_obstacles"]
    if width_min > 0.0 or width_max > 0.0:
        current = tuple(getattr(obstacle_cfg, "obstacle_width_range"))
        obstacle_cfg.obstacle_width_range = (width_min or current[0], width_max or current[1])
    if height_min > 0.0 or height_max > 0.0:
        current = tuple(getattr(obstacle_cfg, "obstacle_height_range"))
        obstacle_cfg.obstacle_height_range = (height_min or current[0], height_max or current[1])
    if num_obstacles > 0:
        obstacle_cfg.num_obstacles = num_obstacles
    if platform_width > 0.0:
        obstacle_cfg.platform_width = platform_width
    if border_width > 0.0:
        obstacle_cfg.border_width = border_width
    if terrain_rows > 0:
        env_cfg.scene.terrain.terrain_generator.num_rows = terrain_rows
    if terrain_cols > 0:
        env_cfg.scene.terrain.terrain_generator.num_cols = terrain_cols


def make_env(args: argparse.Namespace, *, num_envs: int, render: bool):
    _setup_unitree_imports(Path(args.low_level_policy_path).resolve())
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    from src.envs import build_env

    env_cfg = load_env_cfg(args.task, play=render)
    from unitree_nav_layout import configure_terrain_tile_resets

    if hasattr(args, "seed"):
        env_cfg.seed = int(args.seed)
    _apply_debug_obstacle_overrides(args, env_cfg)
    terrain_generator = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
    if terrain_generator is not None and hasattr(terrain_generator, "seed"):
        # Environment reset seeding does not necessarily seed procedural terrain generation.
        terrain_generator.seed = int(args.seed)
    configure_terrain_tile_resets(
        env_cfg,
        enabled=bool(getattr(args, "resample_terrain_tiles", False)),
    )
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.episode_length_s = float(args.episode_length_s)
    if "pose" in env_cfg.commands:
        env_cfg.commands["pose"].resampling_time_range = (float(args.episode_length_s), float(args.episode_length_s))
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
    goal_distance = float(getattr(args, "debug_goal_distance", 3.2))
    min_obstacle_dist = float(getattr(args, "debug_goal_obstacle_min_dist", 0.8))
    max_obstacle_dist = float(getattr(args, "debug_goal_obstacle_max_dist", 2.2))
    min_goal_clearance = float(getattr(args, "min_goal_obstacle_clearance", 0.0))
    selected = set(range(len(obstacle_cells))) if env_ids is None else set(int(i) for i in _to_numpy(env_ids).reshape(-1))
    for i, cells in enumerate(obstacle_cells):
        if i not in selected:
            continue
        if not bool(getattr(args, "debug_goal_through_obstacle", False)) and np.random.random() > probability:
            continue
        if cells.size == 0:
            continue
        start = starts[i]
        rel = cells - start.reshape(1, 2)
        dist = np.linalg.norm(rel, axis=1)
        candidates = np.nonzero((dist >= min_obstacle_dist) & (dist <= max_obstacle_dist))[0]
        if candidates.size == 0:
            candidates = np.argsort(np.abs(dist - 0.5 * (min_obstacle_dist + max_obstacle_dist)))[: max(1, min(8, len(dist)))]
        # Prefer an obstacle direction where the final goal is not itself too
        # close to any obstacle cell.
        chosen_goal = None
        chosen_idx = None
        for idx in candidates[np.argsort(dist[candidates])]:
            direction = rel[idx] / max(float(dist[idx]), 1e-6)
            goal = start + direction * goal_distance
            clearance = float(np.min(np.linalg.norm(cells - goal.reshape(1, 2), axis=1)))
            if clearance >= min_goal_clearance:
                chosen_goal = goal
                chosen_idx = int(idx)
                break
        if chosen_goal is None:
            idx = int(candidates[np.argmin(np.abs(dist[candidates] - min_obstacle_dist))])
            direction = rel[idx] / max(float(dist[idx]), 1e-6)
            chosen_goal = start + direction * goal_distance
            chosen_idx = idx
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
            "nearest_corridor_obstacle_dist": float("inf"),
            "path_length": float(np.linalg.norm(goal_xy - start_xy)),
        }
    vec = goal_xy - start_xy
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return {
            "blocked": False,
            "blocked_cell_count": 0,
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
    return {
        "blocked": bool(count > 0),
        "blocked_cell_count": count,
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
    if min_clearance <= 0.0:
        return None, None
    unwrapped = env.env.unwrapped
    term = unwrapped.command_manager._terms["pose"]
    attempts = int(getattr(args, "goal_clearance_resample_attempts", 50))
    clearances = _goal_clearances(env, obstacle_cells)
    selected = None
    if env_ids is not None:
        selected = torch.zeros_like(clearances, dtype=torch.bool)
        selected[torch.as_tensor(env_ids, device=clearances.device, dtype=torch.long).reshape(-1)] = True
    for _ in range(max(0, attempts)):
        bad_mask = clearances < min_clearance
        if selected is not None:
            bad_mask &= selected
        bad = torch.nonzero(bad_mask, as_tuple=False).flatten()
        if bad.numel() == 0:
            break
        term._resample_command(bad)
        term._update_command()
        clearances = _goal_clearances(env, obstacle_cells)
    return _recompute_observations(env), clearances


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
    for _ in range(blocked_attempts if require_blocked else attempts):
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
    return obs_raw, start_clearances, goal_clearances, layout_stats, obstacle_cells


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
    if args.controller == "direct_goal":
        return direct_goal_action(
            obs,
            max_vx=args.teacher_max_vx,
            max_vy=args.teacher_max_vy,
            yaw_gain=args.teacher_yaw_gain,
            align_angle=args.teacher_align_angle,
        )
    if args.controller == "scan_teacher":
        action, _, _ = scan_teacher_action(
            obs,
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
        return local_geometry_scan_teacher_action(obs, env=env, obstacle_cells=obstacle_cells, args=args, state=geom_state)
    if args.controller == "policy":
        if policy_actor is None:
            raise ValueError("policy_actor is required for --controller policy")
        with torch.no_grad():
            _, _, mean = policy_actor(obs)
        return mean.clamp(-1.0, 1.0)
    raise ValueError(f"Unsupported controller: {args.controller}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "require_blocked_corridor", False)) and int(args.num_envs) != 1:
        raise ValueError("--require-blocked-corridor currently requires --num-envs 1 for unambiguous reset filtering")
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    env = make_env(args, num_envs=args.num_envs, render=False)
    cost_term_names = list(getattr(env.env.unwrapped.cost_manager, "active_terms", []))
    obs_raw, start_clearances, goal_clearances, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
    obs = _extract_actor_obs(obs_raw).to(args.device, dtype=torch.float32)
    num_envs = int(obs.shape[0])
    target_episodes = int(args.num_episodes)
    quota_base, quota_remainder = divmod(target_episodes, num_envs)
    episode_quota = [quota_base + int(i < quota_remainder) for i in range(num_envs)]
    completed_per_env = [0 for _ in range(num_envs)]
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
    ep_collision_steps = torch.zeros(num_envs, device=args.device)
    ep_success = torch.zeros(num_envs, dtype=torch.bool, device=args.device)
    ep_first_success_step = torch.full((num_envs,), -1, dtype=torch.long, device=args.device)
    ep_steps = torch.zeros(num_envs, dtype=torch.long, device=args.device)

    episodes: list[dict[str, float | int | bool | None]] = []
    start = time.time()
    while len(episodes) < int(args.num_episodes):
        pre_dist = _current_goal_distance(env, num_envs, torch.device(args.device))
        pre_success = (pre_dist <= float(args.success_dist)) & (~ep_success)
        ep_first_success_step[pre_success] = ep_steps[pre_success]
        ep_success |= pre_dist <= float(args.success_dist)

        with torch.no_grad():
            action = controller_action(obs, args, policy_actor, teacher_state, env=env, obstacle_cells=obstacle_cells)
        next_raw, reward, done, extras = env.step(action)
        next_obs = _extract_actor_obs(next_raw).to(args.device, dtype=torch.float32)
        reward = reward.to(args.device, dtype=torch.float32).reshape(num_envs)
        done = done.to(args.device).reshape(num_envs).bool()
        cost_vec = _extract_cost(extras, num_envs, torch.device(args.device))
        cost = cost_vec.reshape(num_envs, -1).sum(dim=1) if cost_vec.ndim > 1 else cost_vec.reshape(num_envs)
        cost_terms = _cost_terms(extras, num_envs, torch.device(args.device), cost_term_names)

        ep_steps += 1
        ep_return += reward
        ep_cost += cost
        for name, value in cost_terms.items():
            ep_cost_terms[name] += value
        ep_collision_steps += (cost > 0.0).float()
        dist = torch.linalg.norm(next_obs[:, 6:8], dim=-1)
        just_success = (dist <= float(args.success_dist)) & (~ep_success) & (~done)
        ep_first_success_step[just_success] = ep_steps[just_success]
        ep_success |= dist <= float(args.success_dist)

        if done.any():
            done_idx = torch.nonzero(done, as_tuple=False).flatten()
            for i in done_idx.tolist():
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
                        "nearest_corridor_obstacle_dist": (
                            float(layout_stats[i]["nearest_corridor_obstacle_dist"]) if layout_stats is not None else None
                        ),
                        "straight_path_length": float(layout_stats[i]["path_length"]) if layout_stats is not None else None,
                        **{
                            f"{name}_cost_sum": float(values[i].detach().cpu().item())
                            for name, values in ep_cost_terms.items()
                        },
                        "collision_steps": int(ep_collision_steps[i].detach().cpu().item()),
                        "return": float(ep_return[i].detach().cpu().item()),
                        "env_index": int(i),
                    }
                )
                completed_per_env[i] += 1
                print(
                    "[eval-episode] "
                    f"controller={args.controller} episode={len(episodes)} "
                    f"success={bool(ep_success[i].detach().cpu().item())} "
                    f"cost={float(ep_cost[i].detach().cpu().item()):.1f} "
                    f"collision_steps={int(ep_collision_steps[i].detach().cpu().item())} "
                    f"steps={int(ep_steps[i].detach().cpu().item())}",
                    flush=True,
                )
            ep_return[done_idx] = 0.0
            ep_cost[done_idx] = 0.0
            for values in ep_cost_terms.values():
                values[done_idx] = 0.0
            ep_collision_steps[done_idx] = 0.0
            ep_success[done_idx] = False
            ep_first_success_step[done_idx] = -1
            ep_steps[done_idx] = 0
            if teacher_state is not None:
                teacher_state.reset(done)
            if (
                num_envs == 1
                and (
                    float(getattr(args, "min_start_obstacle_clearance", 0.0)) > 0.0
                    or bool(getattr(args, "require_blocked_corridor", False))
                )
            ):
                next_raw, start_clearances, goal_clearances, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
                next_obs = _extract_actor_obs(next_raw).to(args.device, dtype=torch.float32)
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
                    next_obs = _extract_actor_obs(next_raw).to(args.device, dtype=torch.float32)
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
                    next_obs = _extract_actor_obs(next_raw).to(args.device, dtype=torch.float32)
                if next_goal_clearances is None:
                    next_goal_clearances = _goal_clearances(env, obstacle_cells)
                if goal_clearances is None:
                    goal_clearances = next_goal_clearances
                else:
                    goal_clearances[done_idx] = next_goal_clearances[done_idx]
                next_layout_stats = _layout_blocked_corridor_stats(args, env, obstacle_cells)
                if layout_stats is None:
                    layout_stats = next_layout_stats
                else:
                    for i in done_idx.tolist():
                        layout_stats[i] = next_layout_stats[i]
        obs = next_obs

    env.close()
    successes = [float(ep["success"]) for ep in episodes]
    costs = [float(ep["cost_sum"]) for ep in episodes]
    collision_steps = [float(ep["collision_steps"]) for ep in episodes]
    lengths = [float(ep["episode_length_s"]) for ep in episodes]
    tts = [float(ep["time_to_success_s"]) for ep in episodes if ep["time_to_success_s"] is not None]
    returns = [float(ep["return"]) for ep in episodes]
    blocked_counts = [float(ep.get("blocked_corridor_cell_count") or 0.0) for ep in episodes]
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
        "num_episodes": len(episodes),
        "success_rate": sum(successes) / max(1, len(successes)),
        "mean_time_to_success_s_success_only": sum(tts) / max(1, len(tts)),
        "mean_episode_length_s": sum(lengths) / max(1, len(lengths)),
        "mean_cost_sum": sum(costs) / max(1, len(costs)),
        "costful_episode_rate": sum(1.0 for c in costs if c > 0.0) / max(1, len(costs)),
        "mean_collision_steps": sum(collision_steps) / max(1, len(collision_steps)),
        "mean_return": sum(returns) / max(1, len(returns)),
        "require_blocked_corridor": bool(args.require_blocked_corridor),
        "blocked_corridor_radius": float(args.blocked_corridor_radius),
        "blocked_corridor_ignore_end_radius": float(args.blocked_corridor_ignore_end_radius),
        "blocked_corridor_min_cells": int(args.blocked_corridor_min_cells),
        "mean_blocked_corridor_cell_count": sum(blocked_counts) / max(1, len(blocked_counts)),
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

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    env = make_env(args, num_envs=1, render=True)
    obs_raw, _, _, _, obstacle_cells = _reset_until_feasible(args, env)
    obs = _extract_actor_obs(obs_raw).to(args.device, dtype=torch.float32)
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
        obs = _extract_actor_obs(obs_raw).to(args.device, dtype=torch.float32)
        frame = env.env.render()
        if frame is not None:
            frames.append(frame)
        if bool(done.reshape(-1)[0].item()):
            obs_raw, _, _, _, obstacle_cells = _reset_until_feasible(args, env)
            if teacher_state is not None:
                teacher_state.reset()
            obs = _extract_actor_obs(obs_raw).to(args.device, dtype=torch.float32)
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
    parser.add_argument("--controller", choices=["direct_goal", "scan_teacher", "geom_scan_teacher", "policy"], required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="Unitree-G1-Nav-Obstacles-Safe-Collision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--episode-length-s", type=float, default=16.0)
    parser.add_argument("--resample-terrain-tiles", action="store_true")
    parser.add_argument("--success-dist", type=float, default=0.5)
    parser.add_argument("--goal-distance-min", type=float, default=0.0)
    parser.add_argument("--goal-distance-max", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--use-layer-norm", action="store_true")
    parser.add_argument("--low-level-policy-path", default=str(DEFAULT_LOW_LEVEL))
    parser.add_argument("--output-dir", default=str(ROOT / "logs" / "unitree_mjlab" / "baseline_eval"))
    parser.add_argument("--run-name", default=f"unitree_baselines_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--teacher-scan-block-threshold", type=float, default=0.12)
    parser.add_argument("--teacher-scan-block-delta", type=float, default=0.0)
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
    parser.add_argument("--debug-obstacle-width-min", type=float, default=1.0)
    parser.add_argument("--debug-obstacle-width-max", type=float, default=1.4)
    parser.add_argument("--debug-obstacle-height-min", type=float, default=1.0)
    parser.add_argument("--debug-obstacle-height-max", type=float, default=1.0)
    parser.add_argument("--debug-num-obstacles", type=int, default=6)
    parser.add_argument("--debug-platform-width", type=float, default=2.0)
    parser.add_argument("--debug-obstacle-border-width", type=float, default=0.0)
    parser.add_argument("--debug-terrain-rows", type=int, default=0)
    parser.add_argument("--debug-terrain-cols", type=int, default=0)
    parser.add_argument("--debug-goal-through-obstacle", action="store_true")
    parser.add_argument("--goal-through-obstacle-prob", type=float, default=0.0)
    parser.add_argument("--debug-goal-distance", type=float, default=3.2)
    parser.add_argument("--debug-goal-obstacle-min-dist", type=float, default=0.8)
    parser.add_argument("--debug-goal-obstacle-max-dist", type=float, default=2.2)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", default=str(ROOT / "logs" / "unitree_mjlab" / "baseline_videos"))
    parser.add_argument("--video-length", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--skip-metrics", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_metrics:
        evaluate(args)
    if args.record_video:
        record_video(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Top-down rollout plot for Unitree navigation controllers.

The controller still uses only policy-visible observations. The plot uses
privileged world pose only for debugging/visualization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_unitree_nav_baselines import (
    _current_goal_distance,
    _load_policy_actor,
    _reset_until_feasible,
    _robot_clearances,
    controller_action,
    make_env,
)
from train_unitree_nav_thesis import (
    DEFAULT_LOW_LEVEL,
    ROOT,
    InterventionGateState,
    ScanTeacherState,
    _extract_actor_obs,
    _extract_cost,
    resolve_intervention_clearances,
)
from unitree_nav_observation import prepare_unitree_actor_obs


def _prepare_policy_obs(raw_obs, args):
    obs = _extract_actor_obs(raw_obs).to(args.device, dtype=torch.float32)
    return prepare_unitree_actor_obs(
        obs,
        mask_proprioception=bool(getattr(args, "mask_proprioception", False)),
        mask_goal_heading=bool(args.mask_goal_heading),
        mask_height_scan=bool(getattr(args, "mask_height_scan", False)),
    )
from unitree_nav_geom_teacher import GeomTeacherState, local_geometry_scan_teacher_action


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "data") and hasattr(value.data, "detach"):
        return value.data.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _command_goal_w(env) -> np.ndarray | None:
    try:
        term = env.env.unwrapped.command_manager._terms["pose"]
        return term._goal_pos_w[0].detach().cpu().numpy().copy()
    except Exception:
        return None


def _robot_xy_heading(env) -> tuple[np.ndarray, float]:
    robot = env.env.unwrapped.scene["robot"]
    xy = robot.data.root_link_pos_w[0, :2].detach().cpu().numpy().copy()
    heading = float(robot.data.heading_w[0].detach().cpu().item())
    return xy, heading


def _scan_points_world(
    obs: torch.Tensor,
    xy: np.ndarray,
    heading: float,
    threshold: float,
    *,
    forward_size: float,
    lateral_size: float,
    resolution: float,
) -> np.ndarray:
    scan_values = obs.shape[1] - 9
    forward_size = float(forward_size) if float(forward_size) > 0 else 3.0
    lateral_size = float(lateral_size) if float(lateral_size) > 0 else 3.0
    resolution = float(resolution)
    forward_count = int(round(forward_size / resolution)) + 1
    lateral_count = int(round(lateral_size / resolution)) + 1
    if forward_count * lateral_count != scan_values:
        raise ValueError(
            "Height-scan shape metadata does not match observation: "
            f"{lateral_count}x{forward_count} != {scan_values}"
        )
    scan = obs[0, 9:].detach().cpu().numpy().reshape(lateral_count, forward_count)
    forward_coords = np.linspace(-forward_size / 2.0, forward_size / 2.0, forward_count)
    lateral_coords = np.linspace(-lateral_size / 2.0, lateral_size / 2.0, lateral_count)
    c, s = math.cos(heading), math.sin(heading)
    pts = []
    for i, lateral in enumerate(lateral_coords):
        for j, forward in enumerate(forward_coords):
            if scan[i, j] >= threshold:
                continue
            local = np.array([forward, lateral])
            world = xy + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
            pts.append(world)
    return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)


def _active_terrain_heightfield(env) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray] | None:
    """Return the active env's heightfield, imshow extent, and obstacle cell centers.

    This is privileged debug data for plotting only. The controller path remains
    observation-only through pose_command + height_scan.
    """
    try:
        unwrapped = env.env.unwrapped
        terrain_origins = _to_numpy(unwrapped.scene.terrain.terrain_origins)
        env_origin = _to_numpy(unwrapped.scene.env_origins[0])
        model = unwrapped.sim.model
        nrow = _to_numpy(model.hfield_nrow).astype(int)
        ncol = _to_numpy(model.hfield_ncol).astype(int)
        sizes = _to_numpy(model.hfield_size)
        data = _to_numpy(model.hfield_data)
    except Exception:
        return None

    flat_origins = terrain_origins.reshape(-1, terrain_origins.shape[-1])
    tile_idx = int(np.argmin(np.linalg.norm(flat_origins[:, :2] - env_origin[:2], axis=1)))
    rows = int(nrow[tile_idx])
    cols = int(ncol[tile_idx])
    offset = int(np.sum(nrow[:tile_idx] * ncol[:tile_idx]))
    height = data[offset : offset + rows * cols].reshape(rows, cols)
    size_x, size_y = float(sizes[tile_idx, 0]), float(sizes[tile_idx, 1])
    origin = flat_origins[tile_idx]
    extent = (
        float(origin[0] - size_x),
        float(origin[0] + size_x),
        float(origin[1] - size_y),
        float(origin[1] + size_y),
    )

    # Heightfield cells are normalized; any clearly raised cell is an obstacle.
    obstacle_mask = height > max(0.05, float(np.nanmax(height)) * 0.1)
    if np.any(obstacle_mask):
        # MuJoCo heightfield data is shaped (nrow, ncol), with ncol samples
        # spanning x and nrow samples spanning y. Keep this mapping aligned
        # with mjlab.viewer.viser.conversions._add_hfield.
        xs = np.linspace(extent[0], extent[1], cols)
        ys = np.linspace(extent[2], extent[3], rows)
        grid_x, grid_y = np.meshgrid(xs, ys)
        obstacle_xy = np.stack([grid_x[obstacle_mask], grid_y[obstacle_mask]], axis=-1)
    else:
        obstacle_xy = np.zeros((0, 2), dtype=np.float32)
    return height, extent, obstacle_xy


def _obstacle_component_short_sides(height: np.ndarray, extent: tuple[float, float, float, float]) -> list[float]:
    from scipy import ndimage

    mask = height > max(0.05, float(np.nanmax(height)) * 0.1)
    labels, count = ndimage.label(mask)
    cell_x = (extent[1] - extent[0]) / max(1, height.shape[1] - 1)
    cell_y = (extent[3] - extent[2]) / max(1, height.shape[0] - 1)
    short_sides: list[float] = []
    for label in range(1, count + 1):
        rows, cols = np.where(labels == label)
        if rows.size:
            width = (cols.max() - cols.min() + 1) * cell_x
            height_m = (rows.max() - rows.min() + 1) * cell_y
            short_sides.append(float(min(width, height_m)))
    return short_sides


def _corridor_obstacle_points(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacle_xy: np.ndarray,
    *,
    corridor_radius: float,
    ignore_end_radius: float,
) -> np.ndarray:
    if obstacle_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    vec = goal_xy - start_xy
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return np.zeros((0, 2), dtype=np.float32)
    rel = obstacle_xy - start_xy.reshape(1, 2)
    t = np.clip((rel @ vec) / (length * length), 0.0, 1.0)
    closest = start_xy.reshape(1, 2) + t.reshape(-1, 1) * vec.reshape(1, 2)
    dist_to_segment = np.linalg.norm(obstacle_xy - closest, axis=1)
    dist_to_start = np.linalg.norm(obstacle_xy - start_xy.reshape(1, 2), axis=1)
    dist_to_goal = np.linalg.norm(obstacle_xy - goal_xy.reshape(1, 2), axis=1)
    mask = (
        (t > 0.0)
        & (t < 1.0)
        & (dist_to_segment <= float(corridor_radius))
        & (dist_to_start >= float(ignore_end_radius))
        & (dist_to_goal >= float(ignore_end_radius))
    )
    return obstacle_xy[mask]


def rollout_and_plot(
    args: argparse.Namespace,
    rollout_index: int | None = None,
    *,
    env=None,
) -> Path:
    owns_env = env is None
    if env is None:
        env = make_env(args, num_envs=1, render=False)
    try:
        obs_raw, start_clearances, goal_clearances, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
    except Exception:
        if owns_env:
            env.close()
        raise
    terrain_debug = _active_terrain_heightfield(env)
    from unitree_nav_observation import UnitreeScanHistory, current_unitree_scan_obs

    scan_history = UnitreeScanHistory(
        getattr(args, "scan_history", 1), getattr(args, "action_history", 0), int(env.action_space.shape[-1])
    )
    obs = scan_history.reset(_prepare_policy_obs(obs_raw, args))
    policy_actor = (
        _load_policy_actor(args, obs_dim=int(obs.shape[1]), act_dim=int(env.action_space.shape[-1]))
        if args.controller == "policy"
        else None
    )
    teacher_state = ScanTeacherState(1, torch.device(args.device)) if args.controller == "scan_teacher" else None
    if args.controller == "geom_scan_teacher":
        teacher_state = GeomTeacherState.create(1)
    gate_state = InterventionGateState(1, torch.device(args.device)) if args.policy_teacher_gate else None
    gate_clearance, gate_release_clearance = resolve_intervention_clearances(args)
    gate_teacher_state = GeomTeacherState.create(1) if args.policy_teacher_gate else None

    xy_hist: list[np.ndarray] = []
    goal_hist: list[np.ndarray] = []
    cost_xy: list[np.ndarray] = []
    perceived_obstacles: list[np.ndarray] = []
    arrows: list[tuple[np.ndarray, np.ndarray]] = []
    intervention_hist: list[bool] = []
    total_cost = 0.0
    success_step: int | None = None
    previous_action = torch.zeros(1, int(env.action_space.shape[-1]), device=args.device)

    for step in range(int(args.steps)):
        xy, heading = _robot_xy_heading(env)
        goal = _command_goal_w(env)
        # Plot success against the same world-space marker being rendered. This
        # avoids transient body-frame command-cache values during reset.
        live_dist = (
            float(np.linalg.norm(goal - xy))
            if goal is not None
            else float(torch.linalg.norm(obs[0, 6:8]).detach().cpu().item())
        )
        if success_step is None and live_dist <= args.success_dist:
            success_step = step
        action = controller_action(obs, args, policy_actor, teacher_state, env=env, obstacle_cells=obstacle_cells)
        teacher_intervened = torch.zeros(1, dtype=torch.bool, device=args.device)
        if args.policy_teacher_gate:
            teacher_action = local_geometry_scan_teacher_action(
                current_unitree_scan_obs(
                    obs, scan_history=args.scan_history, action_history=args.action_history
                ),
                env=env,
                obstacle_cells=obstacle_cells,
                args=args,
                state=gate_teacher_state,
            )
            action_delta = torch.linalg.norm(action - teacher_action, dim=-1)
            gate_values = gate_state.update(
                distance=torch.linalg.norm(obs[:, 6:8], dim=-1),
                clearance=_robot_clearances(env, obstacle_cells),
                action_delta=action_delta,
                clearance_threshold=gate_clearance,
                release_clearance=gate_release_clearance,
                stall_steps=args.intervention_stall_steps,
                progress_epsilon=args.intervention_progress_epsilon,
                release_steps=args.intervention_release_steps,
                release_progress_tolerance=args.intervention_release_progress_tolerance,
                release_action_delta_max=args.intervention_release_action_delta_max,
                goal_tolerance=args.success_dist,
            )
            teacher_intervened = gate_values[0]
            action = torch.where(teacher_intervened.unsqueeze(-1), teacher_action, action)
        from eval_unitree_nav_baselines import smooth_policy_action

        action = smooth_policy_action(
            action,
            previous_action,
            controller=args.controller,
            smoothing=args.policy_action_smoothing,
        )
        previous_action.copy_(action)

        xy_hist.append(xy)
        intervention_hist.append(bool(teacher_intervened[0].item()))
        if goal is not None:
            goal_hist.append(goal)
        perceived = _scan_points_world(
            current_unitree_scan_obs(
                obs, scan_history=args.scan_history, action_history=args.action_history
            ),
            xy,
            heading,
            args.teacher_scan_block_threshold,
            forward_size=args.height_scan_forward_size,
            lateral_size=args.height_scan_lateral_size,
            resolution=args.height_scan_resolution,
        )
        if perceived.size:
            perceived_obstacles.append(perceived)

        if step % int(args.arrow_every) == 0:
            local = action[0, :2].detach().cpu().numpy()
            c, s = math.cos(heading), math.sin(heading)
            world_vec = np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
            norm = np.linalg.norm(world_vec)
            if norm > 1e-6:
                arrows.append((xy.copy(), world_vec / norm * 0.35))

        obs_raw, reward, done, extras = env.step(action)
        from eval_unitree_nav_baselines import _goal_termination_mask

        terminal_success = bool(_goal_termination_mask(env, 1, torch.device(args.device))[0].item())
        current_obs = _prepare_policy_obs(obs_raw, args)
        done_mask = done.to(args.device).reshape(-1).bool()
        obs = scan_history.step(current_obs, done_mask, action=action)
        cost = float(_extract_cost(extras, 1, torch.device(args.device))[0].detach().cpu().item())
        total_cost += cost
        if cost > 0.0:
            cost_xy.append(xy.copy())
        if success_step is None and terminal_success:
            success_step = step + 1
        if success_step is not None and not bool(args.continue_after_success):
            break
        if bool(done.reshape(-1)[0].item()):
            break

    if owns_env:
        env.close()
    xy_arr = np.asarray(xy_hist)
    goal_arr = np.asarray(goal_hist) if goal_hist else np.zeros((0, 2))
    obs_arr = np.concatenate(perceived_obstacles, axis=0) if perceived_obstacles else np.zeros((0, 2))
    cost_arr = np.asarray(cost_xy) if cost_xy else np.zeros((0, 2))
    intervention_arr = np.asarray(intervention_hist, dtype=bool)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{rollout_index:02d}" if rollout_index is not None else ""
    out_path = out_dir / f"{args.controller}_topdown_rollout{suffix}.png"

    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    if obs_arr.size:
        ax.scatter(obs_arr[:, 0], obs_arr[:, 1], s=5, c="black", alpha=0.18, label="height-scan blocked cells")
    if goal_arr.size:
        ax.scatter(goal_arr[0, 0], goal_arr[0, 1], s=260, c="limegreen", marker="*", edgecolors="black", linewidths=1.5, label="goal")
        ax.scatter(goal_arr[0, 0], goal_arr[0, 1], s=900, facecolors="none", edgecolors="limegreen", linewidths=1.6, alpha=0.8, label="goal marker")
    ax.plot(xy_arr[:, 0], xy_arr[:, 1], color="dodgerblue", linewidth=2.0, label="robot trajectory")
    teacher_label_added = False
    for idx in range(max(0, len(xy_arr) - 1)):
        if intervention_arr[idx]:
            ax.plot(
                xy_arr[idx : idx + 2, 0],
                xy_arr[idx : idx + 2, 1],
                color="crimson",
                linewidth=4.0,
                alpha=0.9,
                label="teacher intervention" if not teacher_label_added else None,
                zorder=3,
            )
            teacher_label_added = True
    ax.scatter(xy_arr[0, 0], xy_arr[0, 1], c="white", edgecolors="black", s=110, label="start", zorder=4)
    ax.scatter(xy_arr[-1, 0], xy_arr[-1, 1], c="dodgerblue", edgecolors="black", s=70, label="end", zorder=4)
    if cost_arr.size:
        ax.scatter(cost_arr[:, 0], cost_arr[:, 1], c="red", s=55, marker="x", linewidths=2.0, label="cost/collision")
    goal_obstacle_min_dist = None
    start_obstacle_min_dist = None
    corridor_stats = layout_stats[0] if layout_stats else None
    if terrain_debug is not None:
        height, extent, obstacle_xy = terrain_debug
        height_alpha = np.where(height > max(0.05, float(np.nanmax(height)) * 0.1), height, np.nan)
        ax.imshow(
            height_alpha,
            extent=extent,
            origin="lower",
            cmap="Oranges",
            alpha=0.32,
            interpolation="nearest",
            zorder=0,
        )
        if goal_arr.size:
            ax.plot(
                [xy_arr[0, 0], goal_arr[0, 0]],
                [xy_arr[0, 1], goal_arr[0, 1]],
                linestyle="--",
                color="crimson",
                linewidth=1.2,
                alpha=0.7,
                label="straight start-goal path",
            )
        if goal_arr.size and obstacle_xy.size:
            corridor_pts = _corridor_obstacle_points(
                xy_arr[0],
                goal_arr[0, :2],
                obstacle_xy,
                corridor_radius=float(args.blocked_corridor_radius),
                ignore_end_radius=float(args.blocked_corridor_ignore_end_radius),
            )
            if corridor_pts.size:
                ax.scatter(
                    corridor_pts[:, 0],
                    corridor_pts[:, 1],
                    s=18,
                    c="red",
                    alpha=0.75,
                    marker="s",
                    label="straight-path blockers",
                    zorder=3,
                )
            goal_obstacle_min_dist = float(np.min(np.linalg.norm(obstacle_xy - goal_arr[0, :2], axis=1)))
            start_obstacle_min_dist = float(start_clearances[0].detach().cpu().item()) if start_clearances is not None else None
            ax.text(
                0.02,
                0.02,
                (
                    f"goal-nearest-terrain-obstacle: {goal_obstacle_min_dist:.2f} m\n"
                    f"start-nearest-terrain-obstacle: {start_obstacle_min_dist:.2f} m\n"
                    f"straight blockers: {int(corridor_stats['blocked_cell_count']) if corridor_stats else 0}"
                    if start_obstacle_min_dist is not None
                    else f"goal-nearest-terrain-obstacle: {goal_obstacle_min_dist:.2f} m"
                ),
                transform=ax.transAxes,
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
    for origin, vec in arrows:
        ax.arrow(origin[0], origin[1], vec[0], vec[1], color="orange", width=0.01, head_width=0.08, alpha=0.8)
    ax.set_title(
        f"{args.controller}: cost={total_cost:.1f}, success_step={success_step}, steps={len(xy_hist)}, "
        f"teacher={float(intervention_arr.mean()) if intervention_arr.size else 0.0:.1%}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    obstacle_short_sides = _obstacle_component_short_sides(terrain_debug[0], terrain_debug[1]) if terrain_debug else []
    meta = {
        "controller": args.controller,
        "steps": len(xy_hist),
        "total_cost": total_cost,
        "success_step": success_step,
        "teacher_fraction": float(intervention_arr.mean()) if intervention_arr.size else 0.0,
        "goal_nearest_terrain_obstacle_m": goal_obstacle_min_dist,
        "start_nearest_terrain_obstacle_m": start_obstacle_min_dist,
        "blocked_corridor": bool(corridor_stats["blocked"]) if corridor_stats else None,
        "blocked_corridor_cell_count": int(corridor_stats["blocked_cell_count"]) if corridor_stats else None,
        "nearest_corridor_obstacle_dist": (
            float(corridor_stats["nearest_corridor_obstacle_dist"]) if corridor_stats else None
        ),
        "obstacle_component_count": len(obstacle_short_sides),
        "obstacle_component_min_short_side_m": min(obstacle_short_sides) if obstacle_short_sides else None,
        "obstacle_component_short_sides_m": obstacle_short_sides,
        "plot": str(out_path),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta), flush=True)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["direct_goal", "scan_teacher", "geom_scan_teacher", "policy"], default="scan_teacher")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--task", default="Unitree-G1-Nav-Obstacles-Safe-Collision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--episode-length-s", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--layout-generation-attempts", type=int, default=1)
    parser.add_argument("--resample-terrain-tiles", action="store_true")
    parser.add_argument("--success-dist", type=float, default=0.5)
    parser.add_argument("--terminate-on-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--height-scan-resolution", type=float, default=0.5)
    parser.add_argument("--height-scan-forward-size", type=float, default=0.0)
    parser.add_argument("--height-scan-lateral-size", type=float, default=0.0)
    parser.add_argument("--scan-history", type=int, default=1)
    parser.add_argument("--action-history", type=int, default=0)
    parser.add_argument("--mask-height-scan", action="store_true")
    parser.add_argument("--mask-proprioception", action="store_true")
    parser.add_argument("--mask-goal-heading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-env-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-action-smoothing", type=float, default=0.0)
    parser.add_argument("--policy-teacher-gate", action="store_true")
    parser.add_argument("--intervention-clearance-threshold", type=float, default=0.75)
    parser.add_argument("--intervention-release-clearance", type=float, default=0.85)
    parser.add_argument("--intervention-clearance-mode", choices=["fixed", "teacher_ratio"], default="fixed")
    parser.add_argument("--intervention-clearance-trigger-ratio", type=float, default=2.0 / 3.0)
    parser.add_argument("--intervention-clearance-release-ratio", type=float, default=5.0 / 6.0)
    parser.add_argument("--intervention-stall-steps", type=int, default=60)
    parser.add_argument("--intervention-progress-epsilon", type=float, default=0.04)
    parser.add_argument("--intervention-release-steps", type=int, default=4)
    parser.add_argument("--intervention-release-progress-tolerance", type=float, default=0.005)
    parser.add_argument("--intervention-release-action-delta-max", type=float, default=0.8)
    parser.add_argument("--continue-after-success", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--use-layer-norm", action="store_true")
    parser.add_argument("--low-level-policy-path", default=str(DEFAULT_LOW_LEVEL))
    parser.add_argument("--output-dir", default=str(ROOT / "visualizations" / "unitree_nav_debug"))
    parser.add_argument("--teacher-scan-block-threshold", type=float, default=0.12)
    parser.add_argument("--teacher-scan-planner", choices=["heuristic", "astar"], default="heuristic")
    parser.add_argument("--teacher-scan-astar-clearance", type=float, default=0.6)
    parser.add_argument("--teacher-scan-astar-cell-padding", type=float, default=0.35)
    parser.add_argument("--teacher-scan-astar-resolution", type=float, default=0.25)
    parser.add_argument("--teacher-scan-astar-waypoint-index", type=int, default=3)
    parser.add_argument("--teacher-scan-astar-side-penalty", type=float, default=8.0)
    parser.add_argument("--teacher-scan-astar-commit-steps", type=int, default=30)
    parser.add_argument("--teacher-scan-block-delta", type=float, default=0.0)
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
    parser.add_argument(
        "--blocked-goal-placement-mode",
        choices=["obstacle_multiplier", "distance_grid"],
        default="obstacle_multiplier",
    )
    parser.add_argument("--blocked-goal-distance-multiplier-min", type=float, default=1.0)
    parser.add_argument("--blocked-goal-distance-multiplier-max", type=float, default=2.0)
    parser.add_argument("--blocked-goal-candidate-attempts", type=int, default=64)
    parser.add_argument("--debug-obstacle-width-min", type=float, default=1.0)
    parser.add_argument("--debug-obstacle-width-max", type=float, default=1.4)
    parser.add_argument("--strict-min-size-obstacles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-obstacle-height-min", type=float, default=1.0)
    parser.add_argument("--debug-obstacle-height-max", type=float, default=1.0)
    parser.add_argument("--debug-num-obstacles", type=int, default=6)
    parser.add_argument("--debug-platform-width", type=float, default=2.0)
    parser.add_argument("--debug-obstacle-border-width", type=float, default=0.0)
    parser.add_argument("--debug-terrain-rows", type=int, default=0)
    parser.add_argument("--debug-terrain-cols", type=int, default=0)
    parser.add_argument("--debug-goal-through-obstacle", action="store_true")
    parser.add_argument("--goal-through-obstacle-prob", type=float, default=0.0)
    parser.add_argument("--goal-distance-min", type=float, default=0.0)
    parser.add_argument("--goal-distance-max", type=float, default=0.0)
    parser.add_argument("--debug-goal-distance", type=float, default=3.2)
    parser.add_argument("--debug-goal-obstacle-min-dist", type=float, default=0.8)
    parser.add_argument("--debug-goal-obstacle-max-dist", type=float, default=2.2)
    parser.add_argument("--arrow-every", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from unitree_nav_checkpoint import apply_unitree_checkpoint_config

    apply_unitree_checkpoint_config(args)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    paths = []
    failed_layouts = 0
    max_attempts = max(int(args.num_rollouts), int(args.layout_generation_attempts))
    env = make_env(args, num_envs=1, render=False)
    try:
        while len(paths) < int(args.num_rollouts):
            index = len(paths) + 1 if int(args.num_rollouts) > 1 else None
            try:
                paths.append(rollout_and_plot(args, rollout_index=index, env=env))
            except RuntimeError as exc:
                failed_layouts += 1
                print(json.dumps({"discarded_layout": failed_layouts, "reason": str(exc)}), flush=True)
                if len(paths) + failed_layouts >= max_attempts:
                    raise
    finally:
        env.close()

    if len(paths) > 1:
        images = [plt.imread(path) for path in paths]
        cols = min(5, len(images))
        rows = math.ceil(len(images) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.8 * rows), dpi=130)
        axes_arr = np.asarray(axes, dtype=object).reshape(-1)
        for idx, ax in enumerate(axes_arr):
            ax.axis("off")
            if idx < len(images):
                ax.imshow(images[idx])
                ax.set_title(f"rollout {idx + 1}")
        fig.tight_layout()
        contact_sheet = Path(args.output_dir) / f"{args.controller}_topdown_contact_sheet.png"
        fig.savefig(contact_sheet)
        plt.close(fig)
        print(
            json.dumps(
                {"contact_sheet": str(contact_sheet), "rollouts": len(paths), "discarded_layouts": failed_layouts}
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

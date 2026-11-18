#!/usr/bin/env python3
"""Interactive Unitree navigation debugger.

Controls:
- W/S: forward/back
- A/D: yaw left/right
- Q/E: strafe left/right
- T: toggle autopilot controller
- R: reset current layout
- Right: sample next layout
- Left: show previous cached layout metadata (full simulator restore is not available)
- Space: pause/unpause stepping
- N: single step while paused
- V: toggle RGB camera rendering
- Move a gamepad stick past its configured threshold: temporarily override the
  policy with left-stick forward/lateral and right-stick yaw
- Esc: quit
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval_unitree_nav_baselines import (  # noqa: E402
    DEFAULT_LOW_LEVEL,
    _current_goal_distance,
    _load_policy_actor,
    _reset_until_feasible,
    controller_action,
    make_env,
)
from plot_unitree_nav_rollout import _command_goal_w, _robot_xy_heading  # noqa: E402
from train_unitree_nav_thesis import ScanTeacherState, _extract_actor_obs, _extract_cost  # noqa: E402
from unitree_nav_observation import prepare_unitree_actor_obs  # noqa: E402
from unitree_nav_gamepad import (  # noqa: E402
    DEFAULT_UNITREE_GAMEPAD_CONFIG_PATH,
    DEFAULT_UNITREE_GAMEPAD_PORT,
    UnitreeGamepadController,
    UnitreeGamepadSample,
)
from unitree_nav_human_dataset import UnitreeHumanDatasetWriter  # noqa: E402


def _prepare_policy_obs(raw_obs, args):
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


def _prepare_visual_obs(raw_obs, args):
    """Return the current unmasked actor observation for live sensor displays."""
    return _extract_actor_obs(raw_obs).to(args.device, dtype=torch.float32)


def _apply_strict_blocked_obstacle_profile(args: argparse.Namespace) -> None:
    """Restore the validated intervention benchmark after goal-only checkpoint load."""
    values = {
        "disable_obstacles": False,
        "success_dist": 0.40,
        "goal_distance_min": 4.5,
        "goal_distance_max": 8.0,
        "min_goal_obstacle_clearance": 1.00,
        "goal_clearance_resample_attempts": 100,
        "min_start_obstacle_clearance": 1.00,
        "start_clearance_resample_attempts": 100,
        "require_blocked_corridor": True,
        "blocked_corridor_radius": 0.45,
        "blocked_corridor_min_cells": 4,
        "blocked_corridor_resample_attempts": 300,
        "blocked_goal_max_distance": 8.0,
        "blocked_goal_distance_sampling": "uniform",
        "blocked_goal_placement_mode": "obstacle_multiplier",
        "blocked_goal_distance_multiplier_min": 1.0,
        "blocked_goal_distance_multiplier_max": 2.0,
        "blocked_goal_candidate_attempts": 64,
        "debug_goal_through_obstacle": True,
        "goal_through_obstacle_prob": 1.0,
        "debug_goal_obstacle_min_dist": 1.0,
        "debug_goal_obstacle_max_dist": 5.5,
        "debug_num_obstacles": 6,
        "debug_obstacle_width_min": 1.0,
        "debug_obstacle_width_max": 1.4,
        "debug_obstacle_height_min": 1.0,
        "debug_obstacle_height_max": 1.0,
        "strict_min_size_obstacles": True,
        "debug_platform_width": 2.0,
        "debug_terrain_rows": 10,
        "debug_terrain_cols": 20,
    }
    for key, value in values.items():
        setattr(args, key, value)
from unitree_nav_geom_teacher import GeomTeacherState  # noqa: E402
from unitree_nav_layout import set_terrain_tile_indices, terrain_tile_shape  # noqa: E402


def _manual_action(keys: Any, *, speed: float, yaw: float, strafe: float) -> tuple[torch.Tensor, bool]:
    vx = 0.0
    vy = 0.0
    wz = 0.0
    if keys["w"]:
        vx += speed
    if keys["s"]:
        vx -= speed
    if keys["q"]:
        vy += strafe
    if keys["e"]:
        vy -= strafe
    if keys["a"]:
        wz += yaw
    if keys["d"]:
        wz -= yaw
    arr = torch.tensor([[vx, vy, wz]], dtype=torch.float32)
    return arr.clamp(-1.0, 1.0), bool(abs(vx) + abs(vy) + abs(wz) > 1e-6)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "data") and hasattr(value.data, "detach"):
        return value.data.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _blocked_scan_mask(obs: torch.Tensor, threshold: float, delta: float) -> np.ndarray:
    scan = obs[0, 9:].detach().cpu().numpy().astype(np.float64)
    threshold_blocked = scan < float(threshold)
    if float(delta) > 0.0:
        flat_ref = float(np.nanpercentile(scan, 90.0))
        relative_blocked = scan < (flat_ref - float(delta))
    else:
        relative_blocked = np.zeros_like(threshold_blocked, dtype=bool)
    return (threshold_blocked | relative_blocked).reshape(-1)


def _actual_scan_samples_world(env, obs: torch.Tensor, threshold: float, delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        sensor = env.env.unwrapped.scene.sensors.get("terrain_scan")
        hit_xy = _to_numpy(sensor.data.hit_pos_w)[0, :, :2]
    except Exception:
        empty_xy = np.zeros((0, 2), dtype=np.float32)
        empty_val = np.zeros((0,), dtype=np.float32)
        empty_mask = np.zeros((0,), dtype=bool)
        return empty_xy, empty_val, empty_mask
    values = obs[0, 9:].detach().cpu().numpy().astype(np.float32)
    blocked = _blocked_scan_mask(obs, threshold, delta)
    valid = np.isfinite(hit_xy).all(axis=1) & np.isfinite(values)
    return hit_xy[valid].astype(np.float32), values[valid].astype(np.float32), blocked[valid]


def _actual_scan_points_world(env, obs: torch.Tensor, threshold: float, delta: float) -> np.ndarray:
    pts, _, blocked = _actual_scan_samples_world(env, obs, threshold, delta)
    pts = pts[blocked]
    return pts.astype(np.float32) if pts.size else np.zeros((0, 2), dtype=np.float32)


def _reconstructed_scan_points_world(
    obs: torch.Tensor,
    robot_xy: np.ndarray,
    heading: float,
    threshold: float,
    delta: float,
) -> np.ndarray:
    flat = _blocked_scan_mask(obs, threshold, delta)
    side = int(round(flat.size ** 0.5))
    blocked = flat.reshape(side, side)
    coords = np.linspace(-1.5, 1.5, side)
    c, s = math.cos(heading), math.sin(heading)
    pts = []
    for i, lateral in enumerate(coords):
        for j, forward in enumerate(coords):
            if not blocked[i, j]:
                continue
            local = np.array([forward, lateral])
            world = robot_xy + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
            pts.append(world)
    return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)


def _world_bounds(points: list[np.ndarray], *, pad: float = 1.0) -> tuple[float, float, float, float]:
    valid = [p.reshape(-1, 2) for p in points if p is not None and p.size]
    if not valid:
        return -4.0, 4.0, -4.0, 4.0
    xy = np.concatenate(valid, axis=0)
    xmin, ymin = np.nanmin(xy, axis=0)
    xmax, ymax = np.nanmax(xy, axis=0)
    if not np.isfinite([xmin, xmax, ymin, ymax]).all():
        return -4.0, 4.0, -4.0, 4.0
    if xmax - xmin < 1e-3:
        xmax += 1.0
        xmin -= 1.0
    if ymax - ymin < 1e-3:
        ymax += 1.0
        ymin -= 1.0
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def _make_projector(rect, bounds):
    x0, y0, w, h = rect
    xmin, xmax, ymin, ymax = bounds
    sx = w / max(xmax - xmin, 1e-6)
    sy = h / max(ymax - ymin, 1e-6)
    scale = min(sx, sy)
    cx = x0 + w / 2.0
    cy = y0 + h / 2.0
    mx = 0.5 * (xmin + xmax)
    my = 0.5 * (ymin + ymax)

    def project(p):
        p = np.asarray(p, dtype=np.float64)
        return np.stack([cx + (p[..., 0] - mx) * scale, cy - (p[..., 1] - my) * scale], axis=-1)

    return project


def _draw_text(screen, font, lines: list[str], x: int, y: int, color=(230, 230, 230), line_h: int = 20) -> None:
    for i, line in enumerate(lines):
        surf = font.render(line, True, color)
        screen.blit(surf, (x, y + i * line_h))


def _draw_polyline(pygame, screen, pts: np.ndarray, color, width=2) -> None:
    if len(pts) >= 2:
        pygame.draw.lines(screen, color, False, [tuple(x) for x in pts.astype(int)], width)


def _scan_value_color(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    # Lower scan values are treated as more obstacle-like by the live teacher.
    denom = max(float(vmax) - float(vmin), 1e-6)
    t = float(np.clip((value - vmin) / denom, 0.0, 1.0))
    if t < 0.5:
        u = t / 0.5
        return (230, int(60 + 180 * u), 35)
    u = (t - 0.5) / 0.5
    return (int(230 - 180 * u), 230, int(45 + 185 * u))


def _draw_scan_legend(pygame, screen, rect, *, vmin: float, vmax: float) -> None:
    x, y, w, h = rect
    for i in range(w):
        value = vmin + (vmax - vmin) * (i / max(w - 1, 1))
        pygame.draw.line(screen, _scan_value_color(value, vmin, vmax), (x + i, y), (x + i, y + h))
    pygame.draw.rect(screen, (25, 25, 25), rect, width=1)


def _draw_rgb_panel(pygame, screen, frame, rect, *, smooth_scale: bool) -> None:
    if frame is None:
        pygame.draw.rect(screen, (20, 20, 25), rect)
        return
    arr = np.asarray(frame)
    if arr.ndim != 3:
        return
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    surf = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
    scaler = pygame.transform.smoothscale if smooth_scale else pygame.transform.scale
    surf = scaler(surf, (rect[2], rect[3]))
    screen.blit(surf, (rect[0], rect[1]))


def _manager_env(env: Any) -> Any:
    """Return the MJLab manager environment below lightweight wrappers."""
    current = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_offline_renderer") and hasattr(current, "sim"):
            return current
        next_env = getattr(current, "env", None)
        if next_env is None:
            break
        current = next_env
    return current


def _render_rgb_frame(env: Any, *, include_debug_visualizers: bool) -> np.ndarray | None:
    manager_env = _manager_env(env)
    if include_debug_visualizers:
        return manager_env.render()
    renderer = getattr(manager_env, "_offline_renderer", None)
    if renderer is None:
        return manager_env.render()
    # The scan/top-down panel already visualizes diagnostics. Avoid rebuilding
    # thousands of MuJoCo debug geoms for the low-resolution driving camera.
    renderer.update(manager_env.sim.data, debug_vis_callback=None)
    return renderer.render()


def _update_chase_camera(env: Any, *, heading: float, offset_deg: float) -> None:
    manager_env = _manager_env(env)
    renderer = getattr(manager_env, "_offline_renderer", None)
    camera = getattr(renderer, "_cam", None)
    if camera is not None:
        camera.azimuth = math.degrees(float(heading)) + float(offset_deg)


def _draw_topdown(
    pygame,
    screen,
    rect,
    *,
    trajectory: list[np.ndarray],
    cost_points: list[np.ndarray],
    robot_xy: np.ndarray,
    heading: float,
    goal_xy: np.ndarray | None,
    scan_xy: np.ndarray,
    scan_values: np.ndarray,
    scan_blocked: np.ndarray,
    reconstructed_scan_xy: np.ndarray,
    terrain_obstacles: np.ndarray,
    show_scan_samples: bool,
    action: np.ndarray,
    goal_radius: float,
    paused: bool,
    gamepad_takeover: bool,
    viewport_radius: float,
) -> None:
    pygame.draw.rect(screen, (245, 242, 235), rect)
    pygame.draw.rect(screen, (70, 70, 70), rect, width=1)
    pts_for_bounds = [
        np.asarray(trajectory),
        robot_xy.reshape(1, 2),
        scan_xy,
        reconstructed_scan_xy,
        terrain_obstacles,
    ]
    if goal_xy is not None:
        pts_for_bounds.append(goal_xy.reshape(1, 2))
    viewport_radius = max(1.0, float(viewport_radius))
    bounds = (
        float(robot_xy[0] - viewport_radius),
        float(robot_xy[0] + viewport_radius),
        float(robot_xy[1] - viewport_radius),
        float(robot_xy[1] + viewport_radius),
    )
    project = _make_projector(rect, bounds)

    if terrain_obstacles.size:
        p = project(terrain_obstacles)
        for x, y in p.astype(int):
            pygame.draw.rect(screen, (82, 82, 82), (x - 4, y - 4, 8, 8))
    if show_scan_samples and scan_xy.size:
        vmin = float(np.nanpercentile(scan_values, 5.0)) if scan_values.size else 0.0
        vmax = float(np.nanpercentile(scan_values, 95.0)) if scan_values.size else 1.0
        if abs(vmax - vmin) < 1e-6:
            vmin = float(np.nanmin(scan_values)) - 0.05
            vmax = float(np.nanmax(scan_values)) + 0.05
        p = project(scan_xy)
        for idx, (x, y) in enumerate(p.astype(int)):
            is_blocked = bool(scan_blocked[idx])
            color = (235, 45, 35) if is_blocked else _scan_value_color(float(scan_values[idx]), vmin, vmax)
            radius = 11 if is_blocked else 7
            pygame.draw.circle(screen, color, (x, y), radius)
            pygame.draw.circle(screen, (245, 245, 245), (x, y), radius, width=1)
            if is_blocked:
                pygame.draw.circle(screen, (20, 20, 20), (x, y), radius + 2, width=2)
        legend_rect = (rect[0] + 14, rect[1] + rect[3] - 104, 170, 13)
        _draw_scan_legend(pygame, screen, legend_rect, vmin=vmin, vmax=vmax)
    if reconstructed_scan_xy.size:
        p = project(reconstructed_scan_xy)
        for x, y in p.astype(int):
            pygame.draw.circle(screen, (170, 30, 180), (x, y), 5, width=1)
    if len(trajectory) >= 2:
        _draw_polyline(pygame, screen, project(np.asarray(trajectory)), (35, 110, 220), width=3)
    if cost_points:
        for p in project(np.asarray(cost_points)).astype(int):
            pygame.draw.line(screen, (230, 20, 20), (p[0] - 7, p[1] - 7), (p[0] + 7, p[1] + 7), 3)
            pygame.draw.line(screen, (230, 20, 20), (p[0] - 7, p[1] + 7), (p[0] + 7, p[1] - 7), 3)
    rp = project(robot_xy.reshape(1, 2))[0].astype(int)
    if goal_xy is not None:
        gp_unclipped = project(goal_xy.reshape(1, 2))[0]
        inset = 18
        gp = np.array(
            [
                np.clip(gp_unclipped[0], rect[0] + inset, rect[0] + rect[2] - inset),
                np.clip(gp_unclipped[1], rect[1] + inset, rect[1] + rect[3] - inset),
            ],
            dtype=np.int32,
        )
        goal_visible = bool(np.allclose(gp, gp_unclipped, atol=1.0))
        pygame.draw.line(screen, (70, 175, 70), tuple(rp), tuple(gp), 2)
        if goal_visible:
            radius_point = project((goal_xy + np.array([max(0.0, goal_radius), 0.0])).reshape(1, 2))[0]
            goal_radius_px = max(4, int(round(abs(float(radius_point[0] - gp[0])))))
            pygame.draw.circle(screen, (50, 220, 60), tuple(gp), goal_radius_px, width=3)
            pygame.draw.circle(screen, (240, 220, 20), tuple(gp), 7)
        else:
            pygame.draw.circle(screen, (240, 220, 20), tuple(gp), 10)
            pygame.draw.circle(screen, (50, 220, 60), tuple(gp), 14, width=3)
    robot_color = (220, 35, 45) if gamepad_takeover else (40, 190, 240)
    pygame.draw.circle(screen, robot_color, tuple(rp), 9)
    if gamepad_takeover:
        pygame.draw.circle(screen, (120, 10, 15), tuple(rp), 15, width=3)
    forward = np.array([math.cos(heading), math.sin(heading)])
    hp = project((robot_xy + 0.55 * forward).reshape(1, 2))[0].astype(int)
    pygame.draw.line(screen, (0, 0, 0), tuple(rp), tuple(hp), 3)
    if np.linalg.norm(action[:2]) > 1e-6:
        c, s = math.cos(heading), math.sin(heading)
        local = action[:2]
        world = np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
        ap = project((robot_xy + 0.7 * world).reshape(1, 2))[0].astype(int)
        command_color = (230, 25, 35) if gamepad_takeover else (245, 140, 0)
        pygame.draw.line(screen, command_color, tuple(rp), tuple(ap), 4)
    if paused:
        overlay = pygame.Surface((rect[2], 34), pygame.SRCALPHA)
        overlay.fill((200, 80, 20, 165))
        screen.blit(overlay, (rect[0], rect[1]))


def _draw_student_observation_view(
    pygame,
    screen,
    rect,
    *,
    robot_xy: np.ndarray,
    heading: float,
    scan_xy: np.ndarray,
    scan_values: np.ndarray,
    scan_blocked: np.ndarray,
    goal_body: np.ndarray,
    gamepad_takeover: bool,
) -> None:
    """Render only navigation information available in the actor observation."""
    pygame.draw.rect(screen, (16, 22, 25), rect)
    pygame.draw.rect(screen, (90, 105, 108), rect, width=1)
    center = np.array([rect[0] + rect[2] * 0.5, rect[1] + rect[3] * 0.68], dtype=np.float32)
    rel_world = scan_xy - robot_xy.reshape(1, 2) if scan_xy.size else np.zeros((0, 2), dtype=np.float32)
    c, s = math.cos(heading), math.sin(heading)
    scan_body = (
        np.stack(
            [c * rel_world[:, 0] + s * rel_world[:, 1], -s * rel_world[:, 0] + c * rel_world[:, 1]],
            axis=-1,
        )
        if rel_world.size
        else rel_world
    )
    horizon = max(1.0, float(np.max(np.linalg.norm(scan_body, axis=1))) if scan_body.size else 1.75)
    scale = 0.42 * min(rect[2], rect[3]) / horizon

    def project_body(points: np.ndarray) -> np.ndarray:
        result = np.empty_like(points, dtype=np.float32)
        result[:, 0] = center[0] - points[:, 1] * scale
        result[:, 1] = center[1] - points[:, 0] * scale
        return result

    pygame.draw.circle(screen, (70, 85, 88), tuple(center.astype(int)), int(horizon * scale), width=2)
    pygame.draw.line(
        screen,
        (65, 90, 95),
        tuple(center.astype(int)),
        tuple((center + np.array([0.0, -horizon * scale])).astype(int)),
        2,
    )
    if scan_body.size:
        finite = scan_values[np.isfinite(scan_values)]
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0
        for idx, point in enumerate(project_body(scan_body).astype(int)):
            blocked = bool(scan_blocked[idx])
            color = (240, 45, 35) if blocked else _scan_value_color(float(scan_values[idx]), vmin, vmax)
            pygame.draw.circle(screen, color, tuple(point), 7 if blocked else 5)
            if blocked:
                pygame.draw.circle(screen, (5, 5, 5), tuple(point), 9, width=2)
    goal_body = np.asarray(goal_body, dtype=np.float32).reshape(2)
    goal_norm = float(np.linalg.norm(goal_body))
    if goal_norm > 1e-6:
        goal_horizon = goal_body / goal_norm * (0.95 * horizon)
        goal_point = project_body(goal_horizon.reshape(1, 2))[0].astype(int)
        pygame.draw.line(screen, (80, 230, 95), tuple(center.astype(int)), tuple(goal_point), 4)
        pygame.draw.circle(screen, (250, 225, 30), tuple(goal_point), 11)
        pygame.draw.circle(screen, (70, 230, 80), tuple(goal_point), 15, width=3)
    robot_color = (235, 35, 45) if gamepad_takeover else (40, 190, 240)
    pygame.draw.circle(screen, robot_color, tuple(center.astype(int)), 11)
    pygame.draw.polygon(
        screen,
        (235, 235, 235),
        [
            (int(center[0]), int(center[1] - 18)),
            (int(center[0] - 7), int(center[1] - 5)),
            (int(center[0] + 7), int(center[1] - 5)),
        ],
    )


_ACTIVE_INTERACTIVE_CLEANUPS: list[tuple[str, object]] = []


def _register_interactive_cleanup(name: str, callback) -> None:
    _ACTIVE_INTERACTIVE_CLEANUPS.append((name, callback))


def _close_dataset_writer(writer) -> None:
    writer.close()
    print(
        f"[human-dataset] saved {writer.summary.rows} transitions in {writer.root}",
        flush=True,
    )


def _run_interactive(args: argparse.Namespace) -> int:
    import pygame

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    pygame.init()
    _register_interactive_cleanup("pygame", pygame.quit)
    _register_interactive_cleanup("pygame display", pygame.display.quit)
    pygame.display.set_caption("Unitree Navigation Interactive Debugger")
    windowed_size = (int(args.window_width), int(args.window_height))
    fullscreen = False
    screen = pygame.display.set_mode(windowed_size, pygame.RESIZABLE)
    args.window_width, args.window_height = screen.get_size()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    small = pygame.font.Font(None, 18)

    env = make_env(args, num_envs=1, render=bool(args.show_rgb))
    _register_interactive_cleanup("environment", env.close)
    obs_raw, _, _, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
    from eval_unitree_nav_baselines import _terrain_obstacle_cells_global

    global_terrain_xy = _terrain_obstacle_cells_global(env)
    from unitree_nav_observation import UnitreeScanHistory

    scan_history = UnitreeScanHistory(
        getattr(args, "scan_history", 1),
        getattr(args, "action_history", 0),
        int(env.action_space.shape[-1]),
        history_stride=getattr(args, "scan_history_stride", 1)
    )
    obs = scan_history.reset(_prepare_policy_obs(obs_raw, args))
    online_learner = None
    if bool(args.online_train):
        if args.controller != "policy":
            raise ValueError("--online-train requires --controller policy")
        from unitree_nav_online_human import UnitreeOnlineHumanLearner

        online_learner = UnitreeOnlineHumanLearner(
            args, obs_dim=int(obs.shape[1]), act_dim=int(env.action_space.shape[-1])
        )
        _register_interactive_cleanup("online learner", online_learner.close)
        policy_actor = online_learner.actor
    else:
        policy_actor = (
            _load_policy_actor(args, obs_dim=int(obs.shape[1]), act_dim=int(env.action_space.shape[-1]))
            if args.controller == "policy"
            else None
        )
    teacher_state = ScanTeacherState(1, torch.device(args.device)) if args.controller == "scan_teacher" else None
    if args.controller == "geom_scan_teacher":
        teacher_state = GeomTeacherState.create(1)

    human_controller = None
    if args.human_input_device == "gamepad":
        human_controller = UnitreeGamepadController(
            mode=args.gamepad_mode,
            host=args.gamepad_host,
            port=int(args.gamepad_port),
            config_path=args.gamepad_config_path,
            device_index=int(args.gamepad_device_index),
            reconnect_seconds=float(args.gamepad_reconnect_seconds),
            stale_timeout_s=float(args.gamepad_stale_timeout_s),
            intervention_mode=str(args.gamepad_intervention_mode),
            intervention_threshold=float(args.gamepad_intervention_threshold),
            invert_lateral=bool(args.gamepad_invert_lateral),
        )
        _register_interactive_cleanup("human controller", human_controller.close)
        print(
            "[human-gamepad] "
            f"mode={args.gamepad_mode} target={args.gamepad_host}:{args.gamepad_port} "
            f"config={Path(args.gamepad_config_path).expanduser()} "
            f"intervention_mode={args.gamepad_intervention_mode} "
            f"threshold={args.gamepad_intervention_threshold:.3f} "
            f"invert_lateral={int(args.gamepad_invert_lateral)}",
            flush=True,
        )

    dataset_writer = None
    if str(args.human_dataset_dir).strip():
        dataset_writer = UnitreeHumanDatasetWriter(
            args.human_dataset_dir,
            metadata={
                "task": args.task,
                "model_path": str(args.model_path),
                "controller": args.controller,
                "seed": int(args.seed),
                "high_level_control_hz": 20.0,
                "args": vars(args),
                "gamepad_config": (
                    vars(human_controller.config) if human_controller is not None else None
                ),
            },
            chunk_size=int(args.human_dataset_chunk_size),
        )
        _register_interactive_cleanup(
            "human dataset",
            lambda writer=dataset_writer: _close_dataset_writer(writer),
        )
        print(f"[human-dataset] recording immutable raw transitions to {dataset_writer.root}", flush=True)

    keys = {k: False for k in ["w", "a", "s", "d", "q", "e"]}
    autopilot = bool(args.autopilot)
    paused = bool(args.start_paused)
    single_step = False
    episode_idx = 0
    terrain_rows, terrain_cols = terrain_tile_shape(env)
    terrain_tile_count = terrain_rows * terrain_cols
    current_tile = 0
    previous_note = ""
    trajectory: list[np.ndarray] = []
    cost_points: list[np.ndarray] = []
    total_cost = 0.0
    success_step: int | None = None
    step = 0
    frame = None
    last_rgb_render_ms = 0.0
    rgb_render_ema_ms = 0.0
    rgb_render_count = 0
    done_waiting = False
    last_step_time = 0.0
    previous_policy_action = torch.zeros(1, int(env.action_space.shape[-1]), device=args.device)
    held_policy_action = torch.zeros_like(previous_policy_action)
    from unitree_nav_control import navigation_action_repeat, project_navigation_action

    navigation_repeat = navigation_action_repeat(float(args.navigation_policy_hz))
    display_every = max(1, int(args.display_every))
    rgb_every = max(1, int(args.rgb_every))
    viewer_radius = max(1.0, float(args.viewer_map_radius))
    rgb_enabled = bool(args.show_rgb)
    scan_samples_enabled = bool(args.show_scan_samples)
    student_view_enabled = bool(args.student_view)
    previous_human_active = False
    previous_gamepad_active = False
    previous_gamepad_buttons: dict[str, bool] = {}
    ui_details_enabled = not bool(args.compact_ui)
    last_gamepad_status_at = 0.0
    last_gamepad_transport_signature: tuple[object, ...] | None = None

    def reset_episode(note: str, *, advance_tile: bool = False) -> None:
        nonlocal env, obs, obs_raw, layout_stats, obstacle_cells, teacher_state, episode_idx, current_tile
        nonlocal trajectory, cost_points, total_cost, success_step, step, previous_note, frame, done_waiting, last_step_time
        nonlocal previous_policy_action, held_policy_action
        if advance_tile and terrain_tile_count > 0:
            last_error: RuntimeError | None = None
            for _ in range(max(1, int(args.layout_seed_skip_attempts))):
                current_tile = (current_tile + 1) % terrain_tile_count
                unwrapped = env.env.unwrapped
                env_ids = torch.zeros(1, dtype=torch.long, device=unwrapped.device)
                set_terrain_tile_indices(
                    env,
                    env_ids,
                    torch.as_tensor([current_tile], device=unwrapped.device),
                )
                try:
                    obs_raw, _, _, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
                    break
                except RuntimeError as exc:
                    last_error = exc
                    episode_idx += 1
            else:
                raise RuntimeError(
                    f"Could not find a feasible layout after {args.layout_seed_skip_attempts} terrain tiles"
                ) from last_error
        else:
            obs_raw, _, _, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
        obs = scan_history.reset(_prepare_policy_obs(obs_raw, args))
        if teacher_state is not None:
            teacher_state.reset()
        trajectory = []
        cost_points = []
        total_cost = 0.0
        success_step = None
        step = 0
        frame = None
        done_waiting = False
        last_step_time = 0.0
        previous_policy_action.zero_()
        held_policy_action.zero_()
        previous_note = f"{note} | seed={args.seed} tile={current_tile}/{max(0, terrain_tile_count - 1)}"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE and not fullscreen:
                windowed_size = (max(900, int(event.w)), max(600, int(event.h)))
                screen = pygame.display.set_mode(windowed_size, pygame.RESIZABLE)
                args.window_width, args.window_height = screen.get_size()
            elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                down = event.type == pygame.KEYDOWN
                if event.key == pygame.K_F11 and down:
                    if fullscreen:
                        screen = pygame.display.set_mode(windowed_size, pygame.RESIZABLE)
                        fullscreen = False
                    else:
                        windowed_size = screen.get_size()
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        fullscreen = True
                    args.window_width, args.window_height = screen.get_size()
                    previous_note = (
                        "fullscreen enabled" if fullscreen else
                        f"windowed {args.window_width}x{args.window_height}"
                    )
                elif event.key == pygame.K_ESCAPE and down:
                    running = False
                elif event.key == pygame.K_SPACE and down:
                    paused = not paused
                elif event.key == pygame.K_n and down:
                    single_step = True
                elif event.key == pygame.K_t and down:
                    autopilot = not autopilot
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS) and down:
                    display_every = min(128, display_every * 2)
                    previous_note = f"display every {display_every} simulation steps"
                elif event.key == pygame.K_MINUS and down:
                    display_every = max(1, display_every // 2)
                    previous_note = f"display every {display_every} simulation steps"
                elif event.key == pygame.K_PAGEUP and down:
                    viewer_radius = max(2.0, viewer_radius / 1.25)
                    previous_note = f"map radius {viewer_radius:.1f}m"
                elif event.key == pygame.K_PAGEDOWN and down:
                    viewer_radius = min(30.0, viewer_radius * 1.25)
                    previous_note = f"map radius {viewer_radius:.1f}m"
                elif event.key in (pygame.K_UP, pygame.K_RIGHTBRACKET) and down:
                    rgb_every = min(128, rgb_every * 2)
                    previous_note = f"RGB render every {rgb_every} simulation steps"
                elif event.key in (pygame.K_DOWN, pygame.K_LEFTBRACKET) and down:
                    rgb_every = max(1, rgb_every // 2)
                    previous_note = f"RGB render every {rgb_every} simulation steps"
                elif event.key == pygame.K_v and down:
                    if bool(args.show_rgb):
                        rgb_enabled = not rgb_enabled
                        if not rgb_enabled:
                            frame = None
                        previous_note = f"RGB rendering {'enabled' if rgb_enabled else 'disabled'}"
                    else:
                        previous_note = "RGB rendering unavailable; launch with --show-rgb"
                elif event.key == pygame.K_h and down:
                    scan_samples_enabled = not scan_samples_enabled
                    previous_note = f"scan samples {'enabled' if scan_samples_enabled else 'hidden'}"
                elif event.key == pygame.K_i and down:
                    ui_details_enabled = not ui_details_enabled
                    previous_note = f"detailed telemetry {'enabled' if ui_details_enabled else 'hidden'}"
                elif event.key == pygame.K_o and down:
                    student_view_enabled = not student_view_enabled
                    previous_note = f"student-only view {'enabled' if student_view_enabled else 'disabled'}"
                elif event.key == pygame.K_r and down:
                    reset_episode("reset")
                elif event.key == pygame.K_RIGHT and down:
                    episode_idx += 1
                    reset_episode("next layout", advance_tile=True)
                elif event.key == pygame.K_LEFT and down:
                    previous_note = "previous episode restore is not available for this Isaac wrapper; showing current layout"
                elif event.key in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_q, pygame.K_e):
                    keys[pygame.key.name(event.key)] = down

        robot_xy, heading = _robot_xy_heading(env)
        goal = _command_goal_w(env)
        goal_xy = None if goal is None else goal[:2].copy()
        # Sensor diagnostics must remain truthful even when the loaded policy
        # masks its height scan or consumes stacked scan/action history.
        visual_obs = _prepare_visual_obs(obs_raw, args)
        scan_xy, scan_values, scan_blocked = _actual_scan_samples_world(
            env,
            visual_obs,
            args.teacher_scan_block_threshold,
            args.teacher_scan_block_delta,
        )
        reconstructed_scan_xy = (
            _reconstructed_scan_points_world(
                visual_obs,
                robot_xy,
                heading,
                args.teacher_scan_block_threshold,
                args.teacher_scan_block_delta,
            )
            if bool(args.show_reconstructed_scan)
            else np.zeros((0, 2), dtype=np.float32)
        )
        if global_terrain_xy.size:
            relative_terrain = np.abs(global_terrain_xy - robot_xy.reshape(1, 2))
            inside_viewport = np.max(relative_terrain, axis=1) <= viewer_radius
            terrain_xy = global_terrain_xy[inside_viewport]
        else:
            terrain_xy = np.zeros((0, 2), dtype=np.float32)
        goal_clearance = (
            float(np.min(np.linalg.norm(terrain_xy - goal_xy.reshape(1, 2), axis=1)))
            if goal_xy is not None and terrain_xy.size
            else float("inf")
        )
        manual_action, manual_active = _manual_action(
            keys,
            speed=float(args.manual_speed),
            yaw=float(args.manual_yaw),
            strafe=float(args.manual_strafe),
        )
        gamepad_sample = human_controller.sample() if human_controller is not None else None
        if gamepad_sample is not None and bool(args.gamepad_debug_console):
            transport = gamepad_sample.transport
            signature = (
                gamepad_sample.connected,
                gamepad_sample.stale,
                transport.get("peer", ""),
                transport.get("error", ""),
            )
            now_status = time.time()
            status_changed = signature != last_gamepad_transport_signature
            if status_changed or now_status - last_gamepad_status_at >= float(args.gamepad_status_interval_s):
                named_axes = {
                    str(name): round(float(value), 3)
                    for name, value in dict(gamepad_sample.state.get("named_axes", {})).items()
                    if abs(float(value)) >= 0.02
                }
                buttons = [
                    str(name)
                    for name, value in dict(gamepad_sample.state.get("buttons", {})).items()
                    if bool(value)
                ]
                print(
                    "[human-gamepad] "
                    f"target={transport.get('target', '')} transport_connected={int(transport.get('connected', False))} "
                    f"receiving={int(gamepad_sample.connected and not gamepad_sample.stale)} "
                    f"stale={int(gamepad_sample.stale)} age_s={gamepad_sample.state_age_s:.3f} "
                    f"peer={transport.get('peer', '') or '-'} error={transport.get('error', '') or '-'} "
                    f"stick_takeover={int(gamepad_sample.gate_held)} axes={named_axes} buttons={buttons}",
                    flush=True,
                )
                last_gamepad_status_at = now_status
                last_gamepad_transport_signature = signature
        gamepad_buttons = (
            {name: bool(value) for name, value in dict(gamepad_sample.state.get("buttons", {})).items()}
            if gamepad_sample is not None and gamepad_sample.connected and not gamepad_sample.stale
            else {}
        )

        def gamepad_pressed(name: str) -> bool:
            current = bool(gamepad_buttons.get(name, False))
            return current and not bool(previous_gamepad_buttons.get(name, False))

        if gamepad_pressed("back"):
            previous_note = "gamepad Back: stopping collection"
            running = False
        elif gamepad_pressed("start"):
            paused = not paused
            previous_note = f"gamepad Start: {'paused' if paused else 'running'}"
        elif gamepad_pressed("x"):
            reset_episode("gamepad X reset")
            previous_gamepad_buttons = gamepad_buttons
            continue
        elif gamepad_pressed("y"):
            episode_idx += 1
            reset_episode("gamepad Y next layout", advance_tile=True)
            previous_gamepad_buttons = gamepad_buttons
            continue
        elif gamepad_pressed("a"):
            single_step = True
            previous_note = "gamepad A: single step"
        previous_gamepad_buttons = gamepad_buttons
        if not running:
            break
        gamepad_action = (
            torch.as_tensor(gamepad_sample.action, dtype=torch.float32, device=args.device).reshape(1, -1)
            if gamepad_sample is not None
            else torch.zeros_like(manual_action, device=args.device)
        )
        gamepad_active = bool(gamepad_sample.intervening) if gamepad_sample is not None else False
        with torch.no_grad():
            if step % navigation_repeat == 0:
                held_policy_action = controller_action(
                    obs, args, policy_actor, teacher_state, env=env, obstacle_cells=obstacle_cells
                )
            from eval_unitree_nav_baselines import smooth_policy_action

            if step % navigation_repeat == 0:
                held_policy_action = smooth_policy_action(
                    held_policy_action,
                    previous_policy_action,
                    controller=args.controller,
                    smoothing=args.policy_action_smoothing,
                )
                held_policy_action = project_navigation_action(held_policy_action)
            auto_action = held_policy_action
        if manual_active:
            action = manual_action.to(args.device)
            human_action = action
            control_source = 1
        elif gamepad_active:
            action = gamepad_action
            human_action = action
            control_source = 2
        else:
            action = auto_action if autopilot else torch.zeros_like(auto_action)
            human_action = torch.zeros_like(action)
            control_source = 0
        human_active = bool(manual_active or gamepad_active)
        action = project_navigation_action(action)
        if gamepad_active != previous_gamepad_active:
            event = "TAKEOVER START" if gamepad_active else "TAKEOVER END"
            policy_values = np.array2string(auto_action[0].detach().cpu().numpy(), precision=3)
            gamepad_values = np.array2string(gamepad_action[0].detach().cpu().numpy(), precision=3)
            executed_values = np.array2string(action[0].detach().cpu().numpy(), precision=3)
            print(
                f"[human-gamepad] {event} stick_takeover={int(gamepad_active)} "
                f"policy={policy_values} gamepad={gamepad_values} executed={executed_values}",
                flush=True,
            )
        previous_gamepad_active = gamepad_active
        if not manual_active and autopilot:
            previous_policy_action.copy_(action)

        now = time.time()
        sim_fps = float(args.sim_fps)
        if sim_fps <= 0.0:
            should_step = (not paused and not done_waiting) or single_step
        else:
            step_interval = 1.0 / max(sim_fps, 1e-6)
            should_step = (not paused and not done_waiting and (now - last_step_time >= step_interval)) or single_step
        did_step = False
        if should_step:
            did_step = True
            last_step_time = now
            trajectory.append(robot_xy.copy())
            live_dist = float(_current_goal_distance(env, 1, torch.device(args.device))[0].detach().cpu().item())
            record_obs = obs[0].detach().cpu().numpy()
            record_base_obs = visual_obs[0].detach().cpu().numpy()
            record_student_action = auto_action[0].detach().cpu().numpy()
            record_human_action = human_action[0].detach().cpu().numpy()
            record_executed_action = action[0].detach().cpu().numpy()
            if success_step is None and live_dist <= args.success_dist:
                success_step = step
            obs_raw, reward, done, extras = env.step(action)
            from eval_unitree_nav_baselines import _goal_termination_mask

            terminal_success = bool(_goal_termination_mask(env, 1, torch.device(args.device))[0].item())
            current_obs = _prepare_policy_obs(obs_raw, args)
            done_mask = done.to(args.device).reshape(-1).bool()
            obs = scan_history.step(current_obs, done_mask, action=action)
            next_goal_distance = float(_current_goal_distance(env, 1, torch.device(args.device))[0].detach().cpu().item())
            reached_goal = bool(terminal_success or next_goal_distance <= args.success_dist)
            continuous_goals = str(args.navigation_episode_mode) == "continuous_goals"
            learning_boundary = bool(done_mask[0].item() or (continuous_goals and reached_goal))
            cost = float(_extract_cost(extras, 1, torch.device(args.device))[0].detach().cpu().item())
            total_cost += cost
            if cost > 0.0:
                cost_points.append(robot_xy.copy())
            if rgb_enabled and not student_view_enabled and (step % rgb_every == 0):
                try:
                    if bool(args.video_camera_follow_heading):
                        _update_chase_camera(
                            env,
                            heading=heading,
                            offset_deg=float(args.video_camera_heading_offset),
                        )
                    render_started = time.perf_counter()
                    frame = _render_rgb_frame(
                        env,
                        include_debug_visualizers=bool(args.rgb_debug_visualizers),
                    )
                    last_rgb_render_ms = 1000.0 * (time.perf_counter() - render_started)
                    rgb_render_ema_ms = (
                        last_rgb_render_ms
                        if rgb_render_ema_ms <= 0.0
                        else 0.9 * rgb_render_ema_ms + 0.1 * last_rgb_render_ms
                    )
                    rgb_render_count += 1
                    if bool(args.rgb_profile_console) and rgb_render_count % int(args.rgb_profile_interval) == 0:
                        print(
                            f"[rgb-profile] frame={rgb_render_count} last_ms={last_rgb_render_ms:.1f} "
                            f"ema_ms={rgb_render_ema_ms:.1f} size={args.video_width}x{args.video_height} "
                            f"stride={rgb_every}",
                            flush=True,
                        )
                except Exception:
                    frame = None
            if success_step is None and (
                terminal_success
                or next_goal_distance <= args.success_dist
            ):
                success_step = step + 1
            if dataset_writer is not None:
                next_base_obs = _prepare_visual_obs(obs_raw, args)[0].detach().cpu().numpy()
                dataset_writer.add(
                    observations=record_obs,
                    base_observations=record_base_obs,
                    student_actions=record_student_action,
                    human_actions=record_human_action,
                    executed_actions=record_executed_action,
                    next_observations=obs[0].detach().cpu().numpy(),
                    next_base_observations=next_base_obs,
                    reward=float(reward.reshape(-1)[0].detach().cpu().item()),
                    cost=cost,
                    done=learning_boundary,
                    terminal_success=reached_goal,
                    intervened=human_active,
                    intervention_start=human_active and not previous_human_active,
                    intervention_end=previous_human_active and not human_active,
                    control_source=control_source,
                    gamepad_connected=bool(gamepad_sample.connected) if gamepad_sample is not None else False,
                    gamepad_stale=bool(gamepad_sample.stale) if gamepad_sample is not None else False,
                    gamepad_command_norm=(float(gamepad_sample.command_norm) if gamepad_sample is not None else 0.0),
                    action_delta_to_policy=float(np.linalg.norm(record_executed_action - record_student_action)),
                    goal_distance=live_dist,
                    next_goal_distance=next_goal_distance,
                    episode_index=episode_idx,
                    step_index=step,
                    wall_time_unix_s=time.time(),
                )
            if online_learner is not None:
                online_learner.observe(
                    obs=torch.as_tensor(record_obs, dtype=torch.float32, device=args.device).reshape(1, -1),
                    student_action=torch.as_tensor(
                        record_student_action, dtype=torch.float32, device=args.device
                    ).reshape(1, -1),
                    executed_action=torch.as_tensor(
                        record_executed_action, dtype=torch.float32, device=args.device
                    ).reshape(1, -1),
                    next_obs=obs.detach(),
                    done=learning_boundary,
                    terminal_success=reached_goal,
                    intervened=human_active,
                    intervention_start=human_active and not previous_human_active,
                    goal_distance=live_dist,
                    next_goal_distance=next_goal_distance,
                    cost=cost,
                )
                if int(args.online_total_steps) > 0 and online_learner.step >= int(args.online_total_steps):
                    previous_note = f"online training reached {args.online_total_steps} steps"
                    running = False
            previous_human_active = human_active
            step += 1
            single_step = False
            if continuous_goals and reached_goal and not bool(done_mask[0].item()):
                from eval_unitree_nav_baselines import _resample_continuous_goals

                next_raw, goal_clearances, next_layout_stats = _resample_continuous_goals(
                    args,
                    env,
                    obstacle_cells,
                    torch.zeros(1, dtype=torch.long, device=args.device),
                )
                layout_stats[0] = next_layout_stats[0]
                obs = scan_history.reset(_prepare_policy_obs(next_raw, args))
                if teacher_state is not None:
                    teacher_state.reset()
                episode_idx += 1
                trajectory = []
                cost_points = []
                total_cost = 0.0
                success_step = None
                step = 0
                previous_policy_action.zero_()
                held_policy_action.zero_()
                previous_human_active = False
                previous_note = (
                    f"goal reached; continuing from current pose | next clearance="
                    f"{float(goal_clearances[0].item()):.2f}m"
                )
            elif bool(done.reshape(-1)[0].item()) or (
                str(args.navigation_episode_mode) == "episodic" and step >= int(args.max_steps)
            ):
                if bool(args.auto_reset):
                    episode_idx += 1
                    reset_episode("auto-reset after done/timeout")
                else:
                    done_waiting = True
                    paused = True
                    previous_note = "episode ended/timeout; press R to reset or Right for next layout"

        # Keep policy/control updates at every simulation step while allowing the
        # expensive UI and RGB panels to update less frequently.
        if did_step and not (paused or done_waiting or step % display_every == 0):
            continue

        screen.fill((18, 18, 22))
        show_rgb_panel = bool(args.show_rgb) and not student_view_enabled
        top_width = int(args.window_width * (0.68 if show_rgb_panel else 0.76))
        top_rect = (10, 10, top_width, int(args.window_height) - 20)
        side_x = top_rect[0] + top_rect[2] + 10
        side_width = int(args.window_width) - side_x - 10
        if show_rgb_panel:
            rgb_rect = (side_x, 10, side_width, int(args.window_height * 0.48))
            info_rect = (
                side_x,
                rgb_rect[1] + rgb_rect[3] + 10,
                side_width,
                int(args.window_height) - rgb_rect[3] - 30,
            )
        else:
            rgb_rect = (side_x, 10, side_width, 0)
            info_rect = (side_x, 10, side_width, int(args.window_height) - 20)
        if student_view_enabled:
            _draw_student_observation_view(
                pygame,
                screen,
                top_rect,
                robot_xy=robot_xy,
                heading=heading,
                scan_xy=scan_xy,
                scan_values=scan_values,
                scan_blocked=scan_blocked,
                goal_body=visual_obs[0, 6:8].detach().cpu().numpy(),
                gamepad_takeover=gamepad_active,
            )
        else:
            _draw_topdown(
                pygame,
                screen,
                top_rect,
                trajectory=trajectory,
                cost_points=cost_points,
                robot_xy=robot_xy,
                heading=heading,
                goal_xy=goal_xy,
                scan_xy=scan_xy,
                scan_values=scan_values,
                scan_blocked=scan_blocked,
                reconstructed_scan_xy=reconstructed_scan_xy,
                terrain_obstacles=terrain_xy,
                show_scan_samples=scan_samples_enabled,
                action=action[0].detach().cpu().numpy(),
                goal_radius=float(args.success_dist),
                paused=paused,
                gamepad_takeover=gamepad_active,
                viewport_radius=viewer_radius,
            )
        if gamepad_active:
            # This is deliberately large and high-contrast: the human must
            # know immediately that their command, not the policy, is live.
            takeover_overlay = pygame.Surface((int(args.window_width), int(args.window_height)), pygame.SRCALPHA)
            takeover_overlay.fill((210, 15, 25, 42))
            screen.blit(takeover_overlay, (0, 0))
            pygame.draw.rect(screen, (245, 35, 45), (3, 3, int(args.window_width) - 6, int(args.window_height) - 6), width=7)
        if show_rgb_panel:
            _draw_rgb_panel(
                pygame,
                screen,
                frame,
                rgb_rect,
                smooth_scale=bool(args.rgb_smooth_scale),
            )
            if not rgb_enabled:
                _draw_text(
                    screen,
                    font,
                    ["RGB RENDERING OFF", "Press V to resume camera rendering"],
                    rgb_rect[0] + 16,
                    rgb_rect[1] + 16,
                    color=(225, 225, 230),
                )
        pygame.draw.rect(screen, (32, 32, 40), info_rect)
        pygame.draw.rect(screen, (70, 70, 80), info_rect, width=1)
        layout_line = layout_stats[0] if layout_stats else {}
        goal_distance_now = float(_current_goal_distance(env, 1, torch.device(args.device))[0].detach().cpu().item())
        detailed_info_lines = [
            f"episode={episode_idx} step={step} paused={paused} autopilot={autopilot}",
            f"sim_fps={'unlimited' if float(args.sim_fps) <= 0.0 else f'{args.sim_fps:g}'} ui_fps={args.fps:g}",
            f"display_every={display_every} rgb={'ON' if rgb_enabled else 'off'} every={rgb_every} render={rgb_render_ema_ms:.1f}ms",
            f"control={['policy', 'keyboard', 'gamepad'][control_source]} controller={args.controller}",
            (
                "gamepad=disabled"
                if gamepad_sample is None
                else f"takeover={'ON' if gamepad_active else 'off'} connected={int(gamepad_sample.connected)} stale={int(gamepad_sample.stale)}"
            ),
            f"cost={total_cost:.1f} success_step={success_step}",
            f"goal={goal_distance_now:.2f}m clearance={goal_clearance:.2f}m",
            f"sampling={args.continuous_goal_region_mode} {args.continuous_goal_distance_min:g}-{args.continuous_goal_distance_max:g}m",
            f"blocked={layout_line.get('blocked', None)} p={args.continuous_goal_blocked_probability:.2f}",
            f"map_radius={viewer_radius:g}m cells={len(terrain_xy)}",
            "",
            "O student/map | I compact/details | H scan",
            "PgUp/PgDn map zoom | V RGB | Up/Down RGB stride",
            "F11 fullscreen | resize window | Space pause | Esc quit",
            previous_note,
        ]
        compact_info_lines = [
            f"episode={episode_idx} step={step} {'PAUSED' if paused else 'RUNNING'}",
            f"control={['policy', 'keyboard', 'gamepad'][control_source]} takeover={'ON' if gamepad_active else 'off'}",
            f"goal={goal_distance_now:.2f}m clearance={goal_clearance:.2f}m",
            f"cost={total_cost:.1f} success={success_step}",
            f"map={viewer_radius:g}m blocked={layout_line.get('blocked', None)}",
            "",
            "O student/map | I details | H scan",
            "PgUp/PgDn zoom | V RGB",
            "F11 fullscreen | resize | Space pause | Esc quit",
            previous_note,
        ]
        _draw_text(
            screen,
            font if ui_details_enabled else small,
            detailed_info_lines if ui_details_enabled else compact_info_lines,
            info_rect[0] + 10,
            info_rect[1] + 10,
            line_h=20 if ui_details_enabled else 18,
        )
        if student_view_enabled:
            legend_lines = (
                [
                    "STUDENT VIEW: live scan + relative goal only",
                    "crimson: human takeover | O: global map",
                ]
                if not ui_details_enabled
                else [
                    "STUDENT VIEW: body-frame live height scan only; forward is up",
                    "green arrow/yellow marker: relative goal direction projected to scan horizon",
                    "no privileged terrain geometry, trajectory, or cost history is shown",
                    "crimson robot/window: gamepad action is live",
                ]
            )
        else:
            legend_lines = (
                [
                    "colored: live scan | gray: global terrain | red x: cost",
                    "yellow edge marker: off-screen goal | crimson: takeover",
                ]
                if not ui_details_enabled
                else [
                    "round colored points: live terrain scan values",
                    "black outline: blocked terrain-scan ray",
                    "purple rings: reconstructed grid, if enabled",
                    "gray: global heightfield geometry in moving viewport | red x: cost",
                    "yellow edge marker: off-screen goal | crimson: human action",
                ]
            )
        _draw_text(
            screen,
            small,
            legend_lines,
            top_rect[0] + 12,
            top_rect[1] + top_rect[3] - (42 if not ui_details_enabled else 84),
            color=(20, 20, 20),
            line_h=16,
        )
        pygame.display.flip()
        clock.tick(float(args.fps))

    return 0


def _run_viser(args: argparse.Namespace) -> int:
    """Run the exact navigation environment with browser-side WebGL rendering."""
    import viser
    from mjlab.viewer.base import BaseViewer, VerbosityLevel, ViewerAction
    from mjlab.viewer.viser import ViserPlayViewer
    from unitree_viewer_pacing import LatestFramePacing
    from unitree_nav_control import project_navigation_action

    if args.online_train and args.controller != "policy":
        raise ValueError("Browser online training requires --controller policy")
    observation_only = bool(getattr(args, "observation_only", False))
    if observation_only and (args.navigation_observation_mode not in ("rgb_height", "php_depth")
                             or args.controller != "policy" or not (args.online_train or args.autopilot)):
        raise ValueError("Observation-only view requires an RGB+height or PHP policy and training/autopilot")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    env = make_env(args, num_envs=1, render=False)
    _register_interactive_cleanup("environment", env.close)
    initial_raw, _, _, _, obstacle_cells = _reset_until_feasible(args, env)
    training_session = None
    if args.online_train or args.autopilot or str(args.human_dataset_dir).strip():
        from unitree_viser_training import BrowserTrainingSession
        training_session = BrowserTrainingSession(args, env, initial_raw, obstacle_cells)
        _register_interactive_cleanup("browser training and recording", training_session.close)

    terrain_rows, terrain_cols = terrain_tile_shape(env)
    terrain_tile_count = terrain_rows * terrain_cols
    target_layout = None
    target_material_at = None
    target_sand_drag = float(getattr(args, "target_terrain_sand_linear_drag", 0.0))
    if bool(getattr(args, "target_terrain", False)):
        try:
            from unitree_target_terrain import _material_at, generate_target_terrain

            target_seed = int(args.target_terrain_seed)
            if target_seed < 0:
                target_seed = int(args.seed)
            target_layout = generate_target_terrain(
                target_seed,
                preset=str(args.target_terrain_preset),
                arena_size=float(args.target_terrain_arena_size),
            )
            target_material_at = _material_at
        except Exception:
            pass

    def robot_pose() -> tuple[np.ndarray, np.ndarray]:
        try:
            robot = env.env.unwrapped.scene["robot"]
            root_position = robot.data.root_link_pos_w[0].detach().cpu().numpy().copy()
            heading = float(robot.data.heading_w[0].detach().cpu().item())
        except Exception:
            xy, heading = _robot_xy_heading(env)
            root_position = np.asarray([xy[0], xy[1], 0.9], dtype=np.float64)
        forward = np.asarray([math.cos(heading), math.sin(heading), 0.0], dtype=np.float64)
        return root_position.astype(np.float64), forward

    def camera_pose(mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        root_position, forward = robot_pose()
        target = root_position.copy()
        target += float(args.viser_camera_lookahead) * forward
        target[2] += float(args.viser_camera_look_height)
        if mode == "bird":
            position = root_position + np.asarray(
                [0.0, 0.0, float(args.viser_bird_height)], dtype=np.float64
            )
            # Body-forward points to the top of the screen in bird's-eye mode.
            up = forward.copy()
            target = root_position.copy()
        else:
            # Apply the chase offset in the robot's planar local frame. Keeping
            # height world-aligned avoids camera roll/pitch as the humanoid walks.
            position = root_position - float(args.viser_chase_distance) * forward
            position[2] += float(args.viser_chase_height)
            up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        return position, target, up

    def terrain_status() -> str:
        if not bool(getattr(args, "target_terrain", False)):
            return str(args.task)
        try:
            if target_layout is None or target_material_at is None:
                raise RuntimeError("target layout metadata unavailable")
            root_position, _ = robot_pose()
            xy = root_position[:2]
            origin = env.env.unwrapped.scene.env_origins[0, :2].detach().cpu().numpy()
            local_xy = xy - origin
            material = target_material_at(target_layout, float(local_xy[0]), float(local_xy[1]))
            features = [
                brush.kind
                for brush in target_layout.geometry
                if np.linalg.norm(local_xy - np.asarray(brush.center)) <= float(brush.radius)
            ]
            feature = ",".join(sorted(set(features))) if features else "flat"
            return f"{args.target_terrain_preset} / {material} / {feature}"
        except Exception:
            return f"{args.target_terrain_preset} / material unavailable"

    controller = UnitreeGamepadController(
        mode=args.gamepad_mode,
        host=args.gamepad_host,
        port=int(args.gamepad_port),
        config_path=args.gamepad_config_path,
        device_index=int(args.gamepad_device_index),
        reconnect_seconds=float(args.gamepad_reconnect_seconds),
        stale_timeout_s=float(args.gamepad_stale_timeout_s),
        intervention_mode=str(args.gamepad_intervention_mode),
        intervention_threshold=float(args.gamepad_intervention_threshold),
        invert_lateral=bool(args.gamepad_invert_lateral),
    ) if args.human_input_device == "gamepad" else None
    if controller is not None:
        _register_interactive_cleanup("human controller", controller.close)

    class ViserGamepadPolicy:
        def __init__(self) -> None:
            self.last_status_at = 0.0
            self.previous_active = False
            self.previous_buttons: dict[str, bool] = {}
            self.latest_sample = None
            self.control_handler = None
            self.keyboard_action = np.zeros(3, dtype=np.float32)
            self.keyboard_active_until = float("-inf")
            self.last_control_source = "neutral"
            self.proposed_action = torch.zeros((1, 3), device=args.device)
            self.human_active = False

        def reset(self) -> None:
            self.previous_active = False
            self.keyboard_action.fill(0.0)
            self.keyboard_active_until = float("-inf")

        def pulse_keyboard(self, action: tuple[float, float, float], label: str) -> None:
            self.keyboard_action[:] = np.asarray(action, dtype=np.float32)
            self.keyboard_active_until = time.monotonic() + float(args.viser_keyboard_hold_s)
            self.last_control_source = f"keyboard {label}"

        def poll_controls(self) -> None:
            sample = controller.sample() if controller is not None else UnitreeGamepadSample(
                action=np.zeros(3, dtype=np.float32), intervening=False,
                gate_held=False, command_norm=0.0, connected=False, stale=True,
                state_age_s=float("inf"), state={}, transport={"connected": False},
            )
            self.latest_sample = sample
            buttons = (
                {name: bool(value) for name, value in dict(sample.state.get("buttons", {})).items()}
                if sample.connected and not sample.stale
                else {}
            )
            if self.control_handler is not None:
                for name in ("start", "a", "x", "y", "back"):
                    if buttons.get(name, False) and not self.previous_buttons.get(name, False):
                        self.control_handler(name)
            self.previous_buttons = buttons

        def __call__(self, _obs: torch.Tensor) -> torch.Tensor:
            if training_session is not None and training_session.pending is not None:
                pending = training_session.pending
                self.proposed_action = pending['proposed']
                self.human_active = pending['human']
                return pending['executed']
            sample = self.latest_sample
            if sample is None:
                self.poll_controls()
                sample = self.latest_sample
            assert sample is not None
            active = bool(sample.intervening and sample.connected and not sample.stale)
            keyboard_active = time.monotonic() < self.keyboard_active_until
            self.proposed_action = (training_session.proposal() if training_session is not None
                                    else torch.zeros((1, 3), device=args.device))
            self.human_active = active or keyboard_active
            if active:
                action = torch.as_tensor(
                    sample.action,
                    dtype=torch.float32,
                    device=args.device,
                ).reshape(1, -1)
                self.last_control_source = "gamepad"
            elif keyboard_active:
                action = torch.as_tensor(
                    self.keyboard_action,
                    dtype=torch.float32,
                    device=args.device,
                ).reshape(1, -1)
            else:
                action = self.proposed_action if args.autopilot else torch.zeros_like(self.proposed_action)
                self.last_control_source = "policy" if args.autopilot else "neutral"
            action = project_navigation_action(action)
            if active != self.previous_active:
                event = "TAKEOVER START" if active else "TAKEOVER END"
                values = np.array2string(action[0].detach().cpu().numpy(), precision=3)
                print(f"[viser-gamepad] {event} executed={values}", flush=True)
                self.previous_active = active
            now = time.time()
            if bool(args.gamepad_debug_console) and now - self.last_status_at >= float(
                args.gamepad_status_interval_s
            ):
                transport = sample.transport
                print(
                    "[viser-gamepad] "
                    f"transport_connected={int(transport.get('connected', False))} "
                    f"receiving={int(sample.connected and not sample.stale)} "
                    f"stale={int(sample.stale)} age_s={sample.state_age_s:.3f} "
                    f"takeover={int(active)} command_norm={sample.command_norm:.3f}",
                    flush=True,
                )
                self.last_status_at = now
            return action

    gamepad_policy = ViserGamepadPolicy()

    server = None if observation_only else viser.ViserServer(
        host=str(args.viser_host),
        port=int(args.viser_port),
        label="Unitree target-terrain manual control",
    )
    print(
        f"[viser] Open http://localhost:{int(args.viser_port)} in the Windows browser",
        flush=True,
    )
    if not observation_only:
        initial_position, initial_target, initial_up = camera_pose(str(args.viser_camera_mode))
        server.initial_camera.position = tuple(initial_position)
        server.initial_camera.look_at = tuple(initial_target)
        server.initial_camera.up_direction = tuple(initial_up)

    viewer_base = BaseViewer if observation_only else ViserPlayViewer

    class UnitreeViserViewer(LatestFramePacing, viewer_base):
        """MJLab Viser viewer with navigation-specific camera, controls, and status."""

        def __init__(self) -> None:
            super().__init__(
                env,
                gamepad_policy,
                frame_rate=float(args.viser_fps),
                verbosity=(
                    VerbosityLevel.DEBUG if args.viser_profile_console else VerbosityLevel.INFO
                ),
                **({} if observation_only else {"viser_server": server}),
            )
            self.camera_mode = str(args.viser_camera_mode)
            self.init_pacing()
            self.render_stride = max(1, int(args.viser_render_stride))
            self.camera_stride = max(1, int(args.viser_camera_update_every))
            self.parity_frame = 0
            self.current_tile = 0
            self.scenario_index = 0
            self.last_fall = False
            self.last_goal_reached = False
            self.last_fall_at = float("-inf")
            self.last_goal_reached_at = float("-inf")
            self.note = "ready"
            self.stop_requested = False
            self._navigation_status_html = None
            self._goal_column = None
            self._policy_camera_handle = None
            self._policy_camera_capture = -1
            self.show_policy_preview = True
            if observation_only:
                from threading import Lock
                from unitree_observation_view import ObservationServer
                self._sim_lock = Lock()
                self.observation_server = ObservationServer(str(args.viser_host), int(args.viser_port),
                    lambda payload: self.request_action("CUSTOM", {"type": "observation_control", **payload}))
                self._observation_cache_version = -1
                self._observation_cache = None
            robot = env.env.unwrapped.scene["robot"]
            torso_ids, _ = robot.find_bodies("torso_link")
            self._drag_robot = robot
            self._drag_torso_id = int(torso_ids[0])

        def setup(self) -> None:
            if observation_only:
                self.sync_env_to_viewer()
                self.observation_server.start()
                return
            super().setup()
            import trimesh
            column = trimesh.creation.cylinder(radius=float(args.success_dist), height=80.0, sections=32)
            column.vertices[:, 2] += 40.0
            self._goal_column = server.scene.add_mesh_simple(
                "/navigation/goal_column", vertices=column.vertices, faces=column.faces,
                color=(30, 230, 75), opacity=0.18, side="double")
            if training_session is not None and training_session.vision is not None:
                with server.gui.add_folder("Policy camera input"):
                    self._policy_camera_handle = server.gui.add_image(
                        training_session.vision.preview_image(),
                        label=f"{args.navigation_observation_mode}: actual policy frame(s)")
            # MJLab creates the scene during setup and normally recenters it
            # around a tracked body. Our chase camera already uses absolute world
            # coordinates, so enabling both transforms produces a double offset.
            self._scene.camera_tracking_enabled = False
            self._scene.needs_update = True
            for handle in server.gui._gui_input_handle_from_uuid.values():
                if getattr(handle, "label", "") == "Track camera":
                    handle.value = False
                    break
            with server.gui.add_folder("Unitree manual inspection"):
                self._navigation_status_html = server.gui.add_html("")
                autopilot_control = server.gui.add_checkbox("Policy autopilot", initial_value=bool(args.autopilot))

                @autopilot_control.on_update
                def _(_) -> None:
                    args.autopilot = bool(autopilot_control.value)

                stop_control = server.gui.add_button("Stop and save")

                @stop_control.on_click
                def _(_) -> None:
                    self.request_action("CUSTOM", {"type": "stop"})
                controls = server.gui.add_button_group(
                    "Playback",
                    options=["Pause / Run", "Step", "Reset", "Next layout / seed"],
                )

                @controls.on_click
                def _(event) -> None:
                    value = event.target.value
                    if value == "Pause / Run":
                        self.request_toggle_pause()
                    elif value == "Step":
                        self.request_single_step()
                    elif value == "Reset":
                        self.request_reset()
                    else:
                        self.request_action("CUSTOM", {"type": "next_layout"})

                speed = server.gui.add_button_group("Simulation speed", options=["0.5x", "1x", "1.5x", "2x"])

                @speed.on_click
                def _(event) -> None:
                    self.request_action("CUSTOM", {"type": "speed", "value": float(event.target.value[:-1])})

                display_rate = server.gui.add_slider("Display FPS cap", min=5, max=30, step=5,
                                                     initial_value=min(30, max(5, round(self.display_fps / 5) * 5)))

                @display_rate.on_update
                def _(_) -> None:
                    self.request_action("CUSTOM", {"type": "display_fps", "value": display_rate.value})

                adaptive = server.gui.add_checkbox("Adaptive display frame skipping", initial_value=True)

                @adaptive.on_update
                def _(_) -> None:
                    self.request_action("CUSTOM", {"type": "adaptive_display", "value": adaptive.value})

                preview = server.gui.add_checkbox("Show policy camera preview", initial_value=True)

                @preview.on_update
                def _(_) -> None:
                    self.request_action("CUSTOM", {"type": "policy_preview", "value": preview.value})

                server.gui.add_markdown("Speed is a target, not a guarantee. Display frames may be skipped; "
                    "physics, 5 Hz decisions, policy-camera inputs and learning are never skipped. "
                    "At 2x, you have half as much wall-clock time to intervene.")

                camera = server.gui.add_dropdown(
                    "Camera",
                    options=["chase", "bird", "free"],
                    initial_value=self.camera_mode,
                )

                @camera.on_update
                def _(_) -> None:
                    self.camera_mode = str(camera.value)
                    self.note = f"camera: {self.camera_mode}"
                    self._apply_camera(force=True)

                recenter = server.gui.add_button("Recenter camera")

                @recenter.on_click
                def _(_) -> None:
                    self._apply_camera(force=True)

                cadence = server.gui.add_slider(
                    "Scene update cadence",
                    min=1,
                    max=12,
                    step=1,
                    initial_value=self.render_stride,
                )

                @cadence.on_update
                def _(_) -> None:
                    self.render_stride = max(1, int(cadence.value))
                    self.note = f"scene update every {self.render_stride} viewer frame(s)"

                server.gui.add_markdown(
                    "**Steering:** gamepad sticks, or W/S forward/back, A/D turn, "
                    "Q/E strafe. **Playback:** Start pause/run | A step | X reset | "
                    "Y next | Back stop"
                )
            keyboard_commands = (
                ("Drive forward", "W", (1.0, 0.0, 0.0), "forward"),
                ("Drive backward", "S", (-1.0, 0.0, 0.0), "backward"),
                ("Turn left", "A", (0.0, 0.0, 1.0), "turn left"),
                ("Turn right", "D", (0.0, 0.0, -1.0), "turn right"),
                ("Strafe left", "Q", (0.0, 1.0, 0.0), "strafe left"),
                ("Strafe right", "E", (0.0, -1.0, 0.0), "strafe right"),
            )
            for label, hotkey, action, note in keyboard_commands:
                command = server.gui.add_command(label, hotkey=hotkey)

                @command.on_trigger
                def _(_, action=action, note=note) -> None:
                    gamepad_policy.pulse_keyboard(action, note)
                    self.note = f"keyboard: {note}"
            self._apply_camera(force=True)
            self._update_navigation_status()

        def tick(self) -> bool:
            # Poll outside policy inference so Start/A/X/Y still work while paused.
            gamepad_policy.poll_controls()
            return super().tick()

        def is_running(self) -> bool:
            return not self.stop_requested and (observation_only or super().is_running())

        def sync_viewer_to_env(self) -> None:
            if not observation_only:
                super().sync_viewer_to_env()

        def close(self) -> None:
            if observation_only:
                self.observation_server.stop()
            else:
                super().close()

        def set_playback_speed(self, value) -> None:
            super().set_playback_speed(value)
            if training_session is not None:
                training_session.playback_events.append(dict(
                    wall_time_unix_s=time.time(), environment_ticks=training_session.environment_ticks,
                    navigation_decisions=training_session.steps, target_speed=self._time_multiplier))
            self.note = f"requested speed: {self._time_multiplier:g}x"
            print(f"[viewer-speed] target={self._time_multiplier:g}x; "
                  f"environment_ticks={self._step_count}; wall_time={time.time():.6f}", flush=True)

        def _handle_custom_action(self, action, payload) -> bool:
            if action == ViewerAction.CUSTOM and isinstance(payload, dict):
                kind = payload.get("type")
                if kind == "observation_control":
                    command = payload["action"]
                    if command == "pause":
                        self.toggle_pause()
                    elif command == "step":
                        self._single_step()
                    elif command == "reset":
                        self.reset_environment()
                    elif command == "next":
                        self._advance_layout()
                    elif command == "stop":
                        self.stop_requested = True
                    elif command == "speed":
                        self.set_playback_speed(payload["value"])
                    elif command == "release":
                        gamepad_policy.keyboard_active_until = float("-inf")
                    elif command == "keyboard":
                        actions = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 0, 1),
                                   "d": (0, 0, -1), "q": (0, 1, 0), "e": (0, -1, 0)}
                        gamepad_policy.pulse_keyboard(actions[payload["key"]], payload["key"])
                    return True
                if kind == "speed":
                    self.set_playback_speed(payload["value"])
                    return True
                if kind == "display_fps":
                    self.display_fps = float(payload["value"])
                    return True
                if kind == "adaptive_display":
                    self.adaptive_display = bool(payload["value"])
                    return True
                if kind == "policy_preview":
                    self.show_policy_preview = bool(payload["value"])
                    if self._policy_camera_handle is not None:
                        self._policy_camera_handle.visible = self.show_policy_preview
                    return True
                if kind == "next_layout":
                    self._advance_layout()
                    return True
                if kind == "stop":
                    self.stop_requested = True
                    return True
            return super()._handle_custom_action(action, payload)

        def reset_environment(self) -> None:
            with self._sim_lock:
                raw, _, _, _, cells = _reset_until_feasible(args, env)
                if training_session is not None:
                    training_session.cells = cells
                    training_session.reset(raw)
            gamepad_policy.reset()
            self._step_count = 0
            self._sim_budget = 0.0
            self._last_error = None
            self._last_tick_time = time.perf_counter()
            self.last_fall = False
            self.last_goal_reached = False
            self.note = "environment reset"
            if getattr(self, "_term_overlays", None):
                self._term_overlays.clear_histories()
            if getattr(self, "_scene", None):
                self._scene.needs_update = True
            self._apply_camera(force=True)

        def _advance_layout(self) -> None:
            if terrain_tile_count > 1:
                self.current_tile = (self.current_tile + 1) % terrain_tile_count
                unwrapped = env.env.unwrapped
                env_ids = torch.zeros(1, dtype=torch.long, device=unwrapped.device)
                set_terrain_tile_indices(
                    env,
                    env_ids,
                    torch.as_tensor([self.current_tile], device=unwrapped.device),
                )
                next_note = f"layout tile {self.current_tile}/{terrain_tile_count - 1}"
            else:
                # Direct-geometry target arenas are compiled into the MuJoCo model.
                # Reseeding changes starts/goals; changing terrain requires a rebuild.
                self.scenario_index += 1
                scenario_seed = int(args.seed) + self.scenario_index
                np.random.seed(scenario_seed)
                torch.manual_seed(scenario_seed)
                next_note = f"scenario seed {scenario_seed}; terrain geometry unchanged"
            self.reset_environment()
            self.note = next_note

        def _execute_step(self) -> bool:
            # Scene extraction and simulation share state; never run them together.
            with self._sim_lock:
                return self._execute_navigation_step()

        def _execute_navigation_step(self) -> bool:
            if self.stop_requested or (training_session is not None and training_session.limit_reached):
                self.stop_requested = True
                return False
            self._apply_material_drag()
            if training_session is not None:
                try:
                    with torch.no_grad():
                        action = gamepad_policy(training_session.obs)
                    source = 2 if gamepad_policy.last_control_source == "gamepad" else (1 if gamepad_policy.human_active else 0)
                    training_session.step(action, gamepad_policy.proposed_action,
                                          gamepad_policy.human_active, source, gamepad_policy.latest_sample)
                    self._step_count += 1
                    self._stats_steps += 1
                    if training_session.limit_reached:
                        print(f"[browser] Stop and save: {training_session.steps} navigation decisions "
                              f"({training_session.environment_ticks} environment ticks)", flush=True)
                        self.stop_requested = True
                    ok = True
                except Exception:
                    import traceback
                    self._last_error = traceback.format_exc()
                    print(self._last_error, flush=True)
                    self.pause()
                    return False
            else:
                ok = super()._execute_step()
            if not ok:
                return False
            manager = getattr(env.env.unwrapped, "termination_manager", None)
            try:
                if bool(manager.get_term("fell_over")[0].detach().cpu().item()):
                    self.last_fall = True
                    self.last_fall_at = time.monotonic()
            except Exception:
                pass
            try:
                if bool(manager.get_term("goal_reached")[0].detach().cpu().item()):
                    self.last_goal_reached = True
                    self.last_goal_reached_at = time.monotonic()
            except Exception:
                pass
            return True

        def _apply_material_drag(self) -> None:
            """Apply viscous drag only where the visible target material is sand."""
            robot = self._drag_robot
            velocity = robot.data.root_link_lin_vel_w
            forces = torch.zeros((1, 1, 3), dtype=velocity.dtype, device=velocity.device)
            if target_sand_drag > 0.0 and target_layout is not None and target_material_at is not None:
                position = robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()
                origin = env.env.unwrapped.scene.env_origins[0, :2].detach().cpu().numpy()
                local_xy = position - origin
                material = target_material_at(
                    target_layout, float(local_xy[0]), float(local_xy[1])
                )
                if material == "sand":
                    forces[0, 0, :2] = -target_sand_drag * velocity[0, :2]
            robot.write_external_wrench_to_sim(
                forces,
                torch.zeros_like(forces),
                body_ids=[self._drag_torso_id],
            )

        def sync_env_to_viewer(self) -> None:
            if observation_only:
                from unitree_observation_view import observation_payload
                if self._observation_cache_version != training_session.observation_frames.version:
                    self._observation_cache = observation_payload(training_session)
                    self._observation_cache_version = training_session.observation_frames.version
                self.observation_server.publish(dict(self._observation_cache,
                    paused=self._is_paused, human=gamepad_policy.human_active,
                    decisions=training_session.steps, limit=args.online_total_steps,
                    target_speed=self.target_realtime, actual_speed=self.actual_realtime,
                    tick_ms=self.step_seconds * 1000))
                return
            # Keep MJLab's scene-centering transform disabled even if its generic
            # Track camera control was toggled in the browser.
            self._scene.camera_tracking_enabled = False
            self.parity_frame += 1
            if self._policy_camera_handle is not None and self.show_policy_preview:
                vision = training_session.vision
                if self._policy_camera_capture != vision.captures:
                    self._policy_camera_handle.image = vision.preview_image()
                    self._policy_camera_capture = vision.captures
            if self.parity_frame % self.render_stride == 0:
                super().sync_env_to_viewer()
                goal = _command_goal_w(env)
                if self._goal_column is not None and goal is not None:
                    self._goal_column.position = (float(goal[0]), float(goal[1]), 0.0)
                if self.parity_frame % self.camera_stride == 0:
                    self._apply_camera()
            if self.parity_frame % 5 == 0:
                self._update_navigation_status()

        @staticmethod
        def _should_submit_scene_update(counter, paused, has_pending_updates):
            # Our scheduler already controls scene cadence; avoid MJLab's extra /2.
            return not paused or has_pending_updates

        def _apply_camera(self, *, force: bool = False) -> None:
            if observation_only:
                return
            if self.camera_mode == "free" and not force:
                return
            mode = "bird" if self.camera_mode == "bird" else "chase"
            position, target, up = camera_pose(mode)
            for client in server.get_clients().values():
                try:
                    with client.atomic():
                        client.camera.position = position
                        client.camera.look_at = target
                        if hasattr(client.camera, "up_direction"):
                            client.camera.up_direction = up
                        elif hasattr(client.camera, "up"):
                            client.camera.up = up
                except Exception as exc:
                    self.note = f"camera update failed: {exc}"

        def _update_status_display(self) -> None:
            super()._update_status_display()
            content = self._status_html.content.replace(
                "<strong>Steps:</strong>", "<strong>Environment ticks (20 Hz):</strong>")
            if training_session is not None:
                limit = args.online_total_steps if args.online_total_steps > 0 else "unlimited"
                heading = (f"<div style='padding:0.5em 1em;font-size:0.9em'>"
                           f"<strong>Navigation decisions ({args.navigation_policy_hz:g} Hz): "
                           f"{training_session.steps} / {limit}</strong><br/>"
                           f"{'HUMAN TAKEOVER' if gamepad_policy.human_active else 'Policy / idle'} | "
                           f"goals: {training_session.successes}</div>")
                content = heading + content
                mode = getattr(args, "navigation_observation_mode", "height_scan")
                content += f"<div style='padding:0 1em'>Navigation sensors: {mode}</div>"
            self._status_html.content = content

        def _update_navigation_status(self) -> None:
            if self._navigation_status_html is None:
                return
            try:
                goal = _command_goal_w(env)
                goal_text = "unavailable" if goal is None else f"({goal[0]:.2f}, {goal[1]:.2f})"
                goal_distance = float(
                    _current_goal_distance(env, 1, torch.device(args.device))[0]
                    .detach()
                    .cpu()
                    .item()
                )
                unwrapped = env.env.unwrapped
                episode_step = int(unwrapped.episode_length_buf[0].detach().cpu().item())
                horizon = int(getattr(unwrapped, "max_episode_length", args.max_steps))
                sample = gamepad_policy.latest_sample
                receiving = bool(sample and sample.connected and not sample.stale)
                takeover = gamepad_policy.human_active
                training_text = ""
                if training_session is not None:
                    updates = training_session.learner.updates if training_session.learner else 0
                    limit = args.online_total_steps if args.online_total_steps > 0 else "unlimited"
                    training_text = (f"<strong>Navigation decisions ({args.navigation_policy_hz:g} Hz): {training_session.steps} / {limit}</strong><br/>"
                                     f"Environment ticks (20 Hz): {training_session.environment_ticks} | updates: {updates}<br/>"
                                     f"human decisions: {training_session.interventions} | "
                                     f"goals: {training_session.successes} | cost: {training_session.cost_total:.1f}<br/>")
                status = self.get_status()
                performance = (f"Target: {status.target_realtime:.2f}x | achieved: {status.actual_realtime:.2f}x"
                    f" | display: {status.smoothed_fps:.1f} Hz<br/>"
                    f"Tick incl. learning: {self.step_seconds * 1000:.1f} ms | "
                    f"display sync: {self.display_seconds * 1000:.1f} ms | "
                    f"scene worker: {getattr(self, '_scene_update_last_ms', 0):.1f} ms<br/>")
                if training_session is not None and training_session.environment_ticks:
                    timing = training_session.timing_seconds
                    ticks = training_session.environment_ticks
                    performance += ("Mean per tick: " + " | ".join(
                        f"{key}: {value * 1000 / ticks:.1f} ms" for key, value in timing.items()) + "<br/>")
                fall_recent = self.last_fall and time.monotonic() - self.last_fall_at < 2.0
                goal_recent = (
                    self.last_goal_reached
                    and time.monotonic() - self.last_goal_reached_at < 2.0
                )
                self._navigation_status_html.content = f"""
                <div style="border:3px solid {'crimson' if takeover else 'transparent'};font-size:0.84em;line-height:1.35;padding:0 0.5em 0.25em 0.5em">
                  {training_text}
                  {performance}
                  <strong>{'PAUSED' if status.paused else 'RUNNING'}</strong>
                  &nbsp; {status.speed_label} &nbsp; scene / {self.render_stride}<br/>
                  <strong>Goal:</strong> {goal_text}, {goal_distance:.2f} m<br/>
                  <strong>Episode ticks (20 Hz):</strong> {episode_step} / {horizon}<br/>
                  <strong>Fall:</strong> {'YES' if fall_recent else 'no'}
                  &nbsp; <strong>Goal reached:</strong> {'YES' if goal_recent else 'no'}<br/>
                  <strong>Terrain:</strong> {terrain_status()}<br/>
                  <strong>Sand drag:</strong> {target_sand_drag:.0f} N/(m/s)<br/>
                  <strong>Gamepad:</strong> {'TAKEOVER' if takeover else ('ready' if receiving else 'not receiving')}<br/>
                  <strong>Control:</strong> {gamepad_policy.last_control_source}<br/>
                  <span style="color:#777">{self.note}</span>
                </div>
                """
            except Exception as exc:
                self._navigation_status_html.content = f"Status unavailable: {exc}"

    viewer = UnitreeViserViewer()
    gamepad_policy.control_handler = lambda button: {
        "start": viewer.request_toggle_pause,
        "a": viewer.request_single_step,
        "x": viewer.request_reset,
        "y": lambda: viewer.request_action("CUSTOM", {"type": "next_layout"}),
        "back": lambda: viewer.request_action("CUSTOM", {"type": "stop"}),
    }[button]()
    if bool(args.start_paused):
        viewer.pause()
    if server is not None:
        _register_interactive_cleanup("viser server", server.stop)
    viewer.run(num_steps=(int(args.viser_max_steps) if int(args.viser_max_steps) > 0 else None))
    return 0


def run(args: argparse.Namespace) -> int:
    from unitree_nav_camera_features import camera_enabled
    if camera_enabled(args) and args.viewer_backend != "viser":
        raise ValueError("Camera navigation currently uses --viewer-backend viser; pygame is height-only")
    global _ACTIVE_INTERACTIVE_CLEANUPS
    _ACTIVE_INTERACTIVE_CLEANUPS = []
    try:
        if args.viewer_backend == "viser":
            return _run_viser(args)
        return _run_interactive(args)
    finally:
        cleanup_errors: list[str] = []
        for name, callback in reversed(_ACTIVE_INTERACTIVE_CLEANUPS):
            try:
                callback()
            except Exception as exc:
                cleanup_errors.append(f"{name}: {exc}")
        _ACTIVE_INTERACTIVE_CLEANUPS = []
        if cleanup_errors:
            print(
                "[interactive-cleanup] " + " | ".join(cleanup_errors),
                file=sys.stderr,
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--viewer-backend", choices=["pygame", "viser"], default="pygame")
    p.add_argument("--observation-only", action=argparse.BooleanOptionalAction, default=False,
                   help="Camera policy inputs and relative goal only, without a privileged 3D scene")
    p.add_argument("--viser-host", default="0.0.0.0")
    p.add_argument("--viser-port", type=int, default=8080)
    p.add_argument("--viser-fps", type=float, default=30.0)
    p.add_argument("--viser-profile-console", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--viser-max-steps", type=int, default=0)
    p.add_argument("--viser-camera-mode", choices=["chase", "bird", "free"], default="chase")
    p.add_argument("--viser-chase-distance", type=float, default=4.0)
    p.add_argument("--viser-chase-height", type=float, default=3.0)
    p.add_argument("--viser-camera-lookahead", type=float, default=0.0)
    p.add_argument("--viser-camera-look-height", type=float, default=0.85)
    p.add_argument("--viser-bird-height", type=float, default=10.0)
    p.add_argument("--viser-render-stride", type=int, default=1)
    p.add_argument("--viser-camera-update-every", type=int, default=1)
    p.add_argument("--viser-keyboard-hold-s", type=float, default=0.35)
    p.add_argument("--controller", choices=["direct_goal", "scan_teacher", "geom_scan_teacher", "policy"], default="scan_teacher")
    p.add_argument("--model-path", default="")
    p.add_argument("--task", default="Unitree-G1-Nav-Obstacles-Safe-Collision")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layout-seed-skip-attempts", type=int, default=50)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--episode-length-s", type=float, default=16.0)
    p.add_argument("--max-steps", type=int, default=320)
    p.add_argument(
        "--navigation-episode-mode",
        choices=["episodic", "continuous_goals"],
        default="episodic",
    )
    p.add_argument(
        "--runtime-navigation-episode-mode",
        choices=["episodic", "continuous_goals"],
        default=None,
        help="Runtime override applied after checkpoint environment reconstruction.",
    )
    p.add_argument("--continuous-environment-horizon-s", type=float, default=3600.0)
    p.add_argument("--continuous-goal-distance-min", type=float, default=0.0)
    p.add_argument("--continuous-goal-distance-max", type=float, default=0.0)
    p.add_argument(
        "--continuous-goal-region-mode",
        choices=["assigned_tile", "terrain_bank"],
        default="assigned_tile",
)
    p.add_argument("--continuous-goal-resample-attempts", type=int, default=256)
    p.add_argument("--continuous-goal-boundary-margin", type=float, default=0.5)
    p.add_argument("--continuous-goal-require-blocked-corridor", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--continuous-goal-blocked-probability", type=float, default=1.0)
    p.add_argument("--success-dist", type=float, default=0.5)
    p.add_argument("--terminate-on-goal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--height-scan-resolution", type=float, default=0.5)
    p.add_argument("--height-scan-pattern", choices=["grid", "forward_frustum"], default="grid")
    p.add_argument("--height-scan-frustum-near", type=float, default=0.25)
    p.add_argument("--height-scan-frustum-far", type=float, default=4.0)
    p.add_argument("--height-scan-frustum-fov-deg", type=float, default=70.0)
    p.add_argument("--height-scan-frustum-side", type=int, default=17)
    p.add_argument("--height-scan-forward-size", type=float, default=0.0)
    p.add_argument("--height-scan-lateral-size", type=float, default=0.0)
    p.add_argument("--scan-history", type=int, default=1)
    p.add_argument("--scan-history-stride", type=int, default=1)
    p.add_argument("--action-history", type=int, default=0)
    p.add_argument("--mask-height-scan", action="store_true")
    p.add_argument("--mask-proprioception", action="store_true")
    p.add_argument("--mask-goal-heading", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--goal-encoding", choices=["cartesian", "distance_bearing"], default="cartesian")
    p.add_argument("--goal-distance-scale", type=float, default=14.0)
    p.add_argument("--velocity-scale", type=float, default=1.0)
    p.add_argument("--checkpoint-env-config", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--force-obstacles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restore checkpoint architecture/settings but force obstacle navigation for collection/evaluation.",
    )
    p.add_argument(
        "--force-obstacle-profile",
        choices=["strict_blocked_v1", "checkpoint"],
        default="strict_blocked_v1",
        help="Environment profile used after restoring an obstacle-free checkpoint.",
    )
    p.add_argument("--policy-action-smoothing", type=float, default=0.0)
    p.add_argument("--navigation-policy-hz", type=float, default=5.0)
    p.add_argument("--goal-distance-min", type=float, default=0.0)
    p.add_argument("--goal-distance-max", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--policy-encoder", choices=["mlp", "scan_cnn"], default="mlp")
    p.add_argument("--use-layer-norm", action="store_true")
    p.add_argument("--low-level-policy-path", default=str(DEFAULT_LOW_LEVEL))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--sim-fps", type=float, default=0.0, help="Simulation step cap. Use 0 or negative for unlimited stepping.")
    p.add_argument("--display-every", type=int, default=1, help="Draw the UI once per N simulation steps.")
    p.add_argument("--window-width", type=int, default=1500)
    p.add_argument("--window-height", type=int, default=900)
    p.add_argument("--viewer-map-radius", type=float, default=7.0)
    p.add_argument("--compact-ui", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--show-rgb", action="store_true")
    p.add_argument("--rgb-every", type=int, default=5)
    p.add_argument(
        "--rgb-debug-visualizers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include MJLab debug geometry in RGB frames; substantially slower.",
    )
    p.add_argument(
        "--rgb-smooth-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CPU smooth scaling for the RGB panel instead of fast nearest-neighbor scaling.",
    )
    p.add_argument("--rgb-profile-console", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--rgb-profile-interval", type=int, default=10)
    p.add_argument("--video-width", type=int, default=320)
    p.add_argument("--video-height", type=int, default=240)
    p.add_argument("--video-camera-distance", type=float, default=7.0)
    p.add_argument("--video-camera-elevation", type=float, default=-65.0)
    p.add_argument("--video-camera-azimuth", type=float, default=90.0)
    p.add_argument(
        "--video-camera-follow-heading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rotate the tracking camera with the robot heading.",
    )
    p.add_argument(
        "--video-camera-heading-offset",
        type=float,
        default=0.0,
        help="Azimuth offset in degrees for heading-following camera mode.",
    )
    p.add_argument("--video-max-extra-envs", type=int, default=0)
    p.add_argument("--video-enable-shadows", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--video-enable-reflections", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--show-scan-samples", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--show-reconstructed-scan", action="store_true")
    p.add_argument("--student-view", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--start-paused", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-reset", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--autopilot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--manual-speed", type=float, default=0.65)
    p.add_argument("--manual-strafe", type=float, default=0.45)
    p.add_argument("--manual-yaw", type=float, default=0.9)
    p.add_argument("--human-input-device", choices=["none", "gamepad"], default="none")
    p.add_argument("--gamepad-mode", choices=["local", "connect"], default="local")
    p.add_argument("--gamepad-host", default="127.0.0.1")
    p.add_argument("--gamepad-port", type=int, default=DEFAULT_UNITREE_GAMEPAD_PORT)
    p.add_argument("--gamepad-config-path", default=str(DEFAULT_UNITREE_GAMEPAD_CONFIG_PATH))
    p.add_argument("--gamepad-device-index", type=int, default=0)
    p.add_argument("--gamepad-reconnect-seconds", type=float, default=2.0)
    p.add_argument("--gamepad-stale-timeout-s", type=float, default=0.25)
    p.add_argument("--gamepad-intervention-mode", choices=["stick", "button"], default="stick")
    p.add_argument("--gamepad-intervention-threshold", type=float, default=0.05)
    p.add_argument("--gamepad-invert-lateral", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamepad-debug-console", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamepad-status-interval-s", type=float, default=1.0)
    p.add_argument("--human-dataset-dir", default="")
    p.add_argument("--human-dataset-chunk-size", type=int, default=1024)
    p.add_argument("--online-train", action="store_true")
    p.add_argument(
        "--online-objective-mode",
        choices=["pref_bc_rl", "pref_only", "pref_rl", "bc_rl", "bc_only", "rl_only"],
        default="pref_bc_rl",
    )
    p.add_argument("--online-output-dir", default=str(ROOT / "models" / "unitree_mjlab_nav_human"))
    p.add_argument("--online-run-name", default=f"unitree_human_{time.strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--online-session-receipt", default="", help="Write verified final checkpoint/decision receipt on clean close.")
    p.add_argument("--online-restore-full-state", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--online-replay-capacity", type=int, default=100000)
    p.add_argument("--online-learning-starts", type=int, default=500)
    p.add_argument("--online-batch-size", type=int, default=256)
    p.add_argument("--online-updates-per-step", type=int, default=1)
    p.add_argument("--online-n-step", type=int, default=5)
    p.add_argument("--online-policy-frequency", type=int, default=2)
    p.add_argument("--online-gamma", type=float, default=0.99)
    p.add_argument("--online-tau", type=float, default=0.005)
    p.add_argument("--online-lr-actor", type=float, default=3e-4)
    p.add_argument("--online-lr-critic", type=float, default=3e-4)
    from unitree_nav_online_human import add_online_learning_args
    add_online_learning_args(p)
    p.add_argument("--online-max-grad-norm", type=float, default=10.0)
    p.add_argument("--online-project-actor-actions", action=argparse.BooleanOptionalAction, default=True,
                   help="Use executed unit-ball commands in actor/target Q and BC updates.")
    p.add_argument("--online-dense-progress-scale", type=float, default=1.0)
    p.add_argument("--online-success-bonus", type=float, default=20.0)
    p.add_argument("--online-failure-penalty", type=float, default=-20.0)
    p.add_argument("--online-pref-rank-weight", type=float, default=1.0)
    p.add_argument("--online-pref-rank-margin", type=float, default=0.05)
    p.add_argument("--online-pref-lambda-init", type=float, default=1.0)
    p.add_argument("--online-pref-lambda-lr", type=float, default=0.0,
                   help="Dual update rate; 0 fixes lambda at --online-pref-lambda-init, including warm starts.")
    p.add_argument("--online-pref-lambda-max", type=float, default=10.0)
    p.add_argument("--online-pref-action-delta-min", type=float, default=0.05)
    p.add_argument("--online-pref-proxy-value-bound", type=float, default=1.0)
    p.add_argument("--online-pref-stopgrad-positive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--online-actor-bc-weight", type=float, default=0.2)
    p.add_argument("--online-log-interval", type=int, default=100)
    p.add_argument("--online-checkpoint-interval", type=int, default=1000)
    p.add_argument("--online-total-steps", type=int, default=0, help="Stop after this many online transitions; 0 waits for manual stop.")
    p.add_argument("--online-wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--online-wandb-project", default="thesis-unitree-nav-human")
    p.add_argument("--online-wandb-group", default="")

    p.add_argument("--teacher-scan-block-threshold", type=float, default=0.12)
    p.add_argument("--teacher-scan-block-delta", type=float, default=0.025)
    p.add_argument("--teacher-scan-planner", choices=["heuristic", "astar"], default="heuristic")
    p.add_argument("--teacher-scan-astar-clearance", type=float, default=0.6)
    p.add_argument("--teacher-scan-astar-cell-padding", type=float, default=0.35)
    p.add_argument("--teacher-scan-astar-resolution", type=float, default=0.25)
    p.add_argument("--teacher-scan-astar-waypoint-index", type=int, default=3)
    p.add_argument("--teacher-scan-astar-side-penalty", type=float, default=8.0)
    p.add_argument("--teacher-scan-astar-commit-steps", type=int, default=30)
    p.add_argument("--teacher-sector-half-width", type=float, default=0.35)
    p.add_argument("--teacher-align-angle", type=float, default=0.55)
    p.add_argument("--teacher-max-vx", type=float, default=0.95)
    p.add_argument("--teacher-max-vy", type=float, default=0.45)
    p.add_argument("--teacher-yaw-gain", type=float, default=1.2)
    p.add_argument("--teacher-clearance-weight", type=float, default=5.5)
    p.add_argument("--teacher-clearance-power", type=float, default=2.0)
    p.add_argument("--teacher-speed-clearance-scale", type=float, default=0.0)
    p.add_argument("--teacher-num-sectors", type=int, default=13)
    p.add_argument("--teacher-min-forward-scale", type=float, default=0.12)
    p.add_argument("--teacher-escape-risk-threshold", type=float, default=0.0)
    p.add_argument("--teacher-escape-forward-scale", type=float, default=0.0)
    p.add_argument("--teacher-escape-lateral-scale", type=float, default=1.0)
    p.add_argument("--teacher-escape-radius", type=float, default=1.0)
    p.add_argument("--teacher-escape-all-directions", action="store_true")
    p.add_argument("--teacher-bypass-angle", type=float, default=1.15)
    p.add_argument("--teacher-goal-stop-dist", type=float, default=0.55)
    p.add_argument("--teacher-wall-follow-steps", type=int, default=90)
    p.add_argument("--teacher-wall-follow-angle", type=float, default=1.15)
    p.add_argument("--teacher-wall-follow-clear-risk", type=float, default=0.08)
    p.add_argument("--teacher-rollout-horizon", type=float, default=2.4)
    p.add_argument("--teacher-rollout-clearance", type=float, default=1.0)
    p.add_argument("--teacher-rollout-samples", type=int, default=8)
    p.add_argument("--teacher-rollout-clearance-weight", type=float, default=24.0)
    p.add_argument("--teacher-rollout-forward-bias", type=float, default=0.05)
    p.add_argument("--teacher-emergency-radius", type=float, default=0.95)
    p.add_argument("--teacher-emergency-hard-radius", type=float, default=0.9)
    p.add_argument("--teacher-emergency-speed-scale", type=float, default=0.75)
    p.add_argument("--teacher-emergency-repulsion-weight", type=float, default=0.8)
    p.add_argument("--teacher-emergency-tangent-weight", type=float, default=1.2)
    p.add_argument("--teacher-emergency-goal-weight", type=float, default=0.4)
    p.add_argument("--teacher-geom-planner", choices=["astar", "sector"], default="astar")
    p.add_argument("--teacher-geom-scan-margin", type=float, default=0.15)
    p.add_argument("--teacher-geom-back-margin", type=float, default=0.35)
    p.add_argument("--teacher-geom-max-forward", type=float, default=1.75)
    p.add_argument("--teacher-geom-max-lateral", type=float, default=1.75)
    p.add_argument("--teacher-geom-lookahead", type=float, default=1.6)
    p.add_argument("--teacher-geom-clearance", type=float, default=0.62)
    p.add_argument("--teacher-geom-grid-resolution", type=float, default=0.18)
    p.add_argument("--teacher-geom-waypoint-index", type=int, default=3)
    p.add_argument("--teacher-geom-side-penalty", type=float, default=4.0)
    p.add_argument("--teacher-geom-side-frame", choices=["body", "goal"], default="body")
    p.add_argument("--teacher-geom-disengage-clear-steps", type=int, default=12)
    p.add_argument("--teacher-geom-command-smoothing", type=float, default=0.0)
    p.add_argument("--teacher-geom-memory-radius", type=float, default=0.0)
    p.add_argument("--teacher-geom-waypoint-commit-distance", type=float, default=0.0)
    p.add_argument("--teacher-geom-waypoint-reach-dist", type=float, default=0.25)
    p.add_argument("--teacher-geom-stall-window", type=int, default=0)
    p.add_argument("--teacher-geom-stall-progress-epsilon", type=float, default=0.08)
    p.add_argument("--teacher-geom-stall-recovery-steps", type=int, default=60)
    p.add_argument("--teacher-geom-stall-recovery-angle", type=float, default=0.9)
    p.add_argument("--teacher-geom-stall-flip-side", action="store_true")
    p.add_argument("--teacher-geom-obstacle-speed-radius", type=float, default=0.0)
    p.add_argument("--teacher-geom-near-obstacle-vx-scale", type=float, default=0.7)
    p.add_argument("--teacher-geom-max-angle", type=float, default=1.5707963267948966)
    p.add_argument("--teacher-geom-candidate-count", type=int, default=31)
    p.add_argument("--teacher-geom-soft-width", type=float, default=0.9)
    p.add_argument("--teacher-geom-hard-penalty", type=float, default=35.0)
    p.add_argument("--teacher-geom-risk-weight", type=float, default=9.0)
    p.add_argument("--teacher-geom-angle-weight", type=float, default=0.15)
    p.add_argument("--teacher-geom-forward-bias", type=float, default=0.2)
    p.add_argument("--teacher-geom-emergency-radius", type=float, default=0.8)
    p.add_argument("--teacher-geom-emergency-repulsion-weight", type=float, default=1.2)
    p.add_argument("--teacher-geom-emergency-tangent-weight", type=float, default=1.4)
    p.add_argument("--teacher-geom-emergency-goal-weight", type=float, default=0.25)

    p.add_argument("--min-goal-obstacle-clearance", type=float, default=1.0)
    p.add_argument("--goal-clearance-resample-attempts", type=int, default=100)
    p.add_argument("--min-start-obstacle-clearance", type=float, default=0.75)
    p.add_argument("--start-position-range", type=float, default=0.0)
    p.add_argument("--start-clearance-resample-attempts", type=int, default=100)
    p.add_argument("--require-blocked-corridor", action="store_true")
    p.add_argument("--blocked-corridor-radius", type=float, default=0.55)
    p.add_argument("--blocked-corridor-ignore-end-radius", type=float, default=0.75)
    p.add_argument("--blocked-corridor-min-cells", type=int, default=1)
    p.add_argument("--blocked-corridor-resample-attempts", type=int, default=200)
    p.add_argument("--blocked-goal-max-distance", type=float, default=0.0)
    p.add_argument(
        "--blocked-goal-distance-sampling",
        choices=["nearest", "uniform", "farthest"],
        default="nearest",
    )
    p.add_argument("--blocked-goal-placement-mode", choices=["obstacle_multiplier", "distance_grid"], default="obstacle_multiplier")
    p.add_argument("--blocked-goal-distance-multiplier-min", type=float, default=1.0)
    p.add_argument("--blocked-goal-distance-multiplier-max", type=float, default=2.0)
    p.add_argument("--blocked-goal-candidate-attempts", type=int, default=64)
    p.add_argument("--debug-obstacle-width-min", type=float, default=1.0)
    p.add_argument("--debug-obstacle-width-max", type=float, default=1.4)
    p.add_argument("--strict-min-size-obstacles", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--debug-obstacle-height-min", type=float, default=1.0)
    p.add_argument("--debug-obstacle-height-max", type=float, default=1.0)
    p.add_argument("--debug-num-obstacles", type=int, default=6)
    p.add_argument("--debug-platform-width", type=float, default=2.0)
    p.add_argument("--debug-obstacle-border-width", type=float, default=0.0)
    p.add_argument("--debug-terrain-rows", type=int, default=0)
    p.add_argument("--debug-terrain-cols", type=int, default=0)
    p.add_argument("--debug-goal-through-obstacle", action="store_true")
    p.add_argument("--goal-through-obstacle-prob", type=float, default=0.0)
    p.add_argument("--debug-goal-distance", type=float, default=3.2)
    p.add_argument("--debug-goal-obstacle-min-dist", type=float, default=0.8)
    p.add_argument("--debug-goal-obstacle-max-dist", type=float, default=2.2)
    from unitree_target_navigation import add_target_terrain_args

    add_target_terrain_args(p)
    from unitree_nav_camera_features import add_navigation_camera_args
    add_navigation_camera_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from unitree_nav_checkpoint import apply_unitree_checkpoint_config

    runtime_continuous_overrides = {
        key: getattr(args, key)
        for key in (
            "continuous_environment_horizon_s",
            "continuous_goal_distance_min",
            "continuous_goal_distance_max",
            "continuous_goal_region_mode",
            "continuous_goal_resample_attempts",
            "continuous_goal_boundary_margin",
            "continuous_goal_require_blocked_corridor",
            "continuous_goal_blocked_probability",
            "min_goal_obstacle_clearance",
        )
    }
    apply_unitree_checkpoint_config(args)
    if args.runtime_navigation_episode_mode is not None:
        args.navigation_episode_mode = args.runtime_navigation_episode_mode
        for key, value in runtime_continuous_overrides.items():
            setattr(args, key, value)
    if bool(args.force_obstacles):
        # Goal-only pretrains carry disable_obstacles=True. Keep their policy
        # shape but collect takeover data on the actual obstacle benchmark.
        if args.force_obstacle_profile == "strict_blocked_v1":
            _apply_strict_blocked_obstacle_profile(args)
        else:
            args.disable_obstacles = False
            if int(args.debug_num_obstacles) <= 0:
                args.debug_num_obstacles = 6
        print(
            "[human-collection] forced obstacle environment after checkpoint restore "
            f"(profile={args.force_obstacle_profile} obstacles={args.debug_num_obstacles} "
            f"goal_clearance={args.min_goal_obstacle_clearance:.2f} "
            f"blocked={int(args.require_blocked_corridor)})",
            flush=True,
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

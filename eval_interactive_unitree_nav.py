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
from plot_unitree_nav_rollout import (  # noqa: E402
    _active_terrain_heightfield,
    _command_goal_w,
    _robot_xy_heading,
)
from train_unitree_nav_thesis import ScanTeacherState, _extract_actor_obs, _extract_cost  # noqa: E402
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


def _terrain_obstacles(terrain_debug) -> np.ndarray:
    if terrain_debug is None:
        return np.zeros((0, 2), dtype=np.float32)
    return terrain_debug[2]


def _blocked_scan_mask(obs: torch.Tensor, threshold: float, delta: float) -> np.ndarray:
    scan = obs[0, 9:58].detach().cpu().numpy().astype(np.float64)
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
    values = obs[0, 9:58].detach().cpu().numpy().astype(np.float32)
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
    blocked = _blocked_scan_mask(obs, threshold, delta).reshape(7, 7)
    coords = np.linspace(-1.5, 1.5, 7)
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


def _draw_rgb_panel(pygame, screen, frame, rect) -> None:
    if frame is None:
        pygame.draw.rect(screen, (20, 20, 25), rect)
        return
    arr = np.asarray(frame)
    if arr.ndim != 3:
        return
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    surf = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
    surf = pygame.transform.smoothscale(surf, (rect[2], rect[3]))
    screen.blit(surf, (rect[0], rect[1]))


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
    action: np.ndarray,
    paused: bool,
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
    bounds = _world_bounds(pts_for_bounds, pad=1.0)
    project = _make_projector(rect, bounds)

    if terrain_obstacles.size:
        p = project(terrain_obstacles)
        for x, y in p.astype(int):
            pygame.draw.rect(screen, (95, 95, 95), (x - 2, y - 2, 4, 4))
    if scan_xy.size:
        vmin = float(np.nanpercentile(scan_values, 5.0)) if scan_values.size else 0.0
        vmax = float(np.nanpercentile(scan_values, 95.0)) if scan_values.size else 1.0
        if abs(vmax - vmin) < 1e-6:
            vmin = float(np.nanmin(scan_values)) - 0.05
            vmax = float(np.nanmax(scan_values)) + 0.05
        p = project(scan_xy)
        for idx, (x, y) in enumerate(p.astype(int)):
            color = _scan_value_color(float(scan_values[idx]), vmin, vmax)
            radius = 7 if bool(scan_blocked[idx]) else 5
            pygame.draw.circle(screen, color, (x, y), radius)
            pygame.draw.circle(screen, (245, 245, 245), (x, y), radius, width=1)
            if bool(scan_blocked[idx]):
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
    if goal_xy is not None:
        gp = project(goal_xy.reshape(1, 2))[0].astype(int)
        pygame.draw.circle(screen, (50, 220, 60), tuple(gp), 18, width=3)
        pygame.draw.circle(screen, (240, 220, 20), tuple(gp), 7)
    rp = project(robot_xy.reshape(1, 2))[0].astype(int)
    pygame.draw.circle(screen, (40, 190, 240), tuple(rp), 9)
    forward = np.array([math.cos(heading), math.sin(heading)])
    hp = project((robot_xy + 0.55 * forward).reshape(1, 2))[0].astype(int)
    pygame.draw.line(screen, (0, 0, 0), tuple(rp), tuple(hp), 3)
    if np.linalg.norm(action[:2]) > 1e-6:
        c, s = math.cos(heading), math.sin(heading)
        local = action[:2]
        world = np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
        ap = project((robot_xy + 0.7 * world).reshape(1, 2))[0].astype(int)
        pygame.draw.line(screen, (245, 140, 0), tuple(rp), tuple(ap), 4)
    if paused:
        overlay = pygame.Surface((rect[2], 34), pygame.SRCALPHA)
        overlay.fill((200, 80, 20, 165))
        screen.blit(overlay, (rect[0], rect[1]))


def run(args: argparse.Namespace) -> int:
    import pygame

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    pygame.init()
    pygame.display.set_caption("Unitree Navigation Interactive Debugger")
    screen = pygame.display.set_mode((int(args.window_width), int(args.window_height)))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    small = pygame.font.Font(None, 18)

    env = make_env(args, num_envs=1, render=bool(args.show_rgb))
    obs_raw, _, _, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
    terrain_debug = _active_terrain_heightfield(env)
    obs = _extract_actor_obs(obs_raw).to(args.device, dtype=torch.float32)
    policy_actor = (
        _load_policy_actor(args, obs_dim=int(obs.shape[1]), act_dim=int(env.action_space.shape[-1]))
        if args.controller == "policy"
        else None
    )
    teacher_state = ScanTeacherState(1, torch.device(args.device)) if args.controller == "scan_teacher" else None
    if args.controller == "geom_scan_teacher":
        teacher_state = GeomTeacherState.create(1)

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
    done_waiting = False
    last_step_time = 0.0
    display_every = max(1, int(args.display_every))
    rgb_every = max(1, int(args.rgb_every))

    def reset_episode(note: str, *, advance_tile: bool = False) -> None:
        nonlocal env, obs, obs_raw, terrain_debug, layout_stats, obstacle_cells, teacher_state, episode_idx, current_tile
        nonlocal trajectory, cost_points, total_cost, success_step, step, previous_note, frame, done_waiting, last_step_time
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
        terrain_debug = _active_terrain_heightfield(env)
        obs = _extract_actor_obs(obs_raw).to(args.device, dtype=torch.float32)
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
        previous_note = f"{note} | seed={args.seed} tile={current_tile}/{max(0, terrain_tile_count - 1)}"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                down = event.type == pygame.KEYDOWN
                if event.key == pygame.K_ESCAPE and down:
                    running = False
                elif event.key == pygame.K_SPACE and down:
                    paused = not paused
                elif event.key == pygame.K_n and down:
                    single_step = True
                elif event.key == pygame.K_t and down:
                    autopilot = not autopilot
                elif event.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS) and down:
                    display_every = min(128, display_every * 2)
                    previous_note = f"display every {display_every} simulation steps"
                elif event.key in (pygame.K_DOWN, pygame.K_MINUS) and down:
                    display_every = max(1, display_every // 2)
                    previous_note = f"display every {display_every} simulation steps"
                elif event.key == pygame.K_RIGHTBRACKET and down:
                    rgb_every = min(128, rgb_every * 2)
                    previous_note = f"RGB render every {rgb_every} simulation steps"
                elif event.key == pygame.K_LEFTBRACKET and down:
                    rgb_every = max(1, rgb_every // 2)
                    previous_note = f"RGB render every {rgb_every} simulation steps"
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
        scan_xy, scan_values, scan_blocked = _actual_scan_samples_world(
            env,
            obs,
            args.teacher_scan_block_threshold,
            args.teacher_scan_block_delta,
        )
        reconstructed_scan_xy = (
            _reconstructed_scan_points_world(
                obs,
                robot_xy,
                heading,
                args.teacher_scan_block_threshold,
                args.teacher_scan_block_delta,
            )
            if bool(args.show_reconstructed_scan)
            else np.zeros((0, 2), dtype=np.float32)
        )
        terrain_xy = _terrain_obstacles(terrain_debug)
        manual_action, manual_active = _manual_action(
            keys,
            speed=float(args.manual_speed),
            yaw=float(args.manual_yaw),
            strafe=float(args.manual_strafe),
        )
        with torch.no_grad():
            auto_action = controller_action(obs, args, policy_actor, teacher_state, env=env, obstacle_cells=obstacle_cells)
        action = manual_action.to(args.device) if manual_active else (auto_action if autopilot else torch.zeros_like(auto_action))

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
            if success_step is None and live_dist <= args.success_dist:
                success_step = step
            obs_raw, reward, done, extras = env.step(action)
            obs = _extract_actor_obs(obs_raw).to(args.device, dtype=torch.float32)
            cost = float(_extract_cost(extras, 1, torch.device(args.device))[0].detach().cpu().item())
            total_cost += cost
            if cost > 0.0:
                cost_points.append(robot_xy.copy())
            if bool(args.show_rgb) and (step % rgb_every == 0):
                try:
                    frame = env.env.render()
                except Exception:
                    frame = None
            if success_step is None and float(torch.linalg.norm(obs[0, 6:8]).detach().cpu().item()) <= args.success_dist:
                success_step = step + 1
            step += 1
            single_step = False
            if bool(done.reshape(-1)[0].item()) or step >= int(args.max_steps):
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
        top_rect = (10, 10, int(args.window_width * (0.62 if args.show_rgb else 0.72)), int(args.window_height) - 20)
        rgb_rect = (
            top_rect[0] + top_rect[2] + 10,
            10,
            int(args.window_width) - (top_rect[0] + top_rect[2] + 20),
            int(args.window_height * 0.55),
        )
        info_rect = (rgb_rect[0], rgb_rect[1] + rgb_rect[3] + 10, rgb_rect[2], int(args.window_height) - rgb_rect[3] - 30)
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
            action=action[0].detach().cpu().numpy(),
            paused=paused,
        )
        if args.show_rgb:
            _draw_rgb_panel(pygame, screen, frame, rgb_rect)
        pygame.draw.rect(screen, (32, 32, 40), info_rect)
        pygame.draw.rect(screen, (70, 70, 80), info_rect, width=1)
        layout_line = layout_stats[0] if layout_stats else {}
        _draw_text(
            screen,
            font,
            [
                f"episode={episode_idx} step={step} paused={paused} autopilot={autopilot}",
                f"done_waiting={done_waiting} sim_fps={'unlimited' if float(args.sim_fps) <= 0.0 else f'{args.sim_fps:g}'} ui_fps={args.fps:g}",
                f"display_every={display_every} rgb_every={rgb_every}",
                f"manual={'ON' if manual_active else 'off'} controller={args.controller}",
                f"cost={total_cost:.1f} success_step={success_step}",
                f"goal_dist={float(torch.linalg.norm(obs[0, 6:8]).detach().cpu().item()):.2f}",
                f"blocked_corridor={layout_line.get('blocked', None)} cells={layout_line.get('blocked_cell_count', None)}",
                "",
                "W/S forward/back | A/D yaw | Q/E strafe",
                "T autopilot | R reset | Right next | Left note",
                "Up/Down playback speed | [/] RGB render stride",
                "Space pause | N single-step | Esc quit",
                previous_note,
            ],
            info_rect[0] + 10,
            info_rect[1] + 10,
        )
        _draw_text(
            screen,
            small,
            [
                "round colored points: scan samples; cyan/green=clear, yellow=mid, red=low",
                "black outline: threshold-blocked terrain_scan ray",
                "purple rings: reconstructed grid overlay, if enabled",
                "gray square clusters: inferred heightfield obstacle cells | red x: cost",
            ],
            top_rect[0] + 12,
            top_rect[1] + top_rect[3] - 84,
            color=(20, 20, 20),
            line_h=16,
        )
        pygame.display.flip()
        clock.tick(float(args.fps))

    env.close()
    pygame.quit()
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["direct_goal", "scan_teacher", "geom_scan_teacher", "policy"], default="scan_teacher")
    p.add_argument("--model-path", default="")
    p.add_argument("--task", default="Unitree-G1-Nav-Obstacles-Safe-Collision")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layout-seed-skip-attempts", type=int, default=50)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--episode-length-s", type=float, default=16.0)
    p.add_argument("--max-steps", type=int, default=320)
    p.add_argument("--success-dist", type=float, default=0.5)
    p.add_argument("--goal-distance-min", type=float, default=0.0)
    p.add_argument("--goal-distance-max", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--use-layer-norm", action="store_true")
    p.add_argument("--low-level-policy-path", default=str(DEFAULT_LOW_LEVEL))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--sim-fps", type=float, default=0.0, help="Simulation step cap. Use 0 or negative for unlimited stepping.")
    p.add_argument("--display-every", type=int, default=1, help="Draw the UI once per N simulation steps.")
    p.add_argument("--window-width", type=int, default=1500)
    p.add_argument("--window-height", type=int, default=900)
    p.add_argument("--show-rgb", action="store_true")
    p.add_argument("--rgb-every", type=int, default=5)
    p.add_argument("--show-reconstructed-scan", action="store_true")
    p.add_argument("--start-paused", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-reset", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--autopilot", action="store_true", default=True)
    p.add_argument("--manual-speed", type=float, default=0.65)
    p.add_argument("--manual-strafe", type=float, default=0.45)
    p.add_argument("--manual-yaw", type=float, default=0.9)

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
    p.add_argument("--start-clearance-resample-attempts", type=int, default=100)
    p.add_argument("--require-blocked-corridor", action="store_true")
    p.add_argument("--blocked-corridor-radius", type=float, default=0.55)
    p.add_argument("--blocked-corridor-ignore-end-radius", type=float, default=0.75)
    p.add_argument("--blocked-corridor-min-cells", type=int, default=1)
    p.add_argument("--blocked-corridor-resample-attempts", type=int, default=200)
    p.add_argument("--debug-obstacle-width-min", type=float, default=1.0)
    p.add_argument("--debug-obstacle-width-max", type=float, default=1.4)
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
    return p.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

"""Local-geometry scripted teacher for Unitree navigation diagnostics.

This teacher is intentionally privileged, but only within the approximate
footprint currently covered by the terrain scanner. It is a diagnostic bridge:
if this works while the pure height-scan teacher does not, the bottleneck is
the scan representation/thresholding rather than the local planning idea.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class GeomTeacherState:
    """Persistent bypass-side commitment and command hysteresis."""

    bypass_side: np.ndarray
    clear_steps: np.ndarray
    previous_action: np.ndarray
    observed_obstacles_w: list[np.ndarray]
    committed_waypoint_w: np.ndarray
    progress_reference: np.ndarray
    no_progress_steps: np.ndarray
    recovery_steps: np.ndarray

    @classmethod
    def create(cls, num_envs: int) -> "GeomTeacherState":
        return cls(
            np.zeros(num_envs, dtype=np.int8),
            np.zeros(num_envs, dtype=np.int32),
            np.zeros((num_envs, 3), dtype=np.float32),
            [np.zeros((0, 2), dtype=np.float32) for _ in range(num_envs)],
            np.full((num_envs, 2), np.nan, dtype=np.float32),
            np.full(num_envs, np.inf, dtype=np.float32),
            np.zeros(num_envs, dtype=np.int32),
            np.zeros(num_envs, dtype=np.int32),
        )

    def reset(self, done: torch.Tensor | np.ndarray | None = None) -> None:
        if done is None:
            self.bypass_side.fill(0)
            self.clear_steps.fill(0)
            self.previous_action.fill(0.0)
            self.observed_obstacles_w = [np.zeros((0, 2), dtype=np.float32) for _ in self.observed_obstacles_w]
            self.committed_waypoint_w.fill(np.nan)
            self.progress_reference.fill(np.inf)
            self.no_progress_steps.fill(0)
            self.recovery_steps.fill(0)
            return
        mask = _to_numpy(done).reshape(-1).astype(bool)
        self.bypass_side[mask] = 0
        self.clear_steps[mask] = 0
        self.previous_action[mask] = 0.0
        for idx in np.flatnonzero(mask):
            self.observed_obstacles_w[int(idx)] = np.zeros((0, 2), dtype=np.float32)
        self.committed_waypoint_w[mask] = np.nan
        self.progress_reference[mask] = np.inf
        self.no_progress_steps[mask] = 0
        self.recovery_steps[mask] = 0


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "data") and hasattr(value.data, "detach"):
        return value.data.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _robot_xy_heading_batch(env) -> tuple[np.ndarray, np.ndarray]:
    robot = env.env.unwrapped.scene["robot"]
    xy = _to_numpy(robot.data.root_link_pos_w)[:, :2].copy()
    heading = _to_numpy(robot.data.heading_w).reshape(-1).copy()
    return xy, heading


def _body_from_world(points_w: np.ndarray, robot_xy: np.ndarray, heading: float) -> np.ndarray:
    rel = points_w - robot_xy.reshape(1, 2)
    c = math.cos(-heading)
    s = math.sin(-heading)
    return np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], axis=-1)


def _scan_footprint_body(env, robot_xy: np.ndarray, heading: float, env_idx: int) -> tuple[float, float, float]:
    try:
        sensor = env.env.unwrapped.scene.sensors.get("terrain_scan")
        hit_xy_w = _to_numpy(sensor.data.hit_pos_w)[env_idx, :, :2]
        local = _body_from_world(hit_xy_w, robot_xy, heading)
        valid = np.isfinite(local).all(axis=1)
        local = local[valid]
    except Exception:
        local = np.zeros((0, 2), dtype=np.float64)
    if local.size == 0:
        return -1.5, 1.5, 1.5
    min_forward = float(np.nanpercentile(local[:, 0], 2.0))
    max_forward = float(np.nanpercentile(local[:, 0], 98.0))
    max_lateral = float(np.nanpercentile(np.abs(local[:, 1]), 98.0))
    return min_forward, max_forward, max_lateral


def _astar_local_target(
    *,
    obstacles: np.ndarray,
    goal_b: np.ndarray,
    min_forward: float,
    max_forward: float,
    max_lateral: float,
    clearance: float,
    resolution: float,
    waypoint_index: int,
    preferred_side: int = 0,
    side_penalty: float = 4.0,
    side_frame: str = "body",
) -> float | None:
    x_vals = np.arange(min_forward, max_forward + 0.5 * resolution, resolution, dtype=np.float64)
    y_vals = np.arange(-max_lateral, max_lateral + 0.5 * resolution, resolution, dtype=np.float64)
    if len(x_vals) < 3 or len(y_vals) < 3:
        return None
    xx, yy = np.meshgrid(x_vals, y_vals, indexing="ij")
    occ = np.zeros(xx.shape, dtype=bool)
    if obstacles.size:
        # Inflate obstacle cells by clearance; the coarse scan footprint is
        # short-range, so conservative local keepout is intentional here.
        for obs in obstacles:
            occ |= (xx - obs[0]) ** 2 + (yy - obs[1]) ** 2 <= clearance**2

    def nearest_idx(point: np.ndarray) -> tuple[int, int]:
        ix = int(np.argmin(np.abs(x_vals - float(point[0]))))
        iy = int(np.argmin(np.abs(y_vals - float(point[1]))))
        return ix, iy

    start = nearest_idx(np.asarray([0.0, 0.0], dtype=np.float64))
    occ[start] = False
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            sx = start[0] + dx
            sy = start[1] + dy
            if 0 <= sx < occ.shape[0] and 0 <= sy < occ.shape[1]:
                occ[sx, sy] = False

    goal_dist = float(np.linalg.norm(goal_b))
    if goal_dist < 1e-6:
        return 0.0
    direction = goal_b / goal_dist
    t_candidates = []
    if abs(direction[0]) > 1e-6:
        t_candidates.append(max_forward / direction[0] if direction[0] > 0.0 else min_forward / direction[0])
    if abs(direction[1]) > 1e-6:
        boundary_y = max_lateral if direction[1] > 0.0 else -max_lateral
        t_candidates.append(boundary_y / direction[1])
    positive = [t for t in t_candidates if t > 0.0]
    t_edge = min(positive) if positive else max_forward
    target = direction * min(goal_dist, max(0.25, t_edge))
    target[0] = float(np.clip(target[0], min_forward, max_forward))
    target[1] = float(np.clip(target[1], -max_lateral, max_lateral))
    goal = nearest_idx(target)
    if occ[goal]:
        free = np.argwhere(~occ)
        if free.size == 0:
            return None
        pts = np.stack([x_vals[free[:, 0]], y_vals[free[:, 1]]], axis=1)
        # Prefer free cells near the goal-facing edge and close to the true
        # goal direction.
        score = np.linalg.norm(pts - target.reshape(1, 2), axis=1) + 0.15 * np.linalg.norm(
            pts - goal_b.reshape(1, 2), axis=1
        )
        goal = tuple(int(x) for x in free[int(np.argmin(score))])

    import heapq

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))

    neighbors = [
        (-1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
    ]
    open_heap: list[tuple[float, tuple[int, int]]] = [(heuristic(start, goal), start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            idx = min(max(1, int(waypoint_index)), len(path) - 1)
            wp = path[idx]
            return float(math.atan2(y_vals[wp[1]], x_vals[wp[0]]))
        closed.add(current)
        for dx, dy, step_cost in neighbors:
            nb = (current[0] + dx, current[1] + dy)
            if not (0 <= nb[0] < occ.shape[0] and 0 <= nb[1] < occ.shape[1]):
                continue
            if occ[nb]:
                continue
            point = np.asarray([x_vals[nb[0]], y_vals[nb[1]]], dtype=np.float64)
            if side_frame == "goal":
                goal_unit = goal_b / max(float(np.linalg.norm(goal_b)), 1e-6)
                lateral = float(goal_unit[0] * point[1] - goal_unit[1] * point[0])
            else:
                lateral = float(point[1])
            wrong_side = preferred_side != 0 and preferred_side * lateral < -0.5 * resolution
            tentative = g_score[current] + step_cost + (side_penalty if wrong_side else 0.0)
            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(nb, goal), nb))
    return None


def _single_action(
    *,
    goal_b: np.ndarray,
    obstacle_cells_w: np.ndarray,
    robot_xy: np.ndarray,
    heading: float,
    env,
    env_idx: int,
    args,
    state: GeomTeacherState | None,
) -> torch.Tensor:
    goal_dist = float(np.linalg.norm(goal_b))
    goal_angle = float(math.atan2(goal_b[1], goal_b[0])) if goal_dist > 1e-6 else 0.0
    min_forward, max_forward, max_lateral = _scan_footprint_body(env, robot_xy, heading, env_idx)
    margin = float(getattr(args, "teacher_geom_scan_margin", 0.15))
    max_forward = min(max_forward + margin, float(getattr(args, "teacher_geom_max_forward", 1.75)))
    max_lateral = min(max_lateral + margin, float(getattr(args, "teacher_geom_max_lateral", 1.75)))
    min_forward = max(min_forward - margin, -float(getattr(args, "teacher_geom_back_margin", 0.35)))

    visible_world = np.zeros((0, 2), dtype=np.float64)
    if obstacle_cells_w.size:
        local_all = _body_from_world(obstacle_cells_w.astype(np.float64), robot_xy, heading)
        visible = (
            (local_all[:, 0] >= min_forward)
            & (local_all[:, 0] <= max_forward)
            & (np.abs(local_all[:, 1]) <= max_lateral)
        )
        obstacles = local_all[visible]
        visible_world = obstacle_cells_w[visible].astype(np.float64)
    else:
        obstacles = np.zeros((0, 2), dtype=np.float64)

    memory_radius = float(getattr(args, "teacher_geom_memory_radius", 0.0))
    if state is not None and memory_radius > 0.0:
        remembered = state.observed_obstacles_w[env_idx].astype(np.float64)
        if visible_world.size:
            combined = np.concatenate([remembered, visible_world], axis=0) if remembered.size else visible_world
            resolution = max(float(getattr(args, "teacher_geom_grid_resolution", 0.18)), 1e-3)
            keys = np.round(combined / resolution).astype(np.int64)
            _, unique_idx = np.unique(keys, axis=0, return_index=True)
            remembered = combined[np.sort(unique_idx)]
        if remembered.size:
            keep = np.linalg.norm(remembered - robot_xy.reshape(1, 2), axis=1) <= memory_radius
            remembered = remembered[keep]
        state.observed_obstacles_w[env_idx] = remembered.astype(np.float32)
        obstacles = _body_from_world(remembered, robot_xy, heading) if remembered.size else obstacles
        min_forward = min(min_forward, -memory_radius)
        max_forward = max(max_forward, memory_radius)
        max_lateral = max(max_lateral, memory_radius)

    lookahead = min(max_forward, float(getattr(args, "teacher_geom_lookahead", 1.6)))
    clearance = float(getattr(args, "teacher_geom_clearance", 0.62))
    goal_unit = goal_b / max(goal_dist, 1e-6)
    if obstacles.size:
        projection = obstacles @ goal_unit
        lateral_to_goal = np.abs(obstacles[:, 0] * goal_unit[1] - obstacles[:, 1] * goal_unit[0])
        direct_blocked = bool(np.any((projection > 0.0) & (projection <= lookahead) & (lateral_to_goal <= clearance)))
    else:
        direct_blocked = False
    preferred_side = int(state.bypass_side[env_idx]) if state is not None else 0
    target_angle = None
    if str(getattr(args, "teacher_geom_planner", "astar")).lower() == "astar":
        target_angle = _astar_local_target(
            obstacles=obstacles,
            goal_b=goal_b,
            min_forward=min_forward,
            max_forward=max_forward,
            max_lateral=max_lateral,
            clearance=clearance,
            resolution=float(getattr(args, "teacher_geom_grid_resolution", 0.18)),
            waypoint_index=int(getattr(args, "teacher_geom_waypoint_index", 3)),
            preferred_side=preferred_side,
            side_penalty=float(getattr(args, "teacher_geom_side_penalty", 4.0)),
            side_frame=str(getattr(args, "teacher_geom_side_frame", "body")),
        )
    max_angle = float(getattr(args, "teacher_geom_max_angle", math.pi / 2.0))
    candidate_count = max(3, int(getattr(args, "teacher_geom_candidate_count", 31)))
    candidate_angles = np.linspace(-max_angle, max_angle, candidate_count, dtype=np.float64)
    soft_width = max(clearance, float(getattr(args, "teacher_geom_soft_width", 0.9)))
    hard_penalty = float(getattr(args, "teacher_geom_hard_penalty", 35.0))
    risk_weight = float(getattr(args, "teacher_geom_risk_weight", 9.0))
    angle_weight = float(getattr(args, "teacher_geom_angle_weight", 0.15))
    forward_bias = float(getattr(args, "teacher_geom_forward_bias", 0.2))
    if target_angle is None:
        scores: list[float] = []
        for angle in candidate_angles:
            direction = np.array([math.cos(float(angle)), math.sin(float(angle))], dtype=np.float64)
            if obstacles.size:
                t = obstacles @ direction
                lateral = np.abs(obstacles[:, 0] * direction[1] - obstacles[:, 1] * direction[0])
                in_path = (t > 0.0) & (t <= lookahead)
                hard_count = int(np.sum(in_path & (lateral <= clearance)))
                soft = np.exp(-0.5 * (lateral / max(soft_width, 1e-6)) ** 2)
                soft *= np.exp(-np.clip(t, 0.0, None) / max(lookahead, 1e-6))
                risk = float(np.sum(np.where(in_path, soft, 0.0)))
            else:
                hard_count = 0
                risk = 0.0
            endpoint = direction * min(lookahead, max(goal_dist, 0.25))
            endpoint_dist = float(np.linalg.norm(goal_b - endpoint))
            angle_err = abs(float(math.atan2(math.sin(float(angle) - goal_angle), math.cos(float(angle) - goal_angle))))
            progress = float(goal_unit @ direction)
            progress_penalty = max(0.0, -progress) * 3.0 + max(0.0, 0.2 - progress)
            scores.append(
                endpoint_dist
                + hard_penalty * hard_count
                + risk_weight * risk
                + angle_weight * angle_err
                + progress_penalty
                - forward_bias * progress
            )
        target_angle = float(candidate_angles[int(np.argmin(np.asarray(scores)))])
    if state is not None:
        if preferred_side == 0 and direct_blocked:
            if str(getattr(args, "teacher_geom_side_frame", "body")) == "goal":
                side_signal = math.sin(float(target_angle) - goal_angle)
            else:
                side_signal = float(target_angle)
            preferred_side = 1 if side_signal >= 0.0 else -1
            state.bypass_side[env_idx] = preferred_side
            state.clear_steps[env_idx] = 0
        elif preferred_side != 0:
            if direct_blocked:
                state.clear_steps[env_idx] = 0
            else:
                state.clear_steps[env_idx] += 1
                if state.clear_steps[env_idx] >= int(getattr(args, "teacher_geom_disengage_clear_steps", 12)):
                    state.bypass_side[env_idx] = 0
                    state.clear_steps[env_idx] = 0
                    preferred_side = 0
                    state.committed_waypoint_w[env_idx] = np.nan

    waypoint_distance = float(getattr(args, "teacher_geom_waypoint_commit_distance", 0.0))
    if state is not None and waypoint_distance > 0.0 and preferred_side != 0:
        waypoint_w = state.committed_waypoint_w[env_idx].astype(np.float64)
        waypoint_valid = bool(np.isfinite(waypoint_w).all())
        if waypoint_valid:
            waypoint_b = _body_from_world(waypoint_w.reshape(1, 2), robot_xy, heading)[0]
            waypoint_norm = float(np.linalg.norm(waypoint_b))
            reach_dist = float(getattr(args, "teacher_geom_waypoint_reach_dist", 0.25))
            if waypoint_norm <= reach_dist:
                waypoint_valid = False
            elif obstacles.size:
                direction = waypoint_b / max(waypoint_norm, 1e-6)
                projection = obstacles @ direction
                lateral = np.abs(obstacles[:, 0] * direction[1] - obstacles[:, 1] * direction[0])
                waypoint_valid = not bool(
                    np.any((projection > 0.0) & (projection <= waypoint_norm) & (lateral <= clearance))
                )
        if not waypoint_valid:
            direction_b = np.asarray([math.cos(target_angle), math.sin(target_angle)], dtype=np.float64)
            c = math.cos(heading)
            s = math.sin(heading)
            direction_w = np.asarray(
                [c * direction_b[0] - s * direction_b[1], s * direction_b[0] + c * direction_b[1]],
                dtype=np.float64,
            )
            waypoint_w = robot_xy + waypoint_distance * direction_w
            state.committed_waypoint_w[env_idx] = waypoint_w.astype(np.float32)
            waypoint_b = direction_b * waypoint_distance
        target_angle = float(math.atan2(waypoint_b[1], waypoint_b[0]))

    progress_window = int(getattr(args, "teacher_geom_stall_window", 0))
    if state is not None and progress_window > 0:
        progress_eps = float(getattr(args, "teacher_geom_stall_progress_epsilon", 0.08))
        if not np.isfinite(state.progress_reference[env_idx]):
            state.progress_reference[env_idx] = goal_dist
        if goal_dist <= float(state.progress_reference[env_idx]) - progress_eps:
            state.progress_reference[env_idx] = goal_dist
            state.no_progress_steps[env_idx] = 0
        elif state.recovery_steps[env_idx] <= 0:
            state.no_progress_steps[env_idx] += 1
        if state.no_progress_steps[env_idx] >= progress_window:
            state.no_progress_steps[env_idx] = 0
            state.progress_reference[env_idx] = goal_dist
            state.recovery_steps[env_idx] = int(getattr(args, "teacher_geom_stall_recovery_steps", 60))
            if state.bypass_side[env_idx] == 0:
                if str(getattr(args, "teacher_geom_side_frame", "body")) == "goal":
                    side_signal = math.sin(float(target_angle) - goal_angle)
                else:
                    side_signal = float(target_angle)
                state.bypass_side[env_idx] = 1 if side_signal >= 0.0 else -1
            elif bool(getattr(args, "teacher_geom_stall_flip_side", False)):
                state.bypass_side[env_idx] *= -1
            preferred_side = int(state.bypass_side[env_idx])
        if state.recovery_steps[env_idx] > 0:
            recovery_angle = float(getattr(args, "teacher_geom_stall_recovery_angle", 0.9))
            if str(getattr(args, "teacher_geom_side_frame", "body")) == "goal":
                target_angle = goal_angle + preferred_side * recovery_angle
            else:
                target_angle = preferred_side * recovery_angle
            state.recovery_steps[env_idx] -= 1
    emergency_radius = float(getattr(args, "teacher_geom_emergency_radius", 0.8))
    if obstacles.size and emergency_radius > 0.0:
        radius = np.linalg.norm(obstacles, axis=1)
        close = radius < emergency_radius
        if np.any(close):
            weights = ((emergency_radius - radius[close]).clip(min=0.0) / max(emergency_radius, 1e-6)) ** 2 + 1e-3
            centroid = np.sum(obstacles[close] * weights.reshape(-1, 1), axis=0) / max(float(np.sum(weights)), 1e-6)
            away = -centroid / max(float(np.linalg.norm(centroid)), 1e-6)
            tangent_a = np.array([-centroid[1], centroid[0]], dtype=np.float64)
            tangent_a /= max(float(np.linalg.norm(tangent_a)), 1e-6)
            tangent_b = -tangent_a
            if str(getattr(args, "teacher_geom_side_frame", "body")) == "goal":
                tangent_a_side = float(goal_unit[0] * tangent_a[1] - goal_unit[1] * tangent_a[0])
                tangent_b_side = float(goal_unit[0] * tangent_b[1] - goal_unit[1] * tangent_b[0])
            else:
                tangent_a_side = float(tangent_a[1])
                tangent_b_side = float(tangent_b[1])
            if preferred_side > 0:
                tangent = tangent_a if tangent_a_side >= tangent_b_side else tangent_b
            elif preferred_side < 0:
                tangent = tangent_a if tangent_a_side <= tangent_b_side else tangent_b
            else:
                tangent = tangent_a if float(tangent_a @ goal_unit) >= float(tangent_b @ goal_unit) else tangent_b
            blended = (
                float(getattr(args, "teacher_geom_emergency_repulsion_weight", 1.2)) * away
                + float(getattr(args, "teacher_geom_emergency_tangent_weight", 1.4)) * tangent
                + float(getattr(args, "teacher_geom_emergency_goal_weight", 0.25)) * goal_unit
            )
            if np.linalg.norm(blended) > 1e-6:
                target_angle = float(math.atan2(blended[1], blended[0]))

    max_vx = float(getattr(args, "teacher_max_vx", 0.75))
    max_vy = float(getattr(args, "teacher_max_vy", 0.5))
    obstacle_speed_radius = float(getattr(args, "teacher_geom_obstacle_speed_radius", 0.0))
    if obstacles.size and obstacle_speed_radius > 0.0:
        nearest_obstacle = float(np.min(np.linalg.norm(obstacles, axis=1)))
        if nearest_obstacle < obstacle_speed_radius:
            near_scale = float(getattr(args, "teacher_geom_near_obstacle_vx_scale", 0.7))
            blend = float(np.clip(nearest_obstacle / obstacle_speed_radius, 0.0, 1.0))
            max_vx *= near_scale + (1.0 - near_scale) * blend
    yaw_gain = float(getattr(args, "teacher_yaw_gain", 1.2))
    align_angle = float(getattr(args, "teacher_align_angle", 0.45))
    min_forward_scale = float(getattr(args, "teacher_min_forward_scale", 0.25))
    speed_gate = 1.0 if abs(target_angle) <= align_angle else min_forward_scale
    vx = max_vx * max(0.0, math.cos(target_angle)) * speed_gate
    vy = float(np.clip(max_vy * math.sin(target_angle), -1.0, 1.0))
    yaw = float(np.clip(yaw_gain * target_angle, -1.0, 1.0))
    if goal_dist <= float(getattr(args, "teacher_goal_stop_dist", 0.0)):
        vx = 0.0
        vy = 0.0
        yaw = 0.0
    action = np.asarray([vx, vy, yaw], dtype=np.float32)
    smoothing = float(np.clip(getattr(args, "teacher_geom_command_smoothing", 0.0), 0.0, 0.95))
    if state is not None and smoothing > 0.0:
        action = smoothing * state.previous_action[env_idx] + (1.0 - smoothing) * action
    if state is not None:
        state.previous_action[env_idx] = action
    return torch.as_tensor(action, dtype=torch.float32)


def local_geometry_scan_teacher_action(
    obs: torch.Tensor,
    *,
    env,
    obstacle_cells: list[np.ndarray],
    args,
    state: GeomTeacherState | None = None,
) -> torch.Tensor:
    robot_xy, heading = _robot_xy_heading_batch(env)
    actions = []
    for env_idx in range(int(obs.shape[0])):
        cells = obstacle_cells[env_idx] if env_idx < len(obstacle_cells) else np.zeros((0, 2), dtype=np.float32)
        actions.append(
            _single_action(
                goal_b=obs[env_idx, 6:8].detach().cpu().numpy().astype(np.float64),
                obstacle_cells_w=cells,
                robot_xy=robot_xy[env_idx],
                heading=float(heading[env_idx]),
                env=env,
                env_idx=env_idx,
                args=args,
                state=state,
            )
        )
    return torch.stack(actions, dim=0).to(device=obs.device, dtype=torch.float32).clamp(-1.0, 1.0)

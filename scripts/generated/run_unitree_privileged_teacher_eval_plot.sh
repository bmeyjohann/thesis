#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis
export MPLBACKEND=Agg

python - <<'PY'
from __future__ import annotations

import json
import math
import os
import time
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_unitree_nav_baselines import (
    _current_goal_distance,
    _goal_positions_xy,
    _reset_until_feasible,
    _robot_positions_xy,
    _terrain_obstacle_cells_by_env,
    make_env,
)
from train_unitree_nav_thesis import DEFAULT_LOW_LEVEL, ROOT, _extract_actor_obs, _extract_cost


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _robot_xy_heading(env) -> tuple[np.ndarray, float]:
    robot = env.env.unwrapped.scene["robot"]
    xy = robot.data.root_link_pos_w[0, :2].detach().cpu().numpy().copy()
    heading = float(robot.data.heading_w[0].detach().cpu().item())
    return xy, heading


def _distance_point_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        t = np.zeros((len(points),), dtype=np.float32)
        return np.linalg.norm(points - a[None, :], axis=1), t
    t = np.clip(((points - a[None, :]) @ ab) / denom, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - proj, axis=1), t


class PrivilegedWaypointTeacher:
    def __init__(
        self,
        *,
        clearance: float = 0.75,
        waypoint_offset: float = 1.1,
        waypoint_ahead: float = 0.55,
        max_vx: float = 0.55,
        max_vy: float = 0.0,
        yaw_gain: float = 1.35,
        align_angle: float = 0.7,
        waypoint_reached: float = 0.45,
    ) -> None:
        self.clearance = clearance
        self.waypoint_offset = waypoint_offset
        self.waypoint_ahead = waypoint_ahead
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.yaw_gain = yaw_gain
        self.align_angle = align_angle
        self.waypoint_reached = waypoint_reached
        self.side = 1.0
        self.waypoint: np.ndarray | None = None

    def reset(self) -> None:
        self.side = 1.0
        self.waypoint = None

    def _choose_waypoint(self, xy: np.ndarray, goal: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        if obstacles.size == 0:
            self.waypoint = None
            return goal
        seg_dist, seg_t = _distance_point_to_segment(obstacles, xy, goal)
        path_len = max(1e-6, float(np.linalg.norm(goal - xy)))
        along_ok = (seg_t > 0.05) & (seg_t < 0.9)
        blockers = np.nonzero((seg_dist < self.clearance) & along_ok)[0]
        if blockers.size == 0:
            self.waypoint = None
            return goal

        # Use the closest blocker along the current route and place a tangent-like
        # waypoint on a persistent side. This is privileged and diagnostic only.
        idx = blockers[np.argmin(seg_t[blockers])]
        blocker = obstacles[idx]
        direction = (goal - xy) / path_len
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        left_wp = blocker + normal * self.waypoint_offset + direction * self.waypoint_ahead
        right_wp = blocker - normal * self.waypoint_offset + direction * self.waypoint_ahead

        def score(candidate: np.ndarray) -> float:
            d = np.linalg.norm(obstacles - candidate[None, :], axis=1)
            collision_penalty = max(0.0, self.clearance - float(d.min())) * 20.0
            return float(np.linalg.norm(candidate - goal)) + collision_penalty

        if self.waypoint is None or np.linalg.norm(xy - self.waypoint) < self.waypoint_reached:
            left_score = score(left_wp)
            right_score = score(right_wp)
            self.side = 1.0 if left_score <= right_score else -1.0
            self.waypoint = left_wp if self.side > 0 else right_wp
        return self.waypoint

    def action(self, env, obs: torch.Tensor, obstacles: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        xy, heading = _robot_xy_heading(env)
        goal = _goal_positions_xy(env)[0].copy()
        target = self._choose_waypoint(xy, goal, obstacles)
        vec_w = target - xy
        dist = float(np.linalg.norm(vec_w))
        target_angle_w = math.atan2(float(vec_w[1]), float(vec_w[0])) if dist > 1e-6 else heading
        rel_angle = _wrap_angle(target_angle_w - heading)
        yaw = float(np.clip(self.yaw_gain * rel_angle, -1.0, 1.0))
        speed_gate = 1.0 if abs(rel_angle) <= self.align_angle else 0.15
        vx = float(np.clip(self.max_vx * speed_gate, -1.0, 1.0))
        vy = 0.0
        if self.max_vy > 0.0:
            # Optional lateral component toward the target in the robot frame.
            c, s = math.cos(-heading), math.sin(-heading)
            local_y = s * vec_w[0] + c * vec_w[1]
            vy = float(np.clip(local_y * 0.8, -self.max_vy, self.max_vy))
        if dist < 0.25:
            vx = 0.0
            vy = 0.0
        return torch.tensor([[vx, vy, yaw]], device=obs.device, dtype=torch.float32), target


run_name = os.environ.get("RUN_NAME", f"unitree_privileged_teacher_{time.strftime('%Y%m%d_%H%M%S')}")
out_dir = Path(os.environ.get("OUTPUT_DIR", str(ROOT / "visualizations" / "unitree_nav_privileged_teacher" / run_name)))
out_dir.mkdir(parents=True, exist_ok=True)

args = Namespace(
    controller="privileged_teacher",
    model_path="",
    task=os.environ.get("TASK", "Unitree-G1-Nav-Obstacles-Safe-Collision"),
    device=os.environ.get("DEVICE", "cuda:0"),
    num_envs=1,
    num_episodes=_env_int("NUM_EPISODES", 6),
    episode_length_s=_env_float("EPISODE_LENGTH_S", 20.0),
    success_dist=_env_float("SUCCESS_DIST", 0.5),
    hidden_dim=256,
    use_layer_norm=False,
    low_level_policy_path=os.environ.get("LOW_LEVEL_POLICY_PATH", str(DEFAULT_LOW_LEVEL)),
    output_dir=str(out_dir),
    run_name=run_name,
    min_goal_obstacle_clearance=_env_float("MIN_GOAL_OBSTACLE_CLEARANCE", 0.65),
    goal_clearance_resample_attempts=_env_int("GOAL_CLEARANCE_RESAMPLE_ATTEMPTS", 100),
    min_start_obstacle_clearance=_env_float("MIN_START_OBSTACLE_CLEARANCE", 0.65),
    start_clearance_resample_attempts=_env_int("START_CLEARANCE_RESAMPLE_ATTEMPTS", 100),
    require_blocked_corridor=_env_bool("REQUIRE_BLOCKED_CORRIDOR", True),
    blocked_corridor_radius=_env_float("BLOCKED_CORRIDOR_RADIUS", 0.45),
    blocked_corridor_ignore_end_radius=_env_float("BLOCKED_CORRIDOR_IGNORE_END_RADIUS", 0.75),
    blocked_corridor_min_cells=_env_int("BLOCKED_CORRIDOR_MIN_CELLS", 1),
    blocked_corridor_resample_attempts=_env_int("BLOCKED_CORRIDOR_RESAMPLE_ATTEMPTS", 200),
    debug_obstacle_width_min=_env_float("DEBUG_OBSTACLE_WIDTH_MIN", 0.9),
    debug_obstacle_width_max=_env_float("DEBUG_OBSTACLE_WIDTH_MAX", 1.2),
    debug_obstacle_height_min=_env_float("DEBUG_OBSTACLE_HEIGHT_MIN", 0.45),
    debug_obstacle_height_max=_env_float("DEBUG_OBSTACLE_HEIGHT_MAX", 0.55),
    debug_num_obstacles=_env_int("DEBUG_NUM_OBSTACLES", 12),
    debug_platform_width=_env_float("DEBUG_PLATFORM_WIDTH", 2.0),
    debug_obstacle_border_width=_env_float("DEBUG_OBSTACLE_BORDER_WIDTH", 0.5),
    debug_goal_through_obstacle=_env_bool("DEBUG_GOAL_THROUGH_OBSTACLE", True),
    debug_goal_distance=_env_float("DEBUG_GOAL_DISTANCE", 3.4),
    debug_goal_obstacle_min_dist=_env_float("DEBUG_GOAL_OBSTACLE_MIN_DIST", 0.9),
    debug_goal_obstacle_max_dist=_env_float("DEBUG_GOAL_OBSTACLE_MAX_DIST", 2.2),
)

env = make_env(args, num_envs=1, render=False)
teacher = PrivilegedWaypointTeacher(
    clearance=_env_float("PRIV_CLEARANCE", 0.85),
    waypoint_offset=_env_float("PRIV_WAYPOINT_OFFSET", 1.15),
    waypoint_ahead=_env_float("PRIV_WAYPOINT_AHEAD", 0.55),
    max_vx=_env_float("PRIV_MAX_VX", 0.55),
    max_vy=_env_float("PRIV_MAX_VY", 0.0),
    yaw_gain=_env_float("PRIV_YAW_GAIN", 1.4),
    align_angle=_env_float("PRIV_ALIGN_ANGLE", 0.75),
)

episodes = []
for ep_idx in range(args.num_episodes):
    print(f"[episode {ep_idx + 1}/{args.num_episodes}] reset", flush=True)
    obs_raw, _, _, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
    teacher.reset()
    obs = _extract_actor_obs(obs_raw).to(device=args.device, dtype=torch.float32)
    obstacles = obstacle_cells[0]
    xy_hist = []
    cost_xy = []
    waypoint_hist = []
    goal = _goal_positions_xy(env)[0].copy()
    total_cost = 0.0
    success_step = None
    steps = int(round(args.episode_length_s / 0.05))
    for step in range(steps):
        xy, _ = _robot_xy_heading(env)
        xy_hist.append(xy)
        dist = float(_current_goal_distance(env, 1, torch.device(args.device))[0].detach().cpu().item())
        if success_step is None and dist <= args.success_dist:
            success_step = step
        action, waypoint = teacher.action(env, obs, obstacles)
        waypoint_hist.append(waypoint.copy())
        obs_raw, _, done, extras = env.step(action)
        cost = _extract_cost(extras, 1, torch.device(args.device))
        c = float(cost.reshape(-1)[0].detach().cpu().item())
        total_cost += c
        if c > 0:
            cost_xy.append(xy.copy())
        obs = _extract_actor_obs(obs_raw).to(device=args.device, dtype=torch.float32)
        if bool(torch.as_tensor(done).reshape(-1)[0].detach().cpu().item()):
            break
    episodes.append(
        {
            "xy": np.asarray(xy_hist, dtype=np.float32),
            "cost_xy": np.asarray(cost_xy, dtype=np.float32) if cost_xy else np.zeros((0, 2), dtype=np.float32),
            "waypoints": np.asarray(waypoint_hist, dtype=np.float32),
            "goal": goal,
            "obstacles": obstacles,
            "total_cost": total_cost,
            "success_step": success_step,
            "steps": len(xy_hist),
            "layout": layout_stats[0] if layout_stats else {},
        }
    )
    print(
        f"[episode {ep_idx + 1}] cost={total_cost:.1f} success={success_step is not None} steps={len(xy_hist)}",
        flush=True,
    )

cols = 3
rows = math.ceil(len(episodes) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.2 * rows), dpi=150, squeeze=False)
for idx, ep in enumerate(episodes):
    ax = axes[idx // cols][idx % cols]
    obs_cells = ep["obstacles"]
    xy = ep["xy"]
    if obs_cells.size:
        ax.scatter(obs_cells[:, 0], obs_cells[:, 1], s=9, c="black", alpha=0.22, marker="s", label="obstacle cells")
    ax.plot(xy[:, 0], xy[:, 1], c="dodgerblue", lw=2.0, label="trajectory")
    if ep["waypoints"].size:
        ax.plot(ep["waypoints"][:, 0], ep["waypoints"][:, 1], c="orange", lw=1.0, alpha=0.6, label="privileged target")
    if len(xy):
        ax.scatter(xy[0, 0], xy[0, 1], c="white", edgecolors="black", s=80, zorder=4, label="start")
        ax.scatter(xy[-1, 0], xy[-1, 1], c="dodgerblue", edgecolors="black", s=55, zorder=4, label="end")
    ax.scatter(ep["goal"][0], ep["goal"][1], marker="*", s=230, c="limegreen", edgecolors="black", linewidths=1.0, zorder=5, label="goal")
    if ep["cost_xy"].size:
        ax.scatter(ep["cost_xy"][:, 0], ep["cost_xy"][:, 1], c="red", s=70, marker="x", linewidths=2.2, zorder=6, label="cost")
    ax.set_title(f"ep {idx+1}: cost={ep['total_cost']:.1f}, success={ep['success_step'] is not None}, steps={ep['steps']}", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    if idx == 0:
        ax.legend(loc="upper right", fontsize=7)
for idx in range(len(episodes), rows * cols):
    axes[idx // cols][idx % cols].axis("off")

success_rate = sum(ep["success_step"] is not None for ep in episodes) / len(episodes)
costful_rate = sum(ep["total_cost"] > 0 for ep in episodes) / len(episodes)
mean_cost = sum(ep["total_cost"] for ep in episodes) / len(episodes)
fig.suptitle(f"{run_name}: privileged teacher | success={success_rate:.2f} costful={costful_rate:.2f} mean_cost={mean_cost:.2f}")
fig.tight_layout()
plot_path = out_dir / "privileged_teacher_trajectory_contact_sheet.png"
fig.savefig(plot_path)
plt.close(fig)

summary = {
    "run_name": run_name,
    "success_rate": success_rate,
    "costful_episode_rate": costful_rate,
    "mean_cost_sum": mean_cost,
    "plot": str(plot_path),
    "episodes": [
        {
            "total_cost": ep["total_cost"],
            "success_step": ep["success_step"],
            "steps": ep["steps"],
            "blocked_corridor_cell_count": ep["layout"].get("blocked_cell_count"),
        }
        for ep in episodes
    ],
}
(out_dir / "privileged_teacher_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary), flush=True)
PY

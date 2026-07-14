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


def _robot_xy_heading(env) -> tuple[np.ndarray, float]:
    robot = env.env.unwrapped.scene["robot"]
    xy = robot.data.root_link_pos_w[0, :2].detach().cpu().numpy().copy()
    heading = float(robot.data.heading_w[0].detach().cpu().item())
    return xy, heading


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class LocalGridTeacher:
    """Observation-only teacher using the 7x7 height scan as a local occupancy grid."""

    def __init__(
        self,
        *,
        obstacle_delta: float = 0.018,
        clearance: float = 0.75,
        horizon: float = 1.7,
        candidate_count: int = 25,
        path_samples: int = 9,
        clearance_weight: float = 60.0,
        switch_penalty: float = 0.08,
        max_vx: float = 0.55,
        min_vx: float = 0.12,
        yaw_gain: float = 1.25,
        align_angle: float = 0.75,
    ) -> None:
        self.obstacle_delta = obstacle_delta
        self.clearance = clearance
        self.horizon = horizon
        self.candidate_count = candidate_count
        self.path_samples = path_samples
        self.clearance_weight = clearance_weight
        self.switch_penalty = switch_penalty
        self.max_vx = max_vx
        self.min_vx = min_vx
        self.yaw_gain = yaw_gain
        self.align_angle = align_angle
        self.last_angle = 0.0

    def reset(self) -> None:
        self.last_angle = 0.0

    def action(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        obs_np = obs[0].detach().cpu().numpy()
        goal = obs_np[6:8].astype(np.float32)
        goal_dist = float(np.linalg.norm(goal))
        goal_angle = math.atan2(float(goal[1]), float(goal[0])) if goal_dist > 1e-6 else 0.0
        scan = obs_np[9:58].reshape(7, 7)

        coords = np.linspace(-1.5, 1.5, 7, dtype=np.float32)
        occupied = []
        flat_ref = float(np.percentile(scan, 90))
        blocked = scan < (flat_ref - self.obstacle_delta)
        for i, fwd in enumerate(coords):
            for j, lat in enumerate(coords):
                if blocked[i, j]:
                    occupied.append([float(fwd), float(lat)])
        occ = np.asarray(occupied, dtype=np.float32) if occupied else np.zeros((0, 2), dtype=np.float32)

        angles = np.linspace(-math.pi * 0.85, math.pi * 0.85, self.candidate_count)
        path_s = np.linspace(0.25, self.horizon, self.path_samples)
        best_score = float("inf")
        best_angle = goal_angle
        best_clearance = float("inf")
        for angle in angles:
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            path = path_s[:, None] * direction[None, :]
            if occ.size:
                d = np.linalg.norm(path[:, None, :] - occ[None, :, :], axis=-1)
                min_clearance = float(np.min(d))
            else:
                min_clearance = float("inf")
            endpoint = self.horizon * direction
            goal_after = float(np.linalg.norm(goal - endpoint))
            clearance_penalty = max(0.0, self.clearance - min_clearance) ** 2 * self.clearance_weight
            angle_penalty = 0.08 * abs(_wrap_angle(angle - goal_angle))
            switch_penalty = self.switch_penalty * abs(_wrap_angle(angle - self.last_angle))
            score = goal_after + clearance_penalty + angle_penalty + switch_penalty
            if score < best_score:
                best_score = score
                best_angle = float(angle)
                best_clearance = min_clearance

        # Hysteresis: avoid one-step side flips when a previous safe direction is almost as good.
        self.last_angle = 0.75 * self.last_angle + 0.25 * best_angle
        target_angle = self.last_angle
        yaw = float(np.clip(self.yaw_gain * target_angle, -1.0, 1.0))
        turn_scale = max(0.0, 1.0 - abs(target_angle) / max(1e-6, math.pi))
        vx = self.min_vx + (self.max_vx - self.min_vx) * turn_scale
        if abs(target_angle) > self.align_angle:
            vx = self.min_vx
        if goal_dist < 0.35:
            vx = 0.0
            yaw = 0.0
        action = torch.tensor([[vx, 0.0, yaw]], device=obs.device, dtype=torch.float32)
        return action, {
            "flat_ref": flat_ref,
            "blocked": blocked,
            "occupied_local": occ,
            "target_angle": target_angle,
            "raw_best_angle": best_angle,
            "best_clearance": best_clearance,
            "best_score": best_score,
            "goal_angle": goal_angle,
            "goal_dist": goal_dist,
        }


run_name = os.environ.get("RUN_NAME", f"unitree_local_grid_teacher_{time.strftime('%Y%m%d_%H%M%S')}")
out_dir = Path(os.environ.get("OUTPUT_DIR", str(ROOT / "visualizations" / "unitree_local_grid_teacher" / run_name)))
out_dir.mkdir(parents=True, exist_ok=True)

args = Namespace(
    controller="local_grid_teacher",
    model_path="",
    task=os.environ.get("TASK", "Unitree-G1-Nav-Obstacles-Safe-Collision"),
    device=os.environ.get("DEVICE", "cuda:0"),
    num_envs=1,
    num_episodes=_env_int("NUM_EPISODES", 3),
    episode_length_s=_env_float("EPISODE_LENGTH_S", 20.0),
    success_dist=_env_float("SUCCESS_DIST", 0.5),
    hidden_dim=256,
    use_layer_norm=False,
    low_level_policy_path=os.environ.get("LOW_LEVEL_POLICY_PATH", str(DEFAULT_LOW_LEVEL)),
    output_dir=str(out_dir),
    run_name=run_name,
    min_goal_obstacle_clearance=_env_float("MIN_GOAL_OBSTACLE_CLEARANCE", 0.0),
    goal_clearance_resample_attempts=_env_int("GOAL_CLEARANCE_RESAMPLE_ATTEMPTS", 100),
    min_start_obstacle_clearance=_env_float("MIN_START_OBSTACLE_CLEARANCE", 0.0),
    start_clearance_resample_attempts=_env_int("START_CLEARANCE_RESAMPLE_ATTEMPTS", 100),
    require_blocked_corridor=_env_bool("REQUIRE_BLOCKED_CORRIDOR", False),
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
teacher = LocalGridTeacher(
    obstacle_delta=_env_float("LOCAL_GRID_OBSTACLE_DELTA", 0.018),
    clearance=_env_float("LOCAL_GRID_CLEARANCE", 0.70),
    horizon=_env_float("LOCAL_GRID_HORIZON", 1.7),
    candidate_count=_env_int("LOCAL_GRID_CANDIDATES", 25),
    path_samples=_env_int("LOCAL_GRID_PATH_SAMPLES", 9),
    clearance_weight=_env_float("LOCAL_GRID_CLEARANCE_WEIGHT", 60.0),
    switch_penalty=_env_float("LOCAL_GRID_SWITCH_PENALTY", 0.08),
    max_vx=_env_float("LOCAL_GRID_MAX_VX", 0.55),
    min_vx=_env_float("LOCAL_GRID_MIN_VX", 0.12),
    yaw_gain=_env_float("LOCAL_GRID_YAW_GAIN", 1.25),
    align_angle=_env_float("LOCAL_GRID_ALIGN_ANGLE", 0.75),
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
    action_arrows = []
    snapshot_debug = None
    goal = _goal_positions_xy(env)[0].copy()
    total_cost = 0.0
    success_step = None
    steps = int(round(args.episode_length_s / 0.05))
    for step in range(steps):
        xy, heading = _robot_xy_heading(env)
        xy_hist.append(xy)
        dist = float(_current_goal_distance(env, 1, torch.device(args.device))[0].detach().cpu().item())
        if success_step is None and dist <= args.success_dist:
            success_step = step
        action, debug = teacher.action(obs)
        if step == 0:
            snapshot_debug = debug
        if step % 30 == 0:
            action_np = action[0].detach().cpu().numpy()
            local = action_np[:2]
            c, s = math.cos(heading), math.sin(heading)
            vec = np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]], dtype=np.float32)
            if np.linalg.norm(vec) < 1e-6:
                turn_heading = heading + math.copysign(math.pi / 2.0, float(action_np[2]) if abs(float(action_np[2])) > 1e-6 else 1.0)
                vec = np.array([math.cos(turn_heading), math.sin(turn_heading)], dtype=np.float32) * 0.25
            else:
                vec = vec / np.linalg.norm(vec) * 0.45
            action_arrows.append((xy.copy(), vec.copy()))
        obs_raw, _, done, extras = env.step(action)
        cost = _extract_cost(extras, 1, torch.device(args.device))
        cst = float(cost.reshape(-1)[0].detach().cpu().item())
        total_cost += cst
        if cst > 0:
            cost_xy.append(xy.copy())
        obs = _extract_actor_obs(obs_raw).to(device=args.device, dtype=torch.float32)
        if bool(torch.as_tensor(done).reshape(-1)[0].detach().cpu().item()):
            break
    episodes.append(
        {
            "xy": np.asarray(xy_hist, dtype=np.float32),
            "cost_xy": np.asarray(cost_xy, dtype=np.float32) if cost_xy else np.zeros((0, 2), dtype=np.float32),
            "goal": goal,
            "obstacles": obstacles,
            "arrows": action_arrows,
            "total_cost": total_cost,
            "success_step": success_step,
            "steps": len(xy_hist),
            "layout": layout_stats[0] if layout_stats else {},
            "snapshot_debug": snapshot_debug,
        }
    )
    print(f"[episode {ep_idx + 1}] cost={total_cost:.1f} success={success_step is not None} steps={len(xy_hist)}", flush=True)

cols = 3
rows = math.ceil(len(episodes) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 5.2 * rows), dpi=150, squeeze=False)
for idx, ep in enumerate(episodes):
    ax = axes[idx // cols][idx % cols]
    obstacles = ep["obstacles"]
    xy = ep["xy"]
    if obstacles.size:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], s=8, c="black", alpha=0.22, marker="s", label="obstacle cells")
    ax.plot(xy[:, 0], xy[:, 1], c="dodgerblue", lw=2.0, label="trajectory")
    for origin, vec in ep["arrows"]:
        ax.arrow(origin[0], origin[1], vec[0], vec[1], color="orange", width=0.012, head_width=0.08, alpha=0.75)
    if len(xy):
        ax.scatter(xy[0, 0], xy[0, 1], c="white", edgecolors="black", s=80, zorder=4, label="start")
        ax.scatter(xy[-1, 0], xy[-1, 1], c="dodgerblue", edgecolors="black", s=55, zorder=4, label="end")
    ax.scatter(ep["goal"][0], ep["goal"][1], marker="*", s=230, c="limegreen", edgecolors="black", linewidths=1.0, zorder=5, label="goal")
    if ep["cost_xy"].size:
        ax.scatter(ep["cost_xy"][:, 0], ep["cost_xy"][:, 1], c="red", s=70, marker="x", linewidths=2.2, zorder=6, label="cost")
    dbg = ep["snapshot_debug"] or {}
    ax.set_title(
        f"ep {idx+1}: cost={ep['total_cost']:.1f}, success={ep['success_step'] is not None}, "
        f"cells={int(dbg.get('blocked', np.zeros((0,))).sum()) if 'blocked' in dbg else -1}",
        fontsize=9,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    if idx == 0:
        ax.legend(loc="upper right", fontsize=7)
for idx in range(len(episodes), rows * cols):
    axes[idx // cols][idx % cols].axis("off")
success_rate = sum(ep["success_step"] is not None for ep in episodes) / len(episodes)
costful_rate = sum(ep["total_cost"] > 0 for ep in episodes) / len(episodes)
mean_cost = sum(ep["total_cost"] for ep in episodes) / len(episodes)
fig.suptitle(f"{run_name}: local-grid teacher | success={success_rate:.2f} costful={costful_rate:.2f} mean_cost={mean_cost:.2f}")
fig.tight_layout()
plot_path = out_dir / "local_grid_teacher_trajectory_contact_sheet.png"
fig.savefig(plot_path)
plt.close(fig)

summary = {
    "run_name": run_name,
    "plot": str(plot_path),
    "success_rate": success_rate,
    "costful_episode_rate": costful_rate,
    "mean_cost_sum": mean_cost,
    "episodes": [
        {
            "total_cost": ep["total_cost"],
            "success_step": ep["success_step"],
            "steps": ep["steps"],
            "blocked_corridor_cell_count": ep["layout"].get("blocked_cell_count"),
            "initial_blocked_scan_count": int(ep["snapshot_debug"]["blocked"].sum()) if ep["snapshot_debug"] else None,
            "initial_flat_ref": float(ep["snapshot_debug"]["flat_ref"]) if ep["snapshot_debug"] else None,
            "initial_target_angle": float(ep["snapshot_debug"]["target_angle"]) if ep["snapshot_debug"] else None,
            "initial_best_clearance": float(ep["snapshot_debug"]["best_clearance"]) if ep["snapshot_debug"] else None,
        }
        for ep in episodes
    ],
}
(out_dir / "local_grid_teacher_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary), flush=True)
PY

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
    controller_action,
    make_env,
)
from train_unitree_nav_thesis import DEFAULT_LOW_LEVEL, ROOT, ScanTeacherState, _extract_actor_obs, _extract_cost


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


def _scan_grid_world(obs: torch.Tensor, xy: np.ndarray, heading: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scan = obs[0, 9:58].detach().cpu().numpy().reshape(7, 7)
    coords = np.linspace(-1.5, 1.5, 7)
    c, s = math.cos(heading), math.sin(heading)
    pts = []
    vals = []
    local = []
    for i, forward in enumerate(coords):
        for j, lateral in enumerate(coords):
            local_xy = np.array([forward, lateral], dtype=np.float32)
            world = xy + np.array([c * forward - s * lateral, s * forward + c * lateral], dtype=np.float32)
            pts.append(world)
            vals.append(float(scan[i, j]))
            local.append(local_xy)
    return np.asarray(pts), np.asarray(vals), np.asarray(local)


def _action_world_vector(action: torch.Tensor, heading: float) -> np.ndarray:
    action_np = action[0, :3].detach().cpu().numpy()
    local = action_np[:2]
    c, s = math.cos(heading), math.sin(heading)
    world = np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]], dtype=np.float32)
    yaw = float(action_np[2])
    if np.linalg.norm(world) < 1e-6:
        # If teacher rotates in place, draw a short vector in the yaw direction so
        # the suggested turn remains visible in a static plot.
        turn_heading = heading + math.copysign(math.pi / 2.0, yaw if abs(yaw) > 1e-6 else 1.0)
        world = np.array([math.cos(turn_heading), math.sin(turn_heading)], dtype=np.float32) * min(0.35, abs(yaw))
    norm = np.linalg.norm(world)
    if norm > 1e-6:
        world = world / norm * 0.65
    return world


run_name = os.environ.get("RUN_NAME", f"unitree_height_scan_snapshot_{time.strftime('%Y%m%d_%H%M%S')}")
out_dir = Path(os.environ.get("OUTPUT_DIR", str(ROOT / "visualizations" / "unitree_height_scan_snapshots" / run_name)))
out_dir.mkdir(parents=True, exist_ok=True)

args = Namespace(
    controller="scan_teacher",
    model_path="",
    task=os.environ.get("TASK", "Unitree-G1-Nav-Obstacles-Safe-Collision"),
    device=os.environ.get("DEVICE", "cuda:0"),
    num_envs=1,
    num_episodes=1,
    episode_length_s=_env_float("EPISODE_LENGTH_S", 20.0),
    success_dist=_env_float("SUCCESS_DIST", 0.5),
    hidden_dim=256,
    use_layer_norm=False,
    low_level_policy_path=os.environ.get("LOW_LEVEL_POLICY_PATH", str(DEFAULT_LOW_LEVEL)),
    output_dir=str(out_dir),
    run_name=run_name,
    teacher_scan_block_threshold=_env_float("TEACHER_SCAN_BLOCK_THRESHOLD", 0.12),
    teacher_sector_half_width=_env_float("TEACHER_SECTOR_HALF_WIDTH", 0.45),
    teacher_align_angle=_env_float("TEACHER_ALIGN_ANGLE", 0.8),
    teacher_max_vx=_env_float("TEACHER_MAX_VX", 0.55),
    teacher_max_vy=_env_float("TEACHER_MAX_VY", 0.0),
    teacher_yaw_gain=_env_float("TEACHER_YAW_GAIN", 1.2),
    teacher_clearance_weight=_env_float("TEACHER_CLEARANCE_WEIGHT", 6.0),
    teacher_clearance_power=_env_float("TEACHER_CLEARANCE_POWER", 2.0),
    teacher_speed_clearance_scale=_env_float("TEACHER_SPEED_CLEARANCE_SCALE", 8.0),
    teacher_num_sectors=_env_int("TEACHER_NUM_SECTORS", 15),
    teacher_min_forward_scale=_env_float("TEACHER_MIN_FORWARD_SCALE", 0.15),
    teacher_escape_risk_threshold=_env_float("TEACHER_ESCAPE_RISK_THRESHOLD", 0.0),
    teacher_escape_forward_scale=_env_float("TEACHER_ESCAPE_FORWARD_SCALE", 0.0),
    teacher_escape_lateral_scale=_env_float("TEACHER_ESCAPE_LATERAL_SCALE", 1.0),
    teacher_escape_radius=_env_float("TEACHER_ESCAPE_RADIUS", 1.0),
    teacher_escape_all_directions=_env_bool("TEACHER_ESCAPE_ALL_DIRECTIONS", False),
    teacher_bypass_angle=_env_float("TEACHER_BYPASS_ANGLE", 0.0),
    teacher_goal_stop_dist=_env_float("TEACHER_GOAL_STOP_DIST", 0.0),
    teacher_wall_follow_steps=_env_int("TEACHER_WALL_FOLLOW_STEPS", 0),
    teacher_wall_follow_angle=_env_float("TEACHER_WALL_FOLLOW_ANGLE", 0.9),
    teacher_wall_follow_clear_risk=_env_float("TEACHER_WALL_FOLLOW_CLEAR_RISK", 0.15),
    teacher_rollout_horizon=_env_float("TEACHER_ROLLOUT_HORIZON", 1.6),
    teacher_rollout_clearance=_env_float("TEACHER_ROLLOUT_CLEARANCE", 0.75),
    teacher_rollout_samples=_env_int("TEACHER_ROLLOUT_SAMPLES", 8),
    teacher_rollout_clearance_weight=_env_float("TEACHER_ROLLOUT_CLEARANCE_WEIGHT", 25.0),
    teacher_rollout_forward_bias=_env_float("TEACHER_ROLLOUT_FORWARD_BIAS", 0.3),
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

snapshot_steps = [int(s) for s in os.environ.get("SNAPSHOT_STEPS", "0,60,120,200").split(",") if s.strip()]
max_step = max(snapshot_steps)
env = make_env(args, num_envs=1, render=False)
obs_raw, _, _, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
obs = _extract_actor_obs(obs_raw).to(device=args.device, dtype=torch.float32)
teacher_state = ScanTeacherState(1, torch.device(args.device))
teacher_state.reset(torch.tensor([0], device=args.device))

snapshots = []
for step in range(max_step + 1):
    if step in snapshot_steps:
        xy, heading = _robot_xy_heading(env)
        goal = _goal_positions_xy(env)[0].copy()
        action = controller_action(obs, args, None, teacher_state)
        scan_pts, scan_vals, scan_local = _scan_grid_world(obs, xy, heading)
        action_vec = _action_world_vector(action, heading)
        dist = float(_current_goal_distance(env, 1, torch.device(args.device))[0].detach().cpu().item())
        snapshots.append(
            {
                "step": step,
                "xy": xy.copy(),
                "heading": heading,
                "goal": goal,
                "action": action[0].detach().cpu().numpy().copy(),
                "action_vec": action_vec,
                "scan_pts": scan_pts,
                "scan_vals": scan_vals,
                "scan_local": scan_local,
                "goal_distance": dist,
            }
        )
    if step == max_step:
        break
    action = controller_action(obs, args, None, teacher_state)
    obs_raw, _, done, extras = env.step(action)
    obs = _extract_actor_obs(obs_raw).to(device=args.device, dtype=torch.float32)
    if bool(torch.as_tensor(done).reshape(-1)[0].detach().cpu().item()):
        break

obstacles = obstacle_cells[0]
cols = min(2, len(snapshots))
rows = math.ceil(len(snapshots) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(7.0 * cols, 6.6 * rows), dpi=150, squeeze=False)
for idx, snap in enumerate(snapshots):
    ax = axes[idx // cols][idx % cols]
    if obstacles.size:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], s=8, c="black", alpha=0.20, marker="s", label="privileged obstacle cells")
    blocked = snap["scan_vals"] < args.teacher_scan_block_threshold
    sc = ax.scatter(
        snap["scan_pts"][:, 0],
        snap["scan_pts"][:, 1],
        c=snap["scan_vals"],
        s=np.where(blocked, 95, 48),
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.25, float(np.nanmax(snap["scan_vals"]))),
        edgecolors=np.where(blocked, "red", "white"),
        linewidths=np.where(blocked, 1.8, 0.6),
        label="7x7 height scan points",
        zorder=4,
    )
    ax.scatter(snap["xy"][0], snap["xy"][1], c="white", edgecolors="black", s=130, zorder=5, label="robot")
    ax.arrow(
        snap["xy"][0],
        snap["xy"][1],
        math.cos(snap["heading"]) * 0.45,
        math.sin(snap["heading"]) * 0.45,
        color="deepskyblue",
        width=0.015,
        head_width=0.10,
        zorder=6,
        label="robot heading",
    )
    ax.arrow(
        snap["xy"][0],
        snap["xy"][1],
        snap["action_vec"][0],
        snap["action_vec"][1],
        color="orange",
        width=0.025,
        head_width=0.13,
        zorder=7,
        label="scan-teacher action",
    )
    ax.scatter(snap["goal"][0], snap["goal"][1], marker="*", s=260, c="limegreen", edgecolors="black", zorder=6, label="goal")
    ax.plot([snap["xy"][0], snap["goal"][0]], [snap["xy"][1], snap["goal"][1]], c="limegreen", alpha=0.3, linestyle="--")
    ax.set_title(
        f"step={snap['step']} dist={snap['goal_distance']:.2f} "
        f"action=[{snap['action'][0]:.2f}, {snap['action'][1]:.2f}, {snap['action'][2]:.2f}] "
        f"blocked_scan={int(blocked.sum())}/49",
        fontsize=9,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=7)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="scaled height_scan value")
for idx in range(len(snapshots), rows * cols):
    axes[idx // cols][idx % cols].axis("off")

fig.suptitle(
    f"Height scanner snapshots: red-bordered scan points are below threshold {args.teacher_scan_block_threshold}",
    fontsize=12,
)
fig.tight_layout()
plot_path = out_dir / "height_scan_teacher_snapshots.png"
fig.savefig(plot_path)
plt.close(fig)

summary = {
    "run_name": run_name,
    "plot": str(plot_path),
    "snapshot_steps": [s["step"] for s in snapshots],
    "layout_stats": layout_stats,
    "teacher_scan_block_threshold": args.teacher_scan_block_threshold,
    "snapshots": [
        {
            "step": s["step"],
            "xy": s["xy"].tolist(),
            "goal": s["goal"].tolist(),
            "heading": s["heading"],
            "goal_distance": s["goal_distance"],
            "action": s["action"].tolist(),
            "blocked_scan_count": int((s["scan_vals"] < args.teacher_scan_block_threshold).sum()),
            "scan_min": float(np.min(s["scan_vals"])),
            "scan_max": float(np.max(s["scan_vals"])),
            "scan_mean": float(np.mean(s["scan_vals"])),
        }
        for s in snapshots
    ],
}
(out_dir / "height_scan_teacher_snapshots.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary), flush=True)
PY

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
    controller_action,
    make_env,
)
from train_unitree_nav_thesis import DEFAULT_LOW_LEVEL, ROOT, ScanTeacherState, _extract_actor_obs


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


def _sensor(env):
    unwrapped = env.env.unwrapped
    try:
        return unwrapped.scene.sensors["terrain_scan"]
    except Exception:
        return unwrapped.scene["terrain_scan"]


def _action_world_vec(action: torch.Tensor, heading: float) -> np.ndarray:
    a = action[0, :3].detach().cpu().numpy()
    local = a[:2]
    c, s = math.cos(heading), math.sin(heading)
    vec = np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]], dtype=np.float32)
    if np.linalg.norm(vec) < 1e-6:
        turn_heading = heading + math.copysign(math.pi / 2.0, float(a[2]) if abs(float(a[2])) > 1e-6 else 1.0)
        vec = np.array([math.cos(turn_heading), math.sin(turn_heading)], dtype=np.float32) * min(0.35, abs(float(a[2])))
    n = np.linalg.norm(vec)
    return vec / n * 0.7 if n > 1e-6 else vec


run_name = os.environ.get("RUN_NAME", f"unitree_actual_scan_hits_{time.strftime('%Y%m%d_%H%M%S')}")
out_dir = Path(os.environ.get("OUTPUT_DIR", str(ROOT / "visualizations" / "unitree_height_scan_actual_hits" / run_name)))
out_dir.mkdir(parents=True, exist_ok=True)

args = Namespace(
    controller="scan_teacher",
    model_path="",
    task=os.environ.get("TASK", "Unitree-G1-Nav-Obstacles-Safe-Collision"),
    device=os.environ.get("DEVICE", "cuda:0"),
    num_envs=1,
    num_episodes=1,
    episode_length_s=20.0,
    success_dist=0.5,
    hidden_dim=256,
    use_layer_norm=False,
    low_level_policy_path=os.environ.get("LOW_LEVEL_POLICY_PATH", str(DEFAULT_LOW_LEVEL)),
    output_dir=str(out_dir),
    run_name=run_name,
    teacher_scan_block_threshold=_env_float("TEACHER_SCAN_BLOCK_THRESHOLD", 0.12),
    teacher_sector_half_width=0.45,
    teacher_align_angle=0.8,
    teacher_max_vx=0.55,
    teacher_max_vy=0.0,
    teacher_yaw_gain=1.2,
    teacher_clearance_weight=6.0,
    teacher_clearance_power=2.0,
    teacher_speed_clearance_scale=8.0,
    teacher_num_sectors=15,
    teacher_min_forward_scale=0.15,
    teacher_escape_risk_threshold=0.0,
    teacher_escape_forward_scale=0.0,
    teacher_escape_lateral_scale=1.0,
    teacher_escape_radius=1.0,
    teacher_escape_all_directions=False,
    teacher_bypass_angle=0.0,
    teacher_goal_stop_dist=0.0,
    teacher_wall_follow_steps=0,
    teacher_wall_follow_angle=0.9,
    teacher_wall_follow_clear_risk=0.15,
    teacher_rollout_horizon=1.6,
    teacher_rollout_clearance=0.75,
    teacher_rollout_samples=8,
    teacher_rollout_clearance_weight=25.0,
    teacher_rollout_forward_bias=0.3,
    min_goal_obstacle_clearance=0.0,
    goal_clearance_resample_attempts=100,
    min_start_obstacle_clearance=0.0,
    start_clearance_resample_attempts=100,
    require_blocked_corridor=_env_bool("REQUIRE_BLOCKED_CORRIDOR", False),
    blocked_corridor_radius=0.45,
    blocked_corridor_ignore_end_radius=0.75,
    blocked_corridor_min_cells=1,
    blocked_corridor_resample_attempts=200,
    debug_obstacle_width_min=0.9,
    debug_obstacle_width_max=1.2,
    debug_obstacle_height_min=0.45,
    debug_obstacle_height_max=0.55,
    debug_num_obstacles=12,
    debug_platform_width=2.0,
    debug_obstacle_border_width=0.5,
    debug_goal_through_obstacle=True,
    debug_goal_distance=3.4,
    debug_goal_obstacle_min_dist=0.9,
    debug_goal_obstacle_max_dist=2.2,
)

snapshot_steps = [int(s) for s in os.environ.get("SNAPSHOT_STEPS", "0,60,120").split(",") if s.strip()]
env = make_env(args, num_envs=1, render=False)
obs_raw, _, _, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
obs = _extract_actor_obs(obs_raw).to(device=args.device, dtype=torch.float32)
teacher_state = ScanTeacherState(1, torch.device(args.device))
teacher_state.reset(torch.tensor([0], device=args.device))

snapshots = []
max_step = max(snapshot_steps)
for step in range(max_step + 1):
    if step in snapshot_steps:
        xy, heading = _robot_xy_heading(env)
        goal = _goal_positions_xy(env)[0].copy()
        sensor = _sensor(env)
        hits = sensor.data.hit_pos_w[0].detach().cpu().numpy().copy()
        distances = sensor.data.distances[0].detach().cpu().numpy().copy()
        obs_scan = obs[0, 9:58].detach().cpu().numpy().copy()
        action = controller_action(obs, args, None, teacher_state)
        snapshots.append(
            {
                "step": step,
                "xy": xy,
                "heading": heading,
                "goal": goal,
                "hits": hits,
                "distances": distances,
                "obs_scan": obs_scan,
                "action": action[0].detach().cpu().numpy().copy(),
                "action_vec": _action_world_vec(action, heading),
                "goal_distance": float(_current_goal_distance(env, 1, torch.device(args.device))[0].detach().cpu().item()),
            }
        )
    if step == max_step:
        break
    action = controller_action(obs, args, None, teacher_state)
    obs_raw, _, done, _ = env.step(action)
    obs = _extract_actor_obs(obs_raw).to(device=args.device, dtype=torch.float32)
    if bool(torch.as_tensor(done).reshape(-1)[0].detach().cpu().item()):
        break

obstacles = obstacle_cells[0]
fig, axes = plt.subplots(1, len(snapshots), figsize=(7.2 * len(snapshots), 6.6), dpi=150, squeeze=False)
for idx, snap in enumerate(snapshots):
    ax = axes[0][idx]
    hits_xy = snap["hits"][:, :2]
    vals = snap["obs_scan"]
    blocked = vals < args.teacher_scan_block_threshold
    if obstacles.size:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], s=8, c="black", alpha=0.18, marker="s", label="obstacle hfield cells")
    ax.scatter(
        hits_xy[:, 0],
        hits_xy[:, 1],
        c=vals,
        s=np.where(blocked, 105, 50),
        cmap="viridis",
        edgecolors=np.where(blocked, "red", "white"),
        linewidths=np.where(blocked, 1.8, 0.6),
        zorder=4,
        label="ACTUAL ray hit positions",
    )
    # Label a sparse set plus all blocked cells with flattened/r,c index.
    for k, (p, is_blocked) in enumerate(zip(hits_xy, blocked)):
        if is_blocked or k in {0, 6, 24, 42, 48}:
            r, c = divmod(k, 7)
            ax.text(p[0], p[1], f"{r},{c}", fontsize=6, color="black", ha="center", va="center", zorder=8)
    ax.scatter(snap["xy"][0], snap["xy"][1], c="white", edgecolors="black", s=120, zorder=5, label="robot")
    ax.scatter(snap["goal"][0], snap["goal"][1], marker="*", s=260, c="limegreen", edgecolors="black", zorder=5, label="goal")
    ax.arrow(snap["xy"][0], snap["xy"][1], math.cos(snap["heading"]) * 0.45, math.sin(snap["heading"]) * 0.45, color="deepskyblue", width=0.012, head_width=0.08, zorder=6, label="heading")
    ax.arrow(snap["xy"][0], snap["xy"][1], snap["action_vec"][0], snap["action_vec"][1], color="orange", width=0.022, head_width=0.12, zorder=7, label="current scan teacher action")
    max_obs_diff = float(np.max(np.abs(snap["obs_scan"] - snap["distances"] * 0.2)))
    ax.set_title(
        f"step {snap['step']} | actual ray hits | blocked={int(blocked.sum())}/49\n"
        f"obs≈dist*0.2 maxerr={max_obs_diff:.2e} action={np.round(snap['action'], 2).tolist()}",
        fontsize=9,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=7)
fig.suptitle("Height scan using ACTUAL terrain_scan.data.hit_pos_w, no guessed rotation")
fig.tight_layout()
plot_path = out_dir / "height_scan_actual_ray_hits.png"
fig.savefig(plot_path)
plt.close(fig)

summary = {
    "plot": str(plot_path),
    "layout_stats": layout_stats,
    "snapshots": [
        {
            "step": int(s["step"]),
            "goal_distance": float(s["goal_distance"]),
            "blocked_count": int((s["obs_scan"] < args.teacher_scan_block_threshold).sum()),
            "action": s["action"].tolist(),
            "obs_dist_scale_maxerr": float(np.max(np.abs(s["obs_scan"] - s["distances"] * 0.2))),
            "hit_xy_first7": s["hits"][:7, :2].tolist(),
        }
        for s in snapshots
    ],
}
(out_dir / "height_scan_actual_ray_hits.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary), flush=True)
PY

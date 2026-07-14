#!/usr/bin/env python3
"""Diagnose a Unitree navigation teacher from actual terrain-scan ray hits.

This is intentionally a debugging/evidence tool.  It uses the same height-scan
values exposed to the actor, but plots and plans through the simulator-provided
ray hit positions (`terrain_scan.data.hit_pos_w`) so coordinate-frame mistakes
are visible instead of hidden behind guessed 7x7 grid geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_unitree_nav_baselines import (
    DEFAULT_LOW_LEVEL,
    _current_goal_distance,
    _extract_actor_obs,
    _extract_cost,
    _layout_blocked_corridor_stats,
    _reset_until_feasible,
    _terrain_obstacle_cells_by_env,
    make_env,
)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "data") and hasattr(value.data, "detach"):
        return value.data.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _robot_xy_heading(env) -> tuple[np.ndarray, np.ndarray]:
    robot = env.env.unwrapped.scene["robot"]
    xy = _to_numpy(robot.data.root_link_pos_w)[:, :2].copy()
    heading = _to_numpy(robot.data.heading_w).reshape(-1).copy()
    return xy, heading


def _goal_xy(env) -> np.ndarray:
    term = env.env.unwrapped.command_manager._terms["pose"]
    return _to_numpy(term._goal_pos_w)[:, :2].copy()


def _scan_hit_xy(env) -> np.ndarray:
    sensor = env.env.unwrapped.scene.sensors.get("terrain_scan")
    if sensor is None or not hasattr(sensor.data, "hit_pos_w"):
        raise RuntimeError("terrain_scan.data.hit_pos_w is unavailable")
    return _to_numpy(sensor.data.hit_pos_w)[:, :, :2].copy()


def _body_from_world(points_w: np.ndarray, robot_xy: np.ndarray, heading: float) -> np.ndarray:
    rel = points_w - robot_xy.reshape(1, 2)
    c = math.cos(-heading)
    s = math.sin(-heading)
    return np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], axis=-1)


def _world_from_body(points_b: np.ndarray, robot_xy: np.ndarray, heading: float) -> np.ndarray:
    c = math.cos(heading)
    s = math.sin(heading)
    return robot_xy.reshape(1, 2) + np.stack(
        [c * points_b[:, 0] - s * points_b[:, 1], s * points_b[:, 0] + c * points_b[:, 1]],
        axis=-1,
    )


def _angle_wrap(x: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(x), np.cos(x))


def _actor_goal_xy_body(obs: torch.Tensor, env_idx: int) -> np.ndarray:
    return obs[env_idx, 6:8].detach().cpu().numpy().astype(np.float64)


def actual_hit_teacher_action(
    obs: torch.Tensor,
    hit_xy_w: np.ndarray,
    robot_xy: np.ndarray,
    heading: float,
    *,
    env_idx: int,
    scan_block_threshold: float,
    scan_flat_value: float,
    scan_block_delta: float,
    candidate_count: int,
    max_angle: float,
    lookahead: float,
    corridor_width: float,
    clearance_soft_width: float,
    risk_weight: float,
    blocked_penalty: float,
    angle_weight: float,
    forward_bias: float,
    max_vx: float,
    max_vy: float,
    yaw_gain: float,
    align_angle: float,
    min_forward_scale: float,
    goal_stop_dist: float,
    emergency_radius: float,
    emergency_speed_scale: float,
    emergency_repulsion_weight: float,
    emergency_tangent_weight: float,
    emergency_goal_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    scan = obs[env_idx, 9:58].detach().cpu().numpy().astype(np.float64)
    local = _body_from_world(hit_xy_w[env_idx], robot_xy, heading)
    goal_b = _actor_goal_xy_body(obs, env_idx)
    goal_dist = float(np.linalg.norm(goal_b))
    goal_angle = float(math.atan2(goal_b[1], goal_b[0])) if goal_dist > 1e-6 else 0.0

    flat_ref = float(scan_flat_value)
    if flat_ref <= 0.0:
        flat_ref = float(np.nanpercentile(scan, 90.0))
    blocked = (scan < float(scan_block_threshold)) | (scan < flat_ref - float(scan_block_delta))
    finite = np.isfinite(scan) & np.all(np.isfinite(local), axis=1)
    blocked &= finite

    candidate_angles = np.linspace(-float(max_angle), float(max_angle), int(candidate_count), dtype=np.float64)
    goal_unit = goal_b / max(goal_dist, 1e-6)
    scores = []
    risks = []
    blocked_counts = []
    progress_terms = []
    blocked_points = local[blocked]
    for angle in candidate_angles:
        d = np.array([math.cos(float(angle)), math.sin(float(angle))], dtype=np.float64)
        if blocked_points.size:
            t = blocked_points @ d
            lateral = np.abs(blocked_points[:, 0] * d[1] - blocked_points[:, 1] * d[0])
            in_front = (t > 0.05) & (t < float(lookahead))
            hard = in_front & (lateral < float(corridor_width))
            soft = np.exp(-0.5 * (lateral / max(float(clearance_soft_width), 1e-6)) ** 2) * np.exp(
                -np.clip(t, 0.0, None) / max(float(lookahead), 1e-6)
            )
            soft = np.where(in_front, soft, 0.0)
            risk = float(np.sum(soft))
            hard_count = int(np.sum(hard))
        else:
            risk = 0.0
            hard_count = 0
        endpoint = d * min(float(lookahead), max(goal_dist, 0.25))
        endpoint_dist = float(np.linalg.norm(goal_b - endpoint))
        angle_err = abs(float(_angle_wrap(angle - goal_angle)))
        progress = float(goal_unit @ d)
        # Favor directions that preserve goal progress, but allow temporary
        # side-steps when the direct corridor is blocked.
        progress_penalty = max(0.0, -progress) * 3.0 + max(0.0, 0.25 - progress)
        score = (
            endpoint_dist
            + float(risk_weight) * risk
            + float(blocked_penalty) * hard_count
            + float(angle_weight) * angle_err
            + progress_penalty
            - float(forward_bias) * progress
        )
        scores.append(score)
        risks.append(risk)
        blocked_counts.append(hard_count)
        progress_terms.append(progress)

    scores_np = np.asarray(scores)
    best_idx = int(np.argmin(scores_np))
    target_angle = float(candidate_angles[best_idx])
    emergency_active = False
    emergency_min_dist = float("inf")
    if blocked_points.size:
        radius = np.linalg.norm(blocked_points, axis=1)
        emergency_min_dist = float(np.min(radius))
        close = radius < float(emergency_radius)
        if np.any(close):
            weights = (float(emergency_radius) - radius[close]).clip(min=0.0) / max(float(emergency_radius), 1e-6)
            weights = weights**2 + 1e-3
            centroid = np.sum(blocked_points[close] * weights.reshape(-1, 1), axis=0) / max(float(np.sum(weights)), 1e-6)
            away = -centroid / max(float(np.linalg.norm(centroid)), 1e-6)
            tangent_a = np.array([-centroid[1], centroid[0]], dtype=np.float64)
            tangent_a /= max(float(np.linalg.norm(tangent_a)), 1e-6)
            tangent_b = -tangent_a
            tangent = tangent_a if float(tangent_a @ goal_unit) >= float(tangent_b @ goal_unit) else tangent_b
            blended = (
                float(emergency_repulsion_weight) * away
                + float(emergency_tangent_weight) * tangent
                + float(emergency_goal_weight) * goal_unit
            )
            if np.linalg.norm(blended) > 1e-6:
                target_angle = float(math.atan2(blended[1], blended[0]))
                emergency_active = True
    yaw = float(np.clip(float(yaw_gain) * target_angle, -1.0, 1.0))
    speed_gate = 1.0 if abs(target_angle) <= float(align_angle) else float(min_forward_scale)
    if emergency_active:
        vx = float(max_vx) * float(emergency_speed_scale) * math.cos(target_angle)
        vy = float(max_vy) * float(emergency_speed_scale) * math.sin(target_angle)
    else:
        vx = float(max_vx) * max(0.0, math.cos(target_angle)) * speed_gate
        vy = float(max_vy) * math.sin(target_angle)
    if goal_dist <= float(goal_stop_dist):
        vx = 0.0
        vy = 0.0
        yaw = 0.0
    action = torch.as_tensor([[vx, vy, yaw]], device=obs.device, dtype=torch.float32)
    diag = {
        "goal_b": goal_b.tolist(),
        "goal_angle": goal_angle,
        "goal_dist": goal_dist,
        "target_angle": target_angle,
        "target_angle_deg": math.degrees(target_angle),
        "target_score": float(scores_np[best_idx]),
        "target_risk": float(risks[best_idx]),
        "target_blocked_count": int(blocked_counts[best_idx]),
        "target_progress": float(progress_terms[best_idx]),
        "emergency_active": bool(emergency_active),
        "emergency_min_dist": emergency_min_dist,
        "blocked_count": int(np.sum(blocked)),
        "scan_min": float(np.nanmin(scan)),
        "scan_p90": float(np.nanpercentile(scan, 90.0)),
        "local": local.tolist(),
        "blocked": blocked.astype(bool).tolist(),
        "candidate_angles": candidate_angles.tolist(),
        "candidate_scores": scores_np.tolist(),
    }
    return action.clamp(-1.0, 1.0), diag


def _plot_run(out_dir: Path, episodes: list[dict[str, Any]], title: str) -> Path:
    n = min(len(episodes), 6)
    cols = min(3, max(1, n))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for idx, ep in enumerate(episodes[:n]):
        ax = axes.flat[idx]
        ax.axis("on")
        obs_cells = np.asarray(ep["obstacle_cells"], dtype=np.float64)
        if obs_cells.size:
            ax.scatter(obs_cells[:, 0], obs_cells[:, 1], s=8, c="0.15", alpha=0.35, label="heightfield obstacle")
        traj = np.asarray(ep["trajectory"], dtype=np.float64)
        ax.plot(traj[:, 0], traj[:, 1], color="#1f77b4", linewidth=2.0, label="trajectory")
        if len(traj):
            ax.scatter(traj[0, 0], traj[0, 1], s=80, c="limegreen", edgecolors="k", label="start")
            ax.scatter(traj[-1, 0], traj[-1, 1], s=60, c="#1f77b4", edgecolors="k", label="end")
        goal = np.asarray(ep["goal"], dtype=np.float64)
        ax.scatter(goal[0], goal[1], s=120, marker="*", c="gold", edgecolors="k", label="goal")
        costs = np.asarray(ep["cost_points"], dtype=np.float64)
        if costs.size:
            ax.scatter(costs[:, 0], costs[:, 1], s=90, marker="x", c="red", linewidths=2.5, label="cost")
        snaps = ep.get("snapshots", [])
        if snaps:
            snap = snaps[0]
            hit_w = np.asarray(snap["hit_xy_w"], dtype=np.float64)
            blocked = np.asarray(snap["blocked"], dtype=bool)
            if hit_w.size:
                ax.scatter(hit_w[~blocked, 0], hit_w[~blocked, 1], s=14, c="lightgray", alpha=0.65, label="free scan rays")
                ax.scatter(hit_w[blocked, 0], hit_w[blocked, 1], s=30, c="crimson", alpha=0.9, label="blocked scan rays")
            robot = np.asarray(snap["robot_xy"], dtype=np.float64)
            heading = float(snap["heading"])
            target = float(snap["target_angle"])
            vec = _world_from_body(
                np.asarray([[math.cos(target), math.sin(target)]], dtype=np.float64),
                robot,
                heading,
            )[0] - robot
            ax.arrow(robot[0], robot[1], vec[0], vec[1], color="orange", width=0.025, head_width=0.16, length_includes_head=True)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"ep {idx}: success={ep['success']} cost={ep['cost_sum']:.1f} "
            f"steps={ep['steps']} blocked={ep.get('blocked_corridor')}"
        )
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    path = out_dir / "actual_hit_teacher_trajectory_contact_sheet.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_snapshots(out_dir: Path, snapshots: list[dict[str, Any]], title: str) -> Path:
    n = min(len(snapshots), 9)
    cols = 3
    rows = int(math.ceil(max(1, n) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, snap in enumerate(snapshots[:n]):
        ax = axes.flat[i]
        ax.axis("on")
        obs_cells = np.asarray(snap["obstacle_cells"], dtype=np.float64)
        if obs_cells.size:
            ax.scatter(obs_cells[:, 0], obs_cells[:, 1], s=8, c="0.2", alpha=0.35)
        hit_w = np.asarray(snap["hit_xy_w"], dtype=np.float64)
        blocked = np.asarray(snap["blocked"], dtype=bool)
        ax.scatter(hit_w[~blocked, 0], hit_w[~blocked, 1], s=20, c="lightgray", alpha=0.7)
        ax.scatter(hit_w[blocked, 0], hit_w[blocked, 1], s=42, c="crimson", alpha=0.95)
        robot = np.asarray(snap["robot_xy"], dtype=np.float64)
        goal = np.asarray(snap["goal"], dtype=np.float64)
        ax.scatter(robot[0], robot[1], s=85, c="deepskyblue", edgecolors="k")
        ax.scatter(goal[0], goal[1], s=120, marker="*", c="gold", edgecolors="k")
        heading = float(snap["heading"])
        target = float(snap["target_angle"])
        forward = _world_from_body(np.asarray([[1.0, 0.0]]), robot, heading)[0] - robot
        selected = _world_from_body(np.asarray([[math.cos(target), math.sin(target)]]), robot, heading)[0] - robot
        ax.arrow(robot[0], robot[1], forward[0], forward[1], color="black", width=0.015, head_width=0.12)
        ax.arrow(robot[0], robot[1], selected[0], selected[1], color="orange", width=0.025, head_width=0.18)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"ep{snap['episode']} step{snap['step']} target={snap['target_angle_deg']:.0f}deg "
            f"blocked={snap['blocked_count']} risk={snap['target_risk']:.2f}"
        )
    fig.suptitle(title)
    fig.tight_layout()
    path = out_dir / "actual_hit_teacher_snapshots.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    print(f"[actual-hit-teacher] output_dir={out_dir}", flush=True)
    print("[actual-hit-teacher] creating env", flush=True)
    env = make_env(args, num_envs=1, render=False)
    print("[actual-hit-teacher] reset/filtering layout", flush=True)
    obs_raw, start_clearances, goal_clearances, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
    print(
        "[actual-hit-teacher] layout "
        f"start_clearance={float(start_clearances[0].detach().cpu().item()) if start_clearances is not None else None} "
        f"goal_clearance={float(goal_clearances[0].detach().cpu().item()) if goal_clearances is not None else None} "
        f"blocked={layout_stats[0] if layout_stats is not None else None}",
        flush=True,
    )
    obs = _extract_actor_obs(obs_raw).to(args.device, dtype=torch.float32)

    episodes: list[dict[str, Any]] = []
    global_snapshots: list[dict[str, Any]] = []
    ep_return = 0.0
    ep_cost = 0.0
    ep_collision_steps = 0
    ep_success = False
    ep_first_success_step: int | None = None
    ep_steps = 0
    traj: list[list[float]] = []
    cost_points: list[list[float]] = []
    ep_snaps: list[dict[str, Any]] = []
    step_dt = 0.05
    start_time = time.time()

    while len(episodes) < int(args.num_episodes):
        robot_xy_all, heading_all = _robot_xy_heading(env)
        goal_all = _goal_xy(env)
        hit_xy_w = _scan_hit_xy(env)
        robot_xy = robot_xy_all[0]
        heading = float(heading_all[0])
        goal = goal_all[0]
        traj.append(robot_xy.tolist())

        pre_dist = float(_current_goal_distance(env, 1, torch.device(args.device))[0].detach().cpu().item())
        if pre_dist <= float(args.success_dist) and ep_first_success_step is None:
            ep_first_success_step = ep_steps
        ep_success = ep_success or pre_dist <= float(args.success_dist)

        with torch.no_grad():
            action, diag = actual_hit_teacher_action(
                obs,
                hit_xy_w,
                robot_xy,
                heading,
                env_idx=0,
                scan_block_threshold=args.scan_block_threshold,
                scan_flat_value=args.scan_flat_value,
                scan_block_delta=args.scan_block_delta,
                candidate_count=args.candidate_count,
                max_angle=args.max_angle,
                lookahead=args.lookahead,
                corridor_width=args.corridor_width,
                clearance_soft_width=args.clearance_soft_width,
                risk_weight=args.risk_weight,
                blocked_penalty=args.blocked_penalty,
                angle_weight=args.angle_weight,
                forward_bias=args.forward_bias,
                max_vx=args.max_vx,
                max_vy=args.max_vy,
                yaw_gain=args.yaw_gain,
                align_angle=args.align_angle,
                min_forward_scale=args.min_forward_scale,
                goal_stop_dist=args.goal_stop_dist,
                emergency_radius=args.emergency_radius,
                emergency_speed_scale=args.emergency_speed_scale,
                emergency_repulsion_weight=args.emergency_repulsion_weight,
                emergency_tangent_weight=args.emergency_tangent_weight,
                emergency_goal_weight=args.emergency_goal_weight,
            )

        if ep_steps in set(int(x) for x in args.snapshot_steps.split(",") if x.strip()):
            snap = {
                "episode": len(episodes),
                "step": ep_steps,
                "robot_xy": robot_xy.tolist(),
                "heading": heading,
                "goal": goal.tolist(),
                "hit_xy_w": hit_xy_w[0].tolist(),
                "obstacle_cells": obstacle_cells[0].tolist(),
                **{k: v for k, v in diag.items() if k not in {"local", "candidate_angles", "candidate_scores"}},
            }
            ep_snaps.append(snap)
            global_snapshots.append(snap)

        next_raw, reward, done, extras = env.step(action)
        next_obs = _extract_actor_obs(next_raw).to(args.device, dtype=torch.float32)
        cost = float(_extract_cost(extras, 1, torch.device(args.device)).reshape(-1).sum().detach().cpu().item())
        ep_steps += 1
        ep_return += float(reward.reshape(-1)[0].detach().cpu().item())
        ep_cost += cost
        if cost > 0.0:
            ep_collision_steps += 1
            cost_points.append(robot_xy.tolist())
        dist = float(torch.linalg.norm(next_obs[0, 6:8]).detach().cpu().item())
        if dist <= float(args.success_dist) and ep_first_success_step is None:
            ep_first_success_step = ep_steps
        ep_success = ep_success or dist <= float(args.success_dist)

        if bool(done.reshape(-1)[0].detach().cpu().item()):
            print(
                "[actual-hit-teacher] episode "
                f"{len(episodes)} success={ep_success} cost={ep_cost:.1f} "
                f"collision_steps={ep_collision_steps} steps={ep_steps}",
                flush=True,
            )
            episodes.append(
                {
                    "success": bool(ep_success),
                    "time_to_success_s": None if ep_first_success_step is None else ep_first_success_step * step_dt,
                    "episode_length_s": ep_steps * step_dt,
                    "steps": ep_steps,
                    "cost_sum": ep_cost,
                    "collision_steps": ep_collision_steps,
                    "return": ep_return,
                    "trajectory": traj,
                    "cost_points": cost_points,
                    "goal": goal.tolist(),
                    "obstacle_cells": obstacle_cells[0].tolist(),
                    "snapshots": ep_snaps,
                    "start_obstacle_clearance": (
                        float(start_clearances[0].detach().cpu().item()) if start_clearances is not None else None
                    ),
                    "goal_obstacle_clearance": (
                        float(goal_clearances[0].detach().cpu().item()) if goal_clearances is not None else None
                    ),
                    "blocked_corridor": bool(layout_stats[0]["blocked"]) if layout_stats is not None else None,
                    "blocked_corridor_cell_count": (
                        int(layout_stats[0]["blocked_cell_count"]) if layout_stats is not None else None
                    ),
                    "nearest_corridor_obstacle_dist": (
                        float(layout_stats[0]["nearest_corridor_obstacle_dist"]) if layout_stats is not None else None
                    ),
                }
            )
            try:
                obs_raw, start_clearances, goal_clearances, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
            except Exception as exc:
                print(f"[actual-hit-teacher] stopping early: reset failed after partial episodes: {exc}", flush=True)
                break
            obs = _extract_actor_obs(obs_raw).to(args.device, dtype=torch.float32)
            ep_return = 0.0
            ep_cost = 0.0
            ep_collision_steps = 0
            ep_success = False
            ep_first_success_step = None
            ep_steps = 0
            traj = []
            cost_points = []
            ep_snaps = []
            continue
        obs = next_obs

    env.close()
    costs = [float(ep["cost_sum"]) for ep in episodes]
    successes = [float(ep["success"]) for ep in episodes]
    collision_steps = [float(ep["collision_steps"]) for ep in episodes]
    ttfs = [float(ep["time_to_success_s"]) for ep in episodes if ep["time_to_success_s"] is not None]
    summary = {
        "controller": "actual_hit_scan_teacher",
        "episodes": len(episodes),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_cost_sum": float(np.mean(costs)) if costs else 0.0,
        "costful_episode_rate": float(np.mean([c > 0.0 for c in costs])) if costs else 0.0,
        "mean_collision_steps": float(np.mean(collision_steps)) if collision_steps else 0.0,
        "mean_time_to_success_s": float(np.mean(ttfs)) if ttfs else None,
        "elapsed_s": time.time() - start_time,
        "episodes_detail": episodes,
    }
    traj_path = _plot_run(out_dir, episodes, f"Actual-hit scan teacher: {args.run_name}")
    snap_path = _plot_snapshots(out_dir, global_snapshots, f"Actual-hit scan snapshots: {args.run_name}")
    summary["trajectory_plot"] = str(traj_path)
    summary["snapshot_plot"] = str(snap_path)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes_detail"}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Unitree-G1-Nav-Obstacles-Safe-Collision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=6)
    parser.add_argument("--episode-length-s", type=float, default=16.0)
    parser.add_argument("--success-dist", type=float, default=0.5)
    parser.add_argument("--low-level-policy-path", default=str(DEFAULT_LOW_LEVEL))
    parser.add_argument("--output-dir", default=str(ROOT / "visualizations" / "unitree_actual_hit_teacher"))
    parser.add_argument("--run-name", default=f"actual_hit_teacher_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--scan-block-threshold", type=float, default=0.12)
    parser.add_argument("--scan-flat-value", type=float, default=0.16)
    parser.add_argument("--scan-block-delta", type=float, default=0.025)
    parser.add_argument("--candidate-count", type=int, default=41)
    parser.add_argument("--max-angle", type=float, default=2.35)
    parser.add_argument("--lookahead", type=float, default=2.3)
    parser.add_argument("--corridor-width", type=float, default=0.65)
    parser.add_argument("--clearance-soft-width", type=float, default=0.9)
    parser.add_argument("--risk-weight", type=float, default=4.0)
    parser.add_argument("--blocked-penalty", type=float, default=18.0)
    parser.add_argument("--angle-weight", type=float, default=0.18)
    parser.add_argument("--forward-bias", type=float, default=0.35)
    parser.add_argument("--max-vx", type=float, default=0.65)
    parser.add_argument("--max-vy", type=float, default=0.35)
    parser.add_argument("--yaw-gain", type=float, default=1.2)
    parser.add_argument("--align-angle", type=float, default=0.8)
    parser.add_argument("--min-forward-scale", type=float, default=0.2)
    parser.add_argument("--goal-stop-dist", type=float, default=0.25)
    parser.add_argument("--emergency-radius", type=float, default=1.15)
    parser.add_argument("--emergency-speed-scale", type=float, default=0.75)
    parser.add_argument("--emergency-repulsion-weight", type=float, default=0.8)
    parser.add_argument("--emergency-tangent-weight", type=float, default=1.2)
    parser.add_argument("--emergency-goal-weight", type=float, default=0.4)
    parser.add_argument("--snapshot-steps", default="0,40,80,120")
    parser.add_argument("--min-goal-obstacle-clearance", type=float, default=0.75)
    parser.add_argument("--goal-clearance-resample-attempts", type=int, default=80)
    parser.add_argument("--min-start-obstacle-clearance", type=float, default=0.75)
    parser.add_argument("--start-clearance-resample-attempts", type=int, default=40)
    parser.add_argument("--require-blocked-corridor", action="store_true")
    parser.add_argument("--blocked-corridor-radius", type=float, default=0.55)
    parser.add_argument("--blocked-corridor-ignore-end-radius", type=float, default=0.75)
    parser.add_argument("--blocked-corridor-min-cells", type=int, default=1)
    parser.add_argument("--blocked-corridor-resample-attempts", type=int, default=120)
    parser.add_argument("--debug-obstacle-width-min", type=float, default=0.0)
    parser.add_argument("--debug-obstacle-width-max", type=float, default=0.0)
    parser.add_argument("--debug-obstacle-height-min", type=float, default=0.0)
    parser.add_argument("--debug-obstacle-height-max", type=float, default=0.0)
    parser.add_argument("--debug-num-obstacles", type=int, default=0)
    parser.add_argument("--debug-platform-width", type=float, default=0.0)
    parser.add_argument("--debug-obstacle-border-width", type=float, default=0.0)
    parser.add_argument("--debug-goal-through-obstacle", action="store_true")
    parser.add_argument("--debug-goal-distance", type=float, default=3.2)
    parser.add_argument("--debug-goal-obstacle-min-dist", type=float, default=0.8)
    parser.add_argument("--debug-goal-obstacle-max-dist", type=float, default=2.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.num_envs) != 1:
        raise ValueError("This diagnostic uses num-envs=1 so per-episode plots stay unambiguous")
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot Unitree navigation reset layouts before running a policy.

This is a privileged diagnostic tool: it visualizes terrain obstacle cells,
agent pose, goal pose, straight-line corridors, and an estimated robot collision
footprint. The controller/training code must not use these privileged geometry
signals.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval_unitree_nav_baselines import (
    _goal_positions_xy,
    _line_obstacle_stats,
    _reset_until_feasible,
    _robot_positions_xy,
    _terrain_obstacle_cells_by_env,
    _to_numpy,
    make_env,
)
from plot_unitree_nav_rollout import _active_terrain_heightfield, _corridor_obstacle_points
from train_unitree_nav_thesis import DEFAULT_LOW_LEVEL, ROOT


def _geom_model_and_data(env):
    sim = env.env.unwrapped.sim
    model = getattr(sim, "mj_model", None) or getattr(sim, "model", None)
    data = getattr(sim, "mj_data", None) or getattr(sim, "data", None)
    if model is None or data is None:
        raise RuntimeError("Could not access MuJoCo model/data from Unitree env")
    return model, data


def _geom_name(model, geom_id: int) -> str:
    try:
        import mujoco

        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
    except Exception:
        return f"geom_{geom_id}"


def _body_name(model, body_id: int) -> str:
    try:
        import mujoco

        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or f"body_{body_id}"
    except Exception:
        return f"body_{body_id}"


def _robot_collision_geom_xy(env, root_xy: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return approximate world XY centers/radii of robot collision geoms."""
    model, data = _geom_model_and_data(env)
    geom_xpos = _to_numpy(getattr(data, "geom_xpos"))
    geom_rbound = _to_numpy(getattr(model, "geom_rbound"))
    geom_group = _to_numpy(getattr(model, "geom_group")).astype(int)
    geom_bodyid = _to_numpy(getattr(model, "geom_bodyid")).astype(int)

    rows: list[list[float]] = []
    details: list[dict[str, Any]] = []
    for geom_id in range(int(getattr(model, "ngeom"))):
        body = _body_name(model, int(geom_bodyid[geom_id]))
        name = _geom_name(model, geom_id)
        if body == "terrain" or name.startswith("terrain"):
            continue
        # G1 collision geoms are group 3 in the XML. Keep this permissive because
        # compiled models may still include unnamed/default collision geoms.
        if int(geom_group[geom_id]) < 3 and "collision" not in name:
            continue
        # In this mjlab path geom_xpos is reported in the robot-local/env-local
        # frame, while root_link_pos_w and goals include the terrain tile origin.
        # Shift by the current root so all plot layers share the same coordinates.
        xy = root_xy.astype(float) + geom_xpos[geom_id, :2].astype(float)
        radius = float(geom_rbound[geom_id])
        rows.append([float(xy[0]), float(xy[1]), radius])
        details.append({"id": geom_id, "name": name, "body": body, "xy": xy.tolist(), "rbound": radius})
    arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)
    return arr, details


def _estimate_robot_radius(collision_xy_r: np.ndarray, root_xy: np.ndarray) -> float:
    if collision_xy_r.size == 0:
        return 0.35
    dist = np.linalg.norm(collision_xy_r[:, :2] - root_xy.reshape(1, 2), axis=1) + collision_xy_r[:, 2]
    # Use a robust radius for plotting; max rbound includes tall body spheres and
    # can overstate planar gait footprint.
    return float(np.percentile(dist, 90))


def _plot_one(ax, env, args: argparse.Namespace, index: int) -> dict[str, Any]:
    obs_raw, start_clearances, goal_clearances, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
    del obs_raw
    terrain = _active_terrain_heightfield(env)
    starts = _robot_positions_xy(env)
    goals = _goal_positions_xy(env)
    start_xy = starts[0]
    goal_xy = goals[0]
    obstacle_xy = obstacle_cells[0]
    collision_xy_r, collision_details = _robot_collision_geom_xy(env, start_xy)
    robot_radius = _estimate_robot_radius(collision_xy_r, start_xy)

    if terrain is not None:
        height, extent, _ = terrain
        ax.imshow(
            height.T,
            extent=extent,
            origin="lower",
            cmap="Greys",
            alpha=0.32,
            interpolation="nearest",
            aspect="equal",
        )

    if obstacle_xy.size:
        ax.scatter(obstacle_xy[:, 0], obstacle_xy[:, 1], s=8, c="black", alpha=0.55, label="obstacle cells")

    vec = goal_xy - start_xy
    length = float(np.linalg.norm(vec))
    if length > 1e-6:
        unit = vec / length
        normal = np.array([-unit[1], unit[0]])
        for sign in (-1.0, 1.0):
            edge = np.stack([start_xy + sign * normal * robot_radius, goal_xy + sign * normal * robot_radius])
            ax.plot(edge[:, 0], edge[:, 1], color="orange", linestyle=":", linewidth=1.2)
        ax.plot([start_xy[0], goal_xy[0]], [start_xy[1], goal_xy[1]], color="crimson", linestyle="--", linewidth=1.4, label="center path")

    center_blockers = _corridor_obstacle_points(
        start_xy,
        goal_xy,
        obstacle_xy,
        corridor_radius=float(args.center_corridor_radius),
        ignore_end_radius=float(args.blocked_corridor_ignore_end_radius),
    )
    footprint_blockers = _corridor_obstacle_points(
        start_xy,
        goal_xy,
        obstacle_xy,
        corridor_radius=robot_radius,
        ignore_end_radius=float(args.blocked_corridor_ignore_end_radius),
    )
    if footprint_blockers.size:
        ax.scatter(
            footprint_blockers[:, 0],
            footprint_blockers[:, 1],
            s=42,
            facecolors="none",
            edgecolors="orange",
            linewidths=1.2,
            label="footprint-corridor blockers",
        )
    if center_blockers.size:
        ax.scatter(center_blockers[:, 0], center_blockers[:, 1], s=50, c="red", marker="s", label="center blockers")

    ax.scatter(start_xy[0], start_xy[1], c="white", edgecolors="black", s=90, zorder=5, label="agent root")
    ax.scatter(goal_xy[0], goal_xy[1], c="limegreen", edgecolors="black", marker="*", s=160, zorder=5, label="goal")
    ax.add_patch(Circle(start_xy, robot_radius, fill=False, color="dodgerblue", linewidth=1.8, label="est. robot radius"))
    if collision_xy_r.size:
        ax.scatter(collision_xy_r[:, 0], collision_xy_r[:, 1], c="dodgerblue", s=10, alpha=0.35, label="collision geoms")

    center_stats = _line_obstacle_stats(
        start_xy,
        goal_xy,
        obstacle_xy,
        corridor_radius=float(args.center_corridor_radius),
        ignore_end_radius=float(args.blocked_corridor_ignore_end_radius),
    )
    footprint_stats = _line_obstacle_stats(
        start_xy,
        goal_xy,
        obstacle_xy,
        corridor_radius=robot_radius,
        ignore_end_radius=float(args.blocked_corridor_ignore_end_radius),
    )
    ax.set_title(
        f"sample {index}: path={length:.2f}m, robot_r~{robot_radius:.2f}m\n"
        f"center blockers={center_stats['blocked_cell_count']} | footprint blockers={footprint_stats['blocked_cell_count']}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    return {
        "sample": index,
        "start_xy": start_xy.tolist(),
        "goal_xy": goal_xy.tolist(),
        "path_length": length,
        "start_clearance": float(start_clearances[0].detach().cpu().item()) if start_clearances is not None else None,
        "goal_clearance": float(goal_clearances[0].detach().cpu().item()) if goal_clearances is not None else None,
        "center_corridor_radius": float(args.center_corridor_radius),
        "center_blocked": bool(center_stats["blocked"]),
        "center_blocked_cell_count": int(center_stats["blocked_cell_count"]),
        "footprint_radius_estimate": robot_radius,
        "footprint_blocked": bool(footprint_stats["blocked"]),
        "footprint_blocked_cell_count": int(footprint_stats["blocked_cell_count"]),
        "configured_blocked_stats": layout_stats[0] if layout_stats else None,
        "collision_geom_count": len(collision_details),
        "collision_geom_rbound_min": float(np.min(collision_xy_r[:, 2])) if collision_xy_r.size else None,
        "collision_geom_rbound_p50": float(np.percentile(collision_xy_r[:, 2], 50)) if collision_xy_r.size else None,
        "collision_geom_rbound_max": float(np.max(collision_xy_r[:, 2])) if collision_xy_r.size else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Unitree-G1-Nav-Obstacles-Safe-Collision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-layouts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-length-s", type=float, default=16.0)
    parser.add_argument("--low-level-policy-path", default=str(DEFAULT_LOW_LEVEL))
    parser.add_argument("--output-dir", default=str(ROOT / "visualizations" / "unitree_nav_layout_samples"))
    parser.add_argument("--output-name", default="layout_samples")
    parser.add_argument("--min-goal-obstacle-clearance", type=float, default=0.0)
    parser.add_argument("--goal-clearance-resample-attempts", type=int, default=50)
    parser.add_argument("--min-start-obstacle-clearance", type=float, default=0.0)
    parser.add_argument("--start-clearance-resample-attempts", type=int, default=20)
    parser.add_argument("--require-blocked-corridor", action="store_true")
    parser.add_argument("--blocked-corridor-radius", type=float, default=0.45)
    parser.add_argument("--blocked-corridor-ignore-end-radius", type=float, default=0.75)
    parser.add_argument("--blocked-corridor-min-cells", type=int, default=1)
    parser.add_argument("--blocked-corridor-resample-attempts", type=int, default=100)
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
    parser.add_argument("--center-corridor-radius", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    env = make_env(args, num_envs=1, render=False)
    cols = min(3, int(args.num_layouts))
    rows = int(math.ceil(int(args.num_layouts) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows), dpi=150, squeeze=False)
    records = []
    for i in range(int(args.num_layouts)):
        torch.manual_seed(int(args.seed) + i)
        np.random.seed(int(args.seed) + i)
        records.append(_plot_one(axes.flat[i], env, args, i))
    for ax in axes.flat[int(args.num_layouts) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{args.output_name}.png"
    json_path = out_dir / f"{args.output_name}.json"
    fig.savefig(png_path)
    plt.close(fig)
    json_path.write_text(json.dumps({"records": records}, indent=2, sort_keys=True), encoding="utf-8")
    env.close()
    print(json.dumps({"plot": str(png_path), "json": str(json_path), "records": records}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

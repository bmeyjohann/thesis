#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from safetygym_utils.env import (
    extract_goal_distance,
    extract_min_constrained_clearance,
    make_safety_env,
)
from safetygym_utils.policy_viz import (
    _extract_bounds,
    _extract_overlay_specs,
    _overlay_world,
    _set_agent_pose,
    _set_goal_xy,
)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x, 0.0)


def _smooth_clearance_penalty(
    clearance: np.ndarray,
    *,
    scale: float,
    margin: float,
    temperature: float,
) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    # Positive when clearance is below the desired safety margin, smooth everywhere.
    return -float(scale) * temp * _softplus((float(margin) - clearance) / temp)


def _goal_radius(task) -> float:
    goal = getattr(task, "goal", None)
    for attr in ("size", "keepout"):
        try:
            value = float(getattr(goal, attr))
            if np.isfinite(value) and value > 0.0:
                return value
        except Exception:
            pass
    return 0.3


def _compute_surface(
    *,
    env,
    task,
    state_template: dict[str, Any],
    goal_xy: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    agent_z: float,
    dense_scale: float,
    sparse_goal_bonus: float,
    clearance_scale: float,
    clearance_margin: float,
    clearance_temperature: float,
) -> dict[str, np.ndarray]:
    goal_radius = _goal_radius(task)
    shape = (len(ys), len(xs))
    goal_distance = np.full(shape, np.nan, dtype=np.float64)
    clearance = np.full(shape, np.nan, dtype=np.float64)

    _set_goal_xy(task, goal_xy)
    for row_idx, y in enumerate(ys):
        for col_idx, x in enumerate(xs):
            _set_agent_pose(task, state_template, np.asarray([x, y], dtype=np.float64), 0.0, agent_z)
            goal_distance[row_idx, col_idx] = float(extract_goal_distance(env))
            clearance[row_idx, col_idx] = float(extract_min_constrained_clearance(env))

    goal_potential = -float(dense_scale) * goal_distance
    sparse_goal = float(sparse_goal_bonus) * (goal_distance <= goal_radius).astype(np.float64)
    clearance_penalty = _smooth_clearance_penalty(
        clearance,
        scale=float(clearance_scale),
        margin=float(clearance_margin),
        temperature=float(clearance_temperature),
    )
    total = goal_potential + sparse_goal + clearance_penalty
    return {
        "goal_distance": goal_distance,
        "goal_potential": goal_potential,
        "sparse_goal": sparse_goal,
        "clearance": clearance,
        "clearance_penalty": clearance_penalty,
        "total": total,
    }


def _extent(xs: np.ndarray, ys: np.ndarray) -> list[float]:
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    return [
        float(xs.min() - dx / 2.0),
        float(xs.max() + dx / 2.0),
        float(ys.min() - dy / 2.0),
        float(ys.max() + dy / 2.0),
    ]


def _plot_heatmaps(
    *,
    output_path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    grids: dict[str, np.ndarray],
    task,
    overlay_specs: list[dict[str, Any]],
    goal_xy: np.ndarray,
    title_suffix: str,
) -> Path:
    panels = [
        ("goal potential (-distance)", grids["goal_potential"], "viridis", None, None),
        ("smooth clearance penalty", grids["clearance_penalty"], "magma", None, 0.0),
        ("sparse goal bonus", grids["sparse_goal"], "Greens", 0.0, None),
        ("combined surface", grids["total"], "coolwarm", None, None),
        ("distance to goal", grids["goal_distance"], "cividis", 0.0, None),
        ("clearance", grids["clearance"], "RdYlGn", None, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    extent = _extent(xs, ys)
    for ax, (title, grid, cmap, vmin, vmax) in zip(axes.reshape(-1), panels):
        im = ax.imshow(
            np.asarray(grid, dtype=np.float64),
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        _overlay_world(ax, task=task, overlay_specs=overlay_specs, goal_xy=goal_xy)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(f"Safety-Gym reward surface diagnostic\n{title_suffix}", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _plot_surface_3d(
    *,
    output_path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    total: np.ndarray,
    title_suffix: str,
) -> Path:
    xx, yy = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, total, cmap="coolwarm", linewidth=0, antialiased=True, alpha=0.92)
    ax.set_title(f"Combined reward surface\n{title_suffix}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("reward potential")
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _stats(grid: np.ndarray) -> dict[str, float]:
    arr = np.asarray(grid, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot a state-potential reward surface for Safety-Gym.")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal1-v0")
    p.add_argument("--output_dir", type=Path, default=Path("visualizations") / "safetygym_reward_surface")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid_resolution", type=int, default=96)
    p.add_argument("--x_range", type=float, nargs=2, default=None)
    p.add_argument("--y_range", type=float, nargs=2, default=None)
    p.add_argument("--goal", type=float, nargs=2, default=None)
    p.add_argument("--dense_scale", type=float, default=1.0)
    p.add_argument("--sparse_goal_bonus", type=float, default=1.0)
    p.add_argument("--clearance_scale", type=float, default=4.0)
    p.add_argument("--clearance_margin", type=float, default=0.30)
    p.add_argument("--clearance_temperature", type=float, default=0.08)
    p.add_argument("--car_wheel_command_limit", type=float, default=1.0)
    p.add_argument("--car_force_scale", type=float, default=1.0)
    p.add_argument("--surface_mode", type=str, default="default")
    return p


def main() -> int:
    args = build_parser().parse_args()
    env = make_safety_env(
        args.env_name,
        render_mode="none",
        surface_mode=args.surface_mode,
        car_wheel_command_limit=args.car_wheel_command_limit,
        car_force_scale=args.car_force_scale,
        seed=args.seed,
    )
    obs, _ = env.reset(seed=args.seed)
    del obs
    task = env.unwrapped.task
    if not getattr(task, "observation_flatten", True):
        task.toggle_observation_space()
    state_template = task.world.get_state()
    agent_z = float(state_template["qpos"][2])
    overlay_specs = _extract_overlay_specs(task)
    goal_xy = (
        np.asarray(args.goal, dtype=np.float64).reshape(2)
        if args.goal is not None
        else np.asarray(task.goal.pos[:2], dtype=np.float64).copy()
    )
    _set_goal_xy(task, goal_xy)
    bounds = _extract_bounds(task, x_range=args.x_range, y_range=args.y_range)
    xs = np.linspace(bounds[0], bounds[1], int(args.grid_resolution), dtype=np.float64)
    ys = np.linspace(bounds[2], bounds[3], int(args.grid_resolution), dtype=np.float64)
    grids = _compute_surface(
        env=env,
        task=task,
        state_template=state_template,
        goal_xy=goal_xy,
        xs=xs,
        ys=ys,
        agent_z=agent_z,
        dense_scale=args.dense_scale,
        sparse_goal_bonus=args.sparse_goal_bonus,
        clearance_scale=args.clearance_scale,
        clearance_margin=args.clearance_margin,
        clearance_temperature=args.clearance_temperature,
    )
    tag = args.tag or f"{args.env_name}_seed{args.seed}"
    safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tag)
    title_suffix = (
        f"env={args.env_name}, goal=({goal_xy[0]:.2f}, {goal_xy[1]:.2f}), "
        f"dense_scale={args.dense_scale}, sparse={args.sparse_goal_bonus}, "
        f"softplus_clearance scale={args.clearance_scale}, margin={args.clearance_margin}, "
        f"temp={args.clearance_temperature}"
    )
    heatmap_path = Path(args.output_dir).resolve() / f"{safe_tag}_heatmaps.png"
    surface_path = Path(args.output_dir).resolve() / f"{safe_tag}_surface3d.png"
    json_path = Path(args.output_dir).resolve() / f"{safe_tag}_stats.json"
    _plot_heatmaps(
        output_path=heatmap_path,
        xs=xs,
        ys=ys,
        grids=grids,
        task=task,
        overlay_specs=overlay_specs,
        goal_xy=goal_xy,
        title_suffix=title_suffix,
    )
    _plot_surface_3d(output_path=surface_path, xs=xs, ys=ys, total=grids["total"], title_suffix=title_suffix)
    meta = {
        "env_name": args.env_name,
        "seed": int(args.seed),
        "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
        "x_range": [float(bounds[0]), float(bounds[1])],
        "y_range": [float(bounds[2]), float(bounds[3])],
        "grid_resolution": int(args.grid_resolution),
        "reward_definition": {
            "combined": "goal_potential + sparse_goal + smooth_clearance_penalty",
            "goal_potential": "-dense_scale * distance_to_goal; dense progress is the temporal difference of this potential",
            "sparse_goal": "sparse_goal_bonus if distance_to_goal <= goal_radius else 0",
            "smooth_clearance_penalty": "-clearance_scale * temperature * softplus((margin - clearance) / temperature)",
            "excluded": ["step_penalty", "collision_penalty", "heading_alignment_reward"],
        },
        "params": {
            "dense_scale": float(args.dense_scale),
            "sparse_goal_bonus": float(args.sparse_goal_bonus),
            "clearance_scale": float(args.clearance_scale),
            "clearance_margin": float(args.clearance_margin),
            "clearance_temperature": float(args.clearance_temperature),
            "goal_radius": float(_goal_radius(task)),
        },
        "stats": {name: _stats(grid) for name, grid in grids.items()},
        "outputs": {
            "heatmaps": str(heatmap_path),
            "surface3d": str(surface_path),
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    env.close()
    print(f"Saved heatmaps: {heatmap_path}")
    print(f"Saved 3D surface: {surface_path}")
    print(f"Saved stats: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

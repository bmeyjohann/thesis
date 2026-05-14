#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from safetygym_utils.env import make_safety_env
from safetygym_utils.policy_viz import _extract_bounds, _extract_overlay_specs, _overlay_world, _set_goal_xy
from tools.visualize_safetygym_reward_surface import _compute_surface, _stats


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _plot_grid(
    *,
    output_path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    grids: dict[tuple[float, float], np.ndarray],
    margins: list[float],
    scales: list[float],
    task,
    overlay_specs,
    goal_xy: np.ndarray,
    title: str,
    cmap: str,
    symmetric: bool,
) -> Path:
    rows = len(margins)
    cols = len(scales)
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.8 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes, dtype=object).reshape(rows, cols)
    all_vals = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in grids.values()])
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size:
        if symmetric:
            vmax = float(np.quantile(np.abs(finite), 0.98))
            vmin = -vmax
        else:
            vmin = float(np.quantile(finite, 0.02))
            vmax = float(np.quantile(finite, 0.98))
    else:
        vmin, vmax = None, None
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    extent = [float(xs.min() - dx / 2), float(xs.max() + dx / 2), float(ys.min() - dy / 2), float(ys.max() + dy / 2)]
    last_im = None
    for row_idx, margin in enumerate(margins):
        for col_idx, scale in enumerate(scales):
            ax = axes_arr[row_idx, col_idx]
            grid = grids[(margin, scale)]
            last_im = ax.imshow(
                grid,
                extent=extent,
                origin="lower",
                aspect="equal",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            _overlay_world(ax, task=task, overlay_specs=overlay_specs, goal_xy=goal_xy)
            ax.set_title(f"margin={margin:g}, scale={scale:g}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes_arr.reshape(-1).tolist(), shrink=0.72)
    fig.suptitle(title, fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep Safety-Gym smooth-clearance reward surfaces.")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid_resolution", type=int, default=88)
    p.add_argument("--margins", type=str, default="0.02,0.05,0.10,0.15")
    p.add_argument("--scales", type=str, default="1,2,4,8")
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--dense_scale", type=float, default=1.0)
    p.add_argument("--sparse_goal_bonus", type=float, default=1.0)
    p.add_argument("--output_dir", type=Path, default=Path("visualizations") / "safetygym_reward_surface_sweeps")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--x_range", type=float, nargs=2, default=None)
    p.add_argument("--y_range", type=float, nargs=2, default=None)
    p.add_argument("--goal", type=float, nargs=2, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    margins = _parse_floats(args.margins)
    scales = _parse_floats(args.scales)
    env = make_safety_env(args.env_name, render_mode="none", seed=args.seed)
    env.reset(seed=args.seed)
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

    combined: dict[tuple[float, float], np.ndarray] = {}
    centered_combined: dict[tuple[float, float], np.ndarray] = {}
    step_additive: dict[tuple[float, float], np.ndarray] = {}
    clearance: dict[tuple[float, float], np.ndarray] = {}
    stats: dict[str, dict] = {}
    for margin in margins:
        for scale in scales:
            surface = _compute_surface(
                env=env,
                task=task,
                state_template=state_template,
                goal_xy=goal_xy,
                xs=xs,
                ys=ys,
                agent_z=agent_z,
                dense_scale=float(args.dense_scale),
                sparse_goal_bonus=float(args.sparse_goal_bonus),
                clearance_scale=float(scale),
                clearance_margin=float(margin),
                clearance_temperature=float(args.temperature),
            )
            key = (margin, scale)
            combined[key] = surface["total"]
            centered_combined[key] = surface["total"] - float(np.nanmedian(surface["total"]))
            step_additive[key] = surface["sparse_goal"] + surface["clearance_penalty"]
            clearance[key] = surface["clearance_penalty"]
            stats[f"margin={margin:g},scale={scale:g}"] = {
                "total": _stats(surface["total"]),
                "centered_total": _stats(centered_combined[key]),
                "step_additive_sparse_plus_clearance": _stats(step_additive[key]),
                "clearance_penalty": _stats(surface["clearance_penalty"]),
            }

    tag = _safe_tag(args.tag or f"{args.env_name}_seed{args.seed}_temp{args.temperature:g}")
    output_dir = Path(args.output_dir).resolve()
    combined_path = output_dir / f"{tag}_combined_grid.png"
    centered_path = output_dir / f"{tag}_centered_combined_grid.png"
    step_additive_path = output_dir / f"{tag}_step_additive_grid.png"
    clearance_path = output_dir / f"{tag}_clearance_grid.png"
    stats_path = output_dir / f"{tag}_stats.json"
    title_base = (
        f"{args.env_name} smooth-clearance sweep, temp={args.temperature:g}, "
        f"dense={args.dense_scale:g}, sparse={args.sparse_goal_bonus:g}"
    )
    _plot_grid(
        output_path=combined_path,
        xs=xs,
        ys=ys,
        grids=combined,
        margins=margins,
        scales=scales,
        task=task,
        overlay_specs=overlay_specs,
        goal_xy=goal_xy,
        title=f"Combined reward surface\n{title_base}",
        cmap="coolwarm",
        symmetric=True,
    )
    _plot_grid(
        output_path=centered_path,
        xs=xs,
        ys=ys,
        grids=centered_combined,
        margins=margins,
        scales=scales,
        task=task,
        overlay_specs=overlay_specs,
        goal_xy=goal_xy,
        title=f"Median-centered potential surface (visual diagnostic only)\n{title_base}",
        cmap="coolwarm",
        symmetric=True,
    )
    _plot_grid(
        output_path=step_additive_path,
        xs=xs,
        ys=ys,
        grids=step_additive,
        margins=margins,
        scales=scales,
        task=task,
        overlay_specs=overlay_specs,
        goal_xy=goal_xy,
        title=f"Per-step additive terms: sparse goal + clearance penalty\n{title_base}",
        cmap="coolwarm",
        symmetric=True,
    )
    _plot_grid(
        output_path=clearance_path,
        xs=xs,
        ys=ys,
        grids=clearance,
        margins=margins,
        scales=scales,
        task=task,
        overlay_specs=overlay_specs,
        goal_xy=goal_xy,
        title=f"Clearance penalty component\n{title_base}",
        cmap="magma",
        symmetric=False,
    )
    meta = {
        "env_name": args.env_name,
        "seed": int(args.seed),
        "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
        "margins": margins,
        "scales": scales,
        "temperature": float(args.temperature),
        "dense_scale": float(args.dense_scale),
        "sparse_goal_bonus": float(args.sparse_goal_bonus),
        "grid_resolution": int(args.grid_resolution),
        "outputs": {
            "combined_grid": str(combined_path),
            "centered_combined_grid": str(centered_path),
            "step_additive_grid": str(step_additive_path),
            "clearance_grid": str(clearance_path),
        },
        "stats": stats,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    env.close()
    print(f"Saved combined grid: {combined_path}")
    print(f"Saved centered combined grid: {centered_path}")
    print(f"Saved step-additive grid: {step_additive_path}")
    print(f"Saved clearance grid: {clearance_path}")
    print(f"Saved stats: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

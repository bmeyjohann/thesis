#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from safetygym_utils.env import make_safety_env
from safetygym_utils.policy_viz import _extract_bounds, _set_goal_xy
from tools.visualize_safetygym_reward_surface import _compute_surface, _stats


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create an interactive 3D Safety-Gym reward-surface HTML plot.")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid_resolution", type=int, default=80)
    p.add_argument("--dense_scale", type=float, default=1.0)
    p.add_argument("--sparse_goal_bonus", type=float, default=1.0)
    p.add_argument("--clearance_scale", type=float, default=4.0)
    p.add_argument("--clearance_margin", type=float, default=0.01)
    p.add_argument("--clearance_temperature", type=float, default=0.01)
    p.add_argument("--x_range", type=float, nargs=2, default=None)
    p.add_argument("--y_range", type=float, nargs=2, default=None)
    p.add_argument("--goal", type=float, nargs=2, default=None)
    p.add_argument("--output_dir", type=Path, default=Path("visualizations") / "safetygym_reward_surface_interactive")
    p.add_argument("--tag", type=str, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    env = make_safety_env(args.env_name, render_mode="none", seed=args.seed)
    env.reset(seed=args.seed)
    task = env.unwrapped.task
    if not getattr(task, "observation_flatten", True):
        task.toggle_observation_space()
    state_template = task.world.get_state()
    agent_z = float(state_template["qpos"][2])
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
        dense_scale=float(args.dense_scale),
        sparse_goal_bonus=float(args.sparse_goal_bonus),
        clearance_scale=float(args.clearance_scale),
        clearance_margin=float(args.clearance_margin),
        clearance_temperature=float(args.clearance_temperature),
    )
    env.close()

    centered_total = grids["total"] - float(np.nanmedian(grids["total"]))
    step_additive = grids["sparse_goal"] + grids["clearance_penalty"]
    surfaces = {
        "centered potential": centered_total,
        "per-step additive sparse+clearance": step_additive,
        "clearance penalty": grids["clearance_penalty"],
        "raw potential": grids["total"],
        "goal potential": grids["goal_potential"],
        "clearance": grids["clearance"],
    }
    traces = []
    for idx, (name, grid) in enumerate(surfaces.items()):
        traces.append(
            {
                "type": "surface",
                "name": name,
                "x": xs.tolist(),
                "y": ys.tolist(),
                "z": np.asarray(grid, dtype=np.float64).tolist(),
                "visible": idx == 0,
                "colorscale": "RdBu" if name != "clearance" else "Viridis",
                "reversescale": name != "clearance",
                "colorbar": {"title": name},
            }
        )

    buttons = []
    names = list(surfaces.keys())
    for idx, name in enumerate(names):
        visible = [False] * len(names)
        visible[idx] = True
        buttons.append(
            {
                "label": name,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": f"{args.env_name}: {name}"},
                ],
            }
        )

    meta = {
        "env_name": args.env_name,
        "seed": int(args.seed),
        "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
        "params": {
            "dense_scale": float(args.dense_scale),
            "sparse_goal_bonus": float(args.sparse_goal_bonus),
            "clearance_scale": float(args.clearance_scale),
            "clearance_margin": float(args.clearance_margin),
            "clearance_temperature": float(args.clearance_temperature),
        },
        "stats": {name: _stats(grid) for name, grid in surfaces.items()},
    }
    layout = {
        "title": f"{args.env_name}: centered potential",
        "scene": {
            "xaxis": {"title": "x"},
            "yaxis": {"title": "y"},
            "zaxis": {"title": "value"},
            "aspectmode": "cube",
        },
        "updatemenus": [
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.02,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        "margin": {"l": 0, "r": 0, "t": 60, "b": 0},
    }
    tag = _safe_tag(
        args.tag
        or f"{args.env_name}_m{args.clearance_margin:g}_s{args.clearance_scale:g}_t{args.clearance_temperature:g}"
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{tag}.html"
    json_path = output_dir / f"{tag}_stats.json"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{tag}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: sans-serif; }}
    #plot {{ width: 100vw; height: 100vh; }}
    #note {{ position: fixed; left: 12px; bottom: 10px; background: rgba(255,255,255,0.86); padding: 8px 10px; font-size: 13px; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <div id="note">Dropdown switches surfaces. Centered potential is for path-shape visualization; per-step additive is sparse goal + clearance penalty.</div>
  <script>
    const traces = {json.dumps(traces)};
    const layout = {json.dumps(layout)};
    Plotly.newPlot('plot', traces, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved interactive HTML: {html_path}")
    print(f"Saved stats: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from tools.visualize_safetygym_reward_surface import _compute_surface


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create a tunable interactive Safety-Gym reward-surface HTML plot.")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid_resolution", type=int, default=86)
    p.add_argument("--dense_scale", type=float, default=1.0)
    p.add_argument("--sparse_goal_bonus", type=float, default=1.0)
    p.add_argument("--clearance_scale", type=float, default=1.1)
    p.add_argument("--clearance_margin", type=float, default=0.0)
    p.add_argument("--clearance_temperature", type=float, default=0.001)
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

    tag = _safe_tag(args.tag or f"{args.env_name}_tunable_reward_surface")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{tag}.html"
    data = {
        "envName": args.env_name,
        "seed": int(args.seed),
        "x": xs.tolist(),
        "y": ys.tolist(),
        "goalPotential": np.asarray(grids["goal_potential"], dtype=np.float64).tolist(),
        "sparseGoal": np.asarray(grids["sparse_goal"], dtype=np.float64).tolist(),
        "clearance": np.asarray(grids["clearance"], dtype=np.float64).tolist(),
        "defaults": {
            "margin": float(args.clearance_margin),
            "scale": float(args.clearance_scale),
            "temperature": float(args.clearance_temperature),
            "function": "softplus",
            "surface": "centeredPotential",
        },
        "meta": {
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "dense_scale": float(args.dense_scale),
            "sparse_goal_bonus": float(args.sparse_goal_bonus),
            "note": "Dense goal potential is visual only; training uses temporal progress prev_dist - cur_dist.",
        },
    }
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{tag}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: #f5f1e8; color: #17130f; }}
    #controls {{
      position: fixed; z-index: 10; top: 12px; left: 12px; width: 330px;
      background: rgba(255, 252, 245, 0.94); border: 1px solid rgba(60, 45, 30, 0.18);
      border-radius: 14px; padding: 12px 14px; box-shadow: 0 12px 35px rgba(35, 22, 8, 0.14);
    }}
    #plot {{ width: 100vw; height: 100vh; }}
    label {{ display: grid; grid-template-columns: 92px 1fr 54px; gap: 8px; align-items: center; font-size: 13px; margin: 8px 0; }}
    select {{ width: 100%; }}
    input[type="range"] {{ width: 100%; }}
    .row {{ display: grid; grid-template-columns: 92px 1fr; gap: 8px; align-items: center; margin: 8px 0; font-size: 13px; }}
    .title {{ font-weight: 760; margin-bottom: 4px; }}
    .note {{ font-size: 12px; line-height: 1.35; color: #53483c; margin-top: 10px; }}
    .stats {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; white-space: pre; background: #eee4d2; padding: 8px; border-radius: 8px; margin-top: 8px; }}
  </style>
</head>
<body>
  <div id="controls">
    <div class="title">Safety-Gym Tunable Reward Surface</div>
    <div class="row">
      <span>function</span>
      <select id="fn">
        <option value="softplus">softplus</option>
        <option value="hinge">hinge</option>
        <option value="quadratic_hinge">quadratic hinge</option>
        <option value="exp_hinge">exp hinge</option>
        <option value="exp_soft">exp soft</option>
      </select>
    </div>
    <div class="row">
      <span>surface</span>
      <select id="surface">
        <option value="centeredPotential">centered potential</option>
        <option value="stepAdditive">per-step additive</option>
        <option value="clearancePenalty">clearance penalty</option>
        <option value="rawPotential">raw potential</option>
        <option value="clearance">clearance</option>
      </select>
    </div>
    <label><span>margin</span><input id="margin" type="range" min="0" max="0.08" step="0.001"><span id="marginText"></span></label>
    <label><span>scale</span><input id="scale" type="range" min="0" max="10" step="0.1"><span id="scaleText"></span></label>
    <label><span>temp</span><input id="temp" type="range" min="0.001" max="0.08" step="0.001"><span id="tempText"></span></label>
    <div class="stats" id="stats"></div>
    <div class="note">
      Centered potential is for path-shape inspection. The trainer's dense goal term is per-step progress, so use
      <b>per-step additive</b> to judge sparse+clearance reward magnitude.
    </div>
  </div>
  <div id="plot"></div>
  <script>
    const data = {json.dumps(data)};
    const els = {{
      fn: document.getElementById('fn'),
      surface: document.getElementById('surface'),
      margin: document.getElementById('margin'),
      scale: document.getElementById('scale'),
      temp: document.getElementById('temp'),
      marginText: document.getElementById('marginText'),
      scaleText: document.getElementById('scaleText'),
      tempText: document.getElementById('tempText'),
      stats: document.getElementById('stats'),
    }};
    els.margin.value = data.defaults.margin;
    els.scale.value = data.defaults.scale;
    els.temp.value = data.defaults.temperature;
    els.fn.value = data.defaults.function;
    els.surface.value = data.defaults.surface;

    function softplus(x) {{
      if (x > 40) return x;
      if (x < -40) return Math.exp(x);
      return Math.log1p(Math.exp(x));
    }}
    function penalty(c, margin, scale, temp, fn) {{
      const gap = margin - c;
      if (fn === 'softplus') return -scale * temp * softplus(gap / temp);
      if (fn === 'hinge') return -scale * Math.max(0, gap);
      if (fn === 'quadratic_hinge') {{
        const h = Math.max(0, gap);
        return -scale * h * h;
      }}
      if (fn === 'exp_hinge') {{
        const h = Math.max(0, gap);
        return -scale * temp * (Math.exp(Math.min(h / temp, 50)) - 1);
      }}
      if (fn === 'exp_soft') return -scale * temp * Math.exp(Math.min(gap / temp, 50));
      return 0;
    }}
    function flatten(arr) {{
      const out = [];
      for (const row of arr) for (const v of row) if (Number.isFinite(v)) out.push(v);
      return out;
    }}
    function quantile(vals, q) {{
      if (!vals.length) return NaN;
      const sorted = vals.slice().sort((a, b) => a - b);
      const pos = (sorted.length - 1) * q;
      const lo = Math.floor(pos), hi = Math.ceil(pos);
      if (lo === hi) return sorted[lo];
      return sorted[lo] * (hi - pos) + sorted[hi] * (pos - lo);
    }}
    function median2d(arr) {{ return quantile(flatten(arr), 0.5); }}
    function compute() {{
      const margin = parseFloat(els.margin.value);
      const scale = parseFloat(els.scale.value);
      const temp = parseFloat(els.temp.value);
      const fn = els.fn.value;
      const clearancePenalty = data.clearance.map(row => row.map(c => penalty(c, margin, scale, temp, fn)));
      const rawPotential = data.goalPotential.map((row, i) => row.map((v, j) => v + data.sparseGoal[i][j] + clearancePenalty[i][j]));
      const med = median2d(rawPotential);
      const centeredPotential = rawPotential.map(row => row.map(v => v - med));
      const stepAdditive = data.sparseGoal.map((row, i) => row.map((v, j) => v + clearancePenalty[i][j]));
      const maps = {{ centeredPotential, stepAdditive, clearancePenalty, rawPotential, clearance: data.clearance }};
      return {{ z: maps[els.surface.value], clearancePenalty, stepAdditive, rawPotential, centeredPotential }};
    }}
    function summarize(arr) {{
      const vals = flatten(arr);
      return {{
        min: Math.min(...vals),
        p05: quantile(vals, 0.05),
        p50: quantile(vals, 0.5),
        mean: vals.reduce((a, b) => a + b, 0) / vals.length,
        p95: quantile(vals, 0.95),
        max: Math.max(...vals),
      }};
    }}
    function fmt(x) {{ return Number.isFinite(x) ? x.toFixed(3) : 'nan'; }}
    function update() {{
      els.marginText.textContent = parseFloat(els.margin.value).toFixed(3);
      els.scaleText.textContent = parseFloat(els.scale.value).toFixed(1);
      els.tempText.textContent = parseFloat(els.temp.value).toFixed(3);
      const computed = compute();
      const s = summarize(computed.z);
      els.stats.textContent =
        `shown min ${{fmt(s.min)}}  p50 ${{fmt(s.p50)}}  max ${{fmt(s.max)}}\\n` +
        `shown p05 ${{fmt(s.p05)}} mean ${{fmt(s.mean)}} p95 ${{fmt(s.p95)}}`;
      const title = `${{data.envName}} | ${{els.surface.options[els.surface.selectedIndex].text}} | ${{els.fn.value}} margin=${{els.margin.value}} scale=${{els.scale.value}} temp=${{els.temp.value}}`;
      Plotly.react('plot', [{{
        type: 'surface',
        x: data.x,
        y: data.y,
        z: computed.z,
        colorscale: els.surface.value === 'clearance' ? 'Viridis' : 'RdBu',
        reversescale: els.surface.value !== 'clearance',
        colorbar: {{ title: els.surface.options[els.surface.selectedIndex].text }},
      }}], {{
        title,
        scene: {{
          xaxis: {{ title: 'x' }},
          yaxis: {{ title: 'y' }},
          zaxis: {{ title: 'value' }},
          aspectmode: 'cube',
        }},
        margin: {{ l: 0, r: 0, t: 58, b: 0 }},
      }}, {{ responsive: true }});
    }}
    for (const el of [els.fn, els.surface, els.margin, els.scale, els.temp]) el.addEventListener('input', update);
    update();
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    print(f"Saved tunable interactive HTML: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

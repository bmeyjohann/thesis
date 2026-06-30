#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from safetygym_utils.env import extract_goal_distance, extract_min_constrained_clearance, make_safety_env
from safetygym_utils.policy_viz import _body_xy_and_yaw, _extract_bounds, _extract_overlay_specs, _set_agent_pose
from safetygym_utils.wrappers import FixedSafetyLayoutWrapper
from tools.visualize_safetygym_reward_surface import _compute_surface, _goal_radius


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _overlay_payload(task, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for spec in specs:
        pose = _body_xy_and_yaw(task, str(spec["name"]))
        if pose is None:
            continue
        xy, yaw = pose
        payload.append(
            {
                "name": str(spec["name"]),
                "xy": [float(xy[0]), float(xy[1])],
                "yaw": float(yaw),
                "size": np.asarray(spec.get("size", [0.1]), dtype=np.float64).reshape(-1).tolist(),
                "geom_type": str(spec.get("geom_type", "sphere")),
            }
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create a one-obstacle Safety-Gym reward/intervention gate tuner.")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal1-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fixed_layout_preset", type=str, default="car_single_block")
    p.add_argument("--grid_resolution", type=int, default=120)
    p.add_argument("--path_steps", type=int, default=140)
    p.add_argument("--x_range", type=float, nargs=2, default=(-1.7, 1.7))
    p.add_argument("--y_range", type=float, nargs=2, default=(-1.35, 1.35))
    p.add_argument("--dense_scale", type=float, default=1.0)
    p.add_argument("--sparse_goal_bonus", type=float, default=1.0)
    p.add_argument("--clearance_scale", type=float, default=1.1)
    p.add_argument("--clearance_margin", type=float, default=0.0)
    p.add_argument("--clearance_temperature", type=float, default=0.001)
    p.add_argument("--teacher_progress_epsilon", type=float, default=1e-4)
    p.add_argument("--teacher_progress_bad_steps", type=int, default=3)
    p.add_argument("--teacher_progress_good_steps", type=int, default=5)
    p.add_argument("--output_dir", type=Path, default=Path("visualizations") / "safetygym_intervention_tuner")
    p.add_argument("--tag", type=str, default="car_single_block_reward_progress_tuner")
    return p


def main() -> int:
    args = build_parser().parse_args()
    env = make_safety_env(args.env_name, render_mode="none", seed=args.seed)
    env = FixedSafetyLayoutWrapper(env, preset=str(args.fixed_layout_preset))
    env.reset(seed=args.seed)
    task = env.unwrapped.task
    if not getattr(task, "observation_flatten", True):
        task.toggle_observation_space()

    state_template = task.world.get_state()
    agent_z = float(state_template["qpos"][2])
    start_xy = np.asarray(task.world_info.layout.get("agent", [-1.2, 0.0]), dtype=np.float64).reshape(2)
    goal_xy = np.asarray(task.goal.pos[:2], dtype=np.float64).reshape(2)
    bounds = _extract_bounds(task, x_range=list(args.x_range), y_range=list(args.y_range))
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

    path_t = np.linspace(0.0, 1.0, int(args.path_steps), dtype=np.float64)
    path_xy = start_xy[None, :] + path_t[:, None] * (goal_xy[None, :] - start_xy[None, :])
    path_distance = []
    path_clearance = []
    for xy in path_xy:
        _set_agent_pose(task, state_template, xy, 0.0, agent_z)
        path_distance.append(float(extract_goal_distance(env)))
        path_clearance.append(float(extract_min_constrained_clearance(env)))

    overlays = _overlay_payload(task, _extract_overlay_specs(task))
    env.close()

    data = {
        "envName": str(args.env_name),
        "fixedLayoutPreset": str(args.fixed_layout_preset),
        "x": xs.tolist(),
        "y": ys.tolist(),
        "goalDistance": np.asarray(grids["goal_distance"], dtype=np.float64).tolist(),
        "clearance": np.asarray(grids["clearance"], dtype=np.float64).tolist(),
        "start": [float(start_xy[0]), float(start_xy[1])],
        "goal": [float(goal_xy[0]), float(goal_xy[1])],
        "goalRadius": float(_goal_radius(task)),
        "path": [[float(x), float(y)] for x, y in path_xy],
        "pathDistance": path_distance,
        "pathClearance": path_clearance,
        "overlays": overlays,
        "defaults": {
            "margin": float(args.clearance_margin),
            "scale": float(args.clearance_scale),
            "temperature": float(args.clearance_temperature),
            "function": "softplus",
            "surface": "score",
            "epsilon": float(args.teacher_progress_epsilon),
            "badSteps": int(args.teacher_progress_bad_steps),
            "goodSteps": int(args.teacher_progress_good_steps),
        },
        "meta": {
            "score": "-dense_scale * goal_distance + clearance_penalty(clearance)",
            "gate": "intervene after N consecutive score drops below -epsilon; release after M consecutive score improvements above epsilon",
            "dense_scale": float(args.dense_scale),
            "sparse_goal_bonus": float(args.sparse_goal_bonus),
        },
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{_safe_tag(args.tag)}.html"
    html_path.write_text(_build_html(data), encoding="utf-8")
    print(f"Saved intervention tuner HTML: {html_path}")
    return 0


def _build_html(data: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Safety-Gym Intervention Tuner</title>
  <style>
    body {{ margin:0; background:#f3efe4; color:#18120d; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }}
    #wrap {{ display:grid; grid-template-columns: 380px 1fr; height:100vh; }}
    #controls {{ padding:16px; background:rgba(255,250,239,.96); box-shadow:12px 0 30px rgba(50,35,10,.13); overflow:auto; }}
    h1 {{ margin:0 0 10px; font-size:20px; }}
    label {{ display:grid; grid-template-columns:120px 1fr 58px; gap:8px; align-items:center; margin:9px 0; font-size:13px; }}
    select {{ width:100%; }}
    canvas {{ display:block; }}
    #main {{ position:relative; display:grid; grid-template-rows: minmax(420px, 1fr) 210px; gap:10px; padding:12px; }}
    #map, #chart {{ width:100%; height:100%; background:#191d22; border-radius:16px; box-shadow:0 12px 34px rgba(25,20,10,.18); }}
    .stats {{ white-space:pre-wrap; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; background:#ece1cc; padding:10px; border-radius:10px; margin-top:12px; }}
    .note {{ font-size:12px; line-height:1.4; color:#5a4d3e; margin-top:12px; }}
    .pill {{ display:inline-block; padding:2px 7px; border-radius:999px; background:#d71920; color:white; font-weight:700; }}
  </style>
</head>
<body>
<div id="wrap">
  <div id="controls">
    <h1>One-Obstacle Intervention Tuner</h1>
    <div class="note">Layout: agent -> one hazard -> goal. The path assumes the student drives straight toward the goal. Red path segments are where the reward-progress gate would intervene.</div>
    <label><span>surface</span><select id="surface"><option value="score">state score</option><option value="delta">straight-path delta</option><option value="clearancePenalty">clearance penalty</option><option value="clearance">clearance</option><option value="distance">goal distance</option></select><span></span></label>
    <label><span>function</span><select id="fn"><option value="softplus">softplus</option><option value="hinge">hinge</option><option value="quadratic_hinge">quadratic hinge</option><option value="exp_soft">exp soft</option></select><span></span></label>
    <label><span>margin</span><input id="margin" type="range" min="0" max="0.25" step="0.001"><span id="marginText"></span></label>
    <label><span>scale</span><input id="scale" type="range" min="0" max="20" step="0.1"><span id="scaleText"></span></label>
    <label><span>temp</span><input id="temp" type="range" min="0.001" max="0.08" step="0.001"><span id="tempText"></span></label>
    <label><span>epsilon</span><input id="eps" type="range" min="0" max="0.02" step="0.0001"><span id="epsText"></span></label>
    <label><span>bad steps</span><input id="bad" type="range" min="1" max="10" step="1"><span id="badText"></span></label>
    <label><span>good steps</span><input id="good" type="range" min="1" max="15" step="1"><span id="goodText"></span></label>
    <div id="stats" class="stats"></div>
    <div class="note"><span class="pill">red</span> means the teacher would be active. If red appears only after collision, increase scale/margin or use a direct clearance gate in addition to progress.</div>
  </div>
  <div id="main">
    <canvas id="map"></canvas>
    <canvas id="chart"></canvas>
  </div>
</div>
<script>
const data = {json.dumps(data)};
const els = Object.fromEntries(['surface','fn','margin','scale','temp','eps','bad','good','stats','map','chart','marginText','scaleText','tempText','epsText','badText','goodText'].map(id => [id, document.getElementById(id)]));
els.margin.value = data.defaults.margin; els.scale.value = data.defaults.scale; els.temp.value = data.defaults.temperature;
els.eps.value = data.defaults.epsilon; els.bad.value = data.defaults.badSteps; els.good.value = data.defaults.goodSteps;
els.fn.value = data.defaults.function; els.surface.value = data.defaults.surface;

function softplus(x) {{ if (x > 40) return x; if (x < -40) return Math.exp(x); return Math.log1p(Math.exp(x)); }}
function penalty(c, margin, scale, temp, fn) {{
  const gap = margin - c;
  if (fn === 'softplus') return -scale * temp * softplus(gap / temp);
  if (fn === 'hinge') return -scale * Math.max(0, gap);
  if (fn === 'quadratic_hinge') {{ const h = Math.max(0, gap); return -scale * h * h; }}
  if (fn === 'exp_soft') return -scale * temp * Math.exp(Math.min(gap / temp, 50));
  return 0;
}}
function score(distance, clearance, p) {{ return -data.meta.dense_scale * distance + penalty(clearance, p.margin, p.scale, p.temp, p.fn); }}
function params() {{ return {{ margin:+els.margin.value, scale:+els.scale.value, temp:+els.temp.value, eps:+els.eps.value, bad:+els.bad.value, good:+els.good.value, fn:els.fn.value }}; }}
function computePath(p) {{
  const s = data.pathDistance.map((d,i) => score(d, data.pathClearance[i], p));
  const delta = s.map((v,i) => i === 0 ? 0 : v - s[i-1]);
  let bad=0, good=0, active=false;
  const gate = [];
  for (let i=0;i<s.length;i++) {{
    if (i > 0) {{
      if (delta[i] < -p.eps) {{ bad++; good=0; }}
      else if (delta[i] > p.eps) {{ good++; bad=0; }}
      else {{ bad=0; good=0; }}
    }}
    if (!active && bad >= p.bad) {{ active=true; good=0; }}
    else if (active && good >= p.good) {{ active=false; bad=0; }}
    gate.push(active);
  }}
  return {{score:s, delta, gate}};
}}
function resizeCanvas(c) {{
  const r = c.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
  c.width = Math.max(10, Math.floor(r.width*dpr)); c.height = Math.max(10, Math.floor(r.height*dpr));
  const ctx = c.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); return [r.width, r.height, ctx];
}}
function worldToCanvas(x,y,w,h) {{
  const xmin=data.x[0], xmax=data.x[data.x.length-1], ymin=data.y[0], ymax=data.y[data.y.length-1];
  return [30 + (x-xmin)/(xmax-xmin)*(w-60), h-30 - (y-ymin)/(ymax-ymin)*(h-60)];
}}
function color(v, vmin, vmax, reverse=false) {{
  let t = (v-vmin)/Math.max(1e-9, vmax-vmin); t=Math.max(0,Math.min(1,t)); if (reverse) t=1-t;
  const r = Math.floor(255*t), b = Math.floor(255*(1-t)), g = Math.floor(75 + 95*(1-Math.abs(t-.5)*2));
  return [r,g,b,255];
}}
function flattenGrid(g) {{ return g.flat().filter(Number.isFinite); }}
function quantile(vals,q) {{ const a=vals.slice().sort((x,y)=>x-y); const pos=(a.length-1)*q, lo=Math.floor(pos), hi=Math.ceil(pos); return lo===hi?a[lo]:a[lo]*(hi-pos)+a[hi]*(pos-lo); }}
function computeGrid(p, path) {{
  const rows=data.y.length, cols=data.x.length; let grid=[];
  for (let i=0;i<rows;i++) {{
    let row=[]; for (let j=0;j<cols;j++) {{
      const d=data.goalDistance[i][j], c=data.clearance[i][j], sc=score(d,c,p);
      if (els.surface.value==='score') row.push(sc);
      else if (els.surface.value==='clearancePenalty') row.push(penalty(c,p.margin,p.scale,p.temp,p.fn));
      else if (els.surface.value==='clearance') row.push(c);
      else if (els.surface.value==='distance') row.push(d);
      else row.push(sc);
    }} grid.push(row);
  }}
  return grid;
}}
function drawMap(p,path) {{
  const [w,h,ctx]=resizeCanvas(els.map); ctx.clearRect(0,0,w,h);
  const grid=computeGrid(p,path), vals=flattenGrid(grid); let vmin=quantile(vals,.05), vmax=quantile(vals,.95);
  const rows=data.y.length, cols=data.x.length, img=ctx.createImageData(cols,rows);
  for (let i=0;i<rows;i++) for (let j=0;j<cols;j++) {{
    const idx=((rows-1-i)*cols+j)*4; const rgba=color(grid[i][j],vmin,vmax, els.surface.value==='distance');
    img.data[idx]=rgba[0]; img.data[idx+1]=rgba[1]; img.data[idx+2]=rgba[2]; img.data[idx+3]=255;
  }}
  const tmp=document.createElement('canvas'); tmp.width=cols; tmp.height=rows; tmp.getContext('2d').putImageData(img,0,0);
  ctx.drawImage(tmp,30,30,w-60,h-60);
  ctx.strokeStyle='rgba(255,255,255,.32)'; ctx.strokeRect(30,30,w-60,h-60);
  for (const o of data.overlays) {{
    const [cx,cy]=worldToCanvas(o.xy[0],o.xy[1],w,h); const rad=Math.max(3,(o.size[0]||.1)*(w-60)/(data.x[data.x.length-1]-data.x[0]));
    if (Math.abs(cx) > w+80 || Math.abs(cy) > h+80) continue;
    ctx.fillStyle=o.name.includes('hazard')?'rgba(220,50,45,.65)':'rgba(230,160,50,.55)';
    ctx.beginPath(); ctx.arc(cx,cy,rad,0,Math.PI*2); ctx.fill(); ctx.strokeStyle='rgba(0,0,0,.6)'; ctx.stroke();
  }}
  let prev=null;
  for (let i=0;i<data.path.length;i++) {{
    const pt=worldToCanvas(data.path[i][0],data.path[i][1],w,h);
    if (prev) {{ ctx.strokeStyle=path.gate[i]?'#ff1515':'#00d4ff'; ctx.lineWidth=path.gate[i]?5:3; ctx.beginPath(); ctx.moveTo(prev[0],prev[1]); ctx.lineTo(pt[0],pt[1]); ctx.stroke(); }}
    prev=pt;
  }}
  const [sx,sy]=worldToCanvas(data.start[0],data.start[1],w,h), [gx,gy]=worldToCanvas(data.goal[0],data.goal[1],w,h);
  ctx.fillStyle='#ffffff'; ctx.beginPath(); ctx.arc(sx,sy,6,0,Math.PI*2); ctx.fill(); ctx.fillStyle='#76ff5a'; ctx.beginPath(); ctx.arc(gx,gy,9,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='white'; ctx.font='14px ui-monospace, monospace'; ctx.fillText(`${{data.envName}} / ${{data.fixedLayoutPreset}}`, 42, 24);
}}
function drawChart(path) {{
  const [w,h,ctx]=resizeCanvas(els.chart); ctx.clearRect(0,0,w,h); ctx.fillStyle='#191d22'; ctx.fillRect(0,0,w,h);
  const vals=path.score; const vmin=Math.min(...vals), vmax=Math.max(...vals); const x=i=>30+i/(vals.length-1)*(w-60); const y=v=>h-28-(v-vmin)/Math.max(1e-9,vmax-vmin)*(h-56);
  for (let i=1;i<vals.length;i++) if (path.gate[i]) {{ ctx.fillStyle='rgba(255,0,0,.18)'; ctx.fillRect(x(i-1),20,Math.max(1,x(i)-x(i-1)),h-48); }}
  ctx.strokeStyle='#ffe082'; ctx.lineWidth=2; ctx.beginPath(); vals.forEach((v,i)=>{{ if(i===0)ctx.moveTo(x(i),y(v)); else ctx.lineTo(x(i),y(v)); }}); ctx.stroke();
  ctx.fillStyle='white'; ctx.font='14px ui-monospace, monospace'; ctx.fillText('score along straight-to-goal path; red background = intervention active', 30, 18);
}}
function update() {{
  const p=params(); els.marginText.textContent=p.margin.toFixed(3); els.scaleText.textContent=p.scale.toFixed(1); els.tempText.textContent=p.temp.toFixed(3);
  els.epsText.textContent=p.eps.toFixed(4); els.badText.textContent=p.bad; els.goodText.textContent=p.good;
  const path=computePath(p); drawMap(p,path); drawChart(path);
  const interv=path.gate.filter(Boolean).length, first=path.gate.findIndex(Boolean);
  const minClear=Math.min(...data.pathClearance), minDelta=Math.min(...path.delta);
  els.stats.textContent = `intervention steps: ${{interv}} / ${{path.gate.length}}\\nfirst intervention index: ${{first}}\\nmin straight-path clearance: ${{minClear.toFixed(4)}}\\nmin score delta: ${{minDelta.toFixed(5)}}\\nscore start/end: ${{path.score[0].toFixed(3)}} -> ${{path.score[path.score.length-1].toFixed(3)}}`;
}}
for (const id of ['surface','fn','margin','scale','temp','eps','bad','good']) els[id].addEventListener('input', update);
window.addEventListener('resize', update); update();
</script>
</body>
</html>"""


if __name__ == "__main__":
    raise SystemExit(main())

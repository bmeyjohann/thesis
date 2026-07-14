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


def _scan_grid(obs: torch.Tensor, xy: np.ndarray, heading: float):
    scan = obs[0, 9:58].detach().cpu().numpy().reshape(7, 7)
    coords = np.linspace(-1.5, 1.5, 7, dtype=np.float32)
    c, s = math.cos(heading), math.sin(heading)
    pts = []
    local = []
    vals = []
    ij = []
    for i, forward in enumerate(coords):
        for j, lateral in enumerate(coords):
            world = xy + np.array([c * forward - s * lateral, s * forward + c * lateral], dtype=np.float32)
            pts.append(world)
            local.append([float(forward), float(lateral)])
            vals.append(float(scan[i, j]))
            ij.append([i, j])
    return np.asarray(pts), np.asarray(local), np.asarray(vals), ij, scan


def _action_world_vec(action: torch.Tensor, heading: float) -> np.ndarray:
    action_np = action[0, :3].detach().cpu().numpy()
    local = action_np[:2]
    c, s = math.cos(heading), math.sin(heading)
    vec = np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]], dtype=np.float32)
    if np.linalg.norm(vec) < 1e-6:
        yaw = float(action_np[2])
        turn_heading = heading + math.copysign(math.pi / 2.0, yaw if abs(yaw) > 1e-6 else 1.0)
        vec = np.array([math.cos(turn_heading), math.sin(turn_heading)], dtype=np.float32) * min(0.35, abs(yaw))
    norm = float(np.linalg.norm(vec))
    return vec / norm * 0.7 if norm > 1e-6 else vec


run_name = os.environ.get("RUN_NAME", f"unitree_height_scan_tuner_{time.strftime('%Y%m%d_%H%M%S')}")
out_dir = Path(os.environ.get("OUTPUT_DIR", str(ROOT / "visualizations" / "unitree_height_scan_tuner" / run_name)))
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
        scan_teacher_action = controller_action(obs, args, None, teacher_state)
        scan_pts, scan_local, scan_vals, ij, scan = _scan_grid(obs, xy, heading)
        snapshots.append(
            {
                "step": step,
                "xy": xy.tolist(),
                "heading": heading,
                "goal": goal.tolist(),
                "goal_body": obs[0, 6:8].detach().cpu().numpy().tolist(),
                "goal_distance": float(_current_goal_distance(env, 1, torch.device(args.device))[0].detach().cpu().item()),
                "scan_points_world": scan_pts.tolist(),
                "scan_points_local": scan_local.tolist(),
                "scan_values": scan_vals.tolist(),
                "scan_ij": ij,
                "scan_matrix": scan.tolist(),
                "scan_teacher_action": scan_teacher_action[0].detach().cpu().numpy().tolist(),
                "scan_teacher_world_vec": _action_world_vec(scan_teacher_action, heading).tolist(),
            }
        )
    if step == max_step:
        break
    action = controller_action(obs, args, None, teacher_state)
    obs_raw, _, done, _ = env.step(action)
    obs = _extract_actor_obs(obs_raw).to(device=args.device, dtype=torch.float32)
    if bool(torch.as_tensor(done).reshape(-1)[0].detach().cpu().item()):
        break

data = {
    "run_name": run_name,
    "layout_stats": layout_stats,
    "obstacle_cells": obstacle_cells[0].tolist(),
    "snapshots": snapshots,
    "defaults": {
        "threshold": 0.12,
        "flat_percentile": 90,
        "delta": 0.018,
        "required_free_width": 3,
        "corridor_half_width": 0.45,
        "clearance": 0.70,
        "horizon": 1.6,
        "candidates": 41,
        "goal_weight": 1.0,
        "clearance_weight": 45.0,
        "angle_weight": 0.12,
    },
}
json_path = out_dir / "height_scan_tuner_data.json"
json_path.write_text(json.dumps(data), encoding="utf-8")

# Static preview: show current scan teacher and width-3 corridor default from JS-equivalent logic.
fig, axes = plt.subplots(1, len(snapshots), figsize=(6.4 * len(snapshots), 6.2), dpi=150, squeeze=False)
obstacles = np.asarray(data["obstacle_cells"], dtype=np.float32)
for idx, snap in enumerate(snapshots):
    ax = axes[0][idx]
    xy = np.asarray(snap["xy"], dtype=np.float32)
    goal = np.asarray(snap["goal"], dtype=np.float32)
    scan_pts = np.asarray(snap["scan_points_world"], dtype=np.float32)
    vals = np.asarray(snap["scan_values"], dtype=np.float32)
    blocked = vals < data["defaults"]["threshold"]
    if obstacles.size:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], s=8, c="black", alpha=0.18, marker="s", label="obstacles")
    ax.scatter(scan_pts[:, 0], scan_pts[:, 1], c=vals, s=np.where(blocked, 95, 45), cmap="viridis", edgecolors=np.where(blocked, "red", "white"), linewidths=np.where(blocked, 1.6, 0.5), zorder=4)
    ax.scatter(xy[0], xy[1], c="white", edgecolors="black", s=110, zorder=5, label="robot")
    ax.scatter(goal[0], goal[1], marker="*", s=240, c="limegreen", edgecolors="black", zorder=5, label="goal")
    ax.arrow(xy[0], xy[1], math.cos(snap["heading"]) * 0.45, math.sin(snap["heading"]) * 0.45, color="deepskyblue", width=0.012, head_width=0.08, zorder=6, label="heading")
    vec = np.asarray(snap["scan_teacher_world_vec"], dtype=np.float32)
    ax.arrow(xy[0], xy[1], vec[0], vec[1], color="orange", width=0.022, head_width=0.12, zorder=7, label="current scan teacher")
    ax.set_title(f"step {snap['step']} | current action {np.asarray(snap['scan_teacher_action']).round(2).tolist()}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=7)
fig.tight_layout()
png_path = out_dir / "height_scan_tuner_static_preview.png"
fig.savefig(png_path)
plt.close(fig)

html = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Unitree Height Scan Teacher Tuner</title>
  <style>
    body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: #f6f2e8; color: #16140f; }
    #app { display: grid; grid-template-columns: 360px 1fr; gap: 16px; padding: 14px; }
    #panel { background: #fffaf0; border: 1px solid #d6c8aa; border-radius: 12px; padding: 14px; box-shadow: 0 2px 12px #0001; }
    label { display: grid; grid-template-columns: 150px 1fr 58px; gap: 8px; align-items: center; margin: 8px 0; font-size: 13px; }
    input, select { width: 100%; }
    canvas { background: #fbf7ee; border: 1px solid #d6c8aa; border-radius: 12px; width: 100%; height: calc(100vh - 34px); }
    .metric { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #efe4ce; border-radius: 8px; padding: 8px; margin-top: 10px; font-size: 12px; white-space: pre-wrap; }
    h2 { margin: 0 0 10px; font-size: 18px; }
  </style>
</head>
<body>
<div id="app">
  <div id="panel">
    <h2>Height Scan Teacher Tuner</h2>
    <label>snapshot <select id="snapshot"></select><span></span></label>
    <label>obstacle mode <select id="mode"><option value="threshold">absolute threshold</option><option value="delta">flat percentile delta</option></select><span></span></label>
    <label>threshold <input id="threshold" type="range" min="0.04" max="0.18" step="0.001"><span id="thresholdV"></span></label>
    <label>flat percentile <input id="flatPercentile" type="range" min="50" max="98" step="1"><span id="flatPercentileV"></span></label>
    <label>delta <input id="delta" type="range" min="0.002" max="0.06" step="0.001"><span id="deltaV"></span></label>
    <label>free width cells <input id="requiredFreeWidth" type="range" min="1" max="5" step="1"><span id="requiredFreeWidthV"></span></label>
    <label>corridor half width <input id="corridorHalfWidth" type="range" min="0.20" max="0.95" step="0.01"><span id="corridorHalfWidthV"></span></label>
    <label>clearance <input id="clearance" type="range" min="0.20" max="1.20" step="0.01"><span id="clearanceV"></span></label>
    <label>horizon <input id="horizon" type="range" min="0.5" max="2.5" step="0.05"><span id="horizonV"></span></label>
    <label>candidates <input id="candidates" type="range" min="11" max="81" step="2"><span id="candidatesV"></span></label>
    <label>goal weight <input id="goalWeight" type="range" min="0" max="4" step="0.05"><span id="goalWeightV"></span></label>
    <label>clearance weight <input id="clearanceWeight" type="range" min="0" max="150" step="1"><span id="clearanceWeightV"></span></label>
    <label>angle weight <input id="angleWeight" type="range" min="0" max="2" step="0.01"><span id="angleWeightV"></span></label>
    <div class="metric" id="metrics"></div>
  </div>
  <canvas id="canvas" width="1400" height="1000"></canvas>
</div>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const defaults = DATA.defaults;
const controls = {
  threshold: defaults.threshold,
  flatPercentile: defaults.flat_percentile,
  delta: defaults.delta,
  requiredFreeWidth: defaults.required_free_width,
  corridorHalfWidth: defaults.corridor_half_width,
  clearance: defaults.clearance,
  horizon: defaults.horizon,
  candidates: defaults.candidates,
  goalWeight: defaults.goal_weight,
  clearanceWeight: defaults.clearance_weight,
  angleWeight: defaults.angle_weight,
};
for (const [k,v] of Object.entries(controls)) {
  if ($(k)) $(k).value = v;
}
for (const [i,s] of DATA.snapshots.entries()) {
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `step ${s.step}`;
  $('snapshot').appendChild(opt);
}
$('mode').value = 'delta';
function percentile(arr, p) {
  const a = [...arr].sort((x,y)=>x-y);
  const idx = Math.min(a.length-1, Math.max(0, Math.round((p/100)*(a.length-1))));
  return a[idx];
}
function wrap(a) { return ((a + Math.PI) % (2*Math.PI)) - Math.PI; }
function compute(s) {
  const vals = s.scan_values;
  const flat = percentile(vals, +$('flatPercentile').value);
  const mode = $('mode').value;
  const threshold = +$('threshold').value;
  const delta = +$('delta').value;
  const blocked = vals.map(v => mode === 'threshold' ? v < threshold : v < flat - delta);
  const occ = s.scan_points_local.filter((_,i)=>blocked[i]);
  const goal = s.goal_body;
  const goalAngle = Math.atan2(goal[1], goal[0]);
  const candN = +$('candidates').value;
  const horizon = +$('horizon').value;
  const clearance = +$('clearance').value;
  const corridorHalf = +$('corridorHalfWidth').value;
  const requiredFreeWidth = +$('requiredFreeWidth').value;
  const candidates = [];
  for (let i=0;i<candN;i++) {
    const angle = -Math.PI*0.85 + (Math.PI*1.7)*i/(candN-1);
    const dir = [Math.cos(angle), Math.sin(angle)];
    let minD = 1e9, occupiedInCorridor = 0;
    for (const p of occ) {
      const along = p[0]*dir[0] + p[1]*dir[1];
      const lateral = -p[0]*dir[1] + p[1]*dir[0];
      if (along > -0.1 && along < horizon && Math.abs(lateral) < corridorHalf) occupiedInCorridor++;
      const clampedAlong = Math.min(horizon, Math.max(0.2, along));
      const dx = p[0] - clampedAlong*dir[0];
      const dy = p[1] - clampedAlong*dir[1];
      minD = Math.min(minD, Math.hypot(dx, dy));
    }
    // Estimate how many adjacent free cells cross the endpoint corridor.
    let freeWidth = 0;
    for (const p of s.scan_points_local) {
      const along = p[0]*dir[0] + p[1]*dir[1];
      const lateral = -p[0]*dir[1] + p[1]*dir[0];
      const idx = s.scan_points_local.indexOf(p);
      if (along > horizon-0.55 && along < horizon+0.55 && Math.abs(lateral) < corridorHalf && !blocked[idx]) freeWidth++;
    }
    const endpoint = [horizon*dir[0], horizon*dir[1]];
    const goalAfter = Math.hypot(goal[0]-endpoint[0], goal[1]-endpoint[1]);
    const clearPenalty = Math.max(0, clearance - minD) ** 2 * +$('clearanceWeight').value;
    const corridorPenalty = freeWidth >= requiredFreeWidth ? 0 : (requiredFreeWidth - freeWidth) * 8;
    const occPenalty = occupiedInCorridor * 4;
    const anglePenalty = Math.abs(wrap(angle - goalAngle)) * +$('angleWeight').value;
    const score = goalAfter * +$('goalWeight').value + clearPenalty + corridorPenalty + occPenalty + anglePenalty;
    candidates.push({angle, score, minD, freeWidth, occupiedInCorridor, dir});
  }
  candidates.sort((a,b)=>a.score-b.score);
  return {blocked, flat, candidates, best: candidates[0], goalAngle};
}
function bounds(points) {
  let xs = points.map(p=>p[0]), ys = points.map(p=>p[1]);
  return [Math.min(...xs)-1, Math.max(...xs)+1, Math.min(...ys)-1, Math.max(...ys)+1];
}
function draw() {
  for (const k of Object.keys(controls)) if ($(k+'V')) $(k+'V').textContent = (+$(k).value).toFixed(k.includes('Weight') || k === 'candidates' || k.includes('Width') ? 0 : 3);
  const s = DATA.snapshots[+$('snapshot').value];
  const r = compute(s);
  const canvas = $('canvas'), ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const pts = [...DATA.obstacle_cells, ...s.scan_points_world, s.xy, s.goal];
  const [xmin,xmax,ymin,ymax] = bounds(pts);
  const W=canvas.width,H=canvas.height,pad=50;
  const scale = Math.min((W-2*pad)/(xmax-xmin), (H-2*pad)/(ymax-ymin));
  const X = p => pad + (p[0]-xmin)*scale;
  const Y = p => H - (pad + (p[1]-ymin)*scale);
  function circle(p,r,fill,stroke='none',lw=1) {
    ctx.beginPath(); ctx.arc(X(p),Y(p),r,0,2*Math.PI); ctx.fillStyle=fill; ctx.fill();
    if (stroke !== 'none') { ctx.lineWidth=lw; ctx.strokeStyle=stroke; ctx.stroke(); }
  }
  function arrow(origin, vec, color, width=5) {
    const x1=X(origin), y1=Y(origin), x2=X([origin[0]+vec[0],origin[1]+vec[1]]), y2=Y([origin[0]+vec[0],origin[1]+vec[1]]);
    const a=Math.atan2(y2-y1,x2-x1);
    ctx.strokeStyle=color; ctx.lineWidth=width; ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2,y2); ctx.lineTo(x2-16*Math.cos(a-0.4),y2-16*Math.sin(a-0.4)); ctx.lineTo(x2-16*Math.cos(a+0.4),y2-16*Math.sin(a+0.4)); ctx.closePath(); ctx.fillStyle=color; ctx.fill();
  }
  ctx.fillStyle='rgba(0,0,0,0.18)';
  for (const p of DATA.obstacle_cells) ctx.fillRect(X(p)-3,Y(p)-3,6,6);
  for (let i=0;i<s.scan_points_world.length;i++) {
    const v=s.scan_values[i], b=r.blocked[i];
    const t=Math.max(0,Math.min(1,(v-0.05)/0.13));
    const col=`rgb(${Math.round(240*(1-t))},${Math.round(80+150*t)},${Math.round(80+100*t)})`;
    circle(s.scan_points_world[i], b?10:7, col, b?'red':'white', b?3:1);
  }
  const bestVec=[Math.cos(s.heading+r.best.angle)*0.85, Math.sin(s.heading+r.best.angle)*0.85];
  const current=s.scan_teacher_world_vec;
  arrow(s.xy, [Math.cos(s.heading)*0.55, Math.sin(s.heading)*0.55], 'deepskyblue', 4);
  arrow(s.xy, current, 'orange', 5);
  arrow(s.xy, bestVec, 'magenta', 5);
  circle(s.xy,12,'white','black',2);
  circle(s.goal,16,'limegreen','black',2);
  ctx.fillStyle='black'; ctx.font='18px ui-monospace, monospace';
  ctx.fillText('blue=heading orange=current teacher magenta=tuned corridor teacher', 18, 28);
  $('metrics').textContent =
    `step=${s.step}\n`+
    `flat=${r.flat.toFixed(4)} blocked=${r.blocked.filter(Boolean).length}/49\n`+
    `current_action=[${s.scan_teacher_action.map(x=>x.toFixed(2)).join(', ')}]\n`+
    `best_angle=${r.best.angle.toFixed(3)} score=${r.best.score.toFixed(2)} minD=${r.best.minD.toFixed(2)} freeWidth=${r.best.freeWidth} occCorridor=${r.best.occupiedInCorridor}\n`+
    `goal_angle=${r.goalAngle.toFixed(3)}`;
}
for (const id of ['snapshot','mode','threshold','flatPercentile','delta','requiredFreeWidth','corridorHalfWidth','clearance','horizon','candidates','goalWeight','clearanceWeight','angleWeight']) $(id).addEventListener('input', draw);
draw();
</script>
</body>
</html>
"""
html = html.replace("__DATA__", json.dumps(data))
html_path = out_dir / "height_scan_teacher_tuner.html"
html_path.write_text(html, encoding="utf-8")

print(json.dumps({"run_name": run_name, "html": str(html_path), "json": str(json_path), "png": str(png_path), "snapshots": snapshot_steps}), flush=True)
PY

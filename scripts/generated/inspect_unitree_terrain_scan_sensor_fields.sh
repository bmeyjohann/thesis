#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

python - <<'PY'
from argparse import Namespace
import json
import os
from pathlib import Path

import torch

from eval_unitree_nav_baselines import _reset_until_feasible, make_env
from train_unitree_nav_thesis import DEFAULT_LOW_LEVEL, _extract_actor_obs

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
    output_dir="",
    run_name="inspect_scan",
    teacher_scan_block_threshold=0.12,
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
    require_blocked_corridor=False,
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

env = make_env(args, num_envs=1, render=False)
obs_raw, _, _, layout_stats, obstacle_cells = _reset_until_feasible(args, env)
obs = _extract_actor_obs(obs_raw).to(device=args.device, dtype=torch.float32)
unwrapped = env.env.unwrapped

sensor = unwrapped.scene.sensors.get("terrain_scan") if hasattr(unwrapped.scene, "sensors") else None
if sensor is None:
    # Scene also supports item lookup in some mjlab versions.
    try:
        sensor = unwrapped.scene["terrain_scan"]
    except Exception:
        sensor = None
if sensor is None:
    raise SystemExit("terrain_scan sensor not found")

def tensor_info(x):
    if torch.is_tensor(x):
        arr = x.detach().cpu()
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(arr.float().min().item()) if arr.numel() else None,
            "max": float(arr.float().max().item()) if arr.numel() else None,
            "sample": arr.reshape(-1)[:12].tolist(),
        }
    return {"type": type(x).__name__, "repr": repr(x)[:500]}

data = getattr(sensor, "data", None)
fields = {}
for name in sorted(dir(data)):
    if name.startswith("_"):
        continue
    try:
        value = getattr(data, name)
    except Exception as exc:
        fields[name] = {"error": str(exc)}
        continue
    if torch.is_tensor(value) or name in {
        "ray_hits_w", "ray_starts_w", "ray_directions_w", "pos_w", "quat_w", "frame_pos_w", "frame_quat_w"
    }:
        fields[name] = tensor_info(value)
    elif name in {"pattern", "cfg"}:
        fields[name] = repr(value)[:1000]

scene_sensor_attrs = {}
for name in sorted(dir(sensor)):
    if name.startswith("_"):
        continue
    if name in {"data", "cfg"}:
        continue
    try:
        value = getattr(sensor, name)
    except Exception:
        continue
    if torch.is_tensor(value):
        scene_sensor_attrs[name] = tensor_info(value)
    elif name in {"pattern", "ray_starts", "ray_directions"}:
        scene_sensor_attrs[name] = tensor_info(value)

out = {
    "obs_shape": list(obs.shape),
    "obs_height_scan": obs[0, 9:58].detach().cpu().reshape(7, 7).tolist(),
    "layout_stats": layout_stats,
    "obstacle_cell_count": int(len(obstacle_cells[0])),
    "sensor_type": type(sensor).__name__,
    "data_type": type(data).__name__,
    "data_fields": fields,
    "sensor_tensor_attrs": scene_sensor_attrs,
}
print(json.dumps(out, indent=2), flush=True)
PY

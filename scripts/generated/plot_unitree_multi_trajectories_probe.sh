#!/usr/bin/env bash
set -euo pipefail

cd /home/benjamin/thesis

export MPLBACKEND=Agg

python - <<'PY'
from __future__ import annotations

import copy
import json
import math
import os
import time
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import eval_unitree_nav_baselines as eval_mod
from eval_unitree_nav_baselines import (
    _current_goal_distance,
    _goal_positions_xy,
    _reset_until_feasible,
    _robot_positions_xy,
    _terrain_obstacle_cells_by_env,
    controller_action,
    make_env,
)
from train_unitree_nav_thesis import DEFAULT_LOW_LEVEL, ROOT, ScanTeacherState, _extract_actor_obs, _extract_cost


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _install_round_heightfield_patch() -> None:
    original_apply_debug = eval_mod._apply_debug_obstacle_overrides

    def make_round_cfg(obstacle_cfg):
        import mjlab.terrains as terrain_gen

        class HfRoundDiscreteObstaclesTerrainCfg(terrain_gen.HfDiscreteObstaclesTerrainCfg):
            def function(self, difficulty, spec, rng):
                g = terrain_gen.HfDiscreteObstaclesTerrainCfg.function.__globals__
                np_mod = g["np"]
                uuid = g["uuid"]
                mujoco = g["mujoco"]
                color_by_height = g["color_by_height"]
                _compute_flat_patches = g["_compute_flat_patches"]
                TerrainGeometry = g["TerrainGeometry"]
                TerrainOutput = g["TerrainOutput"]

                body = spec.body("terrain")
                if self.border_width > 0 and self.border_width < self.horizontal_scale:
                    raise ValueError(
                        f"Border width ({self.border_width}) must be >= horizontal scale "
                        f"({self.horizontal_scale})"
                    )

                obs_height = self.obstacle_height_range[0] + difficulty * (
                    self.obstacle_height_range[1] - self.obstacle_height_range[0]
                )
                border_pixels = int(self.border_width / self.horizontal_scale)
                width_pixels = int(self.size[0] / self.horizontal_scale)
                length_pixels = int(self.size[1] / self.horizontal_scale)
                obs_h = int(obs_height / self.vertical_scale)
                obs_width_min = int(self.obstacle_width_range[0] / self.horizontal_scale)
                obs_width_max = int(self.obstacle_width_range[1] / self.horizontal_scale)
                platform_pixels = int(self.platform_width / self.horizontal_scale)

                if border_pixels > 0:
                    inner_width_pixels = width_pixels - 2 * border_pixels
                    inner_length_pixels = length_pixels - 2 * border_pixels
                else:
                    inner_width_pixels = width_pixels
                    inner_length_pixels = length_pixels
                noise = np_mod.zeros((inner_width_pixels, inner_length_pixels), dtype=np_mod.int16)

                obs_width_range = np_mod.arange(obs_width_min, obs_width_max + 1, 4)
                if len(obs_width_range) == 0:
                    obs_width_range = np_mod.array([obs_width_min])
                x_range = np_mod.arange(0, inner_width_pixels, 4)
                y_range = np_mod.arange(0, inner_length_pixels, 4)

                for _ in range(self.num_obstacles):
                    if self.obstacle_height_mode == "choice":
                        h = rng.choice(np_mod.array([-obs_h, -obs_h // 2, obs_h // 2, obs_h]))
                    else:
                        h = obs_h
                    diameter = int(rng.choice(obs_width_range))
                    radius = max(1, diameter // 2)
                    if len(x_range) == 0 or len(y_range) == 0:
                        continue
                    cx = int(rng.choice(x_range))
                    cy = int(rng.choice(y_range))
                    x0 = max(cx - radius, 0)
                    x1 = min(cx + radius + 1, inner_width_pixels)
                    y0 = max(cy - radius, 0)
                    y1 = min(cy + radius + 1, inner_length_pixels)
                    xx, yy = np_mod.ogrid[x0:x1, y0:y1]
                    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
                    patch = noise[x0:x1, y0:y1]
                    patch[mask] = h

                cx = inner_width_pixels // 2
                cy = inner_length_pixels // 2
                half_pf = platform_pixels // 2
                x0 = max(cx - half_pf, 0)
                x1 = min(cx + half_pf, inner_width_pixels)
                y0 = max(cy - half_pf, 0)
                y1 = min(cy + half_pf, inner_length_pixels)
                noise[x0:x1, y0:y1] = 0

                if border_pixels > 0:
                    outer_noise = np_mod.zeros((width_pixels, length_pixels), dtype=np_mod.int16)
                    outer_noise[
                        border_pixels : border_pixels + inner_width_pixels,
                        border_pixels : border_pixels + inner_length_pixels,
                    ] = noise
                    noise = outer_noise

                elevation_min = np_mod.min(noise)
                elevation_max = np_mod.max(noise)
                elevation_range = elevation_max - elevation_min if elevation_max != elevation_min else 1
                max_physical_height = elevation_range * self.vertical_scale
                base_thickness = max_physical_height * self.base_thickness_ratio
                normalized_elevation = (noise - elevation_min) / elevation_range if elevation_range > 0 else np_mod.zeros_like(noise)
                unique_id = uuid.uuid4().hex
                field = spec.add_hfield(
                    name=f"hfield_{unique_id}",
                    size=[self.size[0] / 2, self.size[1] / 2, max_physical_height, base_thickness],
                    nrow=noise.shape[0],
                    ncol=noise.shape[1],
                    userdata=normalized_elevation.flatten().astype(np_mod.float32).tolist(),
                )
                hfield_z_offset = elevation_min * self.vertical_scale if self.obstacle_height_mode == "choice" else 0
                material_name = color_by_height(spec, noise, unique_id, normalized_elevation)
                hfield_geom = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_HFIELD,
                    hfieldname=field.name,
                    pos=[self.size[0] / 2, self.size[1] / 2, hfield_z_offset],
                    material=material_name,
                )
                origin = np_mod.array([self.size[0] / 2, self.size[1] / 2, self.origin_z_offset])
                flat_patches = _compute_flat_patches(
                    noise,
                    self.vertical_scale,
                    self.horizontal_scale,
                    hfield_z_offset,
                    self.flat_patch_sampling,
                    rng,
                )
                geom = TerrainGeometry(geom=hfield_geom, hfield=field)
                return TerrainOutput(origin=origin, geometries=[geom], flat_patches=flat_patches)

        return HfRoundDiscreteObstaclesTerrainCfg(
            proportion=obstacle_cfg.proportion,
            size=obstacle_cfg.size,
            flat_patch_sampling=obstacle_cfg.flat_patch_sampling,
            obstacle_height_mode=obstacle_cfg.obstacle_height_mode,
            obstacle_width_range=obstacle_cfg.obstacle_width_range,
            obstacle_height_range=obstacle_cfg.obstacle_height_range,
            num_obstacles=obstacle_cfg.num_obstacles,
            platform_width=obstacle_cfg.platform_width,
            horizontal_scale=obstacle_cfg.horizontal_scale,
            vertical_scale=obstacle_cfg.vertical_scale,
            base_thickness_ratio=obstacle_cfg.base_thickness_ratio,
            border_width=obstacle_cfg.border_width,
            square_obstacles=True,
            origin_z_offset=obstacle_cfg.origin_z_offset,
        )

    def round_apply_debug(args, env_cfg):
        original_apply_debug(args, env_cfg)
        terrain = getattr(env_cfg.scene, "terrain", None)
        generator = getattr(terrain, "terrain_generator", None) if terrain is not None else None
        if generator is None or "discrete_obstacles" not in getattr(generator, "sub_terrains", {}):
            raise ValueError("round obstacle plot requires discrete_obstacles heightfield terrain")
        env_cfg.scene.terrain.terrain_generator = copy.deepcopy(generator)
        obstacle_cfg = env_cfg.scene.terrain.terrain_generator.sub_terrains["discrete_obstacles"]
        env_cfg.scene.terrain.terrain_generator.sub_terrains["discrete_obstacles"] = make_round_cfg(obstacle_cfg)

    eval_mod._apply_debug_obstacle_overrides = round_apply_debug


round_obstacles = _env_bool("ROUND_OBSTACLES", True)
if round_obstacles:
    _install_round_heightfield_patch()

run_name = os.environ.get(
    "RUN_NAME",
    f"unitree_multi_traj_{'round' if round_obstacles else 'box'}_{time.strftime('%Y%m%d_%H%M%S')}",
)
out_dir = Path(os.environ.get("OUTPUT_DIR", str(ROOT / "visualizations" / "unitree_nav_multi_trajectories" / run_name)))
out_dir.mkdir(parents=True, exist_ok=True)

args = Namespace(
    controller=os.environ.get("CONTROLLER", "scan_teacher"),
    model_path=os.environ.get("MODEL_PATH", ""),
    task=os.environ.get("TASK", "Unitree-G1-Nav-Obstacles-Safe-Collision"),
    device=os.environ.get("DEVICE", "cuda:0"),
    num_envs=1,
    num_episodes=_env_int("NUM_EPISODES", 9),
    episode_length_s=_env_float("EPISODE_LENGTH_S", 20.0),
    success_dist=_env_float("SUCCESS_DIST", 0.5),
    hidden_dim=_env_int("HIDDEN_DIM", 256),
    use_layer_norm=_env_bool("USE_LAYER_NORM", False),
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
    min_goal_obstacle_clearance=_env_float("MIN_GOAL_OBSTACLE_CLEARANCE", 0.9),
    goal_clearance_resample_attempts=_env_int("GOAL_CLEARANCE_RESAMPLE_ATTEMPTS", 50),
    min_start_obstacle_clearance=_env_float("MIN_START_OBSTACLE_CLEARANCE", 0.8),
    start_clearance_resample_attempts=_env_int("START_CLEARANCE_RESAMPLE_ATTEMPTS", 20),
    require_blocked_corridor=_env_bool("REQUIRE_BLOCKED_CORRIDOR", True),
    blocked_corridor_radius=_env_float("BLOCKED_CORRIDOR_RADIUS", 0.45),
    blocked_corridor_ignore_end_radius=_env_float("BLOCKED_CORRIDOR_IGNORE_END_RADIUS", 0.75),
    blocked_corridor_min_cells=_env_int("BLOCKED_CORRIDOR_MIN_CELLS", 1),
    blocked_corridor_resample_attempts=_env_int("BLOCKED_CORRIDOR_RESAMPLE_ATTEMPTS", 100),
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

env = make_env(args, num_envs=1, render=False)
teacher_state = ScanTeacherState(1, torch.device(args.device)) if args.controller == "scan_teacher" else None

episodes = []
for ep in range(int(args.num_episodes)):
    obs_raw, obstacle_cells, layout_stats, _, _ = _reset_until_feasible(args, env)
    if teacher_state is not None:
        teacher_state.reset(torch.tensor([0], device=args.device))
    obs = _extract_actor_obs(obs_raw).to(device=args.device, dtype=torch.float32)
    xy_hist = []
    cost_xy = []
    goal_xy = _goal_positions_xy(env)[0].copy()
    success_step = None
    total_cost = 0.0
    total_collision = 0.0
    total_fall = 0.0
    steps = int(round(args.episode_length_s / 0.05))
    for step in range(steps):
        xy = _robot_positions_xy(env)[0].copy()
        xy_hist.append(xy)
        live_dist = float(_current_goal_distance(env, 1, torch.device(args.device))[0].detach().cpu().item())
        if success_step is None and live_dist <= float(args.success_dist):
            success_step = step
        action = controller_action(obs, args, None, teacher_state)
        obs_raw, reward, done, extras = env.step(action)
        cost = _extract_cost(extras, 1, torch.device(args.device))
        c = float(cost[0].detach().cpu().item())
        total_cost += c
        if c > 0:
            cost_xy.append(xy.copy())
        try:
            cost_terms = extras.get("cost", {})
            if isinstance(cost_terms, dict):
                total_collision += float(cost_terms.get("collision", torch.zeros(1, device=args.device))[0].detach().cpu().item())
                total_fall += float(cost_terms.get("fall", torch.zeros(1, device=args.device))[0].detach().cpu().item())
        except Exception:
            pass
        obs = _extract_actor_obs(obs_raw).to(device=args.device, dtype=torch.float32)
        done_tensor = torch.as_tensor(done)
        if bool(done_tensor.reshape(-1)[0].detach().cpu().item()):
            break
    episodes.append(
        {
            "xy": np.asarray(xy_hist, dtype=np.float32),
            "cost_xy": np.asarray(cost_xy, dtype=np.float32) if cost_xy else np.zeros((0, 2), dtype=np.float32),
            "goal": goal_xy,
            "obstacles": obstacle_cells[0],
            "success_step": success_step,
            "total_cost": total_cost,
            "total_collision": total_collision,
            "total_fall": total_fall,
            "steps": len(xy_hist),
            "layout": layout_stats[0] if layout_stats else {},
        }
    )

cols = 3
rows = math.ceil(len(episodes) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.2 * rows), dpi=150, squeeze=False)
for idx, ep in enumerate(episodes):
    ax = axes[idx // cols][idx % cols]
    obstacles = ep["obstacles"]
    xy = ep["xy"]
    cost_xy = ep["cost_xy"]
    if obstacles.size:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], s=10, c="black", alpha=0.22, marker="s", label="obstacle hfield cells")
    ax.plot(xy[:, 0], xy[:, 1], c="dodgerblue", lw=2.0, label="trajectory")
    if len(xy):
        ax.scatter(xy[0, 0], xy[0, 1], c="white", edgecolors="black", s=80, zorder=4, label="start")
        ax.scatter(xy[-1, 0], xy[-1, 1], c="dodgerblue", edgecolors="black", s=55, zorder=4, label="end")
    ax.scatter(ep["goal"][0], ep["goal"][1], marker="*", s=230, c="limegreen", edgecolors="black", linewidths=1.0, zorder=5, label="goal")
    if cost_xy.size:
        ax.scatter(cost_xy[:, 0], cost_xy[:, 1], c="red", s=70, marker="x", linewidths=2.2, zorder=6, label="cost/collision")
    title = (
        f"ep {idx+1}: cost={ep['total_cost']:.1f}, coll={ep['total_collision']:.1f}, "
        f"succ={ep['success_step'] is not None}, steps={ep['steps']}"
    )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    if idx == 0:
        ax.legend(loc="upper right", fontsize=7)

for idx in range(len(episodes), rows * cols):
    axes[idx // cols][idx % cols].axis("off")

success_rate = sum(ep["success_step"] is not None for ep in episodes) / max(1, len(episodes))
costful_rate = sum(ep["total_cost"] > 0 for ep in episodes) / max(1, len(episodes))
mean_cost = sum(ep["total_cost"] for ep in episodes) / max(1, len(episodes))
fig.suptitle(
    f"{run_name} | round={round_obstacles} | success={success_rate:.2f} | costful={costful_rate:.2f} | mean_cost={mean_cost:.2f}",
    fontsize=12,
)
fig.tight_layout()
out_png = out_dir / "multi_trajectory_contact_sheet.png"
fig.savefig(out_png)
plt.close(fig)

summary = {
    "run_name": run_name,
    "round_obstacles": round_obstacles,
    "controller": args.controller,
    "num_episodes": len(episodes),
    "success_rate": success_rate,
    "costful_episode_rate": costful_rate,
    "mean_cost_sum": mean_cost,
    "episodes": [
        {
            "success_step": ep["success_step"],
            "total_cost": ep["total_cost"],
            "total_collision": ep["total_collision"],
            "total_fall": ep["total_fall"],
            "steps": ep["steps"],
            "blocked_corridor_cell_count": ep["layout"].get("blocked_cell_count"),
        }
        for ep in episodes
    ],
    "plot": str(out_png),
}
summary_path = out_dir / "multi_trajectory_summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary), flush=True)
PY

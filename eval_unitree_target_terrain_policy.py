"""Evaluate a flat-trained G1 locomotion policy on a target terrain arena."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from unitree_target_terrain import (
    BidirectionalTraversalTerrainCfg,
    HomogeneousSurfaceTerrainCfg,
    MATERIAL_STYLE,
    TargetArenaTerrainCfg,
    generate_target_terrain,
)


ROOT = Path(__file__).resolve().parent
UNITREE_REPO = ROOT / "external" / "unitree_rl_mjlab"
if str(UNITREE_REPO) not in sys.path:
    sys.path.insert(0, str(UNITREE_REPO))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=UNITREE_REPO / "logs" / "velocity" / "g1_flat" / "model_1499.pt",
    )
    parser.add_argument("--terrain-seed", type=int, default=3)
    parser.add_argument(
        "--terrain-mode", choices=("target", "flat", "surface", "ramp", "stairs"), default="target"
    )
    parser.add_argument("--surface", choices=("rigid", "ice", "sand"), default="rigid")
    parser.add_argument("--surface-sliding-friction", type=float)
    parser.add_argument("--surface-solref", type=float, nargs=2, metavar=("TIMECONST", "DAMPRATIO"))
    parser.add_argument(
        "--surface-solimp",
        type=float,
        nargs=5,
        metavar=("DMIN", "DMAX", "WIDTH", "MIDPOINT", "POWER"),
    )
    parser.add_argument("--surface-margin", type=float, default=0.0)
    parser.add_argument("--surface-gap", type=float, default=0.0)
    parser.add_argument("--surface-contact-priority", type=int, default=0)
    parser.add_argument("--surface-linear-drag", type=float, default=0.0, metavar="N_PER_MPS")
    parser.add_argument("--joint-damping-scale", type=float, default=1.0)
    parser.add_argument("--geometry-rise", type=float, default=0.3)
    parser.add_argument("--geometry-side-length", type=float, default=2.0)
    parser.add_argument("--geometry-width", type=float, default=2.0)
    parser.add_argument("--geometry-plateau-length", type=float, default=0.7)
    parser.add_argument("--geometry-steps-per-side", type=int, default=5)
    parser.add_argument("--preset", choices=("balanced", "traversal", "navigation"), default="balanced")
    parser.add_argument("--command-speed", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--num-paths", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=720)
    return parser.parse_args()


def _material_at(layout, x: float, y: float) -> str:
    material = "rigid"
    for brush in sorted(layout.materials, key=lambda item: item.layer):
        if math.hypot(x - brush.center[0], y - brush.center[1]) <= brush.radius:
            material = brush.kind
    return material


def _path_scenarios(layout, count: int) -> list[tuple[tuple[float, float], float]]:
    """Choose deterministic, spatially separated paths through varied terrain."""
    candidates: dict[str, list[tuple[float, tuple[float, float], float]]] = {
        "rigid": [], "ice": [], "sand": []
    }
    half = layout.arena_size / 2.0
    for x in np.arange(-8.0, 8.01, 1.0):
        for y in np.arange(-8.0, 8.01, 1.0):
            start = (float(x), float(y))
            if any(math.dist(start, obstacle.center) < max(obstacle.size[:2]) + 0.9 for obstacle in layout.obstacles):
                continue
            if any(math.dist(start, brush.center) < brush.radius + 0.45 for brush in layout.geometry):
                continue
            start_material = _material_at(layout, *start)
            for heading_deg in np.arange(-180.0, 180.0, 22.5):
                direction = np.array([math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))])
                points = np.asarray(start)[None, :] + np.linspace(0.0, 6.0, 49)[:, None] * direction[None, :]
                if np.any(np.abs(points) > half - 0.8):
                    continue
                materials = {_material_at(layout, float(point[0]), float(point[1])) for point in points}
                geometry_kinds = {
                    brush.kind
                    for brush in layout.geometry
                    if np.any(np.linalg.norm(points - np.asarray(brush.center), axis=1) <= brush.radius)
                }
                obstacle_crossing = any(
                    np.any(
                        np.linalg.norm(points - np.asarray(obstacle.center), axis=1)
                        <= max(obstacle.size[:2]) + 0.3
                    )
                    for obstacle in layout.obstacles
                )
                score = 3.0 * (len(materials) - 1) + 2.0 * len(geometry_kinds) + float(obstacle_crossing)
                candidates[start_material].append((score, start, float(heading_deg)))
    desired_materials = ("rigid", "ice", "sand", "rigid", "ice", "sand", "rigid", "rigid")
    selected: list[tuple[tuple[float, float], float]] = []
    for material in desired_materials[:count]:
        options = sorted(candidates[material], key=lambda item: (-item[0], item[1], item[2]))
        choice = next(
            (option for option in options if all(math.dist(option[1], prior[0]) >= 2.5 for prior in selected)),
            options[0],
        )
        selected.append((choice[1], choice[2]))
    return selected


def _draw_layout(ax, layout) -> None:
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.transforms import Affine2D

    half = layout.arena_size / 2.0
    ax.set_facecolor(MATERIAL_STYLE["rigid"]["rgba"])
    for brush in sorted(layout.materials, key=lambda item: item.layer):
        ax.add_patch(
            Circle(
                brush.center,
                brush.radius,
                color=MATERIAL_STYLE[brush.kind]["rgba"],
                alpha=0.66,
                linewidth=0,
            )
        )
    geometry_colors = {
        "rough": "#617540",
        "rubble": "#7a573d",
        "stairs": "#7d7d85",
        "ramp": "#868a91",
    }
    for brush in layout.geometry:
        ax.add_patch(
            Circle(
                brush.center,
                brush.radius,
                facecolor=geometry_colors[brush.kind],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.48,
                hatch="//" if brush.kind in {"stairs", "ramp"} else None,
            )
        )
    for obstacle in layout.obstacles:
        color = "#d61f26" if not obstacle.moving else "#d225b3"
        if obstacle.kind == "cylinder":
            patch = Circle(obstacle.center, obstacle.size[0], color=color, alpha=0.92)
        else:
            patch = Rectangle(
                (obstacle.center[0] - obstacle.size[0], obstacle.center[1] - obstacle.size[1]),
                2 * obstacle.size[0],
                2 * obstacle.size[1],
                color=color,
                alpha=0.92,
            )
            patch.set_transform(Affine2D().rotate_around(*obstacle.center, obstacle.yaw) + ax.transData)
        ax.add_patch(patch)
    ax.set(xlim=(-half, half), ylim=(-half, half), aspect="equal", xlabel="x (m)", ylabel="y (m)")
    ax.grid(color="white", alpha=0.14, linewidth=0.5)


def _plot_trajectories(output_path: Path, layout, trajectories, rows) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10.5, 9.2), constrained_layout=True)
    if layout is None:
        ax.set_facecolor(MATERIAL_STYLE["rigid"]["rgba"])
        ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)")
        ax.grid(color="white", alpha=0.14, linewidth=0.5)
    else:
        _draw_layout(ax, layout)
    colors = plt.cm.turbo(np.linspace(0.04, 0.96, len(trajectories)))
    for index, (trajectory, row, color) in enumerate(zip(trajectories, rows, colors, strict=True)):
        points = np.asarray(trajectory)
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2.4, label=f"{row['heading_deg']:.0f} deg")
        ax.scatter(points[0, 0], points[0, 1], s=30, color=color, edgecolor="white", zorder=6)
        if row["fell"]:
            ax.scatter(points[-1, 0], points[-1, 1], marker="X", s=130, color="#ff2020", edgecolor="white", zorder=8)
        else:
            ax.scatter(points[-1, 0], points[-1, 1], marker="*", s=150, color=color, edgecolor="black", zorder=8)
        direction = np.array([math.cos(math.radians(row["heading_deg"])), math.sin(math.radians(row["heading_deg"]))])
        ax.arrow(points[0, 0], points[0, 1], *(0.8 * direction), width=0.025, color=color, zorder=7)
    ax.set_title(
        f"Flat-trained G1 policy on {'continuous target terrain' if layout is not None else 'matched flat terrain'}\n"
        "fixed 0.6 m/s forward command; X = fall, star = survived full horizon"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#ff2020", markeredgecolor="white", markersize=10, label="fall"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="white", markeredgecolor="black", markersize=12, label="survived"),
    ])
    labels.extend(["fall", "survived"])
    ax.legend(handles, labels, ncol=2, loc="upper left", framealpha=0.88)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint_file.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.steps < 1 or args.command_speed <= 0:
        raise ValueError("steps and command speed must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.wrappers import VideoRecorder
    from eval_unitree_velocity_policy import _runner_checkpoint_path, _validate_robot_compatibility

    obs_dim, action_dim = _validate_robot_compatibility("Unitree-G1-Rough", checkpoint)
    if obs_dim != 98:
        raise ValueError(f"Expected the scan-blind flat policy to have 98 observations, got {obs_dim}")

    layout = (
        generate_target_terrain(args.terrain_seed, preset=args.preset, arena_size=24.0)
        if args.terrain_mode == "target"
        else None
    )
    scenarios = (
        _path_scenarios(layout, args.num_paths)
        if layout is not None
        else [
            ((0.0, 0.0), heading)
            for heading in (0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0)[: args.num_paths]
        ]
    )
    headings_deg = np.asarray([scenario[1] for scenario in scenarios])
    start_positions = np.asarray([scenario[0] for scenario in scenarios], dtype=float)
    num_envs = len(headings_deg)
    configure_torch_backends()
    task_id = "Unitree-G1-Flat" if args.terrain_mode == "flat" else "Unitree-G1-Rough"
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.seed = int(args.terrain_seed)
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.nconmax = max(512, int(env_cfg.sim.nconmax or 0))
    env_cfg.viewer.width = int(args.video_width)
    env_cfg.viewer.height = int(args.video_height)
    env_cfg.viewer.body_name = "torso_link"
    env_cfg.curriculum = {}
    env_cfg.events.pop("push_robot", None)
    env_cfg.events.pop("randomize_terrain", None)
    if args.terrain_mode == "surface":
        sliding = float(
            args.surface_sliding_friction
            if args.surface_sliding_friction is not None
            else MATERIAL_STYLE[args.surface]["friction"][0]
        )
        # MuJoCo combines both contacting geoms. Match foot tangential friction
        # to the terrain so a high default foot value cannot mask ice.
        env_cfg.events["foot_friction"].params["ranges"] = (sliding, sliding)
        if args.joint_damping_scale != 1.0:
            env_cfg.events["surface_joint_drag"] = EventTermCfg(
                func=dr.joint_damping,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                    "operation": "scale",
                    "ranges": (args.joint_damping_scale, args.joint_damping_scale),
                },
            )
    env_cfg.observations["actor"].terms.pop("height_scan", None)
    env_cfg.observations["critic"].terms.pop("height_scan", None)

    reset_base = env_cfg.events["reset_base"]
    reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)
    }
    reset_base.params["velocity_range"] = {}
    twist = env_cfg.commands["twist"]
    twist.resampling_time_range = (1e9, 1e9)
    twist.rel_standing_envs = 0.0
    twist.heading_command = False
    twist.debug_vis = False
    twist.ranges.lin_vel_x = (args.command_speed, args.command_speed)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.ranges.heading = None

    arena_size = 24.0
    if args.terrain_mode in {"target", "surface", "ramp", "stairs"}:
        terrain_size = (24.0, 24.0) if args.terrain_mode in {"target", "surface"} else (16.0, 8.0)
        subterrain = (
            TargetArenaTerrainCfg(
                proportion=1.0,
                size=(arena_size, arena_size),
                seed=int(args.terrain_seed),
                preset=args.preset,
            )
            if args.terrain_mode == "target"
            else HomogeneousSurfaceTerrainCfg(
                proportion=1.0,
                size=(arena_size, arena_size),
                material=args.surface,
                friction=(
                    (
                        float(args.surface_sliding_friction),
                        *MATERIAL_STYLE[args.surface]["friction"][1:],
                    )
                    if args.surface_sliding_friction is not None
                    else None
                ),
                contact_solref=tuple(args.surface_solref) if args.surface_solref else None,
                contact_solimp=tuple(args.surface_solimp) if args.surface_solimp else None,
                contact_margin=float(args.surface_margin),
                contact_gap=float(args.surface_gap),
                contact_priority=int(args.surface_contact_priority),
            )
            if args.terrain_mode == "surface"
            else BidirectionalTraversalTerrainCfg(
                proportion=1.0,
                size=terrain_size,
                kind=args.terrain_mode,
                rise=float(args.geometry_rise),
                side_length=float(args.geometry_side_length),
                width=float(args.geometry_width),
                plateau_length=float(args.geometry_plateau_length),
                steps_per_side=int(args.geometry_steps_per_side),
            )
        )
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=int(args.terrain_seed),
            curriculum=False,
            size=terrain_size,
            border_width=1.0,
            num_rows=1,
            num_cols=num_envs,
            # "none" overwrites every terrain geom with neutral gray in
            # MJLab. Height mode preserves our explicit rgba when no generated
            # height color is attached to the TerrainGeometry.
            color_scheme="height",
            sub_terrains={
                args.terrain_mode: subterrain
            },
            add_lights=True,
        )
        env_cfg.scene.terrain.max_init_terrain_level = 0

    render_mode = "rgb_array" if args.video else None
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=render_mode)
    wrapped_env = base_env
    if args.video:
        video_dir = args.output_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        wrapped_env = VideoRecorder(
            wrapped_env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=int(args.steps),
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(wrapped_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    with _runner_checkpoint_path(checkpoint) as runner_checkpoint:
        runner.load(str(runner_checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    obs, _ = env.reset()
    robot = base_env.scene["robot"]
    root_state = torch.cat(
        (
            robot.data.root_link_pos_w.clone(),
            robot.data.root_link_quat_w.clone(),
            robot.data.root_link_lin_vel_w.clone(),
            robot.data.root_link_ang_vel_w.clone(),
        ),
        dim=-1,
    )
    yaw = torch.as_tensor(np.radians(headings_deg), dtype=root_state.dtype, device=root_state.device)
    root_state[:, 0:2] = base_env.scene.env_origins[:, 0:2] + torch.as_tensor(
        start_positions, dtype=root_state.dtype, device=root_state.device
    )
    root_state[:, 3:7] = 0.0
    root_state[:, 3] = torch.cos(yaw / 2.0)
    root_state[:, 6] = torch.sin(yaw / 2.0)
    root_state[:, 7:13] = 0.0
    robot.write_root_state_to_sim(root_state)

    origins = base_env.scene.env_origins[:, :2].detach().cpu().numpy()
    trajectories: list[list[list[float]]] = [[start.tolist()] for start in start_positions]
    speeds: list[list[float]] = [[] for _ in range(num_envs)]
    root_heights: list[list[float]] = [[] for _ in range(num_envs)]
    alive = np.ones(num_envs, dtype=bool)
    fall_step = np.full(num_envs, -1, dtype=int)
    torso_ids, _ = robot.find_bodies("torso_link")
    torso_id = int(torso_ids[0])
    for step in range(int(args.steps)):
        if args.surface_linear_drag > 0.0:
            root_velocity = robot.data.root_link_lin_vel_w
            drag_forces = torch.zeros((num_envs, 1, 3), dtype=root_velocity.dtype, device=root_velocity.device)
            drag_forces[:, 0, :2] = -float(args.surface_linear_drag) * root_velocity[:, :2]
            if layout is not None:
                local_xy = robot.data.root_link_pos_w[:, :2].detach().cpu().numpy() - origins
                sand_mask = torch.as_tensor(
                    [_material_at(layout, float(x), float(y)) == "sand" for x, y in local_xy],
                    dtype=drag_forces.dtype,
                    device=drag_forces.device,
                )
                drag_forces *= sand_mask[:, None, None]
            drag_torques = torch.zeros_like(drag_forces)
            robot.write_external_wrench_to_sim(
                drag_forces,
                drag_torques,
                body_ids=[torso_id],
            )
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        done_np = dones.detach().cpu().numpy().astype(bool)
        newly_done = alive & done_np
        pos = robot.data.root_link_pos_w[:, :2].detach().cpu().numpy() - origins
        root_height = (
            robot.data.root_link_pos_w[:, 2] - base_env.scene.env_origins[:, 2]
        ).detach().cpu().numpy()
        vel = robot.data.root_link_lin_vel_b[:, 0].detach().cpu().numpy()
        for index in range(num_envs):
            # MJLab auto-resets done environments inside env.step(), so their
            # returned position is a reset pose and must not enter the path.
            if alive[index] and not newly_done[index]:
                trajectories[index].append(pos[index].tolist())
                speeds[index].append(float(vel[index]))
                root_heights[index].append(float(root_height[index]))
        fall_step[newly_done] = step + 1
        alive[newly_done] = False

    rows = []
    for index, heading_deg in enumerate(headings_deg):
        points = np.asarray(trajectories[index], dtype=float)
        if len(points) == 0:
            points = np.zeros((1, 2), dtype=float)
        direction = np.asarray([math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))])
        delta = points[-1] - points[0]
        path_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0
        default_material = args.surface if args.terrain_mode == "surface" else "rigid"
        materials = (
            [_material_at(layout, float(point[0]), float(point[1])) for point in points]
            if layout is not None
            else [default_material] * len(points)
        )
        steps_survived = int(fall_step[index]) if fall_step[index] >= 0 else int(args.steps)
        commanded_distance = float(args.command_speed * steps_survived * base_env.step_dt)
        tracking_ratio = float((delta @ direction) / commanded_distance) if commanded_distance > 0 else 0.0
        row = {
            "path_index": index,
            "start_xy": start_positions[index].tolist(),
            "start_material": (
                _material_at(layout, *start_positions[index]) if layout is not None else default_material
            ),
            "heading_deg": float(heading_deg),
            "steps_survived": steps_survived,
            "fell": bool(fall_step[index] >= 0),
            "fall_step": int(fall_step[index]),
            "duration_s": float(steps_survived * base_env.step_dt),
            "command_speed_mps": float(args.command_speed),
            "commanded_distance_m": commanded_distance,
            "command_progress_ratio": tracking_ratio,
            "mean_forward_speed_mps": float(np.mean(speeds[index])) if speeds[index] else 0.0,
            "mean_root_height_m": float(np.mean(root_heights[index])) if root_heights[index] else 0.0,
            "min_root_height_m": float(np.min(root_heights[index])) if root_heights[index] else 0.0,
            "command_direction_progress_m": float(delta @ direction),
            "lateral_drift_m": float(abs(delta @ np.asarray([-direction[1], direction[0]]))),
            "net_displacement_m": float(np.linalg.norm(delta)),
            "path_length_m": path_length,
            "ice_fraction": float(np.mean(np.asarray(materials) == "ice")),
            "sand_fraction": float(np.mean(np.asarray(materials) == "sand")),
            "functional_traversal_success": bool(fall_step[index] < 0 and tracking_ratio >= 0.5),
        }
        rows.append(row)

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_observation_dim": obs_dim,
        "checkpoint_action_dim": action_dim,
        "terrain_seed": int(args.terrain_seed),
        "terrain_mode": args.terrain_mode,
        "surface": args.surface if args.terrain_mode == "surface" else None,
        "surface_friction": (
            [
                float(args.surface_sliding_friction),
                *MATERIAL_STYLE[args.surface]["friction"][1:],
            ]
            if args.terrain_mode == "surface" and args.surface_sliding_friction is not None
            else list(MATERIAL_STYLE[args.surface]["friction"])
            if args.terrain_mode == "surface"
            else None
        ),
        "surface_solref": args.surface_solref,
        "surface_solimp": args.surface_solimp,
        "surface_margin": float(args.surface_margin),
        "surface_gap": float(args.surface_gap),
        "surface_contact_priority": int(args.surface_contact_priority),
        "surface_linear_drag_n_per_mps": float(args.surface_linear_drag),
        "joint_damping_scale": float(args.joint_damping_scale),
        "geometry_kind": args.terrain_mode if args.terrain_mode in {"ramp", "stairs"} else None,
        "geometry_rise": float(args.geometry_rise),
        "geometry_side_length": float(args.geometry_side_length),
        "geometry_width": float(args.geometry_width),
        "geometry_plateau_length": float(args.geometry_plateau_length),
        "geometry_steps_per_side": int(args.geometry_steps_per_side),
        "terrain_preset": args.preset,
        "moving_obstacles": "frozen_at_initial_pose",
        "num_paths": num_envs,
        "horizon_steps": int(args.steps),
        "step_dt": float(base_env.step_dt),
        "survival_rate": float(np.mean(~(fall_step >= 0))),
        "functional_traversal_success_rate": float(
            np.mean([row["functional_traversal_success"] for row in rows])
        ),
        "mean_steps_survived": float(np.mean([row["steps_survived"] for row in rows])),
        "mean_forward_speed_mps": float(np.mean([row["mean_forward_speed_mps"] for row in rows])),
        "mean_command_progress_m": float(np.mean([row["command_direction_progress_m"] for row in rows])),
        "paths": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "trajectories.json").write_text(
        json.dumps({"trajectories": trajectories}, indent=2) + "\n"
    )
    _plot_trajectories(args.output_dir / "trajectory_map.png", layout, trajectories, rows)
    print("[target-terrain-eval] " + json.dumps(summary, sort_keys=True))
    print(f"[target-terrain-eval] output={args.output_dir.resolve()}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

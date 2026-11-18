"""Shared target-terrain configuration for Unitree navigation train/eval."""

from __future__ import annotations

import argparse
import functools
import math

import numpy as np
import torch


def target_mover_spec(obstacle):
    """Physical, ray-visible mocap obstacle, not a viewer-only animation."""
    import mujoco
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="obstacle", mocap=True)
    round_shape = obstacle.kind == "cylinder"
    body.add_geom(name="collision", type=(mujoco.mjtGeom.mjGEOM_CYLINDER if round_shape
                  else mujoco.mjtGeom.mjGEOM_BOX),
                  size=((obstacle.size[0], obstacle.size[2]/2) if round_shape else
                        (obstacle.size[0], obstacle.size[1], obstacle.size[2]/2)),
                  rgba=(0.82, 0.12, 0.72, 1.0), friction=(0.9, 0.01, 0.001))
    return spec


def update_target_movers(env, env_ids, *, obstacles):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    t = env.episode_length_buf[ids].to(torch.float32) * env.step_dt
    for index, obstacle in enumerate(obstacles):
        phase = t * (2 * math.pi / obstacle.motion_period_s) + obstacle.motion_phase
        offset = obstacle.motion_amplitude * torch.sin(phase)
        pose = torch.zeros((len(ids), 7), device=env.device)
        pose[:, :3] = env.scene.env_origins[ids]
        pose[:, 0] += obstacle.center[0] + obstacle.motion_axis[0] * offset
        pose[:, 1] += obstacle.center[1] + obstacle.motion_axis[1] * offset
        pose[:, 2] += obstacle.size[2] / 2
        pose[:, 3] = math.cos(obstacle.yaw / 2)
        pose[:, 6] = math.sin(obstacle.yaw / 2)
        env.scene[f"target_mover_{index}"].write_mocap_pose_to_sim(pose, env_ids=ids)
    # Step events run after the environment's forward pass. Refresh poses so
    # both current height rays and browser meshes see the same physical objects.
    env.sim.forward()


def reset_target_terrain_origins(env, env_ids: torch.Tensor) -> None:
    """Assign each vector environment to its own deterministic arena patch."""
    from unitree_nav_layout import set_terrain_tile_indices

    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
    set_terrain_tile_indices(env, env_ids, env_ids)


def _point_clear_of_features(local: np.ndarray, layout, clearance: float, half: float) -> bool:
    if np.any(np.abs(local) > half):
        return False
    for obstacle in layout.obstacles:
        radius = 0.5 * math.hypot(float(obstacle.size[0]), float(obstacle.size[1]))
        if layout.preset == "challenge":
            radius = (obstacle.size[0] if obstacle.kind == "cylinder" else math.hypot(*obstacle.size[:2]))
            radius += obstacle.motion_amplitude if obstacle.moving else 0.0
        if np.linalg.norm(local - np.asarray(obstacle.center)) < radius + clearance:
            return False
    for brush in layout.geometry:
        if np.linalg.norm(local - np.asarray(brush.center)) < float(brush.radius) + clearance:
            return False
    return True


def _point_clear_of_obstacles(local: np.ndarray, layout, clearance: float, half: float) -> bool:
    """Check endpoint feasibility while allowing traversable terrain geometry."""
    if np.any(np.abs(local) > half):
        return False
    if layout.preset == "challenge":
        # Difficult features are route choices, not mandatory unreachable goals.
        for brush in layout.geometry:
            if brush.kind in {"stairs", "ramp"} and brush.strength >= 2.6:
                if np.linalg.norm(local - np.asarray(brush.center)) < 1.35 * brush.radius + clearance:
                    return False
    for obstacle in layout.obstacles:
        radius = 0.5 * math.hypot(float(obstacle.size[0]), float(obstacle.size[1]))
        if layout.preset == "challenge":
            radius = (obstacle.size[0] if obstacle.kind == "cylinder" else math.hypot(*obstacle.size[:2]))
            radius += obstacle.motion_amplitude if obstacle.moving else 0.0
        if np.linalg.norm(local - np.asarray(obstacle.center)) < radius + clearance:
            return False
    return True


def _sample_safe_local_points(
    count: int,
    layout,
    *,
    boundary_margin: float,
    clearance: float,
    attempts: int,
) -> np.ndarray:
    """Sample safe starts across the whole target arena."""
    half = float(layout.arena_size) / 2.0 - boundary_margin
    points: list[np.ndarray] = []
    for _ in range(count):
        accepted = None
        for _ in range(attempts):
            candidate = np.random.uniform(-half, half, size=2)
            if _point_clear_of_features(candidate, layout, clearance, half):
                accepted = candidate
                break
        if accepted is None:
            grid = np.linspace(-half, half, 41)
            candidates = [np.asarray([x, y]) for x in grid for y in grid]
            accepted = next(
                (candidate for candidate in candidates if _point_clear_of_features(candidate, layout, clearance, half)),
                None,
            )
        if accepted is None:
            raise RuntimeError("Could not sample a geometry-clear target-terrain start")
        points.append(accepted)
    return np.stack(points)


def reset_target_navigation_root_state(
    env,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg=None,
    *,
    terrain_seed: int,
    terrain_preset: str,
    terrain_arena_size: float,
    start_boundary_margin: float,
    start_clearance: float,
    start_attempts: int,
) -> None:
    """Reset the robot at a safe, arena-wide position instead of one corner."""
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform
    from unitree_target_terrain import generate_target_terrain

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    if asset.is_fixed_base:
        raise ValueError("Target navigation reset expects a floating-base robot")

    layout = generate_target_terrain(
        int(terrain_seed), preset=str(terrain_preset), arena_size=float(terrain_arena_size)
    )
    local_xy = _sample_safe_local_points(
        len(env_ids),
        layout,
        boundary_margin=float(start_boundary_margin),
        clearance=float(start_clearance),
        attempts=int(start_attempts),
    )
    default = asset.data.default_root_state[env_ids].clone()
    positions = default[:, :3] + env.scene.env_origins[env_ids]
    positions[:, :2] += torch.as_tensor(local_xy, dtype=positions.dtype, device=positions.device)

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    pose_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 4), device=env.device)
    positions[:, 2] += pose_samples[:, 0]
    delta = quat_from_euler_xyz(pose_samples[:, 1], pose_samples[:, 2], pose_samples[:, 3])
    orientations = quat_mul(default[:, 3:7], delta)

    velocity_range = velocity_range or {}
    velocity_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    velocity_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in velocity_keys], device=env.device
    )
    velocities = default[:, 7:13] + sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=env.device
    )
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def _segment_feature_labels(start: np.ndarray, goal: np.ndarray, layout) -> tuple[str, ...]:
    """Return terrain features intersected by the direct start-goal segment."""
    distance = float(np.linalg.norm(goal - start))
    samples = max(16, int(math.ceil(distance / 0.15)))
    points = start[None, :] + np.linspace(0.0, 1.0, samples)[:, None] * (goal - start)[None, :]
    labels: set[str] = set()
    for brush in layout.materials:
        if np.any(np.linalg.norm(points - np.asarray(brush.center), axis=1) <= float(brush.radius)):
            labels.add(brush.kind)
    for brush in layout.geometry:
        if np.any(np.linalg.norm(points - np.asarray(brush.center), axis=1) <= float(brush.radius)):
            labels.add(brush.kind)
    for obstacle in layout.obstacles:
        radius = 0.5 * math.hypot(float(obstacle.size[0]), float(obstacle.size[1]))
        if np.any(np.linalg.norm(points - np.asarray(obstacle.center), axis=1) <= radius):
            labels.add("moving_obstacle" if obstacle.moving else "static_obstacle")
    return tuple(sorted(labels))


def update_target_goal_curriculum(args, global_step: int) -> None:
    """Update effective goal distance and terrain-crossing targets."""
    if not bool(getattr(args, "target_terrain_goal_curriculum", False)):
        args.target_terrain_goal_distance_min_effective = float(args.goal_distance_min)
        args.target_terrain_goal_distance_max_effective = float(args.goal_distance_max)
        args.target_terrain_goal_feature_fraction_effective = float(
            getattr(args, "target_terrain_goal_feature_fraction", 0.0)
        )
        return
    duration = max(1, int(getattr(args, "target_terrain_goal_curriculum_steps", 20000)))
    progress = min(1.0, max(0.0, float(global_step) / duration))
    start_min = float(getattr(args, "target_terrain_goal_curriculum_start_min", 0.75))
    start_max = float(getattr(args, "target_terrain_goal_curriculum_start_max", 2.0))
    args.target_terrain_goal_distance_min_effective = start_min + progress * (float(args.goal_distance_min) - start_min)
    args.target_terrain_goal_distance_max_effective = start_max + progress * (float(args.goal_distance_max) - start_max)
    start_fraction = float(getattr(args, "target_terrain_goal_curriculum_start_feature_fraction", 0.0))
    final_fraction = float(getattr(args, "target_terrain_goal_feature_fraction", 0.8))
    feature_progress = min(1.0, max(0.0, (progress - 0.5) / 0.5))
    args.target_terrain_goal_feature_fraction_effective = start_fraction + feature_progress * (
        final_fraction - start_fraction
    )


def configure_target_navigation_terrain(env_cfg, args, *, num_envs: int) -> bool:
    """Replace the legacy obstacle tiles with deterministic target arenas."""
    if not bool(getattr(args, "target_terrain", False)):
        return False

    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
    from unitree_target_terrain import TargetArenaTerrainCfg, generate_target_terrain

    arena_size = float(getattr(args, "target_terrain_arena_size", 24.0))
    if float(getattr(args, "target_terrain_material_resolution", 1.0)) <= 0:
        raise ValueError("Target material resolution must be positive")
    if float(getattr(args, "target_terrain_ice_friction", 0.30)) < 0:
        raise ValueError("Ice friction must be non-negative")
    configured_seed = int(getattr(args, "target_terrain_seed", -1))
    seed = configured_seed if configured_seed >= 0 else int(getattr(args, "seed", 0))
    rows = max(1, int(math.floor(math.sqrt(num_envs))))
    cols = int(math.ceil(num_envs / rows))
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=seed,
        curriculum=False,
        size=(arena_size, arena_size),
        border_width=1.0,
        num_rows=rows,
        num_cols=cols,
        color_scheme="height",
        sub_terrains={
            "target": TargetArenaTerrainCfg(
                proportion=1.0,
                size=(arena_size, arena_size),
                seed=seed,
                preset=str(getattr(args, "target_terrain_preset", "balanced")),
                material_resolution=float(getattr(args, "target_terrain_material_resolution", 1.0)),
                dynamic_obstacles=bool(getattr(args, "target_terrain_dynamic_obstacles", False)),
                ice_friction=float(getattr(args, "target_terrain_ice_friction", 0.30)),
                ice_contact_priority=int(getattr(args, "target_terrain_ice_contact_priority", 0)),
            )
        },
        add_lights=True,
    )
    env_cfg.scene.terrain.max_init_terrain_level = None
    layout = generate_target_terrain(
        seed,
        preset=str(getattr(args, "target_terrain_preset", "balanced")),
        arena_size=arena_size,
    )
    args.target_terrain_layout = layout.to_dict()
    start_jitter = max(0.0, float(getattr(args, "target_terrain_start_jitter", 0.25)))
    reset_base = env_cfg.events.get("reset_base")
    if reset_base is not None:
        pose_range = dict(reset_base.params.get("pose_range", {}))
        pose_range["x"] = (-start_jitter, start_jitter)
        pose_range["y"] = (-start_jitter, start_jitter)
        reset_base.func = reset_target_navigation_root_state
        reset_base.params = {
            **reset_base.params,
            "pose_range": pose_range,
            "terrain_seed": seed,
            "terrain_preset": str(getattr(args, "target_terrain_preset", "balanced")),
            "terrain_arena_size": arena_size,
            "start_boundary_margin": float(getattr(args, "target_terrain_start_boundary_margin", 1.25)),
            "start_clearance": float(getattr(args, "target_terrain_start_clearance", 0.75)),
            "start_attempts": int(getattr(args, "target_terrain_start_attempts", 256)),
        }
    from mjlab.managers.event_manager import EventTermCfg

    env_cfg.events = {
        "assign_target_arena": EventTermCfg(
            func=reset_target_terrain_origins,
            mode="reset",
        ),
        **env_cfg.events,
    }
    if bool(getattr(args, "target_terrain_dynamic_obstacles", False)):
        from mjlab.entity import EntityCfg
        movers = tuple(obstacle for obstacle in layout.obstacles if obstacle.moving)
        for index, obstacle in enumerate(movers):
            env_cfg.scene.entities[f"target_mover_{index}"] = EntityCfg(
                spec_fn=functools.partial(target_mover_spec, obstacle=obstacle),
                init_state=EntityCfg.InitialStateCfg(
                    pos=(*obstacle.center, obstacle.size[2]/2),
                    rot=(math.cos(obstacle.yaw/2), 0.0, 0.0, math.sin(obstacle.yaw/2))))
        for mode in ("reset", "step"):
            env_cfg.events[f"target_movers_{mode}"] = EventTermCfg(
                func=update_target_movers, mode=mode, params={"obstacles": movers})
    env_cfg.curriculum.pop("terrain_levels", None)
    env_cfg.events.pop("randomize_terrain", None)
    # The legacy obstacle task treats any non-foot leg contact with the entire
    # terrain body as an obstacle collision. Direct-geom ramps/material tiles
    # therefore produce false collision cost during ordinary traversal. Keep
    # fall cost/termination, but remove that legacy heightfield-specific term.
    if getattr(env_cfg, "costs", None) is not None:
        env_cfg.costs.pop("collision", None)
    env_cfg.scene.sensors = tuple(
        sensor
        for sensor in tuple(env_cfg.scene.sensors or ())
        if getattr(sensor, "name", "") != "obstacle_collision"
    )
    env_cfg.sim.nconmax = max(int(getattr(env_cfg.sim, "nconmax", 0) or 0), 512)
    if bool(getattr(args, "target_terrain_dynamic_obstacles", False)):
        env_cfg.sim.njmax = max(int(getattr(env_cfg.sim, "njmax", 0) or 0), 2048)
    return True


def add_target_terrain_args(parser) -> None:
    parser.add_argument("--target-terrain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-terrain-preset", choices=["balanced", "traversal", "navigation", "challenge"], default="balanced")
    parser.add_argument("--target-terrain-seed", type=int, default=-1)
    parser.add_argument("--target-terrain-arena-size", type=float, default=24.0)
    parser.add_argument("--target-terrain-material-resolution", type=float, default=1.0)
    parser.add_argument("--target-terrain-dynamic-obstacles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-terrain-ice-friction", type=float, default=0.30)
    parser.add_argument("--target-terrain-ice-contact-priority", type=int, default=0)
    parser.add_argument("--target-terrain-sand-linear-drag", type=float, default=120.0)
    parser.add_argument("--target-terrain-start-jitter", type=float, default=0.25)
    parser.add_argument("--target-terrain-start-boundary-margin", type=float, default=1.25)
    parser.add_argument("--target-terrain-start-clearance", type=float, default=0.75)
    parser.add_argument("--target-terrain-start-attempts", type=int, default=256)
    parser.add_argument("--target-terrain-goal-boundary-margin", type=float, default=0.75)
    parser.add_argument("--target-terrain-goal-clearance", type=float, default=0.75)
    parser.add_argument("--target-terrain-goal-attempts", type=int, default=256)
    parser.add_argument("--target-terrain-goal-feature-fraction", type=float, default=0.0)
    parser.add_argument("--target-terrain-goal-curriculum", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-terrain-goal-curriculum-steps", type=int, default=20000)
    parser.add_argument("--target-terrain-goal-curriculum-start-min", type=float, default=0.75)
    parser.add_argument("--target-terrain-goal-curriculum-start-max", type=float, default=2.0)
    parser.add_argument("--target-terrain-goal-curriculum-start-feature-fraction", type=float, default=0.0)


def sample_target_goal_positions(env, args, env_ids: torch.Tensor) -> torch.Tensor:
    """Sample robot-relative goals inside the arena and away from geometry."""
    from unitree_target_terrain import generate_target_terrain

    unwrapped = env.env.unwrapped
    device = unwrapped.scene.env_origins.device
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=device).reshape(-1)
    origins = unwrapped.scene.env_origins[env_ids, :2].detach().cpu().numpy()
    robot_xy = unwrapped.scene["robot"].data.root_link_pos_w[env_ids, :2].detach().cpu().numpy()
    arena_size = float(getattr(args, "target_terrain_arena_size", 24.0))
    seed = int(getattr(args, "target_terrain_seed", -1))
    if seed < 0:
        seed = int(getattr(args, "seed", 0))
    layout = generate_target_terrain(
        seed,
        preset=str(getattr(args, "target_terrain_preset", "balanced")),
        arena_size=arena_size,
    )
    goal_min = float(getattr(args, "target_terrain_goal_distance_min_effective", args.goal_distance_min))
    goal_max = float(getattr(args, "target_terrain_goal_distance_max_effective", args.goal_distance_max))
    if goal_min <= 0.0 or goal_max < goal_min:
        raise ValueError("Target-terrain goals require 0 < goal_distance_min <= goal_distance_max")
    boundary = max(0.0, float(getattr(args, "target_terrain_goal_boundary_margin", 0.75)))
    clearance = max(0.0, float(getattr(args, "target_terrain_goal_clearance", 0.75)))
    attempts = max(1, int(getattr(args, "target_terrain_goal_attempts", 256)))
    half = arena_size / 2.0 - boundary

    feature_fraction = float(
        getattr(args, "target_terrain_goal_feature_fraction_effective", getattr(args, "target_terrain_goal_feature_fraction", 0.0))
    )

    goals: list[np.ndarray] = []
    for origin, robot_world in zip(origins, robot_xy):
        robot_local = robot_world - origin
        accepted = None
        require_feature = bool(np.random.uniform() < feature_fraction)
        for _ in range(attempts):
            angle = float(np.random.uniform(-math.pi, math.pi))
            distance = float(np.random.uniform(goal_min, goal_max))
            candidate = robot_local + distance * np.asarray([math.cos(angle), math.sin(angle)])
            if _point_clear_of_obstacles(candidate, layout, clearance, half) and (
                not require_feature or _segment_feature_labels(robot_local, candidate, layout)
            ):
                accepted = candidate
                break
        if accepted is None:
            radii = np.linspace(goal_min, goal_max, 9)
            angles = np.linspace(-math.pi, math.pi, 128, endpoint=False)
            candidates = [
                robot_local + radius * np.asarray([math.cos(angle), math.sin(angle)])
                for radius in radii
                for angle in angles
            ]
            accepted = next(
                (
                    candidate
                    for candidate in candidates
                    if _point_clear_of_obstacles(candidate, layout, clearance, half)
                    and (not require_feature or _segment_feature_labels(robot_local, candidate, layout))
                ),
                None,
            )
        if accepted is None and require_feature:
            # Intermediate curriculum annuli can contain no feature from a
            # particular random start. Preserve a valid navigation episode
            # rather than crashing; episode metadata records whether the route
            # was actually non-trivial, and the final 4-8 m cohort is audited.
            for _ in range(attempts):
                angle = float(np.random.uniform(-math.pi, math.pi))
                distance = float(np.random.uniform(goal_min, goal_max))
                candidate = robot_local + distance * np.asarray([math.cos(angle), math.sin(angle)])
                if _point_clear_of_obstacles(candidate, layout, clearance, half):
                    accepted = candidate
                    break
            if accepted is not None:
                fallback_count = int(getattr(unwrapped, "_target_goal_feature_fallback_count", 0)) + 1
                unwrapped._target_goal_feature_fallback_count = fallback_count
                if fallback_count <= 10:
                    print(
                        "[target-goal] feature-crossing fallback "
                        f"count={fallback_count} distance={goal_min:.2f}-{goal_max:.2f}m",
                        flush=True,
                    )
        if accepted is None:
            raise RuntimeError("Could not sample an in-bounds, obstacle-clear target-terrain goal")
        goals.append(origin + accepted)
    return torch.as_tensor(np.stack(goals), dtype=torch.float32, device=device)

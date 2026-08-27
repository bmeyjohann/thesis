"""Seeded continuous MuJoCo target terrains for Unitree locomotion research.

The arena is intentionally not a grid of independent training cells.  Circular
brushes paint local surface materials and geometry onto one shared floor, so
the robot has to cross transitions and can encounter combinations created by
overlapping layers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Iterable

import mujoco
import numpy as np

from mjlab.terrains import SubTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput


MATERIAL_STYLE = {
    "rigid": {
        "rgba": (0.42, 0.45, 0.43, 1.0),
        "friction": (0.9, 0.01, 0.001),
        "speed_scale": 1.0,
    },
    "ice": {
        "rgba": (0.20, 0.68, 0.96, 1.0),
        "friction": (0.055, 0.004, 0.0005),
        "speed_scale": 0.72,
    },
    "sand": {
        "rgba": (0.93, 0.72, 0.20, 1.0),
        "friction": (1.25, 0.025, 0.004),
        "speed_scale": 0.55,
    },
}

GEOMETRY_COLORS = {
    "rough": (0.38, 0.46, 0.25, 1.0),
    "rubble": (0.48, 0.34, 0.24, 1.0),
    "stairs": (0.48, 0.48, 0.52, 1.0),
    "ramp": (0.48, 0.50, 0.55, 1.0),
}


def _geometry_color(kind: str, material: str) -> tuple[float, float, float, float]:
    """Tint geometry by the local material so crossed brush layers stay visible."""
    geometry = np.asarray(GEOMETRY_COLORS[kind][:3])
    surface = np.asarray(MATERIAL_STYLE[material]["rgba"][:3])
    rgb = 0.55 * geometry + 0.45 * surface
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def _add_exclusive_material_surface(
    body: mujoco.MjsBody,
    layout: TargetTerrainLayout,
    *,
    min_x: float,
    min_y: float,
    prefix: str,
    resolution: float = 0.5,
) -> list[mujoco.MjsGeom]:
    """Rasterize final paint order into non-overlapping material strips."""
    cells = int(math.ceil(layout.arena_size / resolution))
    edges_x = np.linspace(min_x, min_x + layout.arena_size, cells + 1)
    edges_y = np.linspace(min_y, min_y + layout.arena_size, cells + 1)
    geoms: list[mujoco.MjsGeom] = []
    for row in range(cells):
        y0, y1 = float(edges_y[row]), float(edges_y[row + 1])
        y = (y0 + y1) / 2.0
        materials = [
            _material_at(layout, (float(edges_x[col]) + float(edges_x[col + 1])) / 2.0, y)
            for col in range(cells)
        ]
        run_start = 0
        while run_start < cells:
            material = materials[run_start]
            run_end = run_start + 1
            while run_end < cells and materials[run_end] == material:
                run_end += 1
            x0, x1 = float(edges_x[run_start]), float(edges_x[run_end])
            style = MATERIAL_STYLE[material]
            geoms.append(
                body.add_geom(
                    name=f"{prefix}_{row}_{run_start}_{material}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=((x0 + x1) / 2.0, y, -0.05),
                    size=((x1 - x0) / 2.0, (y1 - y0) / 2.0, 0.05),
                    friction=style["friction"],
                    rgba=style["rgba"],
                )
            )
            run_start = run_end
    return geoms


@dataclass(frozen=True)
class MaterialBrush:
    kind: str
    center: tuple[float, float]
    radius: float
    layer: int


@dataclass(frozen=True)
class GeometryBrush:
    kind: str
    center: tuple[float, float]
    radius: float
    yaw: float
    strength: float


@dataclass(frozen=True)
class Obstacle:
    kind: str
    center: tuple[float, float]
    size: tuple[float, float, float]
    yaw: float
    moving: bool = False
    motion_axis: tuple[float, float] = (1.0, 0.0)
    motion_amplitude: float = 0.0
    motion_period_s: float = 5.0
    motion_peak_speed_mps: float = 0.0
    motion_phase: float = 0.0


@dataclass(frozen=True)
class TargetTerrainLayout:
    seed: int
    preset: str
    arena_size: float
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    materials: tuple[MaterialBrush, ...]
    geometry: tuple[GeometryBrush, ...]
    obstacles: tuple[Obstacle, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PRESET_COUNTS = {
    "balanced": {"materials": 11, "geometry": 9, "static": 8, "moving": 2},
    "traversal": {"materials": 14, "geometry": 13, "static": 5, "moving": 1},
    "navigation": {"materials": 9, "geometry": 7, "static": 13, "moving": 4},
}

MOVING_OBSTACLE_SPEED_RANGE_MPS = (0.65, 1.8)
MOVING_OBSTACLE_AMPLITUDE_RANGE_M = (0.45, 1.4)


def _sample_moving_obstacle_motion(rng: np.random.Generator) -> tuple[float, float]:
    """Return amplitude and period with sweep distance decreasing as speed rises."""
    min_speed, max_speed = MOVING_OBSTACLE_SPEED_RANGE_MPS
    min_amplitude, max_amplitude = MOVING_OBSTACLE_AMPLITUDE_RANGE_M
    peak_speed = float(rng.uniform(min_speed, max_speed))
    speed_fraction = (peak_speed - min_speed) / (max_speed - min_speed)
    amplitude = float(max_amplitude + speed_fraction * (min_amplitude - max_amplitude))
    period = float(2.0 * math.pi * amplitude / peak_speed)
    return amplitude, period


def _sample_center(
    rng: np.random.Generator,
    arena_size: float,
    radius: float,
    protected: Iterable[tuple[tuple[float, float], float]],
    *,
    attempts: int = 200,
) -> tuple[float, float]:
    half = arena_size / 2.0
    for _ in range(attempts):
        value = tuple(rng.uniform(-half + radius, half - radius, size=2).tolist())
        if all(math.dist(value, center) >= radius + margin for center, margin in protected):
            return value
    # Deterministic fallback near the arena center.  This keeps generation total
    # while metadata makes the placement visible to downstream checks.
    return (0.0, 0.0)


def generate_target_terrain(
    seed: int,
    *,
    preset: str = "balanced",
    arena_size: float = 24.0,
) -> TargetTerrainLayout:
    """Generate a deterministic continuous arena from independent brush layers."""
    if preset not in PRESET_COUNTS:
        raise ValueError(f"Unknown preset {preset!r}; expected one of {tuple(PRESET_COUNTS)}")
    if arena_size < 16.0:
        raise ValueError("arena_size must be at least 16 m")

    rng = np.random.default_rng(int(seed))
    counts = PRESET_COUNTS[preset]
    half = arena_size / 2.0
    start_xy = (-half + 2.2, -half + 2.2)
    goal_xy = (half - 2.2, half - 2.2)
    protected = ((start_xy, 1.4), (goal_xy, 1.4))

    materials: list[MaterialBrush] = []
    kinds = ("ice", "sand")
    for index in range(counts["materials"]):
        radius = float(rng.uniform(1.7, 3.5))
        materials.append(
            MaterialBrush(
                kind=kinds[index % len(kinds)],
                center=_sample_center(rng, arena_size, radius, protected),
                radius=radius,
                layer=index,
            )
        )

    geometry: list[GeometryBrush] = []
    # Traversable elevation changes are the primary target-terrain challenge;
    # weight them twice as often as rough/rubble patches.
    geometry_kinds = ("stairs", "ramp", "rough", "rubble", "stairs", "ramp")
    for index in range(counts["geometry"]):
        radius = float(rng.uniform(1.5, 3.1))
        geometry.append(
            GeometryBrush(
                kind=geometry_kinds[index % len(geometry_kinds)],
                center=_sample_center(rng, arena_size, radius, protected),
                radius=radius,
                yaw=float(rng.uniform(-math.pi, math.pi)),
                strength=float(rng.uniform(0.45, 1.0)),
            )
        )

    obstacles: list[Obstacle] = []
    for index in range(counts["static"]):
        is_round = bool(index % 2)
        size_x = float(rng.uniform(0.32, 1.15))
        size_y = size_x if is_round else float(rng.uniform(0.32, 1.15))
        footprint_radius = size_x if is_round else math.hypot(size_x, size_y)
        center = _sample_center(rng, arena_size, footprint_radius, protected)
        obstacles.append(
            Obstacle(
                kind="cylinder" if is_round else "box",
                center=center,
                size=(
                    size_x,
                    size_y,
                    float(rng.uniform(0.40, 1.55)),
                ),
                yaw=float(rng.uniform(-math.pi, math.pi)),
            )
        )

    for index in range(counts["moving"]):
        is_round = bool(index % 2)
        size_x = float(rng.uniform(0.25, 0.72))
        size_y = size_x if is_round else float(rng.uniform(0.25, 0.72))
        amplitude, period = _sample_moving_obstacle_motion(rng)
        footprint_radius = size_x if is_round else math.hypot(size_x, size_y)
        center = _sample_center(rng, arena_size, footprint_radius + amplitude, protected)
        angle = float(rng.uniform(-math.pi, math.pi))
        obstacles.append(
            Obstacle(
                kind="cylinder" if is_round else "box",
                center=center,
                size=(size_x, size_y, float(rng.uniform(0.45, 1.35))),
                yaw=angle,
                moving=True,
                motion_axis=(math.cos(angle), math.sin(angle)),
                motion_amplitude=amplitude,
                motion_period_s=period,
                motion_peak_speed_mps=float(2.0 * math.pi * amplitude / period),
                motion_phase=float(rng.uniform(0.0, 2.0 * math.pi)),
            )
        )

    return TargetTerrainLayout(
        seed=int(seed),
        preset=preset,
        arena_size=float(arena_size),
        start_xy=start_xy,
        goal_xy=goal_xy,
        materials=tuple(materials),
        geometry=tuple(geometry),
        obstacles=tuple(obstacles),
    )


def _quat_z(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _material_at(layout: TargetTerrainLayout, x: float, y: float) -> str:
    result = "rigid"
    for brush in sorted(layout.materials, key=lambda item: item.layer):
        if math.dist((x, y), brush.center) <= brush.radius:
            result = brush.kind
    return result


def layout_statistics(layout: TargetTerrainLayout, *, samples_per_axis: int = 121) -> dict[str, float | int]:
    """Approximate painted-area coverage on a deterministic uniform grid."""
    coordinates = np.linspace(-layout.arena_size / 2.0, layout.arena_size / 2.0, samples_per_axis)
    material_counts = {name: 0 for name in MATERIAL_STYLE}
    geometry_count = 0
    overlap_count = 0
    total = samples_per_axis * samples_per_axis
    for x in coordinates:
        for y in coordinates:
            material = _material_at(layout, float(x), float(y))
            material_counts[material] += 1
            geometry_present = any(
                math.dist((float(x), float(y)), brush.center) <= brush.radius
                for brush in layout.geometry
            )
            geometry_count += int(geometry_present)
            overlap_count += int(geometry_present and material != "rigid")
    result: dict[str, float | int] = {
        f"{name}_fraction": count / total for name, count in material_counts.items()
    }
    result.update(
        {
            "geometry_fraction": geometry_count / total,
            "material_geometry_overlap_fraction": overlap_count / total,
            "static_obstacles": sum(not obstacle.moving for obstacle in layout.obstacles),
            "moving_obstacles": sum(obstacle.moving for obstacle in layout.obstacles),
        }
    )
    return result


def _add_rough_patch(world: mujoco.MjsBody, layout: TargetTerrainLayout, brush: GeometryBrush, index: int) -> None:
    rng = np.random.default_rng(layout.seed * 10_003 + index)
    spacing = 0.42
    count = int((2.0 * brush.radius) / spacing) + 1
    for row in range(count):
        for col in range(count):
            x = brush.center[0] - brush.radius + row * spacing
            y = brush.center[1] - brush.radius + col * spacing
            if math.dist((x, y), brush.center) > brush.radius:
                continue
            height = float(rng.uniform(0.018, 0.075) * brush.strength)
            style = MATERIAL_STYLE[_material_at(layout, x, y)]
            world.add_geom(
                name=f"rough_{index}_{row}_{col}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(x, y, height / 2.0 + 0.003),
                size=(spacing * 0.48, spacing * 0.48, height / 2.0),
                friction=style["friction"],
                rgba=_geometry_color("rough", _material_at(layout, x, y)),
            )


def _add_rubble_patch(world: mujoco.MjsBody, layout: TargetTerrainLayout, brush: GeometryBrush, index: int) -> None:
    rng = np.random.default_rng(layout.seed * 20_003 + index)
    count = max(10, int(brush.radius * brush.radius * 4.0))
    for item in range(count):
        radial = brush.radius * math.sqrt(float(rng.random()))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        x = brush.center[0] + radial * math.cos(angle)
        y = brush.center[1] + radial * math.sin(angle)
        sx, sy = rng.uniform(0.12, 0.32, size=2)
        height = float(rng.uniform(0.05, 0.18) * brush.strength)
        style = MATERIAL_STYLE[_material_at(layout, x, y)]
        world.add_geom(
            name=f"rubble_{index}_{item}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(x, y, height / 2.0 + 0.004),
            quat=_quat_z(float(rng.uniform(-math.pi, math.pi))),
            size=(float(sx), float(sy), height / 2.0),
            friction=style["friction"],
            rgba=_geometry_color("rubble", _material_at(layout, x, y)),
        )


def _add_stairs(world: mujoco.MjsBody, layout: TargetTerrainLayout, brush: GeometryBrush, index: int) -> None:
    rng = np.random.default_rng(layout.seed * 30_011 + index)
    side_steps = int(rng.integers(3, 6))
    plateau_steps = int(rng.integers(1, 3))
    segments = 2 * side_steps + plateau_steps
    usable_length = 2.0 * brush.radius * float(rng.uniform(0.82, 0.98))
    depth = usable_length / segments
    width = brush.radius * float(rng.uniform(1.05, 1.55))
    peak_height = float(rng.uniform(0.20, 0.48) * (0.65 + 0.35 * brush.strength))
    step_height = peak_height / side_steps
    axis = np.array((math.cos(brush.yaw), math.sin(brush.yaw)))
    heights = (
        [step_height * (step + 1) for step in range(side_steps)]
        + [peak_height] * plateau_steps
        + [step_height * step for step in range(side_steps, 0, -1)]
    )
    for step, height in enumerate(heights):
        along = -usable_length / 2.0 + (step + 0.5) * depth
        center = np.asarray(brush.center) + along * axis
        style = MATERIAL_STYLE[_material_at(layout, float(center[0]), float(center[1]))]
        world.add_geom(
            name=f"stairs_{index}_{step}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(float(center[0]), float(center[1]), height / 2.0 + 0.003),
            quat=_quat_z(brush.yaw),
            size=(depth / 2.0, width / 2.0, height / 2.0),
            friction=style["friction"],
            rgba=_geometry_color(
                "stairs", _material_at(layout, float(center[0]), float(center[1]))
            ),
        )


def _add_ramp(world: mujoco.MjsBody, layout: TargetTerrainLayout, brush: GeometryBrush, index: int) -> None:
    rng = np.random.default_rng(layout.seed * 40_009 + index)
    total_length = 2.0 * brush.radius * float(rng.uniform(0.84, 0.98))
    plateau_length = min(float(rng.uniform(0.25, 0.75)), total_length * 0.22)
    side_length = (total_length - plateau_length) / 2.0
    width = brush.radius * float(rng.uniform(1.0, 1.45))
    rise = float(rng.uniform(0.18, 0.52) * (0.65 + 0.35 * brush.strength))
    pitch = math.atan2(rise, side_length)
    slope_length = math.hypot(rise, side_length)
    half_thickness = 0.035
    style = MATERIAL_STYLE[_material_at(layout, *brush.center)]
    axis = np.array((math.cos(brush.yaw), math.sin(brush.yaw)))

    def yaw_pitch_quat(yaw: float, pitch_angle: float) -> tuple[float, float, float, float]:
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        cp, sp = math.cos(pitch_angle / 2.0), math.sin(pitch_angle / 2.0)
        return (cy * cp, -sy * sp, cy * sp, sy * cp)

    for side, direction in (("up", -1.0), ("down", 1.0)):
        offset = direction * (plateau_length / 2.0 + side_length / 2.0)
        local_pitch = -pitch if side == "up" else pitch
        # Place the top contact plane, not the box centerline, exactly between
        # floor level and crest level. This removes the entry curb.
        normal_axis_offset = half_thickness * math.sin(pitch) * (-direction)
        center = np.asarray(brush.center) + (offset + normal_axis_offset) * axis
        world.add_geom(
            name=f"ramp_{index}_{side}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(
                float(center[0]),
                float(center[1]),
                rise / 2.0 - half_thickness * math.cos(pitch),
            ),
            quat=yaw_pitch_quat(brush.yaw, local_pitch),
            size=(slope_length / 2.0, width / 2.0, half_thickness),
            friction=style["friction"],
            rgba=_geometry_color("ramp", _material_at(layout, *brush.center)),
        )
    world.add_geom(
        name=f"ramp_{index}_crest",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(brush.center[0], brush.center[1], rise - half_thickness),
        quat=_quat_z(brush.yaw),
        size=(plateau_length / 2.0, width / 2.0, half_thickness),
        friction=style["friction"],
        rgba=_geometry_color("ramp", _material_at(layout, *brush.center)),
    )


def build_target_terrain_spec(layout: TargetTerrainLayout) -> mujoco.MjSpec:
    """Build an executable MuJoCo model containing the target terrain."""
    spec = mujoco.MjSpec()
    spec.modelname = f"unitree_target_{layout.preset}_{layout.seed}"
    spec.option.timestep = 0.01
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    world = spec.worldbody
    half = layout.arena_size / 2.0

    world.add_light(
        name="key",
        pos=(0.0, -4.0, 18.0),
        dir=(0.0, 0.2, -1.0),
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    world.add_geom(
        name="base_ground",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.0, 0.0, -0.15),
        size=(half, half, 0.05),
        friction=MATERIAL_STYLE["rigid"]["friction"],
        rgba=MATERIAL_STYLE["rigid"]["rgba"],
    )

    # A low physical perimeter makes the arena extent clear and prevents a
    # future locomotion rollout from silently leaving the generated benchmark.
    wall_height = 0.16
    wall_width = 0.12
    for name, pos, size in (
        ("boundary_north", (0.0, half, wall_height / 2.0), (half, wall_width, wall_height / 2.0)),
        ("boundary_south", (0.0, -half, wall_height / 2.0), (half, wall_width, wall_height / 2.0)),
        ("boundary_east", (half, 0.0, wall_height / 2.0), (wall_width, half, wall_height / 2.0)),
        ("boundary_west", (-half, 0.0, wall_height / 2.0), (wall_width, half, wall_height / 2.0)),
    ):
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            friction=MATERIAL_STYLE["rigid"]["friction"],
            rgba=(0.12, 0.14, 0.15, 1.0),
        )

    _add_exclusive_material_surface(
        world,
        layout,
        min_x=-half,
        min_y=-half,
        prefix="material_tile",
    )

    for index, brush in enumerate(layout.geometry):
        if brush.kind == "rough":
            _add_rough_patch(world, layout, brush, index)
        elif brush.kind == "rubble":
            _add_rubble_patch(world, layout, brush, index)
        elif brush.kind == "stairs":
            _add_stairs(world, layout, brush, index)
        elif brush.kind == "ramp":
            _add_ramp(world, layout, brush, index)

    for index, obstacle in enumerate(layout.obstacles):
        body = world.add_body(
            name=f"moving_obstacle_{index}" if obstacle.moving else f"static_obstacle_{index}",
            pos=(obstacle.center[0], obstacle.center[1], obstacle.size[2] / 2.0),
            mocap=obstacle.moving,
        )
        geom_type = mujoco.mjtGeom.mjGEOM_CYLINDER if obstacle.kind == "cylinder" else mujoco.mjtGeom.mjGEOM_BOX
        geom_size = (
            (obstacle.size[0], obstacle.size[2] / 2.0)
            if obstacle.kind == "cylinder"
            else (obstacle.size[0], obstacle.size[1], obstacle.size[2] / 2.0)
        )
        body.add_geom(
            name=f"obstacle_geom_{index}",
            type=geom_type,
            quat=_quat_z(obstacle.yaw),
            size=geom_size,
            friction=MATERIAL_STYLE["rigid"]["friction"],
            rgba=(0.86, 0.16, 0.12, 1.0) if not obstacle.moving else (0.82, 0.12, 0.72, 1.0),
        )

    # Non-colliding markers make start/goal placement visible without creating
    # a special platform or changing the local terrain physics.
    for name, xy, outer, inner in (
        (
            "start_marker",
            layout.start_xy,
            (0.12, 0.84, 0.98, 0.72),
            (1.0, 1.0, 1.0, 0.9),
        ),
        (
            "goal_marker",
            layout.goal_xy,
            (0.18, 0.98, 0.28, 0.72),
            (0.02, 0.22, 0.04, 0.9),
        ),
    ):
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(xy[0], xy[1], 0.015),
            size=(0.55, 0.015),
            contype=0,
            conaffinity=0,
            rgba=outer,
        )
        world.add_geom(
            name=f"{name}_center",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(xy[0], xy[1], 0.035),
            size=(0.16, 0.018),
            contype=0,
            conaffinity=0,
            rgba=inner,
        )
    return spec


def set_moving_obstacles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout: TargetTerrainLayout,
    time_s: float,
) -> None:
    """Place mocap obstacles on their deterministic periodic trajectories."""
    for index, obstacle in enumerate(layout.obstacles):
        if not obstacle.moving:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"moving_obstacle_{index}")
        mocap_id = int(model.body_mocapid[body_id])
        phase = 2.0 * math.pi * time_s / obstacle.motion_period_s + obstacle.motion_phase
        offset = obstacle.motion_amplitude * math.sin(phase)
        data.mocap_pos[mocap_id, 0] = obstacle.center[0] + obstacle.motion_axis[0] * offset
        data.mocap_pos[mocap_id, 1] = obstacle.center[1] + obstacle.motion_axis[1] * offset
        data.mocap_pos[mocap_id, 2] = obstacle.size[2] / 2.0
    mujoco.mj_forward(model, data)


def _shift_layout(layout: TargetTerrainLayout, dx: float, dy: float) -> TargetTerrainLayout:
    def shift(point: tuple[float, float]) -> tuple[float, float]:
        return (point[0] + dx, point[1] + dy)

    return replace(
        layout,
        start_xy=shift(layout.start_xy),
        goal_xy=shift(layout.goal_xy),
        materials=tuple(replace(brush, center=shift(brush.center)) for brush in layout.materials),
        geometry=tuple(replace(brush, center=shift(brush.center)) for brush in layout.geometry),
        obstacles=tuple(replace(obstacle, center=shift(obstacle.center)) for obstacle in layout.obstacles),
    )


@dataclass(kw_only=True)
class TargetArenaTerrainCfg(SubTerrainCfg):
    """MJLab adapter for one continuous target arena.

    MJLab sub-terrain coordinates start at a patch corner, whereas the preview
    generator is centered at the origin.  This adapter translates the same
    deterministic layout and returns every direct geom so TerrainGenerator can
    apply its world-grid offset.  Moving obstacles are frozen at their initial
    positions for the first locomotion-transfer diagnostic.
    """

    seed: int = 3
    preset: str = "balanced"
    material_resolution: float = 0.5

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del difficulty
        patch_token = int(rng.integers(0, np.iinfo(np.int32).max))
        if abs(self.size[0] - self.size[1]) > 1e-6:
            raise ValueError("TargetArenaTerrainCfg currently requires a square patch")
        layout = generate_target_terrain(
            self.seed,
            preset=self.preset,
            arena_size=float(self.size[0]),
        )
        half = float(self.size[0]) / 2.0
        layout = _shift_layout(layout, half, half)
        body = spec.body("terrain")
        before = len(body.geoms)

        body.add_geom(
            name="target_base",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(half, half, -0.15),
            size=(half, half, 0.05),
            friction=MATERIAL_STYLE["rigid"]["friction"],
            rgba=MATERIAL_STYLE["rigid"]["rgba"],
        )
        _add_exclusive_material_surface(
            body,
            layout,
            min_x=0.0,
            min_y=0.0,
            prefix="target_material_tile",
            resolution=float(self.material_resolution),
        )

        for index, brush in enumerate(layout.geometry):
            if brush.kind == "rough":
                _add_rough_patch(body, layout, brush, index)
            elif brush.kind == "rubble":
                _add_rubble_patch(body, layout, brush, index)
            elif brush.kind == "stairs":
                _add_stairs(body, layout, brush, index)
            elif brush.kind == "ramp":
                _add_ramp(body, layout, brush, index)

        for index, obstacle in enumerate(layout.obstacles):
            geom_type = (
                mujoco.mjtGeom.mjGEOM_CYLINDER
                if obstacle.kind == "cylinder"
                else mujoco.mjtGeom.mjGEOM_BOX
            )
            geom_size = (
                (obstacle.size[0], obstacle.size[2] / 2.0)
                if obstacle.kind == "cylinder"
                else (obstacle.size[0], obstacle.size[1], obstacle.size[2] / 2.0)
            )
            body.add_geom(
                name=f"target_obstacle_{index}",
                type=geom_type,
                pos=(obstacle.center[0], obstacle.center[1], obstacle.size[2] / 2.0),
                quat=_quat_z(obstacle.yaw),
                size=geom_size,
                friction=MATERIAL_STYLE["rigid"]["friction"],
                rgba=(0.86, 0.16, 0.12, 1.0)
                if not obstacle.moving
                else (0.82, 0.12, 0.72, 1.0),
            )

        new_geoms = list(body.geoms)[before:]
        for geom_index, geom in enumerate(new_geoms):
            geom.name = f"target_{patch_token}_{geom_index}"
        geometries = [TerrainGeometry(geom=geom) for geom in new_geoms]
        return TerrainOutput(
            origin=np.array((half, half, 0.0)),
            geometries=geometries,
        )


@dataclass(kw_only=True)
class HomogeneousSurfaceTerrainCfg(SubTerrainCfg):
    """One flat patch using exactly one target-arena surface material."""

    material: str = "rigid"
    friction: tuple[float, float, float] | None = None
    contact_solref: tuple[float, float] | None = None
    contact_solimp: tuple[float, float, float, float, float] | None = None
    contact_margin: float = 0.0
    contact_gap: float = 0.0
    contact_priority: int = 0

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del difficulty
        if self.material not in MATERIAL_STYLE:
            raise ValueError(f"Unknown homogeneous material: {self.material}")
        half_x, half_y = float(self.size[0]) / 2.0, float(self.size[1]) / 2.0
        style = MATERIAL_STYLE[self.material]
        body = spec.body("terrain")
        geom_kwargs = {
            "name": f"surface_{self.material}_{int(rng.integers(0, np.iinfo(np.int32).max))}",
            "type": mujoco.mjtGeom.mjGEOM_BOX,
            "pos": (half_x, half_y, -0.045),
            "size": (half_x, half_y, 0.05),
            "friction": self.friction or style["friction"],
            "rgba": style["rgba"],
            "margin": float(self.contact_margin),
            "gap": float(self.contact_gap),
            "priority": int(self.contact_priority),
        }
        if self.contact_solref is not None:
            geom_kwargs["solref"] = self.contact_solref
        if self.contact_solimp is not None:
            geom_kwargs["solimp"] = self.contact_solimp
        geom = body.add_geom(
            **geom_kwargs,
        )
        return TerrainOutput(
            origin=np.array((half_x, half_y, 0.005)),
            geometries=[TerrainGeometry(geom=geom)],
        )


@dataclass(kw_only=True)
class BidirectionalTraversalTerrainCfg(SubTerrainCfg):
    """Flat approach followed by one controlled up-and-down ramp or staircase."""

    kind: str = "ramp"
    rise: float = 0.3
    side_length: float = 2.0
    width: float = 2.0
    plateau_length: float = 0.7
    steps_per_side: int = 5

    def function(
        self,
        difficulty: float,
        spec: mujoco.MjSpec,
        rng: np.random.Generator,
    ) -> TerrainOutput:
        del difficulty, rng
        if self.kind not in {"ramp", "stairs"}:
            raise ValueError(f"Unknown traversal geometry: {self.kind}")
        body = spec.body("terrain")
        half_x, half_y = float(self.size[0]) / 2.0, float(self.size[1]) / 2.0
        geoms = []
        floor = body.add_geom(
            name="traversal_floor",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(half_x, half_y, -0.05),
            size=(half_x, half_y, 0.05),
            friction=MATERIAL_STYLE["rigid"]["friction"],
            rgba=MATERIAL_STYLE["rigid"]["rgba"],
        )
        geoms.append(TerrainGeometry(geom=floor))
        approach_x = 4.0
        center_x = approach_x + self.side_length + self.plateau_length / 2.0

        if self.kind == "ramp":
            pitch = math.atan2(self.rise, self.side_length)
            slope_length = math.hypot(self.rise, self.side_length)
            half_thickness = 0.035

            def pitch_quat(angle: float) -> tuple[float, float, float, float]:
                return (math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0)

            for name, top_mid_x, angle in (
                # Positive MuJoCo rotation around Y slopes local +X downward.
                ("up", approach_x + self.side_length / 2.0, -pitch),
                (
                    "down",
                    approach_x + self.side_length + self.plateau_length + self.side_length / 2.0,
                    pitch,
                ),
            ):
                normal_x = math.sin(angle)
                geom = body.add_geom(
                    name=f"traversal_ramp_{name}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=(
                        top_mid_x - half_thickness * normal_x,
                        half_y,
                        self.rise / 2.0 - half_thickness * math.cos(pitch),
                    ),
                    quat=pitch_quat(angle),
                    size=(slope_length / 2.0, self.width / 2.0, half_thickness),
                    friction=MATERIAL_STYLE["rigid"]["friction"],
                    rgba=GEOMETRY_COLORS["ramp"],
                )
                geoms.append(TerrainGeometry(geom=geom))
            crest = body.add_geom(
                name="traversal_ramp_crest",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(center_x, half_y, self.rise - half_thickness),
                size=(self.plateau_length / 2.0, self.width / 2.0, half_thickness),
                friction=MATERIAL_STYLE["rigid"]["friction"],
                rgba=GEOMETRY_COLORS["ramp"],
            )
            geoms.append(TerrainGeometry(geom=crest))
        else:
            step_depth = self.side_length / self.steps_per_side
            heights = (
                [self.rise * (index + 1) / self.steps_per_side for index in range(self.steps_per_side)]
                + [self.rise]
                + [self.rise * index / self.steps_per_side for index in range(self.steps_per_side, 0, -1)]
            )
            total_length = 2.0 * self.side_length + self.plateau_length
            segment_depths = [step_depth] * self.steps_per_side + [self.plateau_length] + [step_depth] * self.steps_per_side
            cursor = approach_x
            for index, (height, depth) in enumerate(zip(heights, segment_depths, strict=True)):
                geom = body.add_geom(
                    name=f"traversal_stair_{index}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=(cursor + depth / 2.0, half_y, height / 2.0 + 0.003),
                    size=(depth / 2.0, self.width / 2.0, height / 2.0),
                    friction=MATERIAL_STYLE["rigid"]["friction"],
                    rgba=GEOMETRY_COLORS["stairs"],
                )
                geoms.append(TerrainGeometry(geom=geom))
                cursor += depth
            assert math.isclose(cursor, approach_x + total_length)

        # Spawn 2 m before the structure, facing along +x.
        return TerrainOutput(
            origin=np.array((2.0, half_y, 0.0)),
            geometries=geoms,
        )

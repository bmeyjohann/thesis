from __future__ import annotations

from unitree_target_terrain import (
    BidirectionalTraversalTerrainCfg,
    HomogeneousSurfaceTerrainCfg,
    MATERIAL_STYLE,
    MaterialBrush,
    TargetTerrainLayout,
    TargetArenaTerrainCfg,
    _add_exclusive_material_surface,
    _sample_moving_obstacle_motion,
    build_target_terrain_spec,
    generate_target_terrain,
    layout_statistics,
    set_moving_obstacles,
)
import mujoco
import numpy as np


def test_target_terrain_is_deterministic() -> None:
    first = generate_target_terrain(17, preset="balanced")
    second = generate_target_terrain(17, preset="balanced")
    assert first == second
    assert first != generate_target_terrain(18, preset="balanced")


def test_target_terrain_contains_required_semantics() -> None:
    layout = generate_target_terrain(3, preset="balanced")
    assert {brush.kind for brush in layout.materials} == {"ice", "sand"}
    assert {brush.kind for brush in layout.geometry} == {"rough", "rubble", "stairs", "ramp"}
    assert any(obstacle.moving for obstacle in layout.obstacles)
    assert any(not obstacle.moving for obstacle in layout.obstacles)
    assert MATERIAL_STYLE["ice"]["rgba"] != MATERIAL_STYLE["sand"]["rgba"]
    kinds = [brush.kind for brush in layout.geometry]
    assert kinds.count("stairs") >= 3
    assert kinds.count("ramp") >= 3


def test_target_terrain_compiles_as_mujoco() -> None:
    layout = generate_target_terrain(5, preset="balanced")
    model = build_target_terrain_spec(layout).compile()
    assert model.ngeom > 20
    assert model.nmocap == sum(obstacle.moving for obstacle in layout.obstacles)


def test_material_coverage_and_moving_obstacles_are_real() -> None:
    layout = generate_target_terrain(5, preset="balanced")
    statistics = layout_statistics(layout, samples_per_axis=41)
    assert statistics["ice_fraction"] > 0.10
    assert statistics["sand_fraction"] > 0.10
    assert statistics["geometry_fraction"] > 0.15
    assert statistics["material_geometry_overlap_fraction"] > 0.05
    spec = build_target_terrain_spec(layout)
    model = spec.compile()
    data = mujoco.MjData(model)
    set_moving_obstacles(model, data, layout, 0.0)
    first = data.mocap_pos.copy()
    set_moving_obstacles(model, data, layout, 1.25)
    assert not np.allclose(first, data.mocap_pos)


def test_faster_moving_obstacles_have_shorter_sweeps() -> None:
    rng = np.random.default_rng(12)
    samples = [_sample_moving_obstacle_motion(rng) for _ in range(100)]
    samples.sort(key=lambda item: 2.0 * np.pi * item[0] / item[1])
    amplitudes = [sample[0] for sample in samples]
    assert all(left >= right for left, right in zip(amplitudes, amplitudes[1:]))
    speeds = [2.0 * np.pi * amplitude / period for amplitude, period in samples]
    assert min(speeds) >= 0.65
    assert max(speeds) <= 1.8
    layout = generate_target_terrain(12, preset="navigation")
    for obstacle in (item for item in layout.obstacles if item.moving):
        expected = 2.0 * np.pi * obstacle.motion_amplitude / obstacle.motion_period_s
        assert np.isclose(obstacle.motion_peak_speed_mps, expected)


def test_obstacle_dimensions_use_wider_bounded_ranges() -> None:
    obstacles = [
        obstacle
        for seed in range(20)
        for obstacle in generate_target_terrain(seed, preset="navigation").obstacles
    ]
    static = [obstacle for obstacle in obstacles if not obstacle.moving]
    moving = [obstacle for obstacle in obstacles if obstacle.moving]
    assert min(size for obstacle in static for size in obstacle.size[:2]) >= 0.32
    assert max(size for obstacle in static for size in obstacle.size[:2]) > 1.0
    assert min(size for obstacle in moving for size in obstacle.size[:2]) >= 0.25
    assert max(size for obstacle in moving for size in obstacle.size[:2]) > 0.65
    for obstacle in obstacles:
        radius = (
            obstacle.size[0]
            if obstacle.kind == "cylinder"
            else float(np.hypot(obstacle.size[0], obstacle.size[1]))
        )
        sweep = obstacle.motion_amplitude if obstacle.moving else 0.0
        assert abs(obstacle.center[0]) + radius + sweep <= 12.0
        assert abs(obstacle.center[1]) + radius + sweep <= 12.0


def test_mjlab_target_arena_adapter_compiles() -> None:
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = TargetArenaTerrainCfg(
        seed=3,
        preset="balanced",
        size=(24.0, 24.0),
    ).function(0.5, spec, np.random.default_rng(0))
    model = spec.compile()
    assert len(output.geometries) > 100
    assert np.allclose(output.origin, (12.0, 12.0, 0.0))
    assert model.ngeom == len(output.geometries)


def test_homogeneous_surface_adapters_compile_with_exact_friction() -> None:
    for material, style in MATERIAL_STYLE.items():
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        output = HomogeneousSurfaceTerrainCfg(
            proportion=1.0,
            size=(24.0, 24.0),
            material=material,
        ).function(0.0, spec, np.random.default_rng(3))
        model = spec.compile()
        assert model.ngeom == 1
        assert np.allclose(model.geom_friction[0], style["friction"])
        assert len(output.geometries) == 1


def test_homogeneous_surface_accepts_contact_calibration() -> None:
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    HomogeneousSurfaceTerrainCfg(
        proportion=1.0,
        size=(24.0, 24.0),
        material="sand",
        friction=(2.0, 0.03, 0.005),
        contact_solref=(0.08, 0.7),
        contact_solimp=(0.5, 0.85, 0.04, 0.5, 2.0),
        contact_margin=0.01,
        contact_gap=0.002,
        contact_priority=1,
    ).function(0.0, spec, np.random.default_rng(3))
    model = spec.compile()
    assert np.allclose(model.geom_friction[0], (2.0, 0.03, 0.005))
    assert np.allclose(model.geom_solref[0], (0.08, 0.7))
    assert np.allclose(model.geom_solimp[0], (0.5, 0.85, 0.04, 0.5, 2.0))
    assert np.isclose(model.geom_margin[0], 0.01)
    assert np.isclose(model.geom_gap[0], 0.002)
    assert model.geom_priority[0] == 1


def test_controlled_ramp_and_stairs_return_to_ground() -> None:
    for kind in ("ramp", "stairs"):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        output = BidirectionalTraversalTerrainCfg(
            proportion=1.0,
            size=(16.0, 8.0),
            kind=kind,
            rise=0.4,
            side_length=2.0,
            steps_per_side=5,
        ).function(0.0, spec, np.random.default_rng(0))
        model = spec.compile()
        assert model.ngeom >= 4
        assert np.allclose(output.origin, (2.0, 4.0, 0.0))
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) for index in range(model.ngeom)]
        expected_stem = "ramp" if kind == "ramp" else "stair"
        assert any(expected_stem in (name or "") for name in names)
        if kind == "ramp":
            up_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "traversal_ramp_up")
            down_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "traversal_ramp_down")
            up_rotation = np.empty(9)
            down_rotation = np.empty(9)
            mujoco.mju_quat2Mat(up_rotation, model.geom_quat[up_id])
            mujoco.mju_quat2Mat(down_rotation, model.geom_quat[down_id])
            # Local +X must ascend on the approach and descend after the crest.
            assert up_rotation.reshape(3, 3)[2, 0] > 0.0
            assert down_rotation.reshape(3, 3)[2, 0] < 0.0
            expected_half_length = np.hypot(0.4, 2.0) / 2.0
            assert np.isclose(model.geom_size[up_id, 0], expected_half_length)
            assert np.isclose(model.geom_size[down_id, 0], expected_half_length)
            for geom_id, expected_left, expected_right in (
                (up_id, 0.0, 0.4),
                (down_id, 0.4, 0.0),
            ):
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
                rotation = rotation.reshape(3, 3)
                top_midpoint = model.geom_pos[geom_id] + model.geom_size[geom_id, 2] * rotation[:, 2]
                left = top_midpoint - model.geom_size[geom_id, 0] * rotation[:, 0]
                right = top_midpoint + model.geom_size[geom_id, 0] * rotation[:, 0]
                assert np.isclose(left[2], expected_left, atol=1e-7)
                assert np.isclose(right[2], expected_right, atol=1e-7)
            crest_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "traversal_ramp_crest"
            )
            crest_top = model.geom_pos[crest_id, 2] + model.geom_size[crest_id, 2]
            assert np.isclose(crest_top, 0.4, atol=1e-7)


def test_overlapping_material_paint_has_one_exclusive_physics_tile() -> None:
    layout = TargetTerrainLayout(
        seed=0,
        preset="test",
        arena_size=4.0,
        start_xy=(-1.5, -1.5),
        goal_xy=(1.5, 1.5),
        materials=(
            MaterialBrush(kind="ice", center=(0.0, 0.0), radius=1.5, layer=0),
            MaterialBrush(kind="sand", center=(0.0, 0.0), radius=1.0, layer=1),
        ),
        geometry=(),
        obstacles=(),
    )
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="terrain")
    _add_exclusive_material_surface(
        body,
        layout,
        min_x=-2.0,
        min_y=-2.0,
        prefix="target_material_tile",
        resolution=0.5,
    )
    model = spec.compile()
    checked_overlap = False
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith("target_material_tile_"):
            continue
        x, y = model.geom_pos[geom_id, :2]
        covering = [
            brush
            for brush in layout.materials
            if np.hypot(x - brush.center[0], y - brush.center[1]) <= brush.radius
        ]
        if len(covering) < 2:
            continue
        expected = sorted(covering, key=lambda brush: brush.layer)[-1].kind
        assert name.endswith(f"_{expected}")
        assert np.allclose(model.geom_friction[geom_id], MATERIAL_STYLE[expected]["friction"])
        checked_overlap = True
        break
    assert checked_overlap, "test seed must contain at least one raster cell with overlapping paint"

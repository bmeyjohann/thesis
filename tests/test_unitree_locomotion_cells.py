from unitree_locomotion_cells import GEOMETRIES, MATERIALS, make_locomotion_cell_cfg, make_locomotion_mixed_cfg


def test_every_cell_is_single_terrain_and_policy_compatible():
    for geometry in GEOMETRIES:
        for material in MATERIALS:
            cfg = make_locomotion_cell_cfg(geometry, material, num_envs=7)
            generator = cfg.scene.terrain.terrain_generator
            assert generator is not None
            assert tuple(generator.sub_terrains) == (geometry,)
            assert generator.num_cols == 1
            assert cfg.scene.num_envs == 7
            assert cfg.observations["policy"] is cfg.observations["actor"]


def test_material_cells_change_physics_not_only_appearance():
    rigid = make_locomotion_cell_cfg("flat", "rigid")
    slippery = make_locomotion_cell_cfg("flat", "slippery")
    sand = make_locomotion_cell_cfg("flat", "sand_drag")
    assert rigid.events["foot_friction"].params["ranges"] == (0.9, 0.9)
    assert slippery.events["foot_friction"].params["ranges"] == (0.10, 0.10)
    assert sand.events["foot_friction"].params["ranges"] == (0.9, 0.9)
    assert "sand_joint_drag" not in rigid.events
    assert "sand_joint_drag" in sand.events
    assert sand.events["sand_joint_drag"].params["ranges"] == (1.8, 1.8)


def test_mixed_training_cfg_contains_all_geometries_and_material_envelope():
    cfg = make_locomotion_mixed_cfg(num_envs=32)
    generator = cfg.scene.terrain.terrain_generator
    assert generator is not None
    assert set(generator.sub_terrains) == set(GEOMETRIES)
    assert generator.num_rows == 5
    assert generator.num_cols == 5
    assert cfg.scene.num_envs == 32
    assert cfg.events["foot_friction"].params["ranges"] == (0.10, 0.90)
    assert cfg.events["sand_joint_drag"].params["ranges"] == (1.0, 1.8)

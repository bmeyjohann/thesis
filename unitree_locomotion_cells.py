"""Single-cell Unitree G1 locomotion environments for staged experts."""

from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
UNITREE_ROOT = REPO_ROOT / "external" / "unitree_rl_mjlab"
if str(UNITREE_ROOT) not in sys.path:
    sys.path.insert(0, str(UNITREE_ROOT))

from mjlab.envs import ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.envs.mdp import dr  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.terrains import BoxFlatTerrainCfg, TerrainGeneratorCfg  # noqa: E402
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg  # noqa: E402
from src.tasks.velocity.terrain_cfgs import (  # noqa: E402
    ISAACLAB_ROUGH_TERRAINS_CFG,
    PARKOUR_TERRAINS_CFG,
)


GEOMETRIES = ("flat", "random_rough", "cobblestone", "stairs", "stepping_stones")
MATERIALS = ("rigid", "slippery", "sand_drag")


def _subterrain(geometry: str):
    if geometry == "flat":
        return BoxFlatTerrainCfg()
    if geometry == "random_rough":
        return copy.deepcopy(ISAACLAB_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"])
    if geometry == "cobblestone":
        # Low randomized blocks are a reproducible cobblestone geometry proxy.
        value = copy.deepcopy(ISAACLAB_ROUGH_TERRAINS_CFG.sub_terrains["boxes"])
        value.grid_width = 0.30
        value.grid_height_range = (0.025, 0.10)
        value.platform_width = 1.5
        return value
    if geometry == "stairs":
        return copy.deepcopy(PARKOUR_TERRAINS_CFG.sub_terrains["open_stairs"])
    if geometry == "stepping_stones":
        return copy.deepcopy(PARKOUR_TERRAINS_CFG.sub_terrains["stepping_stones"])
    raise ValueError(f"Unknown geometry: {geometry}")


def _material_events(cfg: ManagerBasedRlEnvCfg, material: str) -> None:
    friction = {
        "rigid": 0.9,
        "slippery": 0.10,
        "sand_drag": 0.9,
    }.get(material)
    if friction is None:
        raise ValueError(f"Unknown material: {material}")
    cfg.events["foot_friction"].params["ranges"] = (friction, friction)

    if material == "sand_drag":
        # A global single-cell proxy: high contact grip plus increased joint
        # damping. It models energy loss, not deformable granular mechanics.
        cfg.events["sand_joint_drag"] = EventTermCfg(
            func=dr.joint_damping,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "operation": "scale",
                "ranges": (1.8, 1.8),
            },
        )


def make_locomotion_cell_cfg(
    geometry: str,
    material: str,
    *,
    play: bool = False,
    num_envs: int | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create one geometry/material cell while retaining the height scanner."""
    cfg = unitree_g1_rough_env_cfg(play=play)
    terrain = _subterrain(geometry)
    terrain.proportion = 1.0
    terrain.size = (8.0, 8.0)
    generator = TerrainGeneratorCfg(
        seed=0,
        curriculum=not play,
        size=(8.0, 8.0),
        border_width=8.0,
        num_rows=1 if play else 5,
        num_cols=1,
        difficulty_range=(0.15, 0.65),
        color_scheme="height",
        sub_terrains={geometry: terrain},
        add_lights=True,
    )
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = replace(generator)
    cfg.scene.terrain.max_init_terrain_level = None if play else 1
    cfg.curriculum.pop("terrain_levels", None)
    # Generated rough cells can exceed the G1 default of 70 contacts during
    # initialization. Leave headroom for harder terrain and batched rollouts.
    cfg.sim.nconmax = 256
    _material_events(cfg, material)
    # Installed RSL-RL rollout storage expects the actor group under `policy`.
    cfg.observations["policy"] = cfg.observations["actor"]
    if num_envs is not None:
        cfg.scene.num_envs = int(num_envs)
    return cfg


def make_locomotion_mixed_cfg(
    *, play: bool = False, num_envs: int | None = None
) -> ManagerBasedRlEnvCfg:
    """Create one vectorized environment mixing all geometries and materials."""
    cfg = unitree_g1_rough_env_cfg(play=play)
    sub_terrains = {}
    for geometry in GEOMETRIES:
        terrain = _subterrain(geometry)
        terrain.proportion = 1.0 / len(GEOMETRIES)
        terrain.size = (8.0, 8.0)
        sub_terrains[geometry] = terrain
    generator = TerrainGeneratorCfg(
        seed=0,
        curriculum=not play,
        size=(8.0, 8.0),
        border_width=8.0,
        num_rows=5,
        num_cols=5,
        difficulty_range=(0.15, 0.65),
        color_scheme="height",
        sub_terrains=sub_terrains,
        add_lights=True,
    )
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = replace(generator)
    cfg.scene.terrain.max_init_terrain_level = None if play else 1
    cfg.curriculum.pop("terrain_levels", None)
    cfg.sim.nconmax = 256
    # Sample the full material envelope independently across environments.
    cfg.events["foot_friction"].params["ranges"] = (0.10, 0.90)
    cfg.events["sand_joint_drag"] = EventTermCfg(
        func=dr.joint_damping,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "scale",
            "ranges": (1.0, 1.8),
        },
    )
    cfg.observations["policy"] = cfg.observations["actor"]
    if num_envs is not None:
        cfg.scene.num_envs = int(num_envs)
    return cfg


def cell_metadata(geometry: str, material: str) -> dict[str, object]:
    return {
        "geometry": geometry,
        "material": material,
        "terrain_observation": "privileged_height_scan",
        "material_model": (
            "foot_contact_friction"
            if material != "sand_drag"
            else "foot_contact_friction_plus_global_joint_damping_proxy"
        ),
    }

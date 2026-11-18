"""Gentle rough-terrain curriculum for the Unitree G1 locomotion policy."""

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
from mjlab.managers.curriculum_manager import CurriculumTermCfg  # noqa: E402
from mjlab.tasks.velocity import mdp  # noqa: E402
from mjlab.terrains import BoxFlatTerrainCfg, TerrainGeneratorCfg  # noqa: E402
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg  # noqa: E402
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg  # noqa: E402
from src.tasks.velocity.terrain_cfgs import ISAACLAB_ROUGH_TERRAINS_CFG  # noqa: E402


def _gentle_terrain_generator(*, play: bool) -> TerrainGeneratorCfg:
    source = ISAACLAB_ROUGH_TERRAINS_CFG.sub_terrains

    flat = BoxFlatTerrainCfg(proportion=0.20)

    stairs = copy.deepcopy(source["pyramid_stairs"])
    stairs.proportion = 0.15
    stairs.step_height_range = (0.03, 0.12)
    stairs.step_width = 0.35
    stairs.platform_width = 3.0

    stairs_inv = copy.deepcopy(source["pyramid_stairs_inv"])
    stairs_inv.proportion = 0.15
    stairs_inv.step_height_range = (0.03, 0.12)
    stairs_inv.step_width = 0.35
    stairs_inv.platform_width = 3.0

    rough = copy.deepcopy(source["random_rough"])
    rough.proportion = 0.15
    rough.noise_range = (0.005, 0.05)
    rough.noise_step = 0.01

    slope = copy.deepcopy(source["hf_pyramid_slope"])
    slope.proportion = 0.175
    slope.slope_range = (0.0, 0.18)  # About 10 degrees at maximum difficulty.
    slope.platform_width = 2.5

    slope_inv = copy.deepcopy(source["hf_pyramid_slope_inv"])
    slope_inv.proportion = 0.175
    slope_inv.slope_range = (0.0, 0.18)
    slope_inv.platform_width = 2.5

    return TerrainGeneratorCfg(
        seed=0,
        curriculum=not play,
        size=(8.0, 8.0),
        border_width=12.0,
        num_rows=5 if play else 8,
        num_cols=6 if play else 12,
        difficulty_range=(0.0, 1.0),
        color_scheme="height",
        sub_terrains={
            "flat": flat,
            "stairs": stairs,
            "stairs_down": stairs_inv,
            "random_rough": rough,
            "slope": slope,
            "slope_down": slope_inv,
        },
        add_lights=True,
    )


def make_gentle_locomotion_cfg(
    *, play: bool = False, num_envs: int | None = None
) -> ManagerBasedRlEnvCfg:
    """Build a G1 task that learns walking, low geometry, and responsive turns."""
    cfg = unitree_g1_rough_env_cfg(play=play)
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = replace(_gentle_terrain_generator(play=play))
    cfg.scene.terrain.max_init_terrain_level = None if play else 1
    cfg.sim.nconmax = 256

    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (0.0, 0.7)
    twist.ranges.lin_vel_y = (-0.10, 0.10)
    twist.ranges.ang_vel_z = (-0.75, 0.75)

    # Increase yaw pressure without sacrificing forward velocity tracking.
    cfg.rewards["track_angular_velocity"].weight = 3.0
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {
                    "step": 0,
                    "lin_vel_x": (0.0, 0.7),
                    "lin_vel_y": (-0.10, 0.10),
                    "ang_vel_z": (-0.75, 0.75),
                },
                {
                    "step": 1200 * 24,
                    "lin_vel_x": (0.0, 1.0),
                    "lin_vel_y": (-0.20, 0.20),
                    "ang_vel_z": (-1.20, 1.20),
                },
                {
                    "step": 3000 * 24,
                    "lin_vel_x": (0.0, 1.0),
                    "lin_vel_y": (-0.25, 0.25),
                    "ang_vel_z": (-1.50, 1.50),
                },
            ],
        },
    )

    cfg.observations["policy"] = cfg.observations["actor"]
    if num_envs is not None:
        cfg.scene.num_envs = int(num_envs)
    return cfg


def gentle_locomotion_metadata() -> dict[str, object]:
    return {
        "terrain_mix": {
            "flat": 0.20,
            "stairs_up": 0.15,
            "stairs_down": 0.15,
            "random_rough": 0.15,
            "slope_up": 0.175,
            "slope_down": 0.175,
        },
        "max_step_height_m": 0.12,
        "max_slope": 0.18,
        "max_slope_degrees_approx": 10.2,
        "command_curriculum_iterations": (0, 1200, 3000),
        "final_command_ranges": {
            "forward_mps": (0.0, 1.0),
            "lateral_mps": (-0.25, 0.25),
            "yaw_radps": (-1.5, 1.5),
        },
        "angular_tracking_reward_weight": 3.0,
        "terrain_observation": "privileged_height_scan",
    }

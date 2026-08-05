"""Episode-indexed layout manifests for paired Unitree navigation evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from unitree_nav_layout import set_terrain_tile_indices, terrain_tile_shape


MANIFEST_VERSION = 1


def _tolist(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().tolist()


def capture_layout_entries(env, env_ids: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """Capture replayable tile, root-state, and goal data from active environments."""
    unwrapped = env.env.unwrapped if hasattr(env, "env") else env
    terrain = unwrapped.scene.terrain
    robot = unwrapped.scene["robot"]
    command = unwrapped.command_manager._terms["pose"]
    rows, cols = terrain_tile_shape(env)
    if rows <= 0 or cols <= 0:
        raise RuntimeError("Paired evaluation manifests require a terrain tile bank")
    selected = list(range(unwrapped.num_envs)) if env_ids is None else [int(i) for i in env_ids]
    root_pose = robot.data.root_link_pose_w
    root_velocity = robot.data.root_link_vel_w
    entries: list[dict[str, Any]] = []
    for env_id in selected:
        origin = terrain.env_origins[env_id]
        tile_index = int(terrain.terrain_levels[env_id].item()) * cols + int(terrain.terrain_types[env_id].item())
        pose_relative = root_pose[env_id].clone()
        pose_relative[:3] -= origin
        goal_relative = command._goal_pos_w[env_id].clone()
        goal_relative[:2] -= origin[:2]
        entries.append(
            {
                "tile_index": tile_index,
                "root_pose_relative": _tolist(pose_relative),
                "root_velocity": _tolist(root_velocity[env_id]),
                "goal_xy_relative": _tolist(goal_relative[:2]),
                "goal_heading": float(command._goal_heading_w[env_id].detach().cpu().item()),
            }
        )
    return entries


def apply_layout_entries(env, env_ids: Sequence[int] | torch.Tensor, entries: Sequence[dict[str, Any]]) -> None:
    """Apply manifest entries after an environment reset, independent of reset order."""
    unwrapped = env.env.unwrapped if hasattr(env, "env") else env
    terrain = unwrapped.scene.terrain
    robot = unwrapped.scene["robot"]
    command = unwrapped.command_manager._terms["pose"]
    device = terrain.env_origins.device
    ids = torch.as_tensor(env_ids, device=device, dtype=torch.long).reshape(-1)
    if len(entries) != ids.numel():
        raise ValueError("Manifest replay requires exactly one entry per environment id")
    tile_indices = torch.as_tensor([entry["tile_index"] for entry in entries], device=device, dtype=torch.long)
    set_terrain_tile_indices(env, ids, tile_indices)
    origins = terrain.env_origins[ids]

    root_pose = torch.as_tensor(
        [entry["root_pose_relative"] for entry in entries], device=device, dtype=robot.data.root_link_pose_w.dtype
    )
    root_pose[:, :3] += origins
    root_velocity = torch.as_tensor(
        [entry["root_velocity"] for entry in entries], device=device, dtype=robot.data.root_link_vel_w.dtype
    )
    robot.write_root_state_to_sim(torch.cat([root_pose, root_velocity], dim=-1), env_ids=ids)
    unwrapped.sim.forward()

    goals = torch.as_tensor(
        [entry["goal_xy_relative"] for entry in entries], device=command._goal_pos_w.device, dtype=command._goal_pos_w.dtype
    )
    command._goal_pos_w[ids, :2] = goals + terrain.env_origins[ids, :2]
    command._goal_heading_w[ids] = torch.as_tensor(
        [entry["goal_heading"] for entry in entries],
        device=command._goal_heading_w.device,
        dtype=command._goal_heading_w.dtype,
    )
    command._update_command()


def save_layout_manifest(path: Path, *, entries: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": MANIFEST_VERSION, "metadata": metadata, "episodes": list(entries)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_layout_manifest(path: Path, *, minimum_episodes: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", -1)) != MANIFEST_VERSION:
        raise ValueError(f"Unsupported Unitree evaluation manifest version: {payload.get('version')}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < int(minimum_episodes):
        raise ValueError(f"Manifest {path} has {len(episodes) if isinstance(episodes, list) else 0} episodes; need {minimum_episodes}")
    required = {"tile_index", "root_pose_relative", "root_velocity", "goal_xy_relative", "goal_heading"}
    for index, episode in enumerate(episodes[:minimum_episodes]):
        missing = required.difference(episode)
        if missing:
            raise ValueError(f"Manifest episode {index} is missing fields: {sorted(missing)}")
    return payload

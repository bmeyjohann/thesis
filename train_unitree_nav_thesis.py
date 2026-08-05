#!/usr/bin/env python3
"""Standalone thesis-method trainer for Unitree G1 obstacle navigation.

This intentionally keeps the teacher non-privileged: teacher decisions are
computed from the same actor observation used by the student policy
(`pose_command` + `height_scan`), not from simulator obstacle geometry.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
UNITREE_REPO = ROOT / "external" / "unitree_rl_mjlab"
DEFAULT_LOW_LEVEL = (
    UNITREE_REPO
    / "logs"
    / "rsl_rl"
    / "g1_velocity"
    / "2026-07-12_10-35-19_omni_finetune_model1499_20260712"
)


@dataclass
class TeacherDiagnostics:
    goal_blocked_fraction: float
    intervention_fraction: float
    action_delta_mean: float
    target_angle_abs_mean: float
    min_scan_mean: float


class ScanTeacherState:
    def __init__(self, num_envs: int, device: torch.device):
        self.bypass_side = torch.zeros(num_envs, device=device)
        self.bypass_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            self.bypass_side.zero_()
            self.bypass_steps.zero_()
            return
        mask = mask.to(device=self.bypass_steps.device).reshape(-1).bool()
        self.bypass_side[mask] = 0.0
        self.bypass_steps[mask] = 0


class InterventionGateState:
    """Hysteretic clearance-or-stall intervention state for each env."""

    def __init__(self, num_envs: int, device: torch.device):
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.last_distance = torch.full((num_envs,), float("inf"), device=device)
        self.progress_reference = torch.full((num_envs,), float("inf"), device=device)
        self.bad_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.good_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.active_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            mask = torch.ones_like(self.active)
        mask = mask.to(device=self.active.device).reshape(-1).bool()
        self.active[mask] = False
        self.last_distance[mask] = float("inf")
        self.progress_reference[mask] = float("inf")
        self.bad_steps[mask] = 0
        self.good_steps[mask] = 0
        self.active_steps[mask] = 0

    def update(
        self,
        *,
        distance: torch.Tensor,
        clearance: torch.Tensor,
        action_delta: torch.Tensor,
        clearance_threshold: float,
        release_clearance: float,
        stall_steps: int,
        progress_epsilon: float,
        release_steps: int,
        release_progress_tolerance: float,
        release_action_delta_max: float,
        goal_tolerance: float,
    ) -> tuple[torch.Tensor, ...]:
        distance = distance.reshape(-1)
        clearance = clearance.reshape(-1)
        first = ~torch.isfinite(self.last_distance)
        self.last_distance[first] = distance[first]
        self.progress_reference[first] = distance[first]
        step_progress = self.last_distance - distance
        enough_progress = (self.progress_reference - distance) >= float(progress_epsilon)
        self.progress_reference = torch.where(enough_progress, distance, self.progress_reference)
        self.bad_steps = torch.where(enough_progress, torch.zeros_like(self.bad_steps), self.bad_steps + 1)

        clearance_trigger = clearance < float(clearance_threshold)
        stall_trigger = self.bad_steps >= max(1, int(stall_steps))
        at_goal = distance <= float(goal_tolerance)
        clearance_trigger &= ~at_goal
        stall_trigger &= ~at_goal
        engage = (~self.active) & (clearance_trigger | stall_trigger)
        engage_clearance = engage & clearance_trigger
        engage_stall = engage & stall_trigger & ~clearance_trigger
        self.active |= engage

        clearance_ok = clearance >= float(release_clearance)
        progress_ok = step_progress >= -float(release_progress_tolerance)
        action_delta_ok = action_delta.reshape(-1) <= float(release_action_delta_max)
        release_good = clearance_ok & progress_ok & action_delta_ok
        release_blocked_clearance = self.active & ~clearance_ok
        release_blocked_progress = self.active & ~progress_ok
        release_blocked_action_delta = self.active & ~action_delta_ok
        self.active_steps = torch.where(self.active, self.active_steps + 1, self.active_steps)
        self.good_steps = torch.where(
            self.active & release_good,
            self.good_steps + 1,
            torch.zeros_like(self.good_steps),
        )
        release = self.active & (self.good_steps >= max(1, int(release_steps)))
        release |= self.active & at_goal
        released_duration = torch.where(release, self.active_steps, torch.zeros_like(self.active_steps))
        self.active &= ~release
        self.good_steps[release] = 0
        self.bad_steps[release] = 0
        self.progress_reference[release] = distance[release]
        self.active_steps[release] = 0
        self.bad_steps[at_goal] = 0
        self.progress_reference[at_goal] = distance[at_goal]
        self.last_distance = distance.clone()
        return (
            self.active.clone(),
            clearance_trigger,
            stall_trigger,
            release,
            engage_clearance,
            engage_stall,
            release_blocked_clearance,
            release_blocked_progress,
            release_blocked_action_delta,
            released_duration,
        )


def resolve_intervention_clearances(args: argparse.Namespace) -> tuple[float, float]:
    mode = str(getattr(args, "intervention_clearance_mode", "fixed"))
    if mode == "fixed":
        engage = float(args.intervention_clearance_threshold)
        release = float(args.intervention_release_clearance)
    elif mode == "teacher_ratio":
        if str(getattr(args, "teacher_type", "geom_scan")) != "geom_scan":
            raise ValueError("teacher_ratio clearance currently requires teacher_type=geom_scan")
        reference = float(args.teacher_geom_clearance)
        engage = reference * float(args.intervention_clearance_trigger_ratio)
        release = reference * float(args.intervention_clearance_release_ratio)
    else:
        raise ValueError(f"Unsupported intervention_clearance_mode={mode!r}")
    if engage <= 0.0 or release <= engage:
        raise ValueError(f"Intervention clearances must satisfy 0 < engage < release, got {engage}, {release}")
    return engage, release


class UnitreeReplayBuffer:
    def __init__(self, *, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.empty((capacity, obs_dim), device=device)
        self.next_obs = torch.empty((capacity, obs_dim), device=device)
        self.actions = torch.empty((capacity, act_dim), device=device)
        self.student_actions = torch.empty((capacity, act_dim), device=device)
        self.rewards = torch.empty((capacity,), device=device)
        self.dones = torch.empty((capacity,), dtype=torch.bool, device=device)
        self.truncations = torch.empty((capacity,), dtype=torch.bool, device=device)
        self.effective_n_steps = torch.empty((capacity,), dtype=torch.float32, device=device)
        self.teacher_intervened = torch.empty((capacity,), dtype=torch.bool, device=device)
        self.intervention_start = torch.empty((capacity,), dtype=torch.bool, device=device)
        self.eil_good = torch.empty((capacity,), dtype=torch.bool, device=device)
        self.eil_bad = torch.empty((capacity,), dtype=torch.bool, device=device)
        self.env_ids = torch.empty((capacity,), dtype=torch.long, device=device)
        self.pos = 0
        self.size = 0

    def add(
        self,
        *,
        obs: torch.Tensor,
        actions: torch.Tensor,
        student_actions: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncations: torch.Tensor,
        effective_n_steps: torch.Tensor,
        teacher_intervened: torch.Tensor,
        intervention_start: torch.Tensor | None = None,
        eil_good: torch.Tensor | None = None,
        eil_bad: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = int(obs.shape[0])
        idx = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.student_actions[idx] = student_actions
        self.next_obs[idx] = next_obs
        self.rewards[idx] = rewards.reshape(-1)
        self.dones[idx] = dones.reshape(-1).bool()
        self.truncations[idx] = truncations.reshape(-1).bool()
        self.effective_n_steps[idx] = effective_n_steps.reshape(-1).float()
        self.teacher_intervened[idx] = teacher_intervened.reshape(-1).bool()
        self.intervention_start[idx] = (
            intervention_start.reshape(-1).bool() if intervention_start is not None else False
        )
        self.eil_good[idx] = eil_good.reshape(-1).bool() if eil_good is not None else True
        self.eil_bad[idx] = eil_bad.reshape(-1).bool() if eil_bad is not None else False
        self.env_ids[idx] = (
            env_ids.reshape(-1).long() if env_ids is not None else torch.arange(n, device=self.device)
        )
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        return idx

    def mark_eil_bad(self, indices: list[int]) -> None:
        if not indices:
            return
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        self.eil_good[idx] = False
        self.eil_bad[idx] = True

    def sample(self, batch_size: int) -> dict[str, Any]:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        idx = torch.randint(0, self.size, (int(batch_size),), device=self.device)
        return {
            "observations": self.obs[idx],
            "actions": self.actions[idx],
            "student_actions": self.student_actions[idx],
            "teacher_intervened": self.teacher_intervened[idx],
            "intervention_start": self.intervention_start[idx],
            "eil_good": self.eil_good[idx],
            "eil_bad": self.eil_bad[idx],
            "env_ids": self.env_ids[idx],
            "next": {
                "observations": self.next_obs[idx],
                "rewards": self.rewards[idx],
                "dones": self.dones[idx],
                "truncations": self.truncations[idx],
                "effective_n_steps": self.effective_n_steps[idx],
            },
        }


class UnitreeNStepAccumulator:
    """Build reset-safe n-step replay rows independently for each vector env."""

    def __init__(self, *, num_envs: int, n_step: int, gamma: float):
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.queues: list[list[dict[str, torch.Tensor]]] = [[] for _ in range(int(num_envs))]

    def add(self, **transition: torch.Tensor) -> dict[str, torch.Tensor] | None:
        emitted: list[dict[str, torch.Tensor]] = []
        num_envs = int(transition["obs"].shape[0])
        for env_idx in range(num_envs):
            queue = self.queues[env_idx]
            queue.append({key: value[env_idx].detach().clone() for key, value in transition.items()})
            if bool(transition["dones"][env_idx].item()):
                while queue:
                    emitted.append(self._aggregate(queue[: self.n_step]))
                    queue.pop(0)
            elif len(queue) >= self.n_step:
                emitted.append(self._aggregate(queue[: self.n_step]))
                queue.pop(0)
        if not emitted:
            return None
        return {key: torch.stack([row[key] for row in emitted], dim=0) for key in emitted[0]}

    def _aggregate(self, rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        first, last = rows[0], rows[-1]
        reward = torch.zeros_like(first["rewards"])
        for offset, row in enumerate(rows):
            reward = reward + (self.gamma**offset) * row["rewards"]
        return {
            "obs": first["obs"],
            "actions": first["actions"],
            "student_actions": first["student_actions"],
            "next_obs": last["next_obs"],
            "rewards": reward,
            "dones": last["dones"],
            "truncations": last["truncations"],
            "effective_n_steps": torch.as_tensor(float(len(rows)), device=reward.device),
            "teacher_intervened": first["teacher_intervened"],
            "intervention_start": first.get("intervention_start", torch.zeros_like(first["teacher_intervened"])),
            "env_ids": first.get("env_ids", torch.as_tensor(0, device=reward.device, dtype=torch.long)),
        }


def _setup_unitree_imports(low_level_policy_path: Path) -> None:
    os.environ.setdefault("G1_VELOCITY_POLICY_PATH", str(low_level_policy_path))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/unitree-cache")
    os.environ.setdefault("MUJOCO_GL", "egl")
    for key in ("MPLCONFIGDIR", "WARP_CACHE_PATH", "XDG_CACHE_HOME"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(UNITREE_REPO))


def _extract_actor_obs(obs: Any) -> torch.Tensor:
    if hasattr(obs, "keys") and hasattr(obs, "__getitem__"):
        keys = set(obs.keys())
        for key in ("actor", "policy", "obs"):
            if key in keys:
                return obs[key]
    if isinstance(obs, dict):
        if "actor" in obs:
            return obs["actor"]
        if "policy" in obs:
            return obs["policy"]
        if "obs" in obs:
            return obs["obs"]
    if isinstance(obs, (tuple, list)) and obs:
        return _extract_actor_obs(obs[0])
    if torch.is_tensor(obs):
        return obs
    raise TypeError(f"Unsupported observation type: {type(obs)!r}")


def _prepare_actor_obs(obs: Any, args: argparse.Namespace) -> torch.Tensor:
    from unitree_nav_observation import prepare_unitree_actor_obs

    value = _extract_actor_obs(obs).to(args.device, dtype=torch.float32)
    target_dim = int(getattr(args, "pad_obs_to_dim", 0))
    if target_dim > 0:
        if value.shape[1] > target_dim:
            raise ValueError(f"Observation dim {value.shape[1]} exceeds pad target {target_dim}")
        if value.shape[1] < target_dim:
            value = torch.cat(
                [value, torch.zeros((value.shape[0], target_dim - value.shape[1]), device=value.device)],
                dim=1,
            )
    return prepare_unitree_actor_obs(
        value,
        mask_proprioception=bool(getattr(args, "mask_proprioception", False)),
        mask_goal_heading=bool(getattr(args, "mask_goal_heading", False)),
        mask_height_scan=bool(getattr(args, "mask_height_scan", False)),
    )


def _extract_cost(extras: Any, num_envs: int, device: torch.device) -> torch.Tensor:
    def _as_env_cost(value: torch.Tensor) -> torch.Tensor:
        value = value.to(device=device, dtype=torch.float32)
        if value.ndim == 0:
            return value.reshape(1).expand(num_envs)
        if value.shape[0] != num_envs:
            return value.reshape(num_envs, -1).sum(dim=1)
        if value.ndim > 1:
            return value.reshape(num_envs, -1).sum(dim=1)
        return value.reshape(num_envs)

    if isinstance(extras, dict):
        for key in ("cost", "costs"):
            if key in extras and torch.is_tensor(extras[key]):
                return _as_env_cost(extras[key])
        if "extras" in extras:
            nested = _extract_cost(extras["extras"], num_envs, device)
            if torch.any(nested):
                return nested
        if "cost_manager" in extras and isinstance(extras["cost_manager"], dict):
            vals = [
                _as_env_cost(v)
                for v in extras["cost_manager"].values()
                if torch.is_tensor(v)
            ]
            if vals:
                return torch.stack(vals, dim=0).sum(dim=0)
    return torch.zeros(num_envs, device=device)


def _goal_success(obs: torch.Tensor, threshold: float) -> torch.Tensor:
    goal_xy = obs[:, 6:8]
    return torch.linalg.norm(goal_xy, dim=-1) <= float(threshold)


def _scan_astar_teacher_action(
    obs: torch.Tensor,
    *,
    scan_block_threshold: float,
    scan_block_delta: float,
    align_angle: float,
    max_vx: float,
    max_vy: float,
    yaw_gain: float,
    min_forward_scale: float,
    goal_stop_dist: float,
    state: ScanTeacherState | None,
    clearance: float,
    cell_padding: float,
    resolution: float,
    waypoint_index: int,
    side_penalty: float,
    commit_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, TeacherDiagnostics]:
    """Plan over the exact square scan supplied to the student policy."""
    from unitree_nav_geom_teacher import _astar_local_target

    device = obs.device
    goal_xy = obs[:, 6:8]
    goal_dist = torch.linalg.norm(goal_xy, dim=1)
    scan_dim = obs.shape[1] - 9
    side = int(round(scan_dim ** 0.5))
    if side * side != scan_dim:
        raise ValueError(f"Expected square current scan, got {scan_dim} values")
    scan = torch.nan_to_num(obs[:, 9:].reshape(-1, side, side), nan=1.0)
    flat_scan = scan.reshape(-1, scan_dim)
    reference = torch.quantile(flat_scan, 0.9, dim=1, keepdim=True)
    blocked = flat_scan < float(scan_block_threshold)
    if float(scan_block_delta) > 0.0:
        blocked |= flat_scan < (reference - float(scan_block_delta))

    coords = np.linspace(-1.5, 1.5, side, dtype=np.float64)
    forward, lateral = np.meshgrid(coords, coords, indexing="xy")
    # meshgrid(x=forward, y=lateral) yields rows=lateral, columns=forward,
    # matching the actor observation flattening verified against hit_pos_w.
    scan_points = np.stack([forward.reshape(-1), lateral.reshape(-1)], axis=1)
    blocked_np = blocked.detach().cpu().numpy()
    goals_np = goal_xy.detach().cpu().numpy()
    actions: list[list[float]] = []
    direct_blocked_rows: list[bool] = []
    target_angles: list[float] = []

    effective_clearance = float(clearance) + max(0.0, float(cell_padding))
    for env_idx, goal_b in enumerate(goals_np):
        obstacle_points = scan_points[blocked_np[env_idx]]
        dist = float(np.linalg.norm(goal_b))
        goal_angle = float(math.atan2(goal_b[1], goal_b[0])) if dist > 1e-6 else 0.0
        goal_unit = goal_b / max(dist, 1e-6)
        if obstacle_points.size:
            projection = obstacle_points @ goal_unit
            lateral_to_goal = np.abs(
                obstacle_points[:, 0] * goal_unit[1] - obstacle_points[:, 1] * goal_unit[0]
            )
            direct_blocked = bool(
                np.any((projection > 0.0) & (projection <= 1.75) & (lateral_to_goal <= effective_clearance))
            )
        else:
            direct_blocked = False

        preferred_side = 0
        if state is not None and int(state.bypass_steps[env_idx].item()) > 0:
            preferred_side = int(torch.sign(state.bypass_side[env_idx]).item())
        target_angle = _astar_local_target(
            obstacles=obstacle_points,
            goal_b=goal_b.astype(np.float64),
            min_forward=-1.5,
            max_forward=1.5,
            max_lateral=1.5,
            clearance=effective_clearance,
            resolution=float(resolution),
            waypoint_index=int(waypoint_index),
            preferred_side=preferred_side,
            side_penalty=float(side_penalty),
            side_frame="body",
        )
        if target_angle is None:
            target_angle = float(np.clip(goal_angle, -math.pi / 2.0, math.pi / 2.0))

        if state is not None:
            if direct_blocked:
                if preferred_side == 0:
                    preferred_side = 1 if target_angle >= 0.0 else -1
                    state.bypass_side[env_idx] = float(preferred_side)
                state.bypass_steps[env_idx] = max(1, int(commit_steps))
            elif int(state.bypass_steps[env_idx].item()) > 0:
                state.bypass_steps[env_idx] -= 1
                if int(state.bypass_steps[env_idx].item()) == 0:
                    state.bypass_side[env_idx] = 0.0

        target_angle = float(np.clip(target_angle, -math.pi / 2.0, math.pi / 2.0))
        speed_gate = 1.0 if abs(target_angle) <= float(align_angle) else float(min_forward_scale)
        vx = float(max_vx) * max(0.0, math.cos(target_angle)) * speed_gate
        vy = float(np.clip(float(max_vy) * math.sin(target_angle), -1.0, 1.0))
        yaw = float(np.clip(float(yaw_gain) * target_angle, -1.0, 1.0))
        if float(goal_stop_dist) > 0.0 and dist <= float(goal_stop_dist):
            vx = vy = yaw = 0.0
        actions.append([vx, vy, yaw])
        direct_blocked_rows.append(direct_blocked)
        target_angles.append(target_angle)

    teacher = torch.as_tensor(actions, device=device, dtype=obs.dtype).clamp(-1.0, 1.0)
    goal_blocked = torch.as_tensor(direct_blocked_rows, device=device, dtype=torch.bool)
    diagnostics = TeacherDiagnostics(
        goal_blocked_fraction=float(goal_blocked.float().mean().detach().cpu().item()),
        intervention_fraction=0.0,
        action_delta_mean=0.0,
        target_angle_abs_mean=float(np.mean(np.abs(target_angles))) if target_angles else 0.0,
        min_scan_mean=float(scan.amin(dim=(1, 2)).mean().detach().cpu().item()),
    )
    return teacher, goal_blocked, diagnostics


def scan_teacher_action(
    obs: torch.Tensor,
    *,
    scan_block_threshold: float,
    scan_block_delta: float,
    goal_sector_half_width: float,
    align_angle: float,
    max_vx: float,
    max_vy: float,
    yaw_gain: float,
    intervention_delta: float,
    intervene_on_blocked_goal: bool,
    clearance_weight: float = 0.0,
    clearance_power: float = 2.0,
    speed_clearance_scale: float = 0.0,
    num_sectors: int = 13,
    min_forward_scale: float = 0.25,
    escape_risk_threshold: float = 0.0,
    escape_forward_scale: float = 0.0,
    escape_lateral_scale: float = 1.0,
    escape_radius: float = 1.0,
    escape_all_directions: bool = False,
    bypass_angle: float = 0.0,
    goal_stop_dist: float = 0.0,
    state: ScanTeacherState | None = None,
    wall_follow_steps: int = 0,
    wall_follow_angle: float = 0.9,
    wall_follow_clear_risk: float = 0.15,
    rollout_horizon: float = 0.0,
    rollout_clearance: float = 0.65,
    rollout_samples: int = 8,
    rollout_clearance_weight: float = 20.0,
    rollout_forward_bias: float = 0.05,
    emergency_radius: float = 0.0,
    emergency_hard_radius: float = 0.0,
    emergency_speed_scale: float = 0.75,
    emergency_repulsion_weight: float = 0.8,
    emergency_tangent_weight: float = 1.2,
    emergency_goal_weight: float = 0.4,
    planner: str = "heuristic",
    astar_clearance: float = 0.6,
    astar_cell_padding: float = 0.35,
    astar_resolution: float = 0.25,
    astar_waypoint_index: int = 3,
    astar_side_penalty: float = 8.0,
    astar_commit_steps: int = 30,
) -> tuple[torch.Tensor, torch.Tensor, TeacherDiagnostics]:
    """Observation-only height-scan teacher.

    The 7x7 height grid is interpreted in the robot yaw frame. Lower scan values
    are treated as nearby terrain/obstacle height discontinuities. The teacher
    selects the clear direction closest to the goal direction, then outputs a
    high-level velocity command toward that direction.
    """
    if str(planner).lower() == "astar":
        return _scan_astar_teacher_action(
            obs,
            scan_block_threshold=scan_block_threshold,
            scan_block_delta=scan_block_delta,
            align_angle=align_angle,
            max_vx=max_vx,
            max_vy=max_vy,
            yaw_gain=yaw_gain,
            min_forward_scale=min_forward_scale,
            goal_stop_dist=goal_stop_dist,
            state=state,
            clearance=astar_clearance,
            cell_padding=astar_cell_padding,
            resolution=astar_resolution,
            waypoint_index=astar_waypoint_index,
            side_penalty=astar_side_penalty,
            commit_steps=astar_commit_steps,
        )

    device = obs.device
    n = obs.shape[0]
    goal_xy = obs[:, 6:8]
    goal_dist = torch.linalg.norm(goal_xy, dim=1)
    goal_angle = torch.atan2(goal_xy[:, 1], goal_xy[:, 0])
    scan_dim = obs.shape[1] - 9
    side = int(round(scan_dim ** 0.5))
    if side * side != scan_dim:
        raise ValueError(f"Expected square current scan, got {scan_dim} values")
    scan = obs[:, 9:].reshape(n, side, side)
    min_scan = torch.nan_to_num(scan, nan=1.0).amin(dim=(1, 2))

    coords = torch.linspace(-1.5, 1.5, side, device=device)
    # Isaac's grid flattens with row=lateral and column=forward.  This was
    # verified against terrain_scan.data.hit_pos_w; do not swap these axes.
    forward = coords.view(1, side).expand(side, side)
    lateral = coords.view(side, 1).expand(side, side)
    cell_angle = torch.atan2(lateral.reshape(-1), forward.reshape(-1).clamp_min(1e-4))
    front_mask = forward.reshape(-1) > 0.0
    flat_scan = scan.reshape(n, scan_dim)
    threshold = float(scan_block_threshold)
    if float(scan_block_delta) > 0.0:
        flat_ref = torch.quantile(flat_scan, 0.9, dim=1, keepdim=True)
        relative_blocked = flat_scan < (flat_ref - float(scan_block_delta))
    else:
        relative_blocked = torch.zeros_like(flat_scan, dtype=torch.bool)
    blocked_all = (flat_scan < threshold) | relative_blocked
    blocked_cells = blocked_all & front_mask.unsqueeze(0)
    # Continuous obstacle evidence used for conservative local planning. Values
    # below the threshold are obstacle-like; values just above it still matter
    # through the angular kernel below when clearance_weight > 0.
    obstacle_risk_threshold = torch.relu((threshold - flat_scan) / max(threshold, 1e-6))
    if float(scan_block_delta) > 0.0:
        obstacle_risk_delta = torch.relu((flat_ref - float(scan_block_delta) - flat_scan) / max(float(scan_block_delta), 1e-6))
        obstacle_risk_all = torch.maximum(obstacle_risk_threshold, obstacle_risk_delta).pow(float(clearance_power))
    else:
        obstacle_risk_all = obstacle_risk_threshold.pow(float(clearance_power))
    obstacle_risk = obstacle_risk_all * front_mask.unsqueeze(0).float()

    goal_angle_grid = goal_angle.unsqueeze(1)
    goal_sector = (
        torch.abs(torch.atan2(torch.sin(cell_angle - goal_angle_grid), torch.cos(cell_angle - goal_angle_grid)))
        <= float(goal_sector_half_width)
    )
    goal_blocked = (blocked_cells & goal_sector).any(dim=1)

    sector_angles = torch.linspace(-math.pi / 2.0, math.pi / 2.0, max(3, int(num_sectors)), device=device)
    sector_clear = []
    sector_score = []
    sector_risks = []
    for sector_angle in sector_angles:
        diff = torch.abs(torch.atan2(torch.sin(cell_angle - sector_angle), torch.cos(cell_angle - sector_angle)))
        sector_mask = (diff <= float(goal_sector_half_width)) & front_mask
        blocked = (blocked_cells & sector_mask.unsqueeze(0)).any(dim=1)
        kernel_sigma = max(0.05, float(goal_sector_half_width))
        angular_kernel = torch.exp(-0.5 * (diff / kernel_sigma).pow(2)) * front_mask.float()
        risk = (obstacle_risk * angular_kernel.unsqueeze(0)).sum(dim=1)
        sector_clear.append(~blocked)
        angle_err = torch.abs(torch.atan2(torch.sin(sector_angle - goal_angle), torch.cos(sector_angle - goal_angle)))
        sector_score.append(angle_err + blocked.float() * 10.0 + float(clearance_weight) * risk)
        sector_risks.append(risk)
    clear_matrix = torch.stack(sector_clear, dim=1)
    score_matrix = torch.stack(sector_score, dim=1)
    risk_matrix = torch.stack(sector_risks, dim=1)
    best_idx = torch.argmin(score_matrix, dim=1)
    best_angle = sector_angles[best_idx]
    chosen_risk = risk_matrix.gather(1, best_idx.unsqueeze(1)).squeeze(1)

    flat_forward = forward.reshape(-1)
    flat_lateral = lateral.reshape(-1)
    if bool(escape_all_directions):
        local_radius = torch.sqrt(flat_forward.pow(2) + flat_lateral.pow(2))
        near_corridor = (local_radius <= float(escape_radius)) & (local_radius > 0.25)
        escape_risk_source = obstacle_risk_all
    else:
        near_forward = (flat_forward > 0.0) & (flat_forward <= 1.0)
        near_center = flat_lateral.abs() <= 0.75
        near_corridor = near_forward & near_center
        escape_risk_source = obstacle_risk
    near_weights = escape_risk_source * near_corridor.unsqueeze(0).float()
    near_risk = near_weights.sum(dim=1)
    weighted_forward = (near_weights * flat_forward.unsqueeze(0)).sum(dim=1)
    weighted_lateral = (near_weights * flat_lateral.unsqueeze(0)).sum(dim=1)
    goal_side = torch.sign(goal_angle).masked_fill(torch.sign(goal_angle) == 0.0, 1.0)
    left_mask = (flat_lateral > 0.0) & front_mask
    right_mask = (flat_lateral < 0.0) & front_mask
    left_risk = (obstacle_risk_all * left_mask.unsqueeze(0).float()).sum(dim=1)
    right_risk = (obstacle_risk_all * right_mask.unsqueeze(0).float()).sum(dim=1)
    bypass_side = torch.where(left_risk <= right_risk, torch.ones_like(goal_side), -torch.ones_like(goal_side))
    bypass_target = (goal_angle + bypass_side * float(bypass_angle)).clamp(-math.pi / 2.0, math.pi / 2.0)
    away_forward = -weighted_forward
    away_lateral = -weighted_lateral
    away_norm = torch.sqrt(away_forward.pow(2) + away_lateral.pow(2)).clamp_min(1e-6)
    escape_forward_dir = torch.where(near_risk > 0.0, away_forward / away_norm, torch.zeros_like(away_forward))
    escape_lateral_dir = torch.where(near_risk > 0.0, away_lateral / away_norm, -goal_side)
    escape_side = torch.sign(escape_lateral_dir)
    escape_side = torch.where(escape_side == 0.0, -goal_side, escape_side)
    escape_active = near_risk > float(escape_risk_threshold) if float(escape_risk_threshold) > 0.0 else torch.zeros_like(near_risk, dtype=torch.bool)

    any_clear = clear_matrix.any(dim=1)
    target_angle = torch.where(any_clear, best_angle, torch.sign(goal_angle).clamp(min=-1.0, max=1.0) * (math.pi / 2.0))
    target_angle = torch.where(goal_blocked, target_angle, goal_angle.clamp(-math.pi / 2.0, math.pi / 2.0))
    if float(bypass_angle) > 0.0:
        target_angle = torch.where(goal_blocked, bypass_target, target_angle)
    target_angle = torch.where(escape_active, escape_side * (math.pi / 2.0), target_angle)

    emergency_active = torch.zeros(n, dtype=torch.bool, device=device)
    if float(emergency_radius) > 0.0:
        obstacle_points = torch.stack([flat_forward, flat_lateral], dim=1)
        point_dist = torch.linalg.norm(obstacle_points, dim=1).clamp_min(1e-6)
        close_weight = (
            torch.relu(float(emergency_radius) - point_dist).unsqueeze(0)
            / max(float(emergency_radius), 1e-6)
        ).pow(2)
        # Emergency uses all nearby occupied cells, not only cells in front of
        # the robot. Once close to an obstacle, backing/side-stepping can be the
        # only safe action.
        close_weight = close_weight * blocked_all.float()
        weight_sum = close_weight.sum(dim=1).clamp_min(1e-6)
        close_any = close_weight.sum(dim=1) > 0.0
        centroid = close_weight @ obstacle_points / weight_sum.unsqueeze(1)
        centroid_norm = torch.linalg.norm(centroid, dim=1).clamp_min(1e-6)
        away = -centroid / centroid_norm.unsqueeze(1)
        tangent_a = torch.stack([-centroid[:, 1], centroid[:, 0]], dim=1)
        tangent_a = tangent_a / torch.linalg.norm(tangent_a, dim=1).clamp_min(1e-6).unsqueeze(1)
        tangent_b = -tangent_a
        goal_unit = goal_xy / goal_dist.clamp_min(1e-6).unsqueeze(1)
        tangent = torch.where(
            ((tangent_a * goal_unit).sum(dim=1) >= (tangent_b * goal_unit).sum(dim=1)).unsqueeze(1),
            tangent_a,
            tangent_b,
        )
        blended = (
            float(emergency_repulsion_weight) * away
            + float(emergency_tangent_weight) * tangent
            + float(emergency_goal_weight) * goal_unit
        )
        emergency_angle = torch.atan2(blended[:, 1], blended[:, 0])
        if float(emergency_hard_radius) > 0.0:
            hard_weight = (
                torch.relu(float(emergency_hard_radius) - point_dist).unsqueeze(0)
                / max(float(emergency_hard_radius), 1e-6)
            ).pow(2) * blocked_all.float()
            hard_any = hard_weight.sum(dim=1) > 0.0
            hard_centroid = hard_weight @ obstacle_points / hard_weight.sum(dim=1).clamp_min(1e-6).unsqueeze(1)
            hard_away = -hard_centroid / torch.linalg.norm(hard_centroid, dim=1).clamp_min(1e-6).unsqueeze(1)
            hard_angle = torch.atan2(hard_away[:, 1], hard_away[:, 0])
            emergency_angle = torch.where(hard_any, hard_angle, emergency_angle)
        target_angle = torch.where(close_any, emergency_angle, target_angle)
        emergency_active = close_any

    if state is not None and int(wall_follow_steps) > 0:
        start_follow = goal_blocked | escape_active
        chosen_side = torch.where(left_risk <= right_risk, torch.ones_like(goal_side), -torch.ones_like(goal_side))
        state.bypass_side = torch.where(start_follow, chosen_side, state.bypass_side)
        state.bypass_steps = torch.where(
            start_follow,
            torch.full_like(state.bypass_steps, int(wall_follow_steps)),
            torch.clamp(state.bypass_steps - 1, min=0),
        )
        clear_enough = (~goal_blocked) & (near_risk <= float(wall_follow_clear_risk))
        state.bypass_steps = torch.where(clear_enough, torch.zeros_like(state.bypass_steps), state.bypass_steps)
        follow_active = state.bypass_steps > 0
        follow_angle = torch.clamp(state.bypass_side * float(wall_follow_angle), -math.pi / 2.0, math.pi / 2.0)
        target_angle = torch.where(follow_active, follow_angle, target_angle)

    if float(rollout_horizon) > 0.0:
        obstacle_points = torch.stack([flat_forward, flat_lateral], dim=1)
        obstacle_mask = flat_scan < threshold
        candidate_angles = sector_angles
        candidate_scores = []
        candidate_clearances = []
        num_path_samples = max(2, int(rollout_samples))
        path_s = torch.linspace(0.25, float(rollout_horizon), num_path_samples, device=device)
        for angle in candidate_angles:
            direction = torch.stack([torch.cos(angle), torch.sin(angle)])
            path = path_s.unsqueeze(1) * direction.unsqueeze(0)
            dist = torch.cdist(path.unsqueeze(0).expand(n, -1, -1), obstacle_points.unsqueeze(0).expand(n, -1, -1))
            masked_dist = torch.where(obstacle_mask.unsqueeze(1), dist, torch.full_like(dist, 1e6))
            min_dist = masked_dist.amin(dim=(1, 2))
            endpoint = float(rollout_horizon) * direction
            goal_after = torch.linalg.norm(goal_xy - endpoint.unsqueeze(0), dim=1)
            clearance_violation = torch.relu(float(rollout_clearance) - min_dist)
            angle_err = torch.abs(torch.atan2(torch.sin(angle - goal_angle), torch.cos(angle - goal_angle)))
            forward_penalty = torch.relu(-torch.cos(angle)) * 2.0
            score = (
                goal_after
                + float(rollout_clearance_weight) * clearance_violation.pow(2)
                + float(rollout_forward_bias) * angle_err
                + forward_penalty
            )
            candidate_scores.append(score)
            candidate_clearances.append(min_dist)
        rollout_scores = torch.stack(candidate_scores, dim=1)
        rollout_clearances = torch.stack(candidate_clearances, dim=1)
        rollout_idx = torch.argmin(rollout_scores, dim=1)
        rollout_angle = candidate_angles[rollout_idx]
        rollout_clearance_chosen = rollout_clearances.gather(1, rollout_idx.unsqueeze(1)).squeeze(1)
        use_rollout = goal_blocked | escape_active | (rollout_clearance_chosen < float(rollout_clearance) * 1.5)
        target_angle = torch.where(use_rollout, rollout_angle, target_angle)
        chosen_risk = torch.where(
            use_rollout,
            torch.relu(float(rollout_clearance) - rollout_clearance_chosen).pow(2),
            chosen_risk,
        )

    yaw = torch.clamp(float(yaw_gain) * target_angle, -1.0, 1.0)
    speed_gate = (target_angle.abs() <= float(align_angle)).float()
    clearance_speed = 1.0 / (1.0 + float(speed_clearance_scale) * chosen_risk)
    min_scale = float(min_forward_scale)
    vx = (
        float(max_vx)
        * torch.clamp(torch.cos(target_angle), min=0.0)
        * (min_scale + (1.0 - min_scale) * speed_gate)
        * clearance_speed
    )
    vy = torch.clamp(float(max_vy) * torch.sin(target_angle), -1.0, 1.0)
    if bool(escape_all_directions):
        escape_vx = torch.clamp(float(max_vx) * float(escape_forward_scale) * escape_forward_dir, -1.0, 1.0)
        escape_vy = torch.clamp(float(max_vy) * float(escape_lateral_scale) * escape_lateral_dir, -1.0, 1.0)
        vx = torch.where(escape_active, escape_vx, vx)
        vy = torch.where(escape_active, escape_vy, vy)
    else:
        vx = torch.where(escape_active, vx * float(escape_forward_scale), vx)
        vy = torch.where(escape_active, torch.clamp(float(max_vy) * float(escape_lateral_scale) * escape_side, -1.0, 1.0), vy)
    if float(emergency_radius) > 0.0:
        emergency_vx = torch.clamp(float(max_vx) * float(emergency_speed_scale) * torch.cos(target_angle), -1.0, 1.0)
        emergency_vy = torch.clamp(float(max_vy) * float(emergency_speed_scale) * torch.sin(target_angle), -1.0, 1.0)
        vx = torch.where(emergency_active, emergency_vx, vx)
        vy = torch.where(emergency_active, emergency_vy, vy)
    teacher = torch.stack([vx, vy, yaw], dim=-1).clamp(-1.0, 1.0)
    if float(goal_stop_dist) > 0.0:
        teacher = torch.where((goal_dist <= float(goal_stop_dist)).unsqueeze(1), torch.zeros_like(teacher), teacher)

    # The final gate also depends on the current student action, so this helper
    # returns a delta-threshold function input through `student_delta_mask`.
    diagnostics = TeacherDiagnostics(
        goal_blocked_fraction=float(goal_blocked.float().mean().detach().cpu().item()),
        intervention_fraction=0.0,
        action_delta_mean=0.0,
        target_angle_abs_mean=float(target_angle.abs().mean().detach().cpu().item()),
        min_scan_mean=float(min_scan.mean().detach().cpu().item()),
    )
    return teacher, goal_blocked if intervene_on_blocked_goal else torch.zeros_like(goal_blocked), diagnostics


def make_env(args: argparse.Namespace, *, render: bool = False):
    _setup_unitree_imports(Path(args.low_level_policy_path).resolve())
    from eval_unitree_nav_baselines import _apply_debug_obstacle_overrides, _apply_scan_and_goal_overrides

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.wrappers import VideoRecorder
    from src.envs import build_env

    env_cfg = load_env_cfg(args.task, play=render)
    env_cfg.seed = int(args.seed)
    from unitree_nav_layout import configure_terrain_tile_resets

    _apply_debug_obstacle_overrides(args, env_cfg)
    _apply_scan_and_goal_overrides(args, env_cfg)
    terrain_generator = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
    if terrain_generator is not None and hasattr(terrain_generator, "seed"):
        terrain_generator.seed = int(args.seed)
    configure_terrain_tile_resets(
        env_cfg,
        enabled=bool(getattr(args, "resample_terrain_tiles", False)),
    )
    env_cfg.scene.num_envs = int(args.num_envs if not render else 1)
    env_cfg.episode_length_s = float(args.episode_length_s)
    if "pose" in env_cfg.commands:
        env_cfg.commands["pose"].resampling_time_range = (float(args.episode_length_s), float(args.episode_length_s))
        goal_min = float(getattr(args, "goal_distance_min", 0.0))
        goal_max = float(getattr(args, "goal_distance_max", 0.0))
        if goal_min > 0.0 or goal_max > 0.0:
            if goal_min <= 0.0 or goal_max < goal_min:
                raise ValueError("goal_distance_min/max must satisfy 0 < min <= max")
            env_cfg.commands["pose"].distance_range = (goal_min, goal_max)

    render_mode = "rgb_array" if render else None
    env = build_env(env_cfg, device=args.device, render_mode=render_mode)
    if render:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        env = VideoRecorder(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=int(args.video_length),
            disable_logger=True,
        )
    return RslRlVecEnvWrapper(env, clip_actions=1.0)


def _needs_feasible_reset(args: argparse.Namespace) -> bool:
    if int(getattr(args, "num_envs", 1)) != 1:
        return False
    return (
        float(getattr(args, "min_start_obstacle_clearance", 0.0)) > 0.0
        or float(getattr(args, "min_goal_obstacle_clearance", 0.0)) > 0.0
        or bool(getattr(args, "require_blocked_corridor", False))
        or (int(getattr(args, "num_envs", 1)) == 1 and bool(getattr(args, "debug_goal_through_obstacle", False)))
    )


def _reset_train_env(args: argparse.Namespace, env):
    if not _needs_feasible_reset(args):
        obs_raw, _ = env.reset()
        return obs_raw
    if int(args.num_envs) != 1:
        raise ValueError("Unitree feasible/debug-goal reset filtering currently requires --num-envs 1")
    from eval_unitree_nav_baselines import _reset_until_feasible

    obs_raw, _, _, _, _ = _reset_until_feasible(args, env)
    return obs_raw


def save_checkpoint(
    path: Path,
    *,
    sac,
    args: argparse.Namespace,
    step: int,
    obs_dim: int,
    act_dim: int,
    policy_actor=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actor = policy_actor if policy_actor is not None else sac.actor
    payload = {
        "step": int(step),
        "args": vars(args),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "policy_family": "hg_dagger" if str(getattr(args, "method", "thesis")) == "hg_dagger" else "sac_actor",
        "actor_state_dict": actor.state_dict(),
    }
    if sac is not None:
        payload.update(
            {
            "critic_state_dict": sac.critic.state_dict(),
            "critic_target_state_dict": sac.critic_target.state_dict(),
            "log_alpha": sac.log_alpha.detach().cpu(),
            "pref_lambda": float(sac.pref_lambda),
            "pref_violation_ema": float(sac.pref_violation_ema),
            }
        )
    torch.save(payload, path)


def run_checkpoint_evaluation(
    checkpoint_path: Path,
    *,
    args: argparse.Namespace,
    step: int,
    run_dir: Path,
) -> dict[str, float] | None:
    """Run deterministic policy-only evaluation in an isolated subprocess."""
    eval_root = run_dir / "eval"
    eval_name = f"step_{int(step)}"
    eval_root.mkdir(parents=True, exist_ok=True)
    log_path = eval_root / f"{eval_name}.log"
    cmd = [
        sys.executable,
        str(ROOT / "eval_unitree_nav_baselines.py"),
        "--controller",
        "policy",
        "--model-path",
        str(checkpoint_path.resolve()),
        "--checkpoint-env-config",
        "--device",
        str(args.device),
        "--seed",
        str(int(args.eval_seed)),
        "--num-envs",
        str(int(args.eval_num_envs)),
        "--num-episodes",
        str(int(args.eval_num_episodes)),
        "--output-dir",
        str(eval_root.resolve()),
        "--run-name",
        eval_name,
    ]
    if str(getattr(args, "eval_layout_manifest", "")):
        cmd.extend(["--layout-manifest", str(Path(args.eval_layout_manifest).resolve())])
    print(
        f"[periodic-eval] starting step={step} episodes={args.eval_num_episodes} "
        f"num_envs={args.eval_num_envs} seed={args.eval_seed}",
        flush=True,
    )
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("WARP_CACHE_PATH", "/tmp/warp-cache")
    env.setdefault("XDG_CACHE_HOME", "/tmp/unitree-cache")
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=float(args.eval_timeout_s),
                check=False,
            )
    except subprocess.TimeoutExpired:
        print(f"[periodic-eval] timed out at step={step}; see {log_path}", flush=True)
        if args.eval_fail_fast:
            raise
        return None
    if completed.returncode != 0:
        message = f"periodic evaluation failed at step={step} exit={completed.returncode}; see {log_path}"
        if args.eval_fail_fast:
            raise RuntimeError(message)
        print(f"[periodic-eval] WARNING: {message}", flush=True)
        return None
    metrics_path = eval_root / eval_name / "policy_metrics.json"
    if not metrics_path.exists():
        message = f"periodic evaluation did not produce {metrics_path}"
        if args.eval_fail_fast:
            raise RuntimeError(message)
        print(f"[periodic-eval] WARNING: {message}", flush=True)
        return None
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    episodes = list(summary.get("episodes", []))
    safe_success_rate = sum(
        bool(episode.get("success")) and float(episode.get("cost_sum", 0.0)) == 0.0
        for episode in episodes
    ) / max(1, len(episodes))
    row = {
        "step": float(step),
        "transitions": float(step * int(args.num_envs)),
        "episodes": float(summary.get("num_episodes", len(episodes))),
        "success_rate": float(summary.get("success_rate", 0.0)),
        "safe_success_rate": float(safe_success_rate),
        "mean_cost_sum": float(summary.get("mean_cost_sum", 0.0)),
        "costful_episode_rate": float(summary.get("costful_episode_rate", 0.0)),
        "mean_costful_steps": float(summary.get("mean_costful_steps", 0.0)),
        "mean_collision_steps": float(summary.get("mean_collision_steps", 0.0)),
        "mean_time_to_success_s": float(summary.get("mean_time_to_success_s_success_only", 0.0)),
        "mean_episode_length_s": float(summary.get("mean_episode_length_s", 0.0)),
        "mean_action_delta": float(summary.get("mean_action_delta", 0.0)),
        "mean_action_sign_flips": float(summary.get("mean_action_sign_flips", 0.0)),
        "mean_min_goal_distance": float(summary.get("mean_min_goal_distance", 0.0)),
    }
    with (run_dir / "eval_metrics.jsonl").open("a", encoding="utf-8") as eval_file:
        eval_file.write(json.dumps(row) + "\n")
    print(f"[periodic-eval] {json.dumps(row, sort_keys=True)}", flush=True)
    return row


def run_training(args: argparse.Namespace) -> Path:
    from safetygym_utils.sac import build_sac, sac_update_step
    from unitree_nav_competitors import (
        UnitreeExpertBuffer,
        build_hg_dagger,
        build_pvp_state,
        concat_replay_batches,
        hg_dagger_update,
        pvp_update_step,
    )
    method = str(args.method).strip().lower()
    print(
        "[learner] "
        f"method={method} num_envs={args.num_envs} n_step={args.n_step} "
        f"updates_per_step={args.updates_per_step} init_actor={args.init_actor_checkpoint or '<none>'}",
        flush=True,
    )
    intervention_clearance, intervention_release_clearance = resolve_intervention_clearances(args)
    args.intervention_clearance_threshold_effective = intervention_clearance
    args.intervention_release_clearance_effective = intervention_release_clearance
    print(
        "[intervention-gate] "
        f"mode={args.intervention_clearance_mode} engage={intervention_clearance:.3f}m "
        f"release={intervention_release_clearance:.3f}m teacher_clearance={args.teacher_geom_clearance:.3f}m",
        flush=True,
    )
    from eval_unitree_nav_baselines import (
        _robot_clearances,
        _terrain_obstacle_cells_by_env,
        _layout_blocked_corridor_stats,
        _resample_close_goals,
        _set_goal_through_obstacle,
        _validate_required_blocked_corridors,
        direct_goal_action,
    )
    from unitree_nav_geom_teacher import GeomTeacherState, local_geometry_scan_teacher_action

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            group=args.wandb_group or None,
            mode=args.wandb_mode,
            config=vars(args),
            dir=str(run_dir),
        )

    env = make_env(args, render=False)
    obs_raw = _reset_train_env(args, env)
    obstacle_cells = _terrain_obstacle_cells_by_env(env)
    adjusted_obs = _set_goal_through_obstacle(args, env, obstacle_cells)
    if adjusted_obs is not None:
        obs_raw = adjusted_obs
    clearance_obs, _ = _resample_close_goals(args, env, obstacle_cells)
    if clearance_obs is not None:
        obs_raw = clearance_obs
    _validate_required_blocked_corridors(args, env, obstacle_cells)
    from unitree_nav_observation import UnitreeScanHistory, current_unitree_scan_obs

    scan_history = UnitreeScanHistory(args.scan_history, args.action_history, int(env.action_space.shape[-1]))
    obs = scan_history.reset(_prepare_actor_obs(obs_raw, args))
    num_envs, obs_dim = int(obs.shape[0]), int(obs.shape[1])
    act_dim = int(env.action_space.shape[-1])
    device = torch.device(args.device)

    sac = build_sac(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_actor=args.hidden_dim,
        hidden_critic=args.hidden_dim,
        num_critics=2,
        use_layer_norm=args.use_layer_norm,
        layer_norm_eps=1e-5,
        init_scale=0.01,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        weight_decay=0.0,
        num_envs=num_envs,
        device=device,
        alpha_init=args.alpha_init,
        temporal_encoder="unitree_scan_cnn" if args.policy_encoder == "scan_cnn" else "none",
        obs_frame_stack=args.scan_history,
        unitree_action_history=args.action_history,
    )
    if args.init_checkpoint:
        init_checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if int(init_checkpoint.get("obs_dim", obs_dim)) != obs_dim or int(init_checkpoint.get("act_dim", act_dim)) != act_dim:
            raise ValueError("--init-checkpoint observation/action dimensions do not match the current environment")
        sac.actor.load_state_dict(init_checkpoint["actor_state_dict"])
        sac.critic.load_state_dict(init_checkpoint["critic_state_dict"])
        sac.critic_target.load_state_dict(init_checkpoint["critic_target_state_dict"])
        with torch.no_grad():
            sac.log_alpha.copy_(torch.as_tensor(init_checkpoint["log_alpha"], device=device))
        sac.pref_lambda = float(init_checkpoint.get("pref_lambda", sac.pref_lambda))
        sac.pref_violation_ema = float(init_checkpoint.get("pref_violation_ema", sac.pref_violation_ema))
        print(f"[init] loaded actor, critics, alpha, and preference state from {args.init_checkpoint}", flush=True)
    elif args.init_actor_checkpoint:
        init_checkpoint = torch.load(args.init_actor_checkpoint, map_location=device, weights_only=False)
        sac.actor.load_state_dict(init_checkpoint["actor_state_dict"])
        print(f"[init] loaded actor from {args.init_actor_checkpoint}", flush=True)
    hg_state = None
    pvp_state = None
    if method == "hg_dagger":
        hg_state = build_hg_dagger(
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_envs=num_envs,
            hidden_dim=args.hidden_dim,
            ensemble_size=args.hg_ensemble_size,
            use_layer_norm=args.use_layer_norm,
            policy_encoder=args.policy_encoder,
            scan_history=args.scan_history,
            action_history=args.action_history,
            lr=args.lr_actor,
            device=device,
        )
        for member in hg_state.actor.members:
            member.load_state_dict(sac.actor.state_dict())
        policy_actor = hg_state.actor
    else:
        policy_actor = sac.actor
    if method == "pvp":
        pvp_state = build_pvp_state(sac)
    buffer = UnitreeReplayBuffer(capacity=args.replay_capacity, obs_dim=obs_dim, act_dim=act_dim, device=device)
    human_buffer = (
        UnitreeReplayBuffer(capacity=args.replay_capacity, obs_dim=obs_dim, act_dim=act_dim, device=device)
        if method in {"hilserl", "pvp"}
        else None
    )
    novice_buffer = (
        UnitreeReplayBuffer(capacity=args.replay_capacity, obs_dim=obs_dim, act_dim=act_dim, device=device)
        if method == "pvp"
        else None
    )
    expert_buffer = (
        UnitreeExpertBuffer(capacity=args.replay_capacity, obs_dim=obs_dim, act_dim=act_dim, device=device)
        if method == "hg_dagger"
        else None
    )
    eil_recent_indices = [deque(maxlen=max(1, int(args.eil_bad_pre_steps))) for _ in range(num_envs)]
    nstep_accumulator = UnitreeNStepAccumulator(num_envs=num_envs, n_step=args.n_step, gamma=args.gamma)

    start_time = time.time()
    episode_return = torch.zeros(num_envs, device=device)
    episode_env_return = torch.zeros(num_envs, device=device)
    episode_cost = torch.zeros(num_envs, device=device)
    episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    completed_returns: list[float] = []
    completed_env_returns: list[float] = []
    completed_costs: list[float] = []
    completed_successes: list[float] = []
    update_metrics: list[dict[str, float]] = []
    teacher_state = (
        GeomTeacherState.create(num_envs)
        if args.teacher_type == "geom_scan"
        else ScanTeacherState(num_envs, device)
    )
    gate_state = InterventionGateState(num_envs, device)
    total_interventions = 0
    total_rows = 0
    interval_interventions = 0
    interval_rows = 0
    interval_clearance_triggers = 0
    interval_stall_triggers = 0
    interval_releases = 0
    interval_clearance_engagements = 0
    interval_stall_engagements = 0
    interval_release_blocked_clearance = 0
    interval_release_blocked_progress = 0
    interval_release_blocked_action_delta = 0
    interval_released_duration_sum = 0
    interval_released_duration_count = 0
    interval_teacher_cost = 0.0
    interval_student_cost = 0.0
    interval_teacher_costful_steps = 0
    interval_student_costful_steps = 0
    interval_teacher_rows = 0
    interval_student_rows = 0
    interval_reverse_action_sum = 0.0
    interval_lateral_action_abs_sum = 0.0
    interval_goal_turn_alignment_sum = 0.0
    previous_student_action = torch.zeros(num_envs, act_dim, device=device)
    previous_teacher_intervened = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for global_step in range(1, int(args.total_steps) + 1):
        with torch.no_grad():
            if args.student_controller == "direct_goal":
                student_action = direct_goal_action(
                    current_unitree_scan_obs(
                        obs, scan_history=args.scan_history, action_history=args.action_history
                    ),
                    max_vx=args.student_direct_goal_max_vx,
                    max_vy=args.student_direct_goal_max_vy,
                    yaw_gain=args.student_direct_goal_yaw_gain,
                    align_angle=args.student_direct_goal_align_angle,
                )
            elif global_step <= args.random_steps:
                student_action = torch.empty((num_envs, act_dim), device=device).uniform_(-1.0, 1.0)
            else:
                student_action, _, student_mean = policy_actor(obs)
                if args.deterministic_student:
                    student_action = student_mean
            if float(args.student_action_smoothing) > 0.0:
                keep = float(args.student_action_smoothing)
                student_action = keep * previous_student_action + (1.0 - keep) * student_action
            previous_student_action.copy_(student_action)
            teacher_disabled = args.intervention_gate_mode == "none" and int(args.teacher_warmup_steps) <= 0
            if teacher_disabled:
                teacher_action = student_action
                blocked_gate = torch.zeros(num_envs, dtype=torch.bool, device=device)
                goal_blocked_fraction = 0.0
                teacher_obs = current_unitree_scan_obs(
                    obs, scan_history=args.scan_history, action_history=args.action_history
                )
                min_scan_mean = float(torch.nan_to_num(teacher_obs[:, 9:], nan=1.0).amin(dim=1).mean().cpu().item())
            elif args.teacher_type == "geom_scan":
                teacher_action = local_geometry_scan_teacher_action(
                    current_unitree_scan_obs(
                        obs, scan_history=args.scan_history, action_history=args.action_history
                    ),
                    env=env,
                    obstacle_cells=obstacle_cells,
                    args=args,
                    state=teacher_state,
                )
                blocked_gate = torch.zeros(num_envs, dtype=torch.bool, device=device)
                goal_blocked_fraction = 0.0
                teacher_obs = current_unitree_scan_obs(
                    obs, scan_history=args.scan_history, action_history=args.action_history
                )
                min_scan_mean = float(torch.nan_to_num(teacher_obs[:, 9:], nan=1.0).amin(dim=1).mean().cpu().item())
            else:
                teacher_action, blocked_gate, diag = scan_teacher_action(
                    current_unitree_scan_obs(
                        obs, scan_history=args.scan_history, action_history=args.action_history
                    ),
                    scan_block_threshold=args.teacher_scan_block_threshold,
                    scan_block_delta=args.teacher_scan_block_delta,
                    goal_sector_half_width=args.teacher_sector_half_width,
                    align_angle=args.teacher_align_angle,
                    max_vx=args.teacher_max_vx,
                    max_vy=args.teacher_max_vy,
                    yaw_gain=args.teacher_yaw_gain,
                    intervention_delta=args.intervention_delta,
                    intervene_on_blocked_goal=args.intervene_on_blocked_goal,
                    clearance_weight=args.teacher_clearance_weight,
                    clearance_power=args.teacher_clearance_power,
                    speed_clearance_scale=args.teacher_speed_clearance_scale,
                    num_sectors=args.teacher_num_sectors,
                    min_forward_scale=args.teacher_min_forward_scale,
                    escape_risk_threshold=args.teacher_escape_risk_threshold,
                    escape_forward_scale=args.teacher_escape_forward_scale,
                    escape_lateral_scale=args.teacher_escape_lateral_scale,
                    escape_radius=args.teacher_escape_radius,
                    escape_all_directions=args.teacher_escape_all_directions,
                    bypass_angle=args.teacher_bypass_angle,
                    goal_stop_dist=args.teacher_goal_stop_dist,
                    state=teacher_state,
                    wall_follow_steps=args.teacher_wall_follow_steps,
                    wall_follow_angle=args.teacher_wall_follow_angle,
                    wall_follow_clear_risk=args.teacher_wall_follow_clear_risk,
                    rollout_horizon=args.teacher_rollout_horizon,
                    rollout_clearance=args.teacher_rollout_clearance,
                    rollout_samples=args.teacher_rollout_samples,
                    rollout_clearance_weight=args.teacher_rollout_clearance_weight,
                    rollout_forward_bias=args.teacher_rollout_forward_bias,
                    emergency_radius=args.teacher_emergency_radius,
                    emergency_hard_radius=args.teacher_emergency_hard_radius,
                    emergency_speed_scale=args.teacher_emergency_speed_scale,
                    emergency_repulsion_weight=args.teacher_emergency_repulsion_weight,
                    emergency_tangent_weight=args.teacher_emergency_tangent_weight,
                    emergency_goal_weight=args.teacher_emergency_goal_weight,
                    planner=args.teacher_scan_planner,
                    astar_clearance=args.teacher_scan_astar_clearance,
                    astar_cell_padding=args.teacher_scan_astar_cell_padding,
                    astar_resolution=args.teacher_scan_astar_resolution,
                    astar_waypoint_index=args.teacher_scan_astar_waypoint_index,
                    astar_side_penalty=args.teacher_scan_astar_side_penalty,
                    astar_commit_steps=args.teacher_scan_astar_commit_steps,
                )
                goal_blocked_fraction = diag.goal_blocked_fraction
                min_scan_mean = diag.min_scan_mean
            delta = torch.linalg.norm(student_action - teacher_action, dim=-1)
            delta_gate = delta >= float(args.intervention_delta)
            clearance = _robot_clearances(env, obstacle_cells)
            goal_distance = torch.linalg.norm(obs[:, 6:8], dim=-1)
            if args.intervention_gate_mode == "clearance_or_stall":
                (
                    clearance_gate,
                    clearance_trigger,
                    stall_trigger,
                    release,
                    engage_clearance,
                    engage_stall,
                    release_blocked_clearance,
                    release_blocked_progress,
                    release_blocked_action_delta,
                    released_duration,
                ) = gate_state.update(
                    distance=goal_distance,
                    clearance=clearance,
                    action_delta=delta,
                    clearance_threshold=intervention_clearance,
                    release_clearance=intervention_release_clearance,
                    stall_steps=args.intervention_stall_steps,
                    progress_epsilon=args.intervention_progress_epsilon,
                    release_steps=args.intervention_release_steps,
                    release_progress_tolerance=args.intervention_release_progress_tolerance,
                    release_action_delta_max=args.intervention_release_action_delta_max,
                    goal_tolerance=args.success_dist,
                )
                base_gate = clearance_gate
            else:
                clearance_trigger = torch.zeros_like(delta_gate)
                stall_trigger = torch.zeros_like(delta_gate)
                release = torch.zeros_like(delta_gate)
                engage_clearance = torch.zeros_like(delta_gate)
                engage_stall = torch.zeros_like(delta_gate)
                release_blocked_clearance = torch.zeros_like(delta_gate)
                release_blocked_progress = torch.zeros_like(delta_gate)
                release_blocked_action_delta = torch.zeros_like(delta_gate)
                released_duration = torch.zeros_like(gate_state.bad_steps)
                gate_state.reset()
                if args.intervention_gate_mode == "action_delta":
                    base_gate = blocked_gate | delta_gate
                elif args.intervention_gate_mode == "always":
                    base_gate = torch.ones_like(delta_gate)
                else:
                    base_gate = torch.zeros_like(delta_gate)
            warmup_gate = torch.full_like(delta_gate, global_step <= int(args.teacher_warmup_steps))
            teacher_intervened = warmup_gate | base_gate
        intervention_start = teacher_intervened & ~previous_teacher_intervened
        action = torch.where(teacher_intervened.unsqueeze(-1), teacher_action, student_action).clamp(-1.0, 1.0)

        # Goal-only pretraining should learn to turn and walk forward rather
        # than exploit backwards or lateral locomotion for Euclidean progress.
        goal_angle = torch.atan2(obs[:, 7], obs[:, 6])
        reverse_action = torch.relu(-action[:, 0])
        lateral_action_abs = action[:, 1].abs()
        goal_turn_alignment = torch.sign(goal_angle) * action[:, 2]
        interval_reverse_action_sum += float(reverse_action.sum().detach().cpu().item())
        interval_lateral_action_abs_sum += float(lateral_action_abs.sum().detach().cpu().item())
        interval_goal_turn_alignment_sum += float(goal_turn_alignment.sum().detach().cpu().item())

        if expert_buffer is not None and bool(teacher_intervened.any().item()):
            expert_buffer.add(obs[teacher_intervened], teacher_action[teacher_intervened])
        previous_teacher_intervened.copy_(teacher_intervened)

        intervention_count = int(teacher_intervened.sum().cpu().item())
        total_interventions += intervention_count
        total_rows += num_envs
        interval_interventions += intervention_count
        interval_rows += num_envs
        interval_clearance_triggers += int(clearance_trigger.sum().cpu().item())
        interval_stall_triggers += int(stall_trigger.sum().cpu().item())
        interval_releases += int(release.sum().cpu().item())
        interval_clearance_engagements += int(engage_clearance.sum().cpu().item())
        interval_stall_engagements += int(engage_stall.sum().cpu().item())
        interval_release_blocked_clearance += int(release_blocked_clearance.sum().cpu().item())
        interval_release_blocked_progress += int(release_blocked_progress.sum().cpu().item())
        interval_release_blocked_action_delta += int(release_blocked_action_delta.sum().cpu().item())
        interval_released_duration_sum += int(released_duration.sum().cpu().item())
        interval_released_duration_count += int(release.sum().cpu().item())

        next_raw, env_reward, done, extras = env.step(action)
        from eval_unitree_nav_baselines import _goal_termination_mask

        terminal_success = _goal_termination_mask(env, num_envs, device)
        next_current_obs = _prepare_actor_obs(next_raw, args)
        env_reward = env_reward.to(device=device, dtype=torch.float32).reshape(num_envs)
        done = done.to(device=device).reshape(num_envs).bool()
        next_obs = scan_history.step(next_current_obs, done, action=action)
        previous_student_action[done] = 0.0
        previous_teacher_intervened[done] = False
        cost = _extract_cost(extras, num_envs, device)
        costful = cost > 0.0
        interval_teacher_cost += float(cost[teacher_intervened].sum().detach().cpu().item())
        interval_student_cost += float(cost[~teacher_intervened].sum().detach().cpu().item())
        interval_teacher_costful_steps += int((costful & teacher_intervened).sum().detach().cpu().item())
        interval_student_costful_steps += int((costful & ~teacher_intervened).sum().detach().cpu().item())
        interval_teacher_rows += int(teacher_intervened.sum().detach().cpu().item())
        interval_student_rows += int((~teacher_intervened).sum().detach().cpu().item())
        success = _goal_success(next_obs, args.success_dist) | terminal_success
        first_success = success & ~episode_success
        if args.learner_reward_mode in {"dense_progress", "dense_progress_exp"}:
            pre_distance = torch.linalg.norm(obs[:, 6:8], dim=-1)
            post_distance = torch.linalg.norm(next_obs[:, 6:8], dim=-1)
            post_distance = torch.where(done, pre_distance, post_distance)
            reward = float(args.dense_progress_scale) * (pre_distance - post_distance)
            if args.learner_reward_mode == "dense_progress_exp":
                temperature = max(float(args.dense_progress_exp_temperature), 1e-6)
                pre_potential = torch.exp(-pre_distance / temperature)
                post_potential = torch.exp(-post_distance / temperature)
                reward += float(args.dense_progress_exp_scale) * (post_potential - pre_potential)
            reward += float(args.goal_turn_alignment_scale) * goal_turn_alignment
            reward -= float(args.reverse_action_penalty) * reverse_action
            reward -= float(args.lateral_action_penalty) * lateral_action_abs
            reward += float(args.success_bonus) * first_success.float()
            reward += float(args.failure_penalty) * (done & ~success).float()
        else:
            reward = env_reward
        # The vector environment auto-resets before returning next_obs, so a
        # timeout/fall transition does not contain the terminal observation.
        # Bootstrapping it would connect the old episode to a new random goal.
        trunc = torch.zeros_like(done)

        replay_rows = nstep_accumulator.add(
            obs=obs,
            actions=action,
            student_actions=student_action,
            next_obs=next_obs,
            rewards=reward,
            dones=done,
            truncations=trunc,
            teacher_intervened=teacher_intervened,
            intervention_start=intervention_start,
            env_ids=torch.arange(num_envs, device=device, dtype=torch.long),
        )
        if replay_rows is not None:
            added_indices = buffer.add(**replay_rows)
            if method == "eil":
                for row_idx, buffer_idx in enumerate(added_indices.detach().cpu().tolist()):
                    env_idx = int(replay_rows["env_ids"][row_idx].item())
                    if bool(replay_rows["intervention_start"][row_idx].item()):
                        buffer.mark_eil_bad(list(eil_recent_indices[env_idx]))
                    if bool(replay_rows["teacher_intervened"][row_idx].item()):
                        eil_recent_indices[env_idx].clear()
                    else:
                        eil_recent_indices[env_idx].append(int(buffer_idx))
            if human_buffer is not None:
                human_mask = replay_rows["teacher_intervened"].bool()
                if bool(human_mask.any().item()):
                    human_buffer.add(**{key: value[human_mask] for key, value in replay_rows.items()})
            if novice_buffer is not None:
                novice_mask = ~replay_rows["teacher_intervened"].bool()
                if bool(novice_mask.any().item()):
                    novice_buffer.add(**{key: value[novice_mask] for key, value in replay_rows.items()})

        episode_return += reward
        episode_env_return += env_reward
        episode_cost += cost
        episode_success |= success
        if done.any():
            done_idx = torch.nonzero(done, as_tuple=False).flatten()
            completed_returns.extend(episode_return[done_idx].detach().cpu().tolist())
            completed_env_returns.extend(episode_env_return[done_idx].detach().cpu().tolist())
            completed_costs.extend(episode_cost[done_idx].detach().cpu().tolist())
            completed_successes.extend(episode_success[done_idx].float().detach().cpu().tolist())
            episode_return[done_idx] = 0.0
            episode_env_return[done_idx] = 0.0
            episode_cost[done_idx] = 0.0
            episode_success[done_idx] = False
            teacher_state.reset(done)
            gate_state.reset(done)
            obstacle_cells = _terrain_obstacle_cells_by_env(env)
            adjusted_obs = _set_goal_through_obstacle(args, env, obstacle_cells, env_ids=done_idx)
            if adjusted_obs is not None:
                next_obs = scan_history.reset(_prepare_actor_obs(adjusted_obs, args), done_idx)
            clearance_obs, _ = _resample_close_goals(args, env, obstacle_cells, env_ids=done_idx)
            if clearance_obs is not None:
                next_obs = scan_history.reset(_prepare_actor_obs(clearance_obs, args), done_idx)
            _validate_required_blocked_corridors(args, env, obstacle_cells, done_idx)
            if _needs_feasible_reset(args):
                obs = scan_history.reset(_prepare_actor_obs(_reset_train_env(args, env), args))
                continue

        obs = next_obs

        learner_ready = buffer.size >= args.learning_starts
        if method == "hg_dagger":
            learner_ready = (
                expert_buffer is not None
                and expert_buffer.size >= args.batch_size
                and total_rows >= args.learning_starts
            )
        if learner_ready:
            for _ in range(args.updates_per_step):
                if method == "hg_dagger":
                    metrics_dict = hg_dagger_update(
                        hg_state,
                        expert_buffer,
                        batch_size=args.batch_size,
                        max_grad_norm=args.max_grad_norm,
                    )
                else:
                    batch = buffer.sample(args.batch_size)
                    if method == "hilserl" and human_buffer is not None and human_buffer.size > 0:
                        human_n = min(args.batch_size - 1, max(1, int(round(args.batch_size * args.hilserl_demo_ratio))))
                        batch = concat_replay_batches(
                            buffer.sample(args.batch_size - human_n),
                            human_buffer.sample(human_n),
                        )
                    elif method == "pvp" and human_buffer is not None and novice_buffer is not None:
                        half = max(1, args.batch_size // 2)
                        if human_buffer.size >= half and novice_buffer.size >= args.batch_size - half:
                            batch = concat_replay_batches(
                                novice_buffer.sample(args.batch_size - half),
                                human_buffer.sample(half),
                            )
                        elif human_buffer.size >= args.batch_size:
                            batch = human_buffer.sample(args.batch_size)
                        elif novice_buffer.size >= args.batch_size:
                            batch = novice_buffer.sample(args.batch_size)
                        else:
                            # Do not silently fall back to the shared replay;
                            # PVP's comparison contract is its two-buffer data path.
                            continue
                    if method == "pvp":
                        metrics_dict = pvp_update_step(
                            sac=sac,
                            state=pvp_state,
                            batch=batch,
                            gamma=args.gamma,
                            tau=args.tau,
                            max_grad_norm=args.max_grad_norm,
                            proxy_value_bound=args.pvp_proxy_value_bound,
                            cql_coefficient=args.pvp_cql_coefficient,
                            policy_delay=args.pvp_policy_delay,
                            target_policy_noise=args.pvp_target_policy_noise,
                            target_noise_clip=args.pvp_target_noise_clip,
                            include_env_reward=args.pvp_include_env_reward_in_td,
                            stop_td_on_intervention_start=args.pvp_stop_td_on_intervention_start,
                        )
                    else:
                        is_thesis = method == "thesis"
                        metrics = sac_update_step(
                            sac=sac,
                            batch=batch,
                            gamma=args.gamma,
                            tau=args.tau,
                            max_grad_norm=args.max_grad_norm,
                            pref_sampling_mode="linked" if is_thesis else "separate",
                            pref_rank_weight=args.pref_rank_weight if is_thesis else 0.0,
                            pref_rank_margin=args.pref_rank_margin,
                            pref_loss_type=args.pref_loss_type,
                            pref_stopgrad_positive=args.pref_stopgrad_positive,
                            pref_lambda_lr=args.pref_lambda_lr,
                            pref_lambda_max=args.pref_lambda_max,
                            pref_action_delta_min=args.pref_action_delta_min,
                            actor_bc_weight=args.actor_bc_weight if is_thesis else 0.0,
                            actor_bc_teacher_only=True,
                            actor_bc_only=args.actor_bc_only if is_thesis else False,
                            algo_variant="eil" if method == "eil" else "plain",
                            eil_threshold=args.eil_threshold,
                            eil_good_margin=args.eil_good_margin,
                            eil_bad_margin=args.eil_bad_margin,
                            eil_pair_margin=args.eil_pair_margin,
                            alpha_min=args.alpha_min,
                            alpha_max=args.alpha_max,
                            update_actor=(global_step % args.policy_frequency == 0),
                        )
                        metrics_dict = asdict(metrics)
                update_metrics.append(metrics_dict)
                if len(update_metrics) > 100:
                    update_metrics.pop(0)

        if global_step % args.log_interval == 0 or global_step == 1:
            recent_returns = completed_returns[-50:]
            recent_env_returns = completed_env_returns[-50:]
            recent_costs = completed_costs[-50:]
            recent_successes = completed_successes[-50:]
            latest_updates = update_metrics[-20:]
            avg_update = {
                k: sum(m.get(k, 0.0) for m in latest_updates) / max(1, len(latest_updates))
                for k in (latest_updates[0].keys() if latest_updates else [])
            }
            layout_stats = _layout_blocked_corridor_stats(args, env, obstacle_cells)
            blocked_fraction = sum(float(bool(item["blocked"])) for item in layout_stats) / max(1, len(layout_stats))
            blocked_cells_mean = sum(float(item["blocked_cell_count"]) for item in layout_stats) / max(1, len(layout_stats))
            blocking_components_mean = sum(
                float(item["blocking_component_count"]) for item in layout_stats
            ) / max(1, len(layout_stats))
            straight_path_length_mean = sum(float(item["path_length"]) for item in layout_stats) / max(1, len(layout_stats))
            row = {
                "method": method,
                "step": global_step,
                "transitions": global_step * num_envs,
                "fps": global_step * num_envs / max(1e-6, time.time() - start_time),
                "replay_size": buffer.size,
                "human_buffer_size": 0 if human_buffer is None else human_buffer.size,
                "novice_buffer_size": 0 if novice_buffer is None else novice_buffer.size,
                "expert_buffer_size": 0 if expert_buffer is None else expert_buffer.size,
                "updates_per_transition": float(args.updates_per_step) / float(num_envs),
                "episode_return_mean": float(sum(recent_returns) / max(1, len(recent_returns))) if recent_returns else 0.0,
                "episode_env_return_mean": float(sum(recent_env_returns) / max(1, len(recent_env_returns))) if recent_env_returns else 0.0,
                "episode_cost_mean": float(sum(recent_costs) / max(1, len(recent_costs))) if recent_costs else 0.0,
                "success_rate": float(sum(recent_successes) / max(1, len(recent_successes))) if recent_successes else 0.0,
                "ongoing_episode_cost_mean": float(episode_cost.mean().cpu().item()),
                "ongoing_episode_cost_max": float(episode_cost.max().cpu().item()),
                "ongoing_success_fraction": float(episode_success.float().mean().cpu().item()),
                "layout_blocked_component_count_mean": blocking_components_mean,
                "layout_straight_path_length_mean": straight_path_length_mean,
                "teacher_fraction_interval": interval_interventions / max(1, interval_rows),
                "teacher_fraction_cumulative": total_interventions / max(1, total_rows),
                "intervention_clearance_trigger_fraction": interval_clearance_triggers / max(1, interval_rows),
                "intervention_stall_trigger_fraction": interval_stall_triggers / max(1, interval_rows),
                "intervention_release_fraction": interval_releases / max(1, interval_rows),
                "intervention_clearance_engagement_rate": interval_clearance_engagements / max(1, interval_rows),
                "intervention_stall_engagement_rate": interval_stall_engagements / max(1, interval_rows),
                "intervention_release_blocked_clearance_fraction": interval_release_blocked_clearance / max(1, interval_rows),
                "intervention_release_blocked_progress_fraction": interval_release_blocked_progress / max(1, interval_rows),
                "intervention_release_blocked_action_delta_fraction": interval_release_blocked_action_delta / max(1, interval_rows),
                "intervention_released_duration_mean": interval_released_duration_sum / max(1, interval_released_duration_count),
                "teacher_executed_cost_sum_interval": interval_teacher_cost,
                "student_executed_cost_sum_interval": interval_student_cost,
                "teacher_executed_costful_step_rate": interval_teacher_costful_steps / max(1, interval_teacher_rows),
                "student_executed_costful_step_rate": interval_student_costful_steps / max(1, interval_student_rows),
                "active_gate_fraction": float(gate_state.active.float().mean().cpu().item()),
                "clearance_mean": float(clearance.mean().cpu().item()),
                "clearance_min": float(clearance.min().cpu().item()),
                "goal_distance_mean": float(goal_distance.mean().cpu().item()),
                "blocked_corridor_fraction": blocked_fraction,
                "blocked_corridor_cells_mean": blocked_cells_mean,
                "goal_blocked_fraction": goal_blocked_fraction,
                "action_delta_mean": float(delta.mean().detach().cpu().item()),
                "min_scan_mean": min_scan_mean,
                "reverse_action_mean": interval_reverse_action_sum / max(1, interval_rows),
                "lateral_action_abs_mean": interval_lateral_action_abs_sum / max(1, interval_rows),
                "goal_turn_alignment_mean": interval_goal_turn_alignment_sum / max(1, interval_rows),
            }
            for k, v in avg_update.items():
                if k in {
                    "critic_loss_total",
                    "critic_loss_replay",
                    "actor_loss",
                    "actor_loss_bc",
                    "alpha",
                    "target_q_mean",
                    "replay_reward_mean",
                    "replay_reward_abs_mean",
                    "batch_teacher_fraction",
                    "pref_linked_rows",
                    "pref_lambda",
                    "pref_lambda_delta",
                    "pref_q_delta",
                    "pref_dual_violation",
                    "pref_dual_signal",
                    "pref_violation",
                    "pref_violation_ema",
                    "q_min_data_mean",
                    "q_min_pi_mean",
                    "q_disagreement_data_mean",
                    "q_disagreement_pi_mean",
                    "pvp_proxy_teacher_loss",
                    "pvp_proxy_student_loss",
                    "pvp_intervened_batch_fraction",
                    "eil_good_loss",
                    "eil_bad_loss",
                    "eil_pair_loss",
                    "eil_good_batch_fraction",
                    "eil_bad_batch_fraction",
                    "hg_doubt_mean",
                    "hg_expert_buffer_size",
                }:
                    row[k] = float(v)
            print(json.dumps(row), flush=True)
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in row.items()}, step=global_step * num_envs)
            interval_interventions = 0
            interval_rows = 0
            interval_clearance_triggers = 0
            interval_stall_triggers = 0
            interval_releases = 0
            interval_clearance_engagements = 0
            interval_stall_engagements = 0
            interval_release_blocked_clearance = 0
            interval_release_blocked_progress = 0
            interval_release_blocked_action_delta = 0
            interval_released_duration_sum = 0
            interval_released_duration_count = 0
            interval_teacher_cost = 0.0
            interval_student_cost = 0.0
            interval_teacher_costful_steps = 0
            interval_student_costful_steps = 0
            interval_teacher_rows = 0
            interval_student_rows = 0
            interval_reverse_action_sum = 0.0
            interval_lateral_action_abs_sum = 0.0
            interval_goal_turn_alignment_sum = 0.0

        checkpoint_due = global_step % args.checkpoint_interval == 0
        eval_due = args.eval_interval > 0 and global_step % args.eval_interval == 0
        checkpoint_path = run_dir / f"step_{global_step}.pt"
        if checkpoint_due or eval_due:
            save_checkpoint(
                checkpoint_path,
                sac=sac,
                args=args,
                step=global_step,
                obs_dim=obs_dim,
                act_dim=act_dim,
                policy_actor=policy_actor,
            )
        if eval_due:
            eval_row = run_checkpoint_evaluation(
                checkpoint_path,
                args=args,
                step=global_step,
                run_dir=run_dir,
            )
            if eval_row is not None and wandb_run is not None:
                wandb_run.log(
                    {f"eval/{key}": value for key, value in eval_row.items() if key not in {"step", "transitions"}},
                    step=global_step * num_envs,
                )

    final_path = run_dir / "final.pt"
    save_checkpoint(
        final_path,
        sac=sac,
        args=args,
        step=args.total_steps,
        obs_dim=obs_dim,
        act_dim=act_dim,
        policy_actor=policy_actor,
    )
    if args.eval_interval > 0 and args.eval_at_end and args.total_steps % args.eval_interval != 0:
        eval_row = run_checkpoint_evaluation(
            final_path,
            args=args,
            step=args.total_steps,
            run_dir=run_dir,
        )
        if eval_row is not None and wandb_run is not None:
            wandb_run.log(
                {f"eval/{key}": value for key, value in eval_row.items() if key not in {"step", "transitions"}},
                step=args.total_steps * num_envs,
            )
    env.close()
    if args.eval_video:
        render_policy(final_path, args)
    if wandb_run is not None:
        wandb_run.finish()
    return final_path


def render_policy(checkpoint_path: Path, args: argparse.Namespace) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    eval_args = argparse.Namespace(**checkpoint.get("args", vars(args)))
    eval_args.device = args.device
    eval_args.num_envs = 1
    eval_args.video_dir = str(Path(args.video_dir or checkpoint_path.parent / "video").resolve())
    eval_args.video_length = args.video_length
    env = make_env(eval_args, render=True)
    obs_raw = _reset_train_env(eval_args, env)
    from unitree_nav_observation import UnitreeScanHistory

    scan_history = UnitreeScanHistory(
        getattr(eval_args, "scan_history", 1),
        getattr(eval_args, "action_history", 0),
        int(env.action_space.shape[-1]),
    )
    obs = scan_history.reset(_prepare_actor_obs(obs_raw, eval_args))
    from eval_unitree_nav_baselines import _load_policy_actor

    eval_args.model_path = str(checkpoint_path)
    actor = _load_policy_actor(eval_args, obs_dim=int(checkpoint["obs_dim"]), act_dim=int(checkpoint["act_dim"]))
    for _ in range(int(args.video_length) + 10):
        with torch.no_grad():
            _, _, mean = actor(obs)
        obs_raw, _, done, _ = env.step(mean.clamp(-1.0, 1.0))
        current_obs = _prepare_actor_obs(obs_raw, eval_args)
        done_mask = done.to(eval_args.device).reshape(-1).bool()
        obs = scan_history.step(current_obs, done_mask, action=mean.clamp(-1.0, 1.0))
        if bool(done.reshape(-1)[0].item()):
            obs_raw = _reset_train_env(eval_args, env)
            obs = scan_history.reset(_prepare_actor_obs(obs_raw, eval_args))
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Unitree-G1-Nav-Obstacles-Safe-Collision")
    parser.add_argument(
        "--method",
        choices=["thesis", "hilserl", "eil", "pvp", "hg_dagger", "sac"],
        default="thesis",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--episode-length-s", type=float, default=16.0)
    parser.add_argument("--resample-terrain-tiles", action="store_true")
    parser.add_argument("--low-level-policy-path", default=str(DEFAULT_LOW_LEVEL))
    parser.add_argument("--output-dir", default=str(ROOT / "models" / "unitree_mjlab_nav_thesis"))
    parser.add_argument("--run-name", default=f"unitree_nav_thesis_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "thesis-unitree-nav"))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP", ""))
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--replay-capacity", type=int, default=500000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--random-steps", type=int, default=500)
    parser.add_argument("--teacher-warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--policy-frequency", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--policy-encoder", choices=["mlp", "scan_cnn"], default="mlp")
    parser.add_argument("--height-scan-resolution", type=float, default=0.5)
    parser.add_argument("--height-scan-forward-size", type=float, default=0.0)
    parser.add_argument("--height-scan-lateral-size", type=float, default=0.0)
    parser.add_argument("--scan-history", type=int, default=1)
    parser.add_argument("--action-history", type=int, default=0)
    parser.add_argument("--student-action-smoothing", type=float, default=0.0)
    parser.add_argument("--pad-obs-to-dim", type=int, default=0)
    parser.add_argument("--mask-height-scan", action="store_true")
    parser.add_argument("--mask-proprioception", action="store_true")
    parser.add_argument("--mask-goal-heading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-layer-norm", action="store_true")
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--alpha-init", type=float, default=0.001)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=0.05)
    parser.add_argument("--pref-rank-weight", type=float, default=1.0)
    parser.add_argument("--pref-rank-margin", type=float, default=0.05)
    parser.add_argument("--pref-loss-type", default="lagrangian", choices=["margin", "softplus", "lagrangian"])
    parser.add_argument("--pref-stopgrad-positive", action="store_true")
    parser.add_argument("--pref-lambda-lr", type=float, default=0.01)
    parser.add_argument("--pref-lambda-max", type=float, default=10.0)
    parser.add_argument("--pref-action-delta-min", type=float, default=0.05)
    parser.add_argument("--actor-bc-weight", type=float, default=0.2)
    parser.add_argument("--actor-bc-only", action="store_true")
    parser.add_argument("--hilserl-demo-ratio", type=float, default=0.5)
    parser.add_argument("--eil-threshold", type=float, default=0.0)
    parser.add_argument("--eil-good-margin", type=float, default=0.01)
    parser.add_argument("--eil-bad-margin", type=float, default=0.01)
    parser.add_argument("--eil-pair-margin", type=float, default=0.05)
    parser.add_argument("--eil-bad-pre-steps", type=int, default=8)
    parser.add_argument("--pvp-proxy-value-bound", type=float, default=1.0)
    parser.add_argument("--pvp-cql-coefficient", type=float, default=1.0)
    parser.add_argument("--pvp-policy-delay", type=int, default=2)
    parser.add_argument("--pvp-target-policy-noise", type=float, default=0.2)
    parser.add_argument("--pvp-target-noise-clip", type=float, default=0.5)
    parser.add_argument("--pvp-include-env-reward-in-td", action="store_true")
    parser.add_argument(
        "--pvp-stop-td-on-intervention-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hg-ensemble-size", type=int, default=5)
    parser.add_argument("--init-actor-checkpoint", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--student-controller", choices=["actor", "direct_goal"], default="actor")
    parser.add_argument("--student-direct-goal-max-vx", type=float, default=0.95)
    parser.add_argument("--student-direct-goal-max-vy", type=float, default=0.45)
    parser.add_argument("--student-direct-goal-yaw-gain", type=float, default=1.2)
    parser.add_argument("--student-direct-goal-align-angle", type=float, default=0.55)
    parser.add_argument(
        "--learner-reward-mode",
        choices=["dense_progress", "dense_progress_exp", "env"],
        default="dense_progress",
    )
    parser.add_argument("--dense-progress-scale", type=float, default=1.0)
    parser.add_argument("--dense-progress-exp-scale", type=float, default=1.0)
    parser.add_argument("--dense-progress-exp-temperature", type=float, default=1.0)
    parser.add_argument("--goal-turn-alignment-scale", type=float, default=0.0)
    parser.add_argument("--reverse-action-penalty", type=float, default=0.0)
    parser.add_argument("--lateral-action-penalty", type=float, default=0.0)
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--failure-penalty", type=float, default=0.0)
    parser.add_argument("--teacher-type", choices=["scan", "geom_scan"], default="geom_scan")
    parser.add_argument(
        "--intervention-gate-mode",
        choices=["clearance_or_stall", "action_delta", "always", "none"],
        default="clearance_or_stall",
    )
    parser.add_argument("--intervention-clearance-threshold", type=float, default=0.65)
    parser.add_argument("--intervention-release-clearance", type=float, default=0.8)
    parser.add_argument("--intervention-clearance-mode", choices=["fixed", "teacher_ratio"], default="fixed")
    parser.add_argument("--intervention-clearance-trigger-ratio", type=float, default=2.0 / 3.0)
    parser.add_argument("--intervention-clearance-release-ratio", type=float, default=5.0 / 6.0)
    parser.add_argument("--intervention-stall-steps", type=int, default=30)
    parser.add_argument("--intervention-progress-epsilon", type=float, default=0.04)
    parser.add_argument("--intervention-release-steps", type=int, default=8)
    parser.add_argument("--intervention-release-progress-tolerance", type=float, default=0.005)
    parser.add_argument("--intervention-release-action-delta-max", type=float, default=0.35)
    parser.add_argument("--teacher-scan-block-threshold", type=float, default=0.12)
    parser.add_argument("--teacher-scan-block-delta", type=float, default=0.0)
    parser.add_argument("--teacher-scan-planner", choices=["heuristic", "astar"], default="heuristic")
    parser.add_argument("--teacher-scan-astar-clearance", type=float, default=0.6)
    parser.add_argument("--teacher-scan-astar-cell-padding", type=float, default=0.35)
    parser.add_argument("--teacher-scan-astar-resolution", type=float, default=0.25)
    parser.add_argument("--teacher-scan-astar-waypoint-index", type=int, default=3)
    parser.add_argument("--teacher-scan-astar-side-penalty", type=float, default=8.0)
    parser.add_argument("--teacher-scan-astar-commit-steps", type=int, default=30)
    parser.add_argument("--teacher-sector-half-width", type=float, default=0.35)
    parser.add_argument("--teacher-align-angle", type=float, default=0.55)
    parser.add_argument("--teacher-max-vx", type=float, default=0.95)
    parser.add_argument("--teacher-max-vy", type=float, default=0.45)
    parser.add_argument("--teacher-yaw-gain", type=float, default=1.2)
    parser.add_argument("--teacher-clearance-weight", type=float, default=0.0)
    parser.add_argument("--teacher-clearance-power", type=float, default=2.0)
    parser.add_argument("--teacher-speed-clearance-scale", type=float, default=0.0)
    parser.add_argument("--teacher-num-sectors", type=int, default=13)
    parser.add_argument("--teacher-min-forward-scale", type=float, default=0.12)
    parser.add_argument("--teacher-escape-risk-threshold", type=float, default=0.0)
    parser.add_argument("--teacher-escape-forward-scale", type=float, default=0.0)
    parser.add_argument("--teacher-escape-lateral-scale", type=float, default=1.0)
    parser.add_argument("--teacher-escape-radius", type=float, default=1.0)
    parser.add_argument("--teacher-escape-all-directions", action="store_true")
    parser.add_argument("--teacher-bypass-angle", type=float, default=0.0)
    parser.add_argument("--teacher-goal-stop-dist", type=float, default=0.0)
    parser.add_argument("--teacher-wall-follow-steps", type=int, default=0)
    parser.add_argument("--teacher-wall-follow-angle", type=float, default=0.9)
    parser.add_argument("--teacher-wall-follow-clear-risk", type=float, default=0.15)
    parser.add_argument("--teacher-rollout-horizon", type=float, default=0.0)
    parser.add_argument("--teacher-rollout-clearance", type=float, default=0.65)
    parser.add_argument("--teacher-rollout-samples", type=int, default=8)
    parser.add_argument("--teacher-rollout-clearance-weight", type=float, default=20.0)
    parser.add_argument("--teacher-rollout-forward-bias", type=float, default=0.05)
    parser.add_argument("--teacher-emergency-radius", type=float, default=0.0)
    parser.add_argument("--teacher-emergency-hard-radius", type=float, default=0.0)
    parser.add_argument("--teacher-emergency-speed-scale", type=float, default=0.75)
    parser.add_argument("--teacher-emergency-repulsion-weight", type=float, default=0.8)
    parser.add_argument("--teacher-emergency-tangent-weight", type=float, default=1.2)
    parser.add_argument("--teacher-emergency-goal-weight", type=float, default=0.4)
    parser.add_argument("--teacher-geom-planner", choices=["astar", "sector"], default="astar")
    parser.add_argument("--teacher-geom-scan-margin", type=float, default=0.15)
    parser.add_argument("--teacher-geom-back-margin", type=float, default=0.35)
    parser.add_argument("--teacher-geom-max-forward", type=float, default=1.75)
    parser.add_argument("--teacher-geom-max-lateral", type=float, default=1.75)
    parser.add_argument("--teacher-geom-lookahead", type=float, default=1.6)
    parser.add_argument("--teacher-geom-clearance", type=float, default=0.5)
    parser.add_argument("--teacher-geom-grid-resolution", type=float, default=0.15)
    parser.add_argument("--teacher-geom-waypoint-index", type=int, default=3)
    parser.add_argument("--teacher-geom-side-penalty", type=float, default=8.0)
    parser.add_argument("--teacher-geom-side-frame", choices=["body", "goal"], default="body")
    parser.add_argument("--teacher-geom-disengage-clear-steps", type=int, default=20)
    parser.add_argument("--teacher-geom-command-smoothing", type=float, default=0.0)
    parser.add_argument("--teacher-geom-memory-radius", type=float, default=0.0)
    parser.add_argument("--teacher-geom-waypoint-commit-distance", type=float, default=0.0)
    parser.add_argument("--teacher-geom-waypoint-reach-dist", type=float, default=0.25)
    parser.add_argument("--teacher-geom-stall-window", type=int, default=0)
    parser.add_argument("--teacher-geom-stall-progress-epsilon", type=float, default=0.08)
    parser.add_argument("--teacher-geom-stall-recovery-steps", type=int, default=60)
    parser.add_argument("--teacher-geom-stall-recovery-angle", type=float, default=0.9)
    parser.add_argument("--teacher-geom-stall-flip-side", action="store_true")
    parser.add_argument("--teacher-geom-obstacle-speed-radius", type=float, default=0.0)
    parser.add_argument("--teacher-geom-near-obstacle-vx-scale", type=float, default=0.7)
    parser.add_argument("--teacher-geom-max-angle", type=float, default=math.pi / 2.0)
    parser.add_argument("--teacher-geom-candidate-count", type=int, default=31)
    parser.add_argument("--teacher-geom-soft-width", type=float, default=0.9)
    parser.add_argument("--teacher-geom-hard-penalty", type=float, default=35.0)
    parser.add_argument("--teacher-geom-risk-weight", type=float, default=9.0)
    parser.add_argument("--teacher-geom-angle-weight", type=float, default=0.15)
    parser.add_argument("--teacher-geom-forward-bias", type=float, default=0.2)
    parser.add_argument("--teacher-geom-emergency-radius", type=float, default=0.8)
    parser.add_argument("--teacher-geom-emergency-repulsion-weight", type=float, default=1.2)
    parser.add_argument("--teacher-geom-emergency-tangent-weight", type=float, default=1.4)
    parser.add_argument("--teacher-geom-emergency-goal-weight", type=float, default=0.25)
    parser.add_argument("--intervention-delta", type=float, default=0.35)
    parser.add_argument("--intervene-on-blocked-goal", action="store_true")
    parser.add_argument("--deterministic-student", action="store_true")
    parser.add_argument("--success-dist", type=float, default=0.5)
    parser.add_argument("--terminate-on-goal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--goal-distance-min", type=float, default=0.0)
    parser.add_argument("--goal-distance-max", type=float, default=0.0)
    parser.add_argument("--min-goal-obstacle-clearance", type=float, default=0.0)
    parser.add_argument("--goal-clearance-resample-attempts", type=int, default=50)
    parser.add_argument("--min-start-obstacle-clearance", type=float, default=0.0)
    parser.add_argument("--start-clearance-resample-attempts", type=int, default=20)
    parser.add_argument("--require-blocked-corridor", action="store_true")
    parser.add_argument("--blocked-corridor-radius", type=float, default=0.45)
    parser.add_argument("--blocked-corridor-ignore-end-radius", type=float, default=0.75)
    parser.add_argument("--blocked-corridor-min-cells", type=int, default=1)
    parser.add_argument("--blocked-corridor-resample-attempts", type=int, default=100)
    parser.add_argument("--blocked-goal-max-distance", type=float, default=0.0)
    parser.add_argument(
        "--blocked-goal-distance-sampling",
        choices=["nearest", "uniform", "farthest"],
        default="nearest",
    )
    parser.add_argument("--blocked-goal-placement-mode", choices=["obstacle_multiplier", "distance_grid"], default="obstacle_multiplier")
    parser.add_argument("--blocked-goal-distance-multiplier-min", type=float, default=1.0)
    parser.add_argument("--blocked-goal-distance-multiplier-max", type=float, default=2.0)
    parser.add_argument("--blocked-goal-candidate-attempts", type=int, default=64)
    parser.add_argument("--debug-obstacle-width-min", type=float, default=1.0)
    parser.add_argument("--debug-obstacle-width-max", type=float, default=1.4)
    parser.add_argument("--strict-min-size-obstacles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-obstacle-height-min", type=float, default=1.0)
    parser.add_argument("--debug-obstacle-height-max", type=float, default=1.0)
    parser.add_argument("--debug-num-obstacles", type=int, default=6)
    parser.add_argument("--disable-obstacles", action="store_true")
    parser.add_argument("--debug-platform-width", type=float, default=2.0)
    parser.add_argument("--debug-obstacle-border-width", type=float, default=0.0)
    parser.add_argument("--debug-terrain-rows", type=int, default=0)
    parser.add_argument("--debug-terrain-cols", type=int, default=0)
    parser.add_argument("--debug-goal-through-obstacle", action="store_true")
    parser.add_argument("--goal-through-obstacle-prob", type=float, default=0.0)
    parser.add_argument("--debug-goal-distance", type=float, default=3.2)
    parser.add_argument("--debug-goal-obstacle-min-dist", type=float, default=0.8)
    parser.add_argument("--debug-goal-obstacle-max-dist", type=float, default=2.2)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-num-envs", type=int, default=8)
    parser.add_argument("--eval-num-episodes", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=941)
    parser.add_argument("--eval-layout-manifest", default="")
    parser.add_argument("--eval-timeout-s", type=float, default=3600.0)
    parser.add_argument("--eval-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-fail-fast", action="store_true")
    parser.add_argument("--eval-video", action="store_true")
    parser.add_argument("--video-dir", default="")
    parser.add_argument("--video-length", type=int, default=300)
    parser.add_argument("--render-checkpoint", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.render_checkpoint:
        render_policy(Path(args.render_checkpoint), args)
    else:
        run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Callable, Optional

from ogbench.wrappers import FlexibleObsWrapper, DetailedRewardWrapper, InterventionWrapper

_GOAL_COLOR_MAP = {
    "red": (0.85, 0.2, 0.2, 1.0),
    "green": (0.1, 0.8, 0.2, 1.0),
    "blue": (0.2, 0.5, 1.0, 1.0),
}


def _yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class CubeTeacherInfoAdapter(gym.Wrapper):
    """
    Ensure cube teacher oracles get target-block info in task-mode environments.

    In cube task mode, info often omits privileged target-block fields that cube oracles expect.
    This adapter reconstructs them and can select a dynamic target block sequentially.
    """

    def __init__(self, env: gym.Env, *, target_mode: str = "sequential", success_tolerance: float = 0.04):
        super().__init__(env)
        if target_mode not in {"fixed", "sequential"}:
            raise ValueError(f"Unknown target_mode={target_mode}")
        self.target_mode = target_mode
        self.success_tolerance = float(success_tolerance)

    def _cube_target_errors(self, out: dict, unwrapped) -> np.ndarray:
        num_cubes = int(getattr(unwrapped, "_num_cubes", 0))
        if num_cubes <= 0:
            return np.zeros(0, dtype=np.float32)
        errs: list[float] = []
        for i in range(num_cubes):
            try:
                obj = np.asarray(out[f"privileged/block_{i}_pos"], dtype=np.float32)
            except Exception:
                try:
                    obj = np.asarray(unwrapped._data.joint(f"object_joint_{i}").qpos[:3], dtype=np.float32)
                except Exception:
                    obj = np.zeros(3, dtype=np.float32)
            try:
                mocap_id = int(unwrapped._cube_target_mocap_ids[i])
                tar = np.asarray(unwrapped._data.mocap_pos[mocap_id], dtype=np.float32)
            except Exception:
                tar = obj
            errs.append(float(np.linalg.norm(obj - tar)))
        return np.asarray(errs, dtype=np.float32)

    def _select_target_block(self, unwrapped, errs: np.ndarray) -> int:
        base_target = int(getattr(unwrapped, "_target_block", 0))
        if self.target_mode != "sequential" or errs.size == 0:
            return base_target
        unresolved = np.where(errs > self.success_tolerance)[0]
        if unresolved.size == 0:
            return base_target
        return int(unresolved[0])

    def _augment_info(self, info):
        if not isinstance(info, dict):
            return info
        out = dict(info)
        unwrapped = self.unwrapped
        errs = self._cube_target_errors(out, unwrapped)
        out["diag/cube_target_errors"] = errs
        out["diag/cubes_solved"] = int(np.sum(errs <= self.success_tolerance)) if errs.size else 0
        out["diag/cube_max_target_error"] = float(np.max(errs)) if errs.size else 0.0

        target_idx = self._select_target_block(unwrapped, errs)
        out["privileged/target_block"] = int(target_idx)
        out["diag/target_block_dynamic"] = int(target_idx)

        try:
            mocap_pos = np.asarray(unwrapped._data.mocap_pos, dtype=np.float32)
            if target_idx < mocap_pos.shape[0]:
                out["privileged/target_block_pos"] = mocap_pos[target_idx].copy()
        except Exception:
            pass

        try:
            mocap_quat = np.asarray(unwrapped._data.mocap_quat, dtype=np.float32)
            if target_idx < mocap_quat.shape[0]:
                yaw = _yaw_from_quat_wxyz(mocap_quat[target_idx])
                out["privileged/target_block_yaw"] = np.asarray([yaw], dtype=np.float32)
        except Exception:
            pass
        return out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._augment_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._augment_info(info)


def build_ogbench_wrapper(
    *,
    obs_mode: str,
    include_goal: bool = True,
    include_distance: bool = False,
    include_direction: bool = False,
    include_velocity: bool = False,
    reward_type: str,
    dense_reward_scale: float,
    step_penalty: float,
    reward_switch_after_steps: int = 0,
    intervention_mode: str = "none",
    teacher_type: str = "bfs",
    tolerance_type: str = "angle",
    tolerance_value: float = 30.0,
    hard_block_lethal: bool = True,
    intervention_enable_after_steps: int = 0,
    intervention_agent_mode: str = "divergence",
    intervention_safety_margin_frac: float = 0.0,
    intervention_release_steps: int = 3,
    intervention_episode_prob: float = 1.0,
    intervention_episode_prob_min: float = 0.0,
    intervention_episode_prob_decay_steps: int = 0,
    intervention_episode_prob_decay_start: int = 0,
    intervention_episode_prob_seed: Optional[int] = None,
    teacher_target_mode: str = "sequential",
    cube_success_tolerance: float = 0.04,
    teleop_interface: Optional[object] = None,
) -> Callable:
    """Return a wrapper function that mirrors training/eval environment stacking."""

    def _apply(env):
        if obs_mode == "state":
            env = FlexibleObsWrapper(
                env,
                include_goal=include_goal,
                include_distance=include_distance,
                include_direction=include_direction,
                include_velocity=include_velocity,
            )
        env = DetailedRewardWrapper(
            env,
            reward_type=reward_type,
            dense_reward_scale=dense_reward_scale,
            step_penalty=step_penalty,
            switch_reward_to_sparse_after_steps_per_env=reward_switch_after_steps,
        )
        if intervention_mode in {"agent", "agent_safety_align", "agent_safety_progress"} and teacher_type in {
            "cube_plan",
            "cube_markov",
        }:
            env = CubeTeacherInfoAdapter(
                env,
                target_mode=teacher_target_mode,
                success_tolerance=cube_success_tolerance,
            )
        if intervention_mode == "human":
            env = InterventionWrapper(
                env,
                teleop_interface=teleop_interface,
                mode="human",
                threshold=0.1,
                hold_time=0.5,
            )
        elif intervention_mode == "agent":
            env = InterventionWrapper(
                env,
                mode="agent",
                teacher_type=teacher_type,
                tolerance_type=tolerance_type,
                tolerance_value=tolerance_value,
                hard_block_lethal=hard_block_lethal,
                enable_after_steps=intervention_enable_after_steps,
                agent_mode=intervention_agent_mode,
                safety_margin_frac=intervention_safety_margin_frac,
                release_steps=intervention_release_steps,
                episode_intervention_prob=intervention_episode_prob,
                episode_intervention_prob_min=intervention_episode_prob_min,
                episode_intervention_prob_decay_steps=intervention_episode_prob_decay_steps,
                episode_intervention_prob_decay_start=intervention_episode_prob_decay_start,
                episode_intervention_seed=intervention_episode_prob_seed,
            )
        elif intervention_mode in ("agent_safety_align", "agent_safety_progress"):
            agent_mode = "safety_align" if intervention_mode == "agent_safety_align" else "safety_progress"
            env = InterventionWrapper(
                env,
                mode="agent",
                teacher_type=teacher_type,
                tolerance_type=tolerance_type,
                tolerance_value=tolerance_value,
                hard_block_lethal=hard_block_lethal,
                enable_after_steps=intervention_enable_after_steps,
                agent_mode=agent_mode,
                safety_margin_frac=intervention_safety_margin_frac,
                release_steps=intervention_release_steps,
                episode_intervention_prob=intervention_episode_prob,
                episode_intervention_prob_min=intervention_episode_prob_min,
                episode_intervention_prob_decay_steps=intervention_episode_prob_decay_steps,
                episode_intervention_prob_decay_start=intervention_episode_prob_decay_start,
                episode_intervention_seed=intervention_episode_prob_seed,
            )
        return env

    return _apply


def maybe_set_goal_color(env, color_name: str) -> bool:
    if not color_name:
        return False
    normalized = color_name.strip().lower()
    if normalized in ("", "auto", "default"):
        return False
    rgba = _GOAL_COLOR_MAP.get(normalized)
    if rgba is None:
        return False

    base = env
    visited = set()
    while hasattr(base, "unwrapped") and getattr(base, "unwrapped") is not base and getattr(base, "unwrapped") not in visited:
        visited.add(base)
        base = base.unwrapped
    model = getattr(base, "model", None)
    if model is None:
        return False
    geom_name = getattr(base, "_goal_geom_name", "target")
    try:
        geom = model.geom(geom_name)
    except Exception:
        return False
    try:
        geom.rgba[:] = np.array(rgba, dtype=np.float32)
    except Exception:
        try:
            geom.rgba = np.array(rgba, dtype=np.float32)
        except Exception:
            return False
    return True

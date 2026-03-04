from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env_wrappers_common import FixedResetSeedWrapper, maybe_wrap_intervention

_GOAL_COLOR_MAP = {
    "red": (0.85, 0.2, 0.2, 1.0),
    "green": (0.1, 0.8, 0.2, 1.0),
    "blue": (0.2, 0.5, 1.0, 1.0),
}

_WRAPPER_STACK_PRINTED: set[tuple] = set()


class MazeFlexibleObsWrapper(gym.ObservationWrapper):
    """Maze-only configurable state observation wrapper."""

    def __init__(
        self,
        env: gym.Env,
        *,
        include_goal: bool = True,
        include_distance: bool = False,
        include_direction: bool = False,
        include_velocity: bool = False,
    ):
        super().__init__(env)
        self.include_goal = bool(include_goal)
        self.include_distance = bool(include_distance)
        self.include_direction = bool(include_direction)
        self.include_velocity = bool(include_velocity)
        self._last_goal = None
        self._last_velocity = None

        obs_dim = 2
        if self.include_goal:
            obs_dim += 2
        if self.include_distance:
            obs_dim += 1
        if self.include_direction:
            obs_dim += 2
        if self.include_velocity:
            obs_dim += 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        agent_pos = obs[:2] if obs.size >= 2 else np.zeros((2,), dtype=np.float32)
        components = [agent_pos]

        if self.include_goal:
            if self._last_goal is not None:
                goal_pos = np.asarray(self._last_goal, dtype=np.float32).reshape(-1)[:2]
                if goal_pos.size < 2:
                    goal_pos = np.pad(goal_pos, (0, 2 - goal_pos.size), mode="constant")
            else:
                goal_pos = np.zeros((2,), dtype=np.float32)
            components.append(goal_pos.astype(np.float32, copy=False))

        if self.include_distance or self.include_direction:
            if self._last_goal is not None:
                goal_pos = np.asarray(self._last_goal, dtype=np.float32).reshape(-1)[:2]
                if goal_pos.size < 2:
                    goal_pos = np.pad(goal_pos, (0, 2 - goal_pos.size), mode="constant")
                delta = goal_pos - agent_pos
                dist = float(np.linalg.norm(delta))
                if self.include_distance:
                    components.append(np.asarray([dist], dtype=np.float32))
                if self.include_direction:
                    if dist > 1e-8:
                        components.append((delta / dist).astype(np.float32, copy=False))
                    else:
                        components.append(np.zeros((2,), dtype=np.float32))
            else:
                if self.include_distance:
                    components.append(np.asarray([0.0], dtype=np.float32))
                if self.include_direction:
                    components.append(np.zeros((2,), dtype=np.float32))

        if self.include_velocity:
            if self._last_velocity is not None:
                vel = np.asarray(self._last_velocity, dtype=np.float32).reshape(-1)[:2]
                if vel.size < 2:
                    vel = np.pad(vel, (0, 2 - vel.size), mode="constant")
            else:
                vel = np.zeros((2,), dtype=np.float32)
            components.append(vel.astype(np.float32, copy=False))
        return np.concatenate(components, axis=0).astype(np.float32, copy=False)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_goal = info.get("goal") if isinstance(info, dict) else None
        self._last_velocity = np.zeros((2,), dtype=np.float32)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(info, dict):
            if "goal" in info:
                self._last_goal = info["goal"]
            if "qvel" in info:
                self._last_velocity = np.asarray(info["qvel"], dtype=np.float32).reshape(-1)[:2]
            elif "velocity" in info:
                self._last_velocity = np.asarray(info["velocity"], dtype=np.float32).reshape(-1)[:2]
        return self.observation(obs), reward, terminated, truncated, info


class MazeDetailedRewardWrapper(gym.Wrapper):
    """Maze-only reward wrapper with sparse/dense/combined/none modes."""

    def __init__(
        self,
        env: gym.Env,
        *,
        reward_type: str = "sparse",
        dense_reward_scale: float = 0.1,
        goal_reward: float = 1.0,
        step_penalty: float = 0.0,
        switch_reward_to_sparse_after_steps_per_env: int = 0,
    ):
        super().__init__(env)
        self.reward_type = str(reward_type)
        self.dense_reward_scale = float(dense_reward_scale)
        self.goal_reward = float(goal_reward)
        self.step_penalty = float(step_penalty)
        self.switch_reward_to_sparse_after_steps_per_env = int(switch_reward_to_sparse_after_steps_per_env or 0)
        self._global_step_env = 0
        self._prev_distance = None
        self._goal_reached = False
        self._episode_steps = 0
        self._episode_sparse_reward = 0.0
        self._episode_dense_reward = 0.0
        self._last_sparse_reward_step = 0.0
        self._last_dense_reward_step = 0.0

    def _effective_reward_type(self) -> str:
        if self.reward_type == "none":
            return "none"
        if (
            self.switch_reward_to_sparse_after_steps_per_env > 0
            and self._global_step_env >= self.switch_reward_to_sparse_after_steps_per_env
        ):
            return "sparse"
        return self.reward_type

    def _extract_agent_goal(self, obs, info):
        agent_pos = None
        goal_pos = None
        try:
            if hasattr(self.unwrapped, "get_xy"):
                agent_pos = np.asarray(self.unwrapped.get_xy(), dtype=np.float32).reshape(-1)[:2]
        except Exception:
            agent_pos = None
        try:
            if hasattr(self.unwrapped, "cur_goal_xy"):
                goal_pos = np.asarray(self.unwrapped.cur_goal_xy, dtype=np.float32).reshape(-1)[:2]
        except Exception:
            goal_pos = None

        if agent_pos is None:
            obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
            if obs_vec.size >= 2:
                agent_pos = obs_vec[:2]
        if goal_pos is None:
            if isinstance(info, dict) and "goal" in info:
                goal_pos = np.asarray(info["goal"], dtype=np.float32).reshape(-1)[:2]
            else:
                obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
                if obs_vec.size >= 4:
                    goal_pos = obs_vec[2:4]
        return agent_pos, goal_pos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._goal_reached = False
        self._episode_steps = 0
        self._episode_sparse_reward = 0.0
        self._episode_dense_reward = 0.0
        self._last_sparse_reward_step = 0.0
        self._last_dense_reward_step = 0.0
        agent_pos, goal_pos = self._extract_agent_goal(obs, info)
        if agent_pos is not None and goal_pos is not None:
            self._prev_distance = float(np.linalg.norm(goal_pos - agent_pos))
        else:
            self._prev_distance = None
        if isinstance(info, dict):
            info.update(
                {
                    "episode_sparse_reward": 0.0,
                    "episode_dense_reward": 0.0,
                    "episode_steps": 0,
                    "goal_reached": False,
                    "distance_to_goal": self._prev_distance if self._prev_distance is not None else 0.0,
                    "reward_type": self._effective_reward_type(),
                }
            )
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info) if isinstance(info, dict) else {}

        sparse_reward_raw = 0.0
        success = bool(info.get("success", False)) or bool(info.get("goal_reached", False))
        if success or float(env_reward) > 0.0:
            sparse_reward_raw = self.goal_reward
            self._goal_reached = True

        agent_pos, goal_pos = self._extract_agent_goal(obs, info)
        current_distance = None
        if agent_pos is not None and goal_pos is not None:
            current_distance = float(np.linalg.norm(goal_pos - agent_pos))

        dense_reward_raw = 0.0
        if self._prev_distance is not None and current_distance is not None:
            dense_reward_raw = (self._prev_distance - current_distance) * self.dense_reward_scale
        sparse_reward = float(sparse_reward_raw)
        dense_reward = float(dense_reward_raw)

        effective_type = self._effective_reward_type()
        if effective_type == "sparse":
            dense_reward = 0.0
        elif effective_type == "dense":
            sparse_reward = 0.0
        elif effective_type == "none":
            sparse_reward = 0.0
            dense_reward = 0.0
        elif effective_type != "combined":
            raise ValueError(f"Unknown reward_type: {effective_type}")

        total_reward_raw = float(sparse_reward_raw + dense_reward_raw - self.step_penalty)
        total_reward = float(sparse_reward + dense_reward - self.step_penalty)
        self._last_sparse_reward_step = float(sparse_reward)
        self._last_dense_reward_step = float(dense_reward)
        self._episode_sparse_reward += float(sparse_reward)
        self._episode_dense_reward += float(dense_reward)
        self._episode_steps += 1
        self._global_step_env += 1
        self._prev_distance = current_distance

        info.update(
            {
                "sparse_reward": float(self._last_sparse_reward_step),
                "dense_reward": float(self._last_dense_reward_step),
                "total_reward": float(total_reward),
                "sparse_reward_raw": float(sparse_reward_raw),
                "dense_reward_raw": float(dense_reward_raw),
                "total_reward_raw": float(total_reward_raw),
                "episode_sparse_reward": float(self._episode_sparse_reward),
                "episode_dense_reward": float(self._episode_dense_reward),
                "episode_steps": int(self._episode_steps),
                "goal_reached": bool(self._goal_reached),
                "distance_to_goal": float(current_distance) if current_distance is not None else 0.0,
                "reward_type": effective_type,
            }
        )
        return obs, total_reward, terminated, truncated, info


def build_ogbench_maze_wrapper(
    *,
    env_name: Optional[str] = None,
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
    tolerance_channel_weights: Optional[str] = None,
    binary_gripper_actions: bool = False,
    binary_gripper_threshold: float = 0.0,
    hard_gripper_intervention: bool = False,
    gripper_intervene_pick_radius: float = 0.06,
    gripper_intervene_place_radius: float = 0.06,
    gripper_intervene_contact_threshold: float = 0.3,
    hard_block_lethal: bool = True,
    intervention_enable_after_steps: int = 0,
    intervention_agent_mode: str = "divergence",
    intervention_safety_margin_frac: float = 0.0,
    intervention_release_steps: int = 3,
    intervention_reward_patience_steps: int = 5,
    intervention_reward_improvement_epsilon: float = 1e-6,
    intervention_episode_prob: float = 1.0,
    intervention_episode_prob_min: float = 0.0,
    intervention_episode_prob_decay_steps: int = 0,
    intervention_episode_prob_decay_start: int = 0,
    intervention_episode_prob_seed: Optional[int] = None,
    teacher_target_mode: str = "sequential",
    cube_success_tolerance: float = 0.04,
    static_reset_seed: Optional[int] = None,
    teleop_interface: Optional[object] = None,
) -> Callable:
    """Build a maze-only wrapper stack."""

    del teacher_target_mode, cube_success_tolerance

    def _apply(env: gym.Env):
        wrapper_chain: list[str] = []
        if static_reset_seed is not None:
            env = FixedResetSeedWrapper(env, reset_seed=int(static_reset_seed))
            wrapper_chain.append(f"FixedResetSeed({int(static_reset_seed)})")

        if obs_mode == "state":
            env = MazeFlexibleObsWrapper(
                env,
                include_goal=include_goal,
                include_distance=include_distance,
                include_direction=include_direction,
                include_velocity=include_velocity,
            )
            wrapper_chain.append("MazeFlexibleObsWrapper")

        env = MazeDetailedRewardWrapper(
            env,
            reward_type=reward_type,
            dense_reward_scale=dense_reward_scale,
            step_penalty=step_penalty,
            switch_reward_to_sparse_after_steps_per_env=reward_switch_after_steps,
        )
        wrapper_chain.append("MazeDetailedRewardWrapper")

        env, intervention_name = maybe_wrap_intervention(
            env,
            intervention_mode=intervention_mode,
            teacher_type=teacher_type,
            tolerance_type=tolerance_type,
            tolerance_value=tolerance_value,
            tolerance_channel_weights=tolerance_channel_weights,
            binary_gripper_actions=binary_gripper_actions,
            binary_gripper_threshold=binary_gripper_threshold,
            hard_gripper_intervention=hard_gripper_intervention,
            gripper_intervene_pick_radius=gripper_intervene_pick_radius,
            gripper_intervene_place_radius=gripper_intervene_place_radius,
            gripper_intervene_contact_threshold=gripper_intervene_contact_threshold,
            hard_block_lethal=hard_block_lethal,
            intervention_enable_after_steps=intervention_enable_after_steps,
            intervention_agent_mode=intervention_agent_mode,
            intervention_safety_margin_frac=intervention_safety_margin_frac,
            intervention_release_steps=intervention_release_steps,
            intervention_reward_patience_steps=intervention_reward_patience_steps,
            intervention_reward_improvement_epsilon=intervention_reward_improvement_epsilon,
            intervention_episode_prob=intervention_episode_prob,
            intervention_episode_prob_min=intervention_episode_prob_min,
            intervention_episode_prob_decay_steps=intervention_episode_prob_decay_steps,
            intervention_episode_prob_decay_start=intervention_episode_prob_decay_start,
            intervention_episode_prob_seed=intervention_episode_prob_seed,
            teleop_interface=teleop_interface,
        )
        if intervention_name is not None:
            wrapper_chain.append(intervention_name)

        env_id = env_name or str(getattr(getattr(env, "spec", None), "id", "") or "")
        stack_key = (env_id, obs_mode, intervention_mode, teacher_type, reward_type, tuple(wrapper_chain))
        if stack_key not in _WRAPPER_STACK_PRINTED:
            _WRAPPER_STACK_PRINTED.add(stack_key)
            print(f"[WrapperStack] env={env_id or '<unknown>'} chain=" + " -> ".join(wrapper_chain))
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

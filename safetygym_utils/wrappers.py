from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np

from .env import clip_action_to_space, extract_goal_distance


@dataclass
class InterventionEpisodeStats:
    episode_steps: int = 0
    intervention_steps: int = 0
    num_interventions: int = 0
    burst_count: int = 0
    burst_steps_sum: int = 0
    _in_burst: bool = False
    _cur_burst_steps: int = 0

    def on_step(self, intervened: bool) -> None:
        self.episode_steps += 1
        if intervened:
            self.intervention_steps += 1
            if not self._in_burst:
                self._in_burst = True
                self._cur_burst_steps = 0
                self.burst_count += 1
                self.num_interventions += 1
            self._cur_burst_steps += 1
        else:
            self._close_burst()

    def _close_burst(self) -> None:
        if self._in_burst:
            self._in_burst = False
            self.burst_steps_sum += self._cur_burst_steps
            self._cur_burst_steps = 0

    def finalize(self) -> dict:
        self._close_burst()
        frac = float(self.intervention_steps) / max(1, int(self.episode_steps))
        avg_burst = float(self.burst_steps_sum) / max(1, int(self.burst_count))
        return {
            "teacher_episode_steps": int(self.episode_steps),
            "teacher_intervention_steps": int(self.intervention_steps),
            "teacher_fraction_steps": float(frac),
            "teacher_num_interventions": int(self.num_interventions),
            "teacher_num_bursts": int(self.burst_count),
            "teacher_avg_burst_len": float(avg_burst),
        }


class HumanInterventionWrapper(gym.Wrapper):
    """Overlay human teleop intervention on top of student policy actions."""

    def __init__(
        self,
        env: gym.Env,
        *,
        controller,
        threshold: float = 0.1,
        hold_seconds: float = 0.25,
    ):
        super().__init__(env)
        self.controller = controller
        self.threshold = float(threshold)
        self.hold_seconds = float(hold_seconds)

        self._last_override_ts = -1e9
        self._last_override_action: Optional[np.ndarray] = None
        self._stats = InterventionEpisodeStats()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_override_ts = -1e9
        self._last_override_action = None
        self._stats = InterventionEpisodeStats()
        return obs, info

    def _current_teacher_action(self) -> Optional[np.ndarray]:
        human = np.asarray(self.controller.get_action(), dtype=np.float32)
        now = time.perf_counter()
        if float(np.linalg.norm(human)) > self.threshold:
            self._last_override_ts = now
            self._last_override_action = human
        active = (now - self._last_override_ts) <= self.hold_seconds
        if active and self._last_override_action is not None:
            return np.asarray(self._last_override_action, dtype=np.float32)
        return None

    def step(self, student_action):
        student_action = np.asarray(student_action, dtype=np.float32)
        t_ctrl0 = time.perf_counter()
        teacher_action = self._current_teacher_action()
        controller_ms = (time.perf_counter() - t_ctrl0) * 1e3
        if teacher_action is not None:
            applied_action = clip_action_to_space(teacher_action, self.action_space)
            intervened = True
        else:
            applied_action = clip_action_to_space(student_action, self.action_space)
            intervened = False

        obs, reward, cost, terminated, truncated, info = self.env.step(applied_action)

        self._stats.on_step(intervened)
        info = dict(info)
        info["teacher_intervened"] = bool(intervened)
        info["teacher_reason"] = "human" if intervened else None
        info["teacher_action"] = np.asarray(teacher_action if teacher_action is not None else applied_action, dtype=np.float32)
        info["student_action"] = np.asarray(student_action, dtype=np.float32)
        info["teacher_delta_l2"] = float(np.linalg.norm(info["teacher_action"] - info["student_action"]))
        info["teacher_controller_ms"] = float(controller_ms)

        if terminated or truncated:
            info.update(self._stats.finalize())

        return obs, reward, cost, terminated, truncated, info


class RewardModeWrapper(gym.Wrapper):
    """Apply sparse/dense/none reward modes for Safety-Gymnasium."""

    def __init__(
        self,
        env: gym.Env,
        *,
        reward_mode: str,
        dense_reward_scale: float = 1.0,
        step_penalty: float = 0.0,
    ):
        super().__init__(env)
        mode = str(reward_mode).lower()
        if mode not in {"sparse", "dense", "none"}:
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")
        self.reward_mode = mode
        self.dense_reward_scale = float(dense_reward_scale)
        self.step_penalty = float(step_penalty)
        self._prev_goal_distance = float("nan")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_goal_distance = extract_goal_distance(self.env)
        return obs, info

    def step(self, action):
        obs, env_reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        goal_met = bool(info.get("goal_met", False))
        cur_dist = extract_goal_distance(self.env)
        rew = 0.0

        if self.reward_mode == "sparse":
            rew = 1.0 if goal_met else 0.0
        elif self.reward_mode == "dense":
            prev = self._prev_goal_distance
            if np.isfinite(prev) and np.isfinite(cur_dist):
                rew = self.dense_reward_scale * float(prev - cur_dist)
            else:
                rew = 0.0
        elif self.reward_mode == "none":
            rew = 0.0

        rew += self.step_penalty

        info["reward_mode"] = self.reward_mode
        info["reward_raw_env"] = float(env_reward)
        info["reward_shaped"] = float(rew)
        if np.isfinite(cur_dist):
            info["goal_distance"] = float(cur_dist)

        self._prev_goal_distance = cur_dist
        return obs, float(rew), cost, terminated, truncated, info

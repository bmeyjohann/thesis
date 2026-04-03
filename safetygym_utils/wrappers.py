from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np

from .env import (
    clip_action_to_space,
    extract_agent_forward_xy,
    extract_agent_velocity_xy,
    extract_agent_xy,
    extract_goal_distance,
    extract_goal_xy,
    extract_min_constrained_clearance,
)


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
        clearance_override_threshold: float = -1.0,
    ):
        super().__init__(env)
        self.controller = controller
        self.threshold = float(threshold)
        self.hold_seconds = float(hold_seconds)
        self.clearance_override_threshold = float(clearance_override_threshold)

        self._last_override_ts = -1e9
        self._last_override_action: Optional[np.ndarray] = None
        self._stats = InterventionEpisodeStats()
        self._last_obs: Optional[np.ndarray] = None
        self._last_teacher_reason: Optional[str] = None
        self._last_trigger_clearance: Optional[float] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_override_ts = -1e9
        self._last_override_action = None
        self._stats = InterventionEpisodeStats()
        self._last_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._last_teacher_reason = None
        self._last_trigger_clearance = None
        return obs, info

    def _clearance_override_active(self) -> tuple[bool, Optional[float]]:
        if self.clearance_override_threshold < 0.0:
            return True, None
        clearance = extract_min_constrained_clearance(self.env)
        if clearance is None:
            return False, None
        return bool(clearance <= self.clearance_override_threshold), float(clearance)

    def _current_teacher_action(self) -> Optional[np.ndarray]:
        if getattr(self.controller, "always_active", False):
            active, clearance = self._clearance_override_active()
            self._last_trigger_clearance = clearance
            if not active:
                self._last_teacher_reason = None
                return None
            if self._last_obs is None:
                self._last_teacher_reason = None
                return None
            try:
                action = self.controller.get_action(obs=self._last_obs, env=self.env)
            except TypeError:
                action = self.controller.get_action(obs=self._last_obs)
            if action is None:
                self._last_teacher_reason = None
                return None
            self._last_teacher_reason = (
                "expert_clearance_override" if clearance is not None and self.clearance_override_threshold >= 0.0 else "expert"
            )
            return np.asarray(action, dtype=np.float32)

        try:
            human_raw = self.controller.get_action(obs=self._last_obs)
        except TypeError:
            human_raw = self.controller.get_action()
        human = np.asarray(human_raw, dtype=np.float32)
        now = time.perf_counter()
        if float(np.linalg.norm(human)) > self.threshold:
            self._last_override_ts = now
            self._last_override_action = human
        active = (now - self._last_override_ts) <= self.hold_seconds
        if active and self._last_override_action is not None:
            self._last_teacher_reason = "human"
            self._last_trigger_clearance = None
            return np.asarray(self._last_override_action, dtype=np.float32)
        self._last_teacher_reason = None
        self._last_trigger_clearance = None
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
        self._last_obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        self._stats.on_step(intervened)
        info = dict(info)
        info["teacher_intervened"] = bool(intervened)
        info["teacher_reason"] = self._last_teacher_reason if intervened else None
        info["teacher_trigger_clearance"] = (
            float(self._last_trigger_clearance) if intervened and self._last_trigger_clearance is not None else None
        )
        info["teacher_action"] = np.asarray(teacher_action if teacher_action is not None else applied_action, dtype=np.float32)
        info["student_action"] = np.asarray(student_action, dtype=np.float32)
        info["teacher_delta_l2"] = float(np.linalg.norm(info["teacher_action"] - info["student_action"]))
        info["teacher_controller_ms"] = float(controller_ms)

        if terminated or truncated:
            info.update(self._stats.finalize())

        return obs, reward, cost, terminated, truncated, info


class RewardModeWrapper(gym.Wrapper):
    """Apply sparse/dense reward modes for Safety-Gymnasium."""

    def __init__(
        self,
        env: gym.Env,
        *,
        reward_mode: str,
        dense_reward_scale: float = 1.0,
        step_penalty: float = 0.0,
        cost_penalty: float = 0.0,
        cost_penalty_warmup_steps: int = 0,
        cost_penalty_ramp_steps: int = 0,
        clearance_penalty_scale: float = 0.0,
        clearance_margin: float = 0.0,
        clearance_penalty_power: float = 1.0,
        clearance_penalty_warmup_steps: int = 0,
        clearance_penalty_ramp_steps: int = 0,
        forward_reward_scale: float = 0.0,
        backward_penalty_scale: float = 0.0,
        heading_reward_scale: float = 0.0,
        heading_positive_only: bool = True,
    ):
        super().__init__(env)
        mode = str(reward_mode).lower()
        if mode == "dual":
            mode = "dense_plus_sparse"
        if mode not in {"sparse", "dense", "dense_plus_sparse", "native", "none"}:
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")
        self.reward_mode = mode
        self.dense_reward_scale = float(dense_reward_scale)
        self.step_penalty = float(step_penalty)
        self.cost_penalty = float(cost_penalty)
        self.cost_penalty_warmup_steps = int(max(0, cost_penalty_warmup_steps))
        self.cost_penalty_ramp_steps = int(max(0, cost_penalty_ramp_steps))
        self.clearance_penalty_scale = float(clearance_penalty_scale)
        self.clearance_margin = float(max(0.0, clearance_margin))
        self.clearance_penalty_power = float(max(1.0, clearance_penalty_power))
        self.clearance_penalty_warmup_steps = int(max(0, clearance_penalty_warmup_steps))
        self.clearance_penalty_ramp_steps = int(max(0, clearance_penalty_ramp_steps))
        self.forward_reward_scale = float(forward_reward_scale)
        self.backward_penalty_scale = float(backward_penalty_scale)
        self.heading_reward_scale = float(heading_reward_scale)
        self.heading_positive_only = bool(heading_positive_only)
        self._prev_goal_distance = float("nan")
        self._total_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_goal_distance = extract_goal_distance(self.env)
        return obs, info

    def set_total_steps(self, total_steps: int) -> None:
        self._total_steps = int(max(0, total_steps))

    def _curriculum_scale(self, warmup_steps: int, ramp_steps: int) -> float:
        if warmup_steps <= 0 and ramp_steps <= 0:
            return 1.0
        steps = int(max(0, self._total_steps))
        if steps <= warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return 1.0
        return float(
            min(
                1.0,
                max(0.0, (steps - warmup_steps) / max(1, ramp_steps)),
            )
        )

    def _cost_penalty_scale(self) -> float:
        return self._curriculum_scale(self.cost_penalty_warmup_steps, self.cost_penalty_ramp_steps)

    def _clearance_penalty_scale(self) -> float:
        return self._curriculum_scale(self.clearance_penalty_warmup_steps, self.clearance_penalty_ramp_steps)

    def step(self, action):
        obs, env_reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        self._total_steps += 1

        goal_met = bool(info.get("goal_met", False))
        cur_dist = extract_goal_distance(self.env)
        prev_dist = self._prev_goal_distance
        rew = 0.0
        dense_component = 0.0
        sparse_component = 0.0
        cost_penalty_component = 0.0
        clearance_penalty_component = 0.0
        forward_reward_component = 0.0
        backward_penalty_component = 0.0
        heading_reward_component = 0.0
        dense_goal_resample_skip = False

        if self.reward_mode == "sparse":
            sparse_component = 1.0 if goal_met else 0.0
            rew = sparse_component
        elif self.reward_mode == "dense":
            if goal_met:
                dense_goal_resample_skip = True
                dense_component = 0.0
                rew = 0.0
            elif np.isfinite(prev_dist) and np.isfinite(cur_dist):
                dense_component = self.dense_reward_scale * float(prev_dist - cur_dist)
                rew = dense_component
            else:
                rew = 0.0
        elif self.reward_mode == "dense_plus_sparse":
            if goal_met:
                dense_goal_resample_skip = True
                dense_component = 0.0
                rew = 0.0
            elif np.isfinite(prev_dist) and np.isfinite(cur_dist):
                dense_component = self.dense_reward_scale * float(prev_dist - cur_dist)
                rew = dense_component
            else:
                rew = 0.0
            if goal_met:
                sparse_component = 1.0
                rew += sparse_component
        elif self.reward_mode == "native":
            rew = float(env_reward)
        elif self.reward_mode == "none":
            rew = 0.0

        rew += self.step_penalty
        if self.cost_penalty != 0.0:
            cost_penalty_component = float(self.cost_penalty * self._cost_penalty_scale() * float(cost))
            rew += cost_penalty_component

        min_constrained_clearance = extract_min_constrained_clearance(self.env)
        if (
            self.clearance_penalty_scale != 0.0
            and self.clearance_margin > 0.0
            and np.isfinite(min_constrained_clearance)
        ):
            clearance_violation = max(0.0, self.clearance_margin - float(min_constrained_clearance))
            if clearance_violation > 0.0:
                clearance_scale = self._clearance_penalty_scale()
                clearance_penalty_component = float(
                    -self.clearance_penalty_scale * clearance_scale * (clearance_violation ** self.clearance_penalty_power)
                )
                rew += clearance_penalty_component

        agent_xy = extract_agent_xy(self.env)
        goal_xy = extract_goal_xy(self.env)
        forward_xy = extract_agent_forward_xy(self.env)
        vel_xy = extract_agent_velocity_xy(self.env)
        forward_speed = float("nan")
        goal_heading_alignment = float("nan")
        goal_direction_speed = float("nan")
        if vel_xy is not None and forward_xy is not None:
            forward_speed = float(np.dot(vel_xy, forward_xy))
            if self.forward_reward_scale != 0.0 and forward_speed > 0.0:
                forward_reward_component = float(self.forward_reward_scale * forward_speed)
                rew += forward_reward_component
            if self.backward_penalty_scale != 0.0 and forward_speed < 0.0:
                backward_penalty_component = float(-self.backward_penalty_scale * (-forward_speed))
                rew += backward_penalty_component
        if agent_xy is not None and goal_xy is not None and forward_xy is not None:
            goal_vec = np.asarray(goal_xy - agent_xy, dtype=np.float64).reshape(-1)
            goal_norm = float(np.linalg.norm(goal_vec))
            if goal_norm > 1e-6:
                goal_dir = goal_vec / goal_norm
                goal_heading_alignment = float(np.dot(forward_xy, goal_dir))
                if vel_xy is not None:
                    goal_direction_speed = float(np.dot(vel_xy, goal_dir))
                if self.heading_reward_scale != 0.0 and not goal_met:
                    heading_signal = goal_heading_alignment
                    if self.heading_positive_only:
                        heading_signal = max(0.0, heading_signal)
                    heading_reward_component = float(self.heading_reward_scale * heading_signal)
                    rew += heading_reward_component

        info["reward_mode"] = self.reward_mode
        info["reward_raw_env"] = float(env_reward)
        info["reward_shaped"] = float(rew)
        info["reward_dense_component"] = float(dense_component)
        info["reward_sparse_component"] = float(sparse_component)
        info["reward_step_penalty_component"] = float(self.step_penalty)
        info["reward_cost_penalty_component"] = float(cost_penalty_component)
        info["reward_clearance_penalty_component"] = float(clearance_penalty_component)
        info["reward_cost_penalty_scale"] = float(self._cost_penalty_scale())
        info["reward_clearance_penalty_scale"] = float(self._clearance_penalty_scale())
        info["reward_forward_component"] = float(forward_reward_component)
        info["reward_backward_penalty_component"] = float(backward_penalty_component)
        info["reward_heading_component"] = float(heading_reward_component)
        info["reward_dense_goal_resample_skip"] = 1.0 if dense_goal_resample_skip else 0.0
        if np.isfinite(forward_speed):
            info["agent_forward_speed"] = float(forward_speed)
        if np.isfinite(goal_heading_alignment):
            info["goal_heading_alignment"] = float(goal_heading_alignment)
        if np.isfinite(goal_direction_speed):
            info["goal_direction_speed"] = float(goal_direction_speed)
        if np.isfinite(prev_dist):
            info["goal_distance_prev"] = float(prev_dist)
        if np.isfinite(cur_dist) and np.isfinite(prev_dist):
            raw_delta = float(cur_dist - prev_dist)
            info["goal_distance_delta_raw"] = raw_delta
            info["goal_distance_delta"] = 0.0 if dense_goal_resample_skip else raw_delta
        if np.isfinite(cur_dist):
            info["goal_distance"] = float(cur_dist)
        if np.isfinite(min_constrained_clearance):
            info["min_constrained_clearance"] = float(min_constrained_clearance)

        self._prev_goal_distance = cur_dist
        return obs, float(rew), cost, terminated, truncated, info


class TerminateOnGoalWrapper(gym.Wrapper):
    """End the episode immediately after the first goal hit."""

    def set_total_steps(self, total_steps: int) -> None:
        if hasattr(self.env, "set_total_steps"):
            self.env.set_total_steps(total_steps)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        if bool(info.get("goal_met", False)):
            terminated = True
            info["terminated_on_goal"] = True
        else:
            info["terminated_on_goal"] = False
        return obs, reward, cost, bool(terminated), bool(truncated), info

from __future__ import annotations

from collections import deque
from typing import Deque, Dict

import numpy as np


MANDATORY_EPISODE_KEYS = {
    "episode_return",
    "episode_cost_sum",
    "episode_cost_rate",
    "episode_length",
    "intervention_steps",
    "intervention_fraction",
    "intervention_num_bursts",
    "intervention_avg_burst_len",
    "goal_met",
    "goal_met_count",
    "first_goal_success",
    "first_goal_hit_step",
    "first_goal_hit_step_success_only",
    "first_goal_within_100",
    "first_goal_within_200",
    "first_goal_reward_sum",
    "first_goal_dense_reward_sum",
    "final_distance_to_goal",
    "outcome_success",
    "outcome_timeout",
    "outcome_kill",
    "outcome_other_failure",
    "terminated",
    "truncated",
}


def classify_outcome(*, goal_met: bool, episode_steps: int, max_episode_steps: int) -> str:
    if goal_met:
        return "success"
    if int(episode_steps) >= int(max_episode_steps):
        return "timeout"
    return "kill"


def validate_episode_metrics(ep: Dict[str, float]) -> None:
    missing = [k for k in sorted(MANDATORY_EPISODE_KEYS) if k not in ep]
    if missing:
        raise ValueError(f"Missing mandatory episode metrics: {missing}")
    for key in sorted(MANDATORY_EPISODE_KEYS):
        value = float(ep[key])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite mandatory metric: {key}={value}")


class EpisodeWindow:
    def __init__(self, size: int = 100):
        self._eps: Deque[Dict[str, float]] = deque(maxlen=int(max(1, size)))

    def add(self, ep: Dict[str, float]) -> None:
        validate_episode_metrics(ep)
        self._eps.append(dict(ep))

    def __len__(self) -> int:
        return len(self._eps)

    def summary(self, prefix: str) -> Dict[str, float]:
        if not self._eps:
            return {}
        keys = sorted({k for ep in self._eps for k in ep.keys()})
        out: Dict[str, float] = {}
        for key in keys:
            vals = np.asarray([float(ep.get(key, 0.0)) for ep in self._eps], dtype=np.float64)
            out[f"{prefix}/{key}_mean"] = float(np.mean(vals))
        out[f"{prefix}/episodes"] = float(len(self._eps))
        return out


def augment_rollout_summary(summary: Dict[str, float], prefix: str) -> Dict[str, float]:
    if not summary:
        return {}
    out = dict(summary)
    episodes = float(out.get(f"{prefix}/episodes", 0.0))
    goal_rate = float(out.get(f"{prefix}/goal_met_mean", out.get(f"{prefix}/outcome_success_mean", 0.0)))
    out[f"{prefix}/goal_success_rate"] = goal_rate
    goals_per_episode = float(out.get(f"{prefix}/goal_met_count_mean", goal_rate))
    out[f"{prefix}/goals_per_episode"] = goals_per_episode
    out[f"{prefix}/goals_solved"] = float(goals_per_episode * episodes)
    out[f"{prefix}/goals_reached"] = float(out[f"{prefix}/goals_solved"])
    out[f"{prefix}/goals_attempted"] = float(episodes)
    out[f"{prefix}/teacher_fraction_steps"] = float(out.get(f"{prefix}/intervention_fraction_mean", 0.0))
    out[f"{prefix}/teacher_intervention_steps"] = float(out.get(f"{prefix}/intervention_steps_mean", 0.0))
    if f"{prefix}/episode_return_mean" in out:
        out[f"{prefix}/mean_reward"] = float(out[f"{prefix}/episode_return_mean"])
    if f"{prefix}/episode_length_mean" in out:
        out[f"{prefix}/mean_episode_length"] = float(out[f"{prefix}/episode_length_mean"])
    if f"{prefix}/reward_raw_env_sum_mean" in out:
        out[f"{prefix}/mean_env_reward"] = float(out[f"{prefix}/reward_raw_env_sum_mean"])
    if f"{prefix}/reward_dense_sum_mean" in out:
        out[f"{prefix}/mean_dense_reward"] = float(out[f"{prefix}/reward_dense_sum_mean"])
    if f"{prefix}/reward_sparse_sum_mean" in out:
        out[f"{prefix}/mean_sparse_reward"] = float(out[f"{prefix}/reward_sparse_sum_mean"])
    if f"{prefix}/reward_step_penalty_sum_mean" in out:
        out[f"{prefix}/mean_step_penalty_reward"] = float(out[f"{prefix}/reward_step_penalty_sum_mean"])
    if f"{prefix}/reward_cost_penalty_sum_mean" in out:
        out[f"{prefix}/mean_cost_penalty_reward"] = float(out[f"{prefix}/reward_cost_penalty_sum_mean"])
    if f"{prefix}/reward_clearance_penalty_sum_mean" in out:
        out[f"{prefix}/mean_clearance_penalty_reward"] = float(out[f"{prefix}/reward_clearance_penalty_sum_mean"])
    if f"{prefix}/reward_forward_sum_mean" in out:
        out[f"{prefix}/mean_forward_reward"] = float(out[f"{prefix}/reward_forward_sum_mean"])
    if f"{prefix}/reward_backward_penalty_sum_mean" in out:
        out[f"{prefix}/mean_backward_penalty_reward"] = float(out[f"{prefix}/reward_backward_penalty_sum_mean"])
    if f"{prefix}/reward_heading_sum_mean" in out:
        out[f"{prefix}/mean_heading_reward"] = float(out[f"{prefix}/reward_heading_sum_mean"])
    if f"{prefix}/reward_cost_penalty_scale_mean" in out:
        out[f"{prefix}/safety_cost_scale"] = float(out[f"{prefix}/reward_cost_penalty_scale_mean"])
    if f"{prefix}/reward_clearance_penalty_scale_mean" in out:
        out[f"{prefix}/safety_clearance_scale"] = float(out[f"{prefix}/reward_clearance_penalty_scale_mean"])
    if f"{prefix}/reward_adaptive_safety_scale_mean" in out:
        out[f"{prefix}/adaptive_safety_scale"] = float(out[f"{prefix}/reward_adaptive_safety_scale_mean"])
    if f"{prefix}/reward_adaptive_goal_window_mean_mean" in out:
        out[f"{prefix}/adaptive_goal_window_mean"] = float(out[f"{prefix}/reward_adaptive_goal_window_mean_mean"])
    if f"{prefix}/reward_adaptive_cost_window_mean_mean" in out:
        out[f"{prefix}/adaptive_cost_window_mean"] = float(out[f"{prefix}/reward_adaptive_cost_window_mean_mean"])
    if f"{prefix}/episode_cost_sum_mean" in out:
        out[f"{prefix}/mean_episode_cost"] = float(out[f"{prefix}/episode_cost_sum_mean"])
        out[f"{prefix}/collision_cost_sum"] = float(out[f"{prefix}/episode_cost_sum_mean"])
    if f"{prefix}/episode_cost_rate_mean" in out:
        out[f"{prefix}/mean_episode_cost_rate"] = float(out[f"{prefix}/episode_cost_rate_mean"])
        out[f"{prefix}/collision_cost_rate"] = float(out[f"{prefix}/episode_cost_rate_mean"])
    if f"{prefix}/mean_constrained_clearance_mean" in out:
        out[f"{prefix}/mean_constrained_clearance"] = float(out[f"{prefix}/mean_constrained_clearance_mean"])
    if f"{prefix}/min_constrained_clearance_mean" in out:
        out[f"{prefix}/min_constrained_clearance"] = float(out[f"{prefix}/min_constrained_clearance_mean"])
    if f"{prefix}/first_goal_success_mean" in out:
        out[f"{prefix}/first_goal_success_rate"] = float(out[f"{prefix}/first_goal_success_mean"])
    if f"{prefix}/first_goal_hit_step_mean" in out:
        out[f"{prefix}/mean_first_goal_hit_step"] = float(out[f"{prefix}/first_goal_hit_step_mean"])
    if f"{prefix}/first_goal_reward_sum_mean" in out:
        out[f"{prefix}/mean_reward_to_first_goal"] = float(out[f"{prefix}/first_goal_reward_sum_mean"])
    if f"{prefix}/first_goal_dense_reward_sum_mean" in out:
        out[f"{prefix}/mean_dense_reward_to_first_goal"] = float(out[f"{prefix}/first_goal_dense_reward_sum_mean"])
    return out

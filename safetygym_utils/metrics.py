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
    out[f"{prefix}/goals_solved"] = float(goal_rate * episodes)
    out[f"{prefix}/goals_attempted"] = float(episodes)
    out[f"{prefix}/teacher_fraction_steps"] = float(out.get(f"{prefix}/intervention_fraction_mean", 0.0))
    out[f"{prefix}/teacher_intervention_steps"] = float(out.get(f"{prefix}/intervention_steps_mean", 0.0))
    if f"{prefix}/reward_shaped_sum_mean" in out:
        out[f"{prefix}/reward_shaped_return_mean"] = float(out[f"{prefix}/reward_shaped_sum_mean"])
    if f"{prefix}/reward_raw_env_sum_mean" in out:
        out[f"{prefix}/reward_raw_env_return_mean"] = float(out[f"{prefix}/reward_raw_env_sum_mean"])
    if f"{prefix}/reward_dense_sum_mean" in out:
        out[f"{prefix}/reward_dense_return_mean"] = float(out[f"{prefix}/reward_dense_sum_mean"])
    if f"{prefix}/reward_sparse_sum_mean" in out:
        out[f"{prefix}/reward_sparse_return_mean"] = float(out[f"{prefix}/reward_sparse_sum_mean"])
    if f"{prefix}/reward_step_penalty_sum_mean" in out:
        out[f"{prefix}/reward_step_penalty_return_mean"] = float(out[f"{prefix}/reward_step_penalty_sum_mean"])
    return out

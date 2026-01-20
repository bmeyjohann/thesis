from __future__ import annotations

from typing import Callable, Optional

from ogbench.wrappers import FlexibleObsWrapper, DetailedRewardWrapper, InterventionWrapper
import numpy as np

_GOAL_COLOR_MAP = {
    "red": (0.85, 0.2, 0.2, 1.0),
    "green": (0.1, 0.8, 0.2, 1.0),
    "blue": (0.2, 0.5, 1.0, 1.0),
}


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

from __future__ import annotations

from typing import Callable, Optional

from ogbench.wrappers import FlexibleObsWrapper, DetailedRewardWrapper, InterventionWrapper


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
            )
        return env

    return _apply

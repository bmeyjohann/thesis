from __future__ import annotations

from typing import Optional

import gymnasium as gym

from .intervention_wrappers import InterventionWrapper


class FixedResetSeedWrapper(gym.Wrapper):
    """Force deterministic reset seed on every episode reset."""

    def __init__(self, env: gym.Env, *, reset_seed: int):
        super().__init__(env)
        self._reset_seed = int(reset_seed)

    def reset(self, **kwargs):
        kwargs = dict(kwargs or {})
        kwargs["seed"] = self._reset_seed
        return self.env.reset(**kwargs)


def infer_ogbench_env_family(env_name: Optional[str]) -> str:
    """Infer broad OGBench family from env id/name."""
    name = str(env_name or "").lower()
    if any(tok in name for tok in ("cube", "scene", "puzzle", "manip")):
        return "manip"
    if any(tok in name for tok in ("maze", "pointmaze", "antmaze")):
        return "maze"
    return "unknown"


def infer_ogbench_env_family_from_env(env: gym.Env, env_name: Optional[str] = None) -> str:
    """Infer family from explicit name first, then fallback to env module path."""
    family_from_name = infer_ogbench_env_family(env_name)
    if family_from_name != "unknown":
        return family_from_name

    mod = str(getattr(type(env.unwrapped), "__module__", "")).lower()
    if "manipspace" in mod:
        return "manip"
    if "maze" in mod:
        return "maze"
    raise ValueError(
        "Unable to infer OGBench env family from env_name/module. "
        "Pass env_family explicitly or use a known maze/manip env id."
    )


def maybe_wrap_intervention(
    env: gym.Env,
    *,
    intervention_mode: str,
    teacher_type: str,
    tolerance_type: str,
    tolerance_value: float,
    tolerance_channel_weights: Optional[str],
    binary_gripper_actions: bool,
    binary_gripper_threshold: float,
    hard_gripper_intervention: bool,
    gripper_intervene_pick_radius: float,
    gripper_intervene_place_radius: float,
    gripper_intervene_contact_threshold: float,
    hard_block_lethal: bool,
    intervention_enable_after_steps: int,
    intervention_agent_mode: str,
    intervention_safety_margin_frac: float,
    intervention_release_steps: int,
    intervention_reward_patience_steps: int,
    intervention_reward_improvement_epsilon: float,
    intervention_episode_prob: float,
    intervention_episode_prob_min: float,
    intervention_episode_prob_decay_steps: int,
    intervention_episode_prob_decay_start: int,
    intervention_episode_prob_seed: Optional[int],
    teleop_interface: Optional[object] = None,
) -> tuple[gym.Env, Optional[str]]:
    """Apply intervention wrapper when configured and return (env, wrapper-name)."""
    if intervention_mode == "human":
        env = InterventionWrapper(
            env,
            teleop_interface=teleop_interface,
            mode="human",
            threshold=0.1,
            hold_time=0.5,
            binary_gripper_actions=binary_gripper_actions,
            binary_gripper_threshold=binary_gripper_threshold,
            hard_gripper_intervention=hard_gripper_intervention,
            gripper_intervene_pick_radius=gripper_intervene_pick_radius,
            gripper_intervene_place_radius=gripper_intervene_place_radius,
            gripper_intervene_contact_threshold=gripper_intervene_contact_threshold,
        )
        return env, "InterventionWrapper(human)"

    if intervention_mode == "agent":
        env = InterventionWrapper(
            env,
            mode="agent",
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
            enable_after_steps=intervention_enable_after_steps,
            agent_mode=intervention_agent_mode,
            safety_margin_frac=intervention_safety_margin_frac,
            release_steps=intervention_release_steps,
            reward_patience_steps=intervention_reward_patience_steps,
            reward_improvement_epsilon=intervention_reward_improvement_epsilon,
            episode_intervention_prob=intervention_episode_prob,
            episode_intervention_prob_min=intervention_episode_prob_min,
            episode_intervention_prob_decay_steps=intervention_episode_prob_decay_steps,
            episode_intervention_prob_decay_start=intervention_episode_prob_decay_start,
            episode_intervention_seed=intervention_episode_prob_seed,
        )
        return env, "InterventionWrapper(agent)"

    if intervention_mode in (
        "agent_always",
        "agent_safety_align",
        "agent_safety_progress",
        "agent_reward_progress",
        "agent_manual_gripper",
    ):
        if intervention_mode == "agent_always":
            agent_mode = "always"
        elif intervention_mode == "agent_safety_align":
            agent_mode = "safety_align"
        elif intervention_mode == "agent_safety_progress":
            agent_mode = "safety_progress"
        elif intervention_mode == "agent_manual_gripper":
            agent_mode = "manual_gripper"
        else:
            agent_mode = "reward_progress"
        env = InterventionWrapper(
            env,
            mode="agent",
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
            enable_after_steps=intervention_enable_after_steps,
            agent_mode=agent_mode,
            safety_margin_frac=intervention_safety_margin_frac,
            release_steps=intervention_release_steps,
            reward_patience_steps=intervention_reward_patience_steps,
            reward_improvement_epsilon=intervention_reward_improvement_epsilon,
            episode_intervention_prob=intervention_episode_prob,
            episode_intervention_prob_min=intervention_episode_prob_min,
            episode_intervention_prob_decay_steps=intervention_episode_prob_decay_steps,
            episode_intervention_prob_decay_start=intervention_episode_prob_decay_start,
            episode_intervention_seed=intervention_episode_prob_seed,
        )
        return env, f"InterventionWrapper({agent_mode})"

    return env, None

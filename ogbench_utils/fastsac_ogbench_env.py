"""Environment/wrapper construction utilities for FastSAC OGBench."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import torch

from fast_sac_utils import EmpiricalNormalization
from fasttd3.fast_sac.environments.ogbench_env import OGBenchVecEnvAdapter

from .env_wrappers_manip import CubeRewardModeTracker, build_ogbench_manip_wrapper, cube_reward_mode_active
from .env_wrappers_manip import canonicalize_cube_reward_mode
from .env_wrappers_maze import build_ogbench_maze_wrapper


def _select_wrapper_builder(env_family: str):
    family = str(env_family).lower()
    if family == "maze":
        return build_ogbench_maze_wrapper
    if family == "manip":
        return build_ogbench_manip_wrapper
    raise ValueError(f"Unknown env_family={env_family!r}")


def make_wrappers(args, env_family: str):
    """Training wrapper stack for one explicit environment family."""
    build_wrapper = _select_wrapper_builder(env_family)
    reward_switch = args.reward_switch_after_steps // max(1, args.num_envs)
    intervention_mode = args.intervention_mode if args.use_intervention else "none"
    wrapper_kwargs = dict(
        env_name=args.env_name,
        obs_mode=args.obs_mode,
        include_goal=args.include_goal,
        include_distance=args.include_distance,
        include_direction=args.include_direction,
        include_velocity=args.include_velocity,
        reward_type=args.reward_type,
        dense_reward_scale=args.dense_reward_scale,
        step_penalty=args.step_penalty,
        reward_switch_after_steps=reward_switch,
        intervention_mode=intervention_mode,
        teacher_type=args.teacher_type,
        tolerance_type=args.tolerance_type,
        tolerance_value=args.tolerance_value,
        tolerance_channel_weights=args.tolerance_channel_weights,
        binary_gripper_actions=args.binary_gripper_actions,
        binary_gripper_threshold=args.binary_gripper_threshold,
        hard_gripper_intervention=args.hard_gripper_intervention,
        gripper_intervene_pick_radius=args.gripper_intervene_pick_radius,
        gripper_intervene_place_radius=args.gripper_intervene_place_radius,
        gripper_intervene_contact_threshold=args.gripper_intervene_contact_threshold,
        hard_block_lethal=args.hard_block_lethal,
        intervention_enable_after_steps=args.intervention_enable_after_steps,
        intervention_safety_margin_frac=args.intervention_safety_margin_frac,
        intervention_release_steps=args.intervention_release_steps,
        intervention_reward_patience_steps=args.intervention_reward_patience_steps,
        intervention_reward_improvement_epsilon=args.intervention_reward_improvement_epsilon,
        intervention_episode_prob=args.intervention_episode_prob,
        intervention_episode_prob_min=args.intervention_episode_prob_min,
        intervention_episode_prob_decay_steps=args.intervention_episode_prob_decay_steps,
        intervention_episode_prob_decay_start=args.intervention_episode_prob_decay_start,
        intervention_episode_prob_seed=args.intervention_episode_prob_seed,
        teacher_target_mode=args.teacher_target_mode,
        cube_success_tolerance=args.cube_success_tolerance,
    )
    if str(env_family).lower() == "manip":
        wrapper_kwargs["cube_reward_mode"] = args.cube_reward_mode
        wrapper_kwargs["include_relative_cube_features"] = bool(
            getattr(args, "include_relative_cube_features", False)
        )
        wrapper_kwargs["relative_only_obs"] = bool(
            getattr(args, "relative_only_obs", False)
        )
        wrapper_kwargs["disable_rotation"] = bool(getattr(args, "disable_rotation", False))
        wrapper_kwargs["tolerance_xyz_value"] = getattr(args, "tolerance_xyz_value", -1.0)
        wrapper_kwargs["tolerance_yaw_value"] = getattr(args, "tolerance_yaw_value", -1.0)
        wrapper_kwargs["tolerance_gripper_value"] = getattr(args, "tolerance_gripper_value", -1.0)
        wrapper_kwargs["tolerance_adaptive_enable"] = bool(
            getattr(args, "tolerance_adaptive_enable", True)
        )
        wrapper_kwargs["tolerance_adaptive_near_distance"] = getattr(
            args, "tolerance_adaptive_near_distance", 0.08
        )
        wrapper_kwargs["tolerance_adaptive_far_distance"] = getattr(
            args, "tolerance_adaptive_far_distance", 0.30
        )
        wrapper_kwargs["tolerance_adaptive_near_scale"] = getattr(
            args, "tolerance_adaptive_near_scale", 0.35
        )
        wrapper_kwargs["intervention_agent_mode"] = getattr(args, "intervention_agent_mode", "divergence")
        wrapper_kwargs["human_intervention_threshold"] = getattr(args, "human_intervention_threshold", 0.1)
        wrapper_kwargs["human_intervention_hold_time"] = getattr(args, "human_intervention_hold_time", 0.5)
        wrapper_kwargs["static_reset_seed"] = getattr(args, "static_reset_seed", None)
    wrapper = build_wrapper(**wrapper_kwargs)
    return [wrapper]


def make_eval_wrappers(args, env_family: str):
    """Evaluation wrapper stack for one explicit environment family."""
    build_wrapper = _select_wrapper_builder(env_family)
    reward_switch = args.reward_switch_after_steps // max(1, args.num_envs)
    wrapper_kwargs = dict(
        env_name=args.env_name,
        obs_mode=args.obs_mode,
        include_goal=args.include_goal,
        include_distance=args.include_distance,
        include_direction=args.include_direction,
        include_velocity=args.include_velocity,
        reward_type=args.reward_type,
        dense_reward_scale=args.dense_reward_scale,
        step_penalty=args.step_penalty,
        reward_switch_after_steps=reward_switch,
        intervention_mode="none",
        teacher_type=args.teacher_type,
        tolerance_type=args.tolerance_type,
        tolerance_value=args.tolerance_value,
        tolerance_channel_weights=args.tolerance_channel_weights,
        binary_gripper_actions=args.binary_gripper_actions,
        binary_gripper_threshold=args.binary_gripper_threshold,
        hard_gripper_intervention=args.hard_gripper_intervention,
        gripper_intervene_pick_radius=args.gripper_intervene_pick_radius,
        gripper_intervene_place_radius=args.gripper_intervene_place_radius,
        gripper_intervene_contact_threshold=args.gripper_intervene_contact_threshold,
        hard_block_lethal=args.hard_block_lethal,
        intervention_enable_after_steps=args.intervention_enable_after_steps,
        intervention_safety_margin_frac=args.intervention_safety_margin_frac,
        intervention_release_steps=args.intervention_release_steps,
        intervention_reward_patience_steps=args.intervention_reward_patience_steps,
        intervention_reward_improvement_epsilon=args.intervention_reward_improvement_epsilon,
        intervention_episode_prob=args.intervention_episode_prob,
        intervention_episode_prob_min=args.intervention_episode_prob_min,
        intervention_episode_prob_decay_steps=args.intervention_episode_prob_decay_steps,
        intervention_episode_prob_decay_start=args.intervention_episode_prob_decay_start,
        intervention_episode_prob_seed=args.intervention_episode_prob_seed,
        teacher_target_mode=args.teacher_target_mode,
        cube_success_tolerance=args.cube_success_tolerance,
    )
    if str(env_family).lower() == "manip":
        wrapper_kwargs["cube_reward_mode"] = args.cube_reward_mode
        wrapper_kwargs["include_relative_cube_features"] = bool(
            getattr(args, "include_relative_cube_features", False)
        )
        wrapper_kwargs["relative_only_obs"] = bool(
            getattr(args, "relative_only_obs", False)
        )
        wrapper_kwargs["disable_rotation"] = bool(getattr(args, "disable_rotation", False))
        wrapper_kwargs["tolerance_xyz_value"] = getattr(args, "tolerance_xyz_value", -1.0)
        wrapper_kwargs["tolerance_yaw_value"] = getattr(args, "tolerance_yaw_value", -1.0)
        wrapper_kwargs["tolerance_gripper_value"] = getattr(args, "tolerance_gripper_value", -1.0)
        wrapper_kwargs["tolerance_adaptive_enable"] = bool(
            getattr(args, "tolerance_adaptive_enable", True)
        )
        wrapper_kwargs["tolerance_adaptive_near_distance"] = getattr(
            args, "tolerance_adaptive_near_distance", 0.08
        )
        wrapper_kwargs["tolerance_adaptive_far_distance"] = getattr(
            args, "tolerance_adaptive_far_distance", 0.30
        )
        wrapper_kwargs["tolerance_adaptive_near_scale"] = getattr(
            args, "tolerance_adaptive_near_scale", 0.35
        )
        wrapper_kwargs["intervention_agent_mode"] = getattr(args, "intervention_agent_mode", "divergence")
        wrapper_kwargs["human_intervention_threshold"] = getattr(args, "human_intervention_threshold", 0.1)
        wrapper_kwargs["human_intervention_hold_time"] = getattr(args, "human_intervention_hold_time", 0.5)
        wrapper_kwargs["static_reset_seed"] = getattr(args, "static_reset_seed", None)
    wrapper = build_wrapper(**wrapper_kwargs)
    return [wrapper]


def _build_env_kwargs(args) -> Dict[str, Any]:
    env_kwargs: Dict[str, Any] = {}
    if hasattr(args, "hold_targets_on_zero_action"):
        env_kwargs["hold_targets_on_zero_action"] = bool(getattr(args, "hold_targets_on_zero_action", False))
    if hasattr(args, "noop_action_threshold"):
        env_kwargs["noop_action_threshold"] = float(getattr(args, "noop_action_threshold", 1e-6))
    if hasattr(args, "disable_rotation"):
        env_kwargs["disable_rotation"] = bool(getattr(args, "disable_rotation", False))
    max_episode_steps = int(getattr(args, "max_episode_steps", 0) or 0)
    if max_episode_steps > 0:
        env_kwargs["max_episode_steps"] = max_episode_steps
    return env_kwargs


def _default_clip_actions(env_family: str) -> float | None:
    """Select vector-env action clipping policy by environment family.

    Maze tasks benefit from global L2 clipping to avoid diagonal-speed artifacts.
    Manipulation tasks use native per-dimension action bounds and should not be
    constrained by a global action-norm cap.
    """
    family = str(env_family).lower()
    if family == "manip":
        return None
    if family == "maze":
        return 1.0
    raise ValueError(f"Unknown env_family={env_family!r}")


def _build_vec_env_with_fallback(
    *,
    env_name: str,
    num_envs: int,
    device: torch.device,
    wrappers,
    clip_actions: float,
    env_kwargs: Dict[str, Any],
) -> OGBenchVecEnvAdapter:
    return OGBenchVecEnvAdapter(
        env_name=env_name,
        num_envs=num_envs,
        device=device,
        wrappers=wrappers,
        clip_actions=clip_actions,
        **env_kwargs,
    )


def _as_tensor_batch(
    value: Any,
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    num_envs: Optional[int] = None,
) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(device=device, dtype=dtype)
    else:
        tensor = tensor.to(device=device)
    if num_envs is not None:
        target = int(num_envs)
        if tensor.ndim == 0:
            tensor = tensor.repeat(target)
        elif tensor.shape[0] == 1 and target > 1:
            reps = [1] * tensor.ndim
            reps[0] = target
            tensor = tensor.repeat(*reps)
    return tensor


def _build_cube_reward_tracker(
    args,
    *,
    num_envs: int,
    device: torch.device,
    context: str,
    env_name_override: Optional[str] = None,
) -> Optional[CubeRewardModeTracker]:
    mode = canonicalize_cube_reward_mode(str(getattr(args, "cube_reward_mode", "dense")))
    setattr(args, "cube_reward_mode", mode)
    env_name = env_name_override if env_name_override is not None else args.env_name
    if not cube_reward_mode_active(env_name=env_name, obs_mode=args.obs_mode, reward_mode=mode):
        warnings.warn(
            f"{context}: --cube_reward_mode={mode} only applies to cube state environments; disabling cube reward tracker."
        )
        return None
    return CubeRewardModeTracker(
        mode=mode,
        num_envs=num_envs,
        device=device,
        success_reward=float(args.cube_success_reward),
        grasp_reward=float(args.cube_subgoal_grasp_reward),
        place_reward=float(args.cube_subgoal_place_reward),
        drop_penalty=float(args.cube_subgoal_drop_penalty),
        grasp_error_threshold=float(args.cube_subgoal_grasp_error_threshold),
        progress_scale=float(args.cube_dense_progress_scale),
        progress_clip=float(args.cube_dense_progress_clip),
    )


def _reward_window_to_log_tensors(
    *,
    sums: Dict[str, float],
    count: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if count <= 0.0:
        return {}
    out: Dict[str, torch.Tensor] = {}
    for key, value in sums.items():
        mean_value = torch.tensor([float(value) / count], device=device, dtype=torch.float32)
        out[f"/RewardMode/{key}"] = mean_value
        # Alias without a leading slash to make W&B panel discovery easier.
        out[f"Train/reward_mode_{key}"] = mean_value
    return out


def _apply_binary_gripper_action(action: torch.Tensor, *, enabled: bool, threshold: float) -> torch.Tensor:
    """Map the final gripper channel to {-1, +1} when enabled."""
    if not enabled:
        return action
    if action.ndim == 0 or action.shape[-1] < 4:
        return action
    out = action.clone()
    gripper_idx = out.shape[-1] - 1
    out[..., gripper_idx] = torch.where(out[..., gripper_idx] >= float(threshold), 1.0, -1.0)
    return out


def build_environment(
    args,
    device: torch.device,
    record_progress,
    env_family: str,
) -> Tuple[
    OGBenchVecEnvAdapter,
    list,
    Any,
    Any,
    int,
    int,
    torch.Tensor | None,
]:
    """Build training vector env + normalizers for FastSAC OGBench pipeline."""
    wrappers = make_wrappers(args, env_family)
    env_kwargs = _build_env_kwargs(args)
    clip_actions = _default_clip_actions(env_family)
    record_progress("[Init] constructing vector env adapter")
    envs = _build_vec_env_with_fallback(
        env_name=args.env_name,
        num_envs=args.num_envs,
        device=device,
        wrappers=wrappers,
        clip_actions=clip_actions,
        env_kwargs=env_kwargs,
    )
    record_progress("[Init] env adapter constructed")
    print("[Init] Env adapter constructed", flush=True)
    initial_obs_raw = envs.reset()
    record_progress("[Init] env reset for initial observation sample")

    obs_dim = envs.num_obs
    obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    critic_obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)

    return (
        envs,
        wrappers,
        obs_normalizer,
        critic_obs_normalizer,
        obs_dim,
        envs.num_actions,
        initial_obs_raw,
    )

def build_eval_environment(args, device: torch.device, env_family: str) -> OGBenchVecEnvAdapter:
    """Build evaluation vector env for FastSAC OGBench pipeline."""
    wrappers = make_eval_wrappers(args, env_family)
    env_kwargs = _build_env_kwargs(args)
    clip_actions = _default_clip_actions(env_family)
    eval_envs = _build_vec_env_with_fallback(
        env_name=args.env_name,
        num_envs=max(1, int(args.eval_num_envs)),
        device=device,
        wrappers=wrappers,
        clip_actions=clip_actions,
        env_kwargs=env_kwargs,
    )
    return eval_envs

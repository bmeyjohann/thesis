"""Maze-specific environment and wrapper assembly for FastSAC OGBench."""

from __future__ import annotations

from typing import Any, Tuple

import torch

from fast_sac_utils import EmpiricalNormalization
from fasttd3.fast_sac.environments.ogbench_env import OGBenchVecEnvAdapter

from .env_wrappers_maze import build_ogbench_maze_wrapper

MAZE_CLIP_ACTIONS: float = 1.0


def make_maze_wrappers(args):
    reward_switch = args.reward_switch_after_steps // max(1, args.num_envs)
    intervention_mode = args.intervention_mode if args.use_intervention else "none"
    wrapper = build_ogbench_maze_wrapper(
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
    return [wrapper]


def make_maze_eval_wrappers(args):
    reward_switch = args.reward_switch_after_steps // max(1, args.num_envs)
    wrapper = build_ogbench_maze_wrapper(
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
    return [wrapper]


def build_maze_environment(
    args,
    device: torch.device,
    record_progress,
) -> Tuple[
    OGBenchVecEnvAdapter,
    list,
    Any,
    Any,
    int,
    int,
    torch.Tensor | None,
]:
    wrappers = make_maze_wrappers(args)
    record_progress("[Init] constructing maze vector env adapter")
    envs = OGBenchVecEnvAdapter(
        env_name=args.env_name,
        num_envs=args.num_envs,
        device=device,
        wrappers=wrappers,
        clip_actions=MAZE_CLIP_ACTIONS,
    )
    record_progress("[Init] maze env adapter constructed")
    print("[Init] Maze env adapter constructed", flush=True)
    initial_obs_raw = envs.reset()
    record_progress("[Init] maze env reset for initial observation sample")

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


def build_maze_eval_environment(args, device: torch.device) -> OGBenchVecEnvAdapter:
    wrappers = make_maze_eval_wrappers(args)
    return OGBenchVecEnvAdapter(
        env_name=args.env_name,
        num_envs=max(1, int(args.eval_num_envs)),
        device=device,
        wrappers=wrappers,
        clip_actions=MAZE_CLIP_ACTIONS,
    )

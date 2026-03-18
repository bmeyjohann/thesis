"""Rollout, evaluation, and training loop utilities for FastSAC OGBench."""

from __future__ import annotations

import copy
import time
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from tensordict import TensorDict

from fast_sac_utils import SimpleReplayBuffer
from fasttd3.fast_sac.environments.ogbench_env import OGBenchVecEnvAdapter

from .env_manager import maybe_switch_env
from .fastsac_ogbench_env import (
    _apply_binary_gripper_action,
    _as_tensor_batch,
    _build_cube_reward_tracker,
    _build_env_kwargs,
    _build_vec_env_with_fallback,
    _default_clip_actions,
    _reward_window_to_log_tensors,
    make_wrappers,
)
from .fastsac_ogbench_setup import build_updater_from_components, create_replay_buffer
from .fastsac_ogbench_types import AMPComponents, BufferComponents, ModelComponents
from .intervention_wrappers import InterventionWrapper
from .logging import CheckpointManager, TeacherMetricsAccumulator, TimingWindow, TrainingLogger
from .obs import prepare_observation
from .update import FastSACUpdater

def prefill_replay_buffer_with_demos(
    *,
    args,
    device: torch.device,
    env_family: str,
    replay_buffer: SimpleReplayBuffer,
    demo_buffer: Optional[SimpleReplayBuffer],
    obs_normalizer,
    model: ModelComponents,
    record_progress,
    training_logger=None,
) -> None:
    """Optional teacher-demo prefill stage executed before main training."""
    max_prefill_steps = int(max(0, args.demo_prefill_steps))
    target_prefill_episodes = int(max(0, args.demo_prefill_episodes))
    if max_prefill_steps <= 0 and target_prefill_episodes <= 0:
        return

    if args.demo_prefill_target == "demo":
        if demo_buffer is None:
            raise ValueError("demo_prefill_target=demo requires --demo_buffer_enable")
        target_buffer = demo_buffer
        target_label = "demo"
    else:
        target_buffer = replay_buffer
        target_label = "replay"

    demo_args = copy.deepcopy(args)
    demo_num_envs = int(getattr(args, "demo_prefill_num_envs", 0))
    if demo_num_envs > 0:
        demo_args.num_envs = demo_num_envs
    demo_args.use_intervention = True
    demo_args.intervention_mode = args.demo_prefill_intervention_mode
    demo_args.intervention_enable_after_steps = args.demo_prefill_enable_after_steps
    demo_args.intervention_episode_prob = args.demo_prefill_episode_prob
    demo_args.intervention_episode_prob_min = args.demo_prefill_episode_prob
    demo_args.intervention_episode_prob_decay_steps = 0
    demo_args.intervention_episode_prob_decay_start = 0
    demo_args.hard_block_lethal = args.demo_prefill_hard_block_lethal
    demo_args.train_render_mode = "none"

    wrappers = make_wrappers(demo_args, env_family)
    env_kwargs = _build_env_kwargs(demo_args)
    clip_actions = _default_clip_actions(env_family)
    demo_envs = _build_vec_env_with_fallback(
        env_name=demo_args.env_name,
        num_envs=demo_args.num_envs,
        device=device,
        wrappers=wrappers,
        clip_actions=clip_actions,
        env_kwargs=env_kwargs,
    )
    demo_reward_tracker = _build_cube_reward_tracker(
        demo_args,
        num_envs=demo_envs.num_envs,
        device=device,
        context="demo_prefill",
    )

    if target_prefill_episodes > 0:
        prefill_goal = f"{target_prefill_episodes} episodes"
    else:
        prefill_goal = f"{max_prefill_steps} env steps"
    record_progress(
        f"[Demo] Prefilling {target_label} buffer for {prefill_goal} using {demo_args.num_envs} envs "
        f"(main training uses {args.num_envs})"
    )
    obs_raw = demo_envs.reset()
    obs = prepare_observation(
        obs_raw,
        device=device,
        obs_mode='state',
        pixel_shape=None,
        flatten=True,
    )

    total_steps = 0
    total_episodes = 0
    total_teacher_steps = 0
    raw_total_steps = 0
    pending_episode_rows: list[list[TensorDict]] = [[] for _ in range(demo_args.num_envs)]
    pending_episode_teacher_steps = [0 for _ in range(demo_args.num_envs)]
    next_prefill_log_step = max(1, int(getattr(args, "log_interval", 1)))
    last_status_time = time.time()
    actor_backbone = model.actor_backbone
    actor_head = model.actor_head
    actor_backbone.eval()
    actor_head.eval()
    try:
        start_line = (
            f"[Demo] rollout started: target={prefill_goal}, prefill_envs={demo_args.num_envs}, "
            f"train_envs={args.num_envs}, target_buffer={target_label}"
        )
        print(start_line, flush=True)
        record_progress(start_line)
        while True:
            if target_prefill_episodes > 0:
                if total_episodes >= target_prefill_episodes:
                    break
            elif total_steps >= max_prefill_steps:
                break
            with torch.no_grad():
                norm_obs = obs_normalizer(obs)
                pi_action, _, _ = actor_head(actor_backbone(norm_obs))
                pi_action = _apply_binary_gripper_action(
                    pi_action,
                    enabled=demo_args.binary_gripper_actions,
                    threshold=demo_args.binary_gripper_threshold,
                )
            next_obs_raw, rewards, dones, infos = demo_envs.step(pi_action.float())
            rewards = _as_tensor_batch(rewards, device=device, dtype=torch.float32, num_envs=demo_envs.num_envs)
            dones = _as_tensor_batch(dones, device=device, dtype=torch.bool, num_envs=demo_envs.num_envs)
            if demo_reward_tracker is not None:
                rewards, _ = demo_reward_tracker.compute(base_rewards=rewards, infos=infos, dones=dones)
            next_obs = prepare_observation(
                next_obs_raw,
                device=device,
                obs_mode='state',
                pixel_shape=None,
                flatten=True,
            )

            trunc_raw = infos.get('time_outs') if isinstance(infos, dict) else None
            truncations = (
                _as_tensor_batch(trunc_raw, device=device, dtype=torch.bool, num_envs=demo_envs.num_envs)
                if trunc_raw is not None
                else torch.zeros_like(dones, device=device, dtype=torch.bool)
            )
            if truncations.shape[0] != demo_envs.num_envs:
                done_ids = torch.nonzero(dones, as_tuple=False).flatten()
                if truncations.ndim == 1 and truncations.numel() == done_ids.numel():
                    trunc_full = torch.zeros_like(dones, device=device, dtype=torch.bool)
                    trunc_full[done_ids] = truncations.to(dtype=torch.bool)
                    truncations = trunc_full
                else:
                    truncations = torch.zeros_like(dones, device=device, dtype=torch.bool)
            applied_raw = infos.get('applied_actions', pi_action) if isinstance(infos, dict) else pi_action
            applied_actions = _as_tensor_batch(
                applied_raw, device=device, dtype=torch.float32, num_envs=demo_envs.num_envs
            )
            obs_detached_flat = obs.detach()
            next_obs_detached_flat = next_obs.detach()

            # Guard against malformed vector-env outputs (e.g. shape [1] instead of [num_envs]).
            if obs_detached_flat.shape[0] != demo_envs.num_envs:
                raise RuntimeError(
                    f"obs batch mismatch: expected {demo_envs.num_envs}, got {tuple(obs_detached_flat.shape)}"
                )
            if applied_actions.shape[0] != demo_envs.num_envs:
                raise RuntimeError(
                    f"action batch mismatch: expected {demo_envs.num_envs}, got {tuple(applied_actions.shape)}"
                )
            if next_obs_detached_flat.shape[0] != demo_envs.num_envs:
                raise RuntimeError(
                    f"next_obs batch mismatch: expected {demo_envs.num_envs}, got {tuple(next_obs_detached_flat.shape)}"
                )
            if rewards.shape[0] != demo_envs.num_envs:
                raise RuntimeError(
                    f"reward batch mismatch: expected {demo_envs.num_envs}, got {tuple(rewards.shape)}"
                )
            if dones.shape[0] != demo_envs.num_envs:
                raise RuntimeError(
                    f"done batch mismatch: expected {demo_envs.num_envs}, got {tuple(dones.shape)}"
                )
            if truncations.shape[0] != demo_envs.num_envs:
                raise RuntimeError(
                    f"truncation batch mismatch: expected {demo_envs.num_envs}, got {tuple(truncations.shape)}"
                )

            transition = TensorDict(
                {
                    'observations': obs_detached_flat,
                    'actions': applied_actions.detach(),
                    'next': {
                        'observations': next_obs_detached_flat,
                        'rewards': rewards.detach(),
                        'truncations': _as_tensor_batch(
                            truncations, device=device, dtype=torch.bool, num_envs=demo_envs.num_envs
                        ).long(),
                        'dones': _as_tensor_batch(
                            dones, device=device, dtype=torch.bool, num_envs=demo_envs.num_envs
                        ).long(),
                    },
                },
                batch_size=(demo_envs.num_envs,),
                device=device,
            )

            obs = next_obs
            step_teacher_mask = torch.zeros(demo_envs.num_envs, device=device, dtype=torch.bool)
            if isinstance(infos, dict):
                teacher_mask_raw = infos.get("teacher_intervened")
                if teacher_mask_raw is not None:
                    teacher_mask = _as_tensor_batch(
                        teacher_mask_raw, device=device, dtype=torch.bool, num_envs=demo_envs.num_envs
                    )
                    step_teacher_mask = teacher_mask.to(torch.bool)
            raw_total_steps += demo_envs.num_envs

            if target_prefill_episodes > 0:
                for env_idx in range(demo_envs.num_envs):
                    row_td = transition[env_idx : env_idx + 1]
                    pending_episode_rows[env_idx].append(row_td)
                    if bool(step_teacher_mask[env_idx].item()):
                        pending_episode_teacher_steps[env_idx] += 1
                    if bool(dones[env_idx].item()):
                        if total_episodes < target_prefill_episodes:
                            episode_rows = pending_episode_rows[env_idx]
                            for episode_row in episode_rows:
                                target_buffer.extend(episode_row)
                            total_steps += len(episode_rows)
                            total_teacher_steps += int(pending_episode_teacher_steps[env_idx])
                            total_episodes += 1
                        pending_episode_rows[env_idx] = []
                        pending_episode_teacher_steps[env_idx] = 0
            else:
                buffer_envs = int(getattr(target_buffer, "n_env", demo_envs.num_envs))
                if buffer_envs == demo_envs.num_envs:
                    target_buffer.extend(transition)
                else:
                    for env_idx in range(demo_envs.num_envs):
                        target_buffer.extend(transition[env_idx : env_idx + 1])
                total_teacher_steps += int(step_teacher_mask.to(torch.int32).sum().item())
                total_episodes += int(dones.sum().item())
                total_steps += demo_envs.num_envs

            now = time.time()
            if (now - last_status_time) >= 2.0:
                teacher_fraction = float(total_teacher_steps / max(1, total_steps))
                progress_line = (
                    f"[Demo] progress raw_env_steps={raw_total_steps} committed_env_steps={total_steps} episodes={total_episodes}"
                    f"/{target_prefill_episodes if target_prefill_episodes > 0 else '-'}"
                    f" teacher_frac={teacher_fraction:.3f}"
                    f" buffer_size={int(getattr(target_buffer, 'size', 0))}"
                )
                print(progress_line, flush=True)
                record_progress(progress_line)
                last_status_time = now
            if training_logger is not None and total_steps >= next_prefill_log_step:
                teacher_fraction = float(total_teacher_steps / max(1, total_steps))
                episode_progress = (
                    float(total_episodes / max(1, target_prefill_episodes))
                    if target_prefill_episodes > 0
                    else 0.0
                )
                target_buffer_size = float(getattr(target_buffer, "size", 0))
                target_buffer_capacity = float(getattr(target_buffer, "capacity", 0))
                training_logger.log_prefill(
                    pseudo_step=-1_000_000 + total_steps,
                    payload={
                        "DemoPrefill/raw_env_steps": float(raw_total_steps),
                        "DemoPrefill/env_steps": float(total_steps),
                        "DemoPrefill/episodes": float(total_episodes),
                        "DemoPrefill/episodes_target": float(target_prefill_episodes),
                        "DemoPrefill/episodes_progress": float(episode_progress),
                        "DemoPrefill/teacher_fraction": float(teacher_fraction),
                        "DemoPrefill/target_buffer_size": target_buffer_size,
                        "DemoPrefill/target_buffer_capacity": target_buffer_capacity,
                        "DemoPrefill/num_envs": float(demo_args.num_envs),
                    },
                )
                next_prefill_log_step += max(1, int(getattr(args, "log_interval", 1)))
    finally:
        demo_envs.close()
        actor_backbone.train()
        actor_head.train()
        completion_line = (
            f"[Demo] Prefill complete: {total_steps} env steps, {total_episodes} episodes added to {target_label} buffer"
        )
        print(completion_line, flush=True)
        record_progress(completion_line)
        if training_logger is not None:
            teacher_fraction = float(total_teacher_steps / max(1, total_steps))
            episode_progress = (
                float(total_episodes / max(1, target_prefill_episodes))
                if target_prefill_episodes > 0
                else 0.0
            )
            training_logger.log_prefill(
                pseudo_step=-999999 + total_steps,
                payload={
                    "DemoPrefill/raw_env_steps": float(raw_total_steps),
                    "DemoPrefill/env_steps": float(total_steps),
                    "DemoPrefill/episodes": float(total_episodes),
                    "DemoPrefill/episodes_target": float(target_prefill_episodes),
                    "DemoPrefill/episodes_progress": float(episode_progress),
                    "DemoPrefill/teacher_fraction": float(teacher_fraction),
                    "DemoPrefill/target_buffer_size": float(getattr(target_buffer, "size", 0)),
                    "DemoPrefill/target_buffer_capacity": float(getattr(target_buffer, "capacity", 0)),
                    "DemoPrefill/num_envs": float(demo_args.num_envs),
                    "DemoPrefill/completed": 1.0,
                },
            )


def run_eval_metrics(
    *,
    args,
    device: torch.device,
    eval_envs: OGBenchVecEnvAdapter,
    actor_backbone: nn.Module,
    actor_head: nn.Module,
    obs_normalizer,
    amp: AMPComponents,
) -> Dict[str, float]:
    """Run policy evaluation rollouts and aggregate episode-level metrics."""
    num_eval_episodes = max(1, int(args.num_eval_episodes))
    obs_normalizer_was_training = obs_normalizer.training
    actor_backbone_was_training = actor_backbone.training
    actor_head_was_training = actor_head.training
    obs_normalizer.eval()
    actor_backbone.eval()
    actor_head.eval()

    obs_raw = eval_envs.reset()
    obs = prepare_observation(
        obs_raw,
        device=device,
        obs_mode='state',
        pixel_shape=None,
        flatten=True,
    )
    eval_reward_tracker = _build_cube_reward_tracker(
        args,
        num_envs=eval_envs.num_envs,
        device=device,
        context="eval_metrics",
    )

    episode_returns = torch.zeros(eval_envs.num_envs, device=device)
    episode_lengths = torch.zeros(eval_envs.num_envs, device=device)
    total_reward = 0.0
    total_length = 0.0
    success_count = 0
    lethal_count = 0
    timeout_count = 0
    distance_sum = 0.0
    distance_count = 0
    success_length_sum = 0.0
    cubes_solved_sum = 0.0
    cubes_solved_count = 0
    cubes_total_sum = 0.0
    cubes_total_count = 0
    cubes_solved_fraction_sum = 0.0
    cubes_solved_fraction_count = 0
    cube_max_error_sum = 0.0
    cube_max_error_count = 0
    dense_phase_cumulative_sum = 0.0
    dense_phase_cumulative_count = 0
    partial_progress_episode_count = 0
    tracker_last_target_sum = 0.0
    tracker_last_target_count = 0
    tracker_grasp_events_sum = 0.0
    tracker_drop_events_sum = 0.0
    tracker_place_events_sum = 0.0
    tracker_grasp_episode_count = 0
    tracker_drop_episode_count = 0
    tracker_place_episode_count = 0
    tracker_max_cubes_solved_sum = 0.0
    tracker_max_cubes_solved_count = 0
    episodes_completed = 0
    tracker_cur_last_target = torch.zeros(eval_envs.num_envs, device=device)
    tracker_cur_grasp_events = torch.zeros(eval_envs.num_envs, device=device)
    tracker_cur_drop_events = torch.zeros(eval_envs.num_envs, device=device)
    tracker_cur_place_events = torch.zeros(eval_envs.num_envs, device=device)
    tracker_cur_max_cubes_solved = torch.zeros(eval_envs.num_envs, device=device)

    while episodes_completed < num_eval_episodes:
        norm_obs = obs_normalizer(obs)
        with torch.no_grad(), autocast(device_type=amp.device_type, dtype=amp.dtype, enabled=amp.enabled):
            _, _, mean_actions = actor_head(actor_backbone(norm_obs))
            mean_actions = _apply_binary_gripper_action(
                mean_actions,
                enabled=args.binary_gripper_actions,
                threshold=args.binary_gripper_threshold,
            )
        next_obs_raw, rewards, dones, infos = eval_envs.step(mean_actions.float())
        rewards = _as_tensor_batch(rewards, device=device, dtype=torch.float32, num_envs=eval_envs.num_envs)
        dones = _as_tensor_batch(dones, device=device, dtype=torch.bool, num_envs=eval_envs.num_envs)
        reward_components: Dict[str, torch.Tensor] = {}
        if eval_reward_tracker is not None:
            rewards, reward_components = eval_reward_tracker.compute(base_rewards=rewards, infos=infos, dones=dones)
            target_block = reward_components.get("target_block")
            if target_block is not None:
                tracker_cur_last_target = target_block.to(device=device, dtype=torch.float32)
            grasp_event = reward_components.get("dense_phase_grasp_event")
            if grasp_event is not None:
                tracker_cur_grasp_events = tracker_cur_grasp_events + grasp_event.to(device=device, dtype=torch.float32)
            drop_event = reward_components.get("drop_event")
            if drop_event is not None:
                tracker_cur_drop_events = tracker_cur_drop_events + drop_event.to(device=device, dtype=torch.float32)
            place_event = reward_components.get("dense_phase_place_event")
            if place_event is not None:
                tracker_cur_place_events = tracker_cur_place_events + place_event.to(device=device, dtype=torch.float32)
            cubes_solved_tracker = reward_components.get("cubes_solved")
            if cubes_solved_tracker is not None:
                tracker_cur_max_cubes_solved = torch.maximum(
                    tracker_cur_max_cubes_solved,
                    cubes_solved_tracker.to(device=device, dtype=torch.float32),
                )
        next_obs = prepare_observation(
            next_obs_raw,
            device=device,
            obs_mode='state',
            pixel_shape=None,
            flatten=True,
        )

        episode_returns += rewards
        episode_lengths += 1

        done_indices = torch.nonzero(dones).flatten().tolist()
        if done_indices:
            completed_returns = [float(episode_returns[i].item()) for i in done_indices]
            completed_lengths = [float(episode_lengths[i].item()) for i in done_indices]
            for i in done_indices:
                episode_returns[i] = 0.0
                episode_lengths[i] = 0.0

            total_reward += float(np.sum(completed_returns))
            total_length += float(np.sum(completed_lengths))
            episodes_completed += len(done_indices)

            goals = infos.get("goals_reached") or []
            lethals = infos.get("lethal_terminations") or []
            timeouts = infos.get("timeouts") or []
            distances = infos.get("distances_to_goal") or []
            cubes_solved = infos.get("episode_cubes_solved") or []
            cubes_total = infos.get("episode_cubes_total") or []
            cubes_solved_fractions = infos.get("episode_cubes_solved_fraction") or []
            cube_max_errors = infos.get("episode_cube_max_target_error") or []

            count = min(len(goals), len(completed_lengths))
            for idx in range(count):
                if float(goals[idx]) > 0.0:
                    success_count += 1
                    success_length_sum += completed_lengths[idx]
            for val in lethals[: len(completed_lengths)]:
                if float(val) > 0.0:
                    lethal_count += 1
            for val in timeouts[: len(completed_lengths)]:
                if float(val) > 0.0:
                    timeout_count += 1
            for val in distances[: len(completed_lengths)]:
                distance_sum += float(val)
                distance_count += 1
            for val in cubes_solved[: len(completed_lengths)]:
                solved = float(val)
                cubes_solved_sum += solved
                cubes_solved_count += 1
                if solved >= 1.0:
                    partial_progress_episode_count += 1
            for val in cubes_total[: len(completed_lengths)]:
                cubes_total_sum += float(val)
                cubes_total_count += 1
            for val in cubes_solved_fractions[: len(completed_lengths)]:
                cubes_solved_fraction_sum += float(val)
                cubes_solved_fraction_count += 1
            for val in cube_max_errors[: len(completed_lengths)]:
                cube_max_error_sum += float(val)
                cube_max_error_count += 1
            if reward_components and "dense_phase_cumulative" in reward_components:
                dense_phase_final = reward_components["dense_phase_cumulative"][done_indices]
                dense_phase_cumulative_sum += float(dense_phase_final.sum().item())
                dense_phase_cumulative_count += int(dense_phase_final.numel())
            if eval_reward_tracker is not None and done_indices:
                done_tensor = torch.tensor(done_indices, device=device, dtype=torch.long)
                last_target = tracker_cur_last_target.index_select(0, done_tensor)
                grasp_counts = tracker_cur_grasp_events.index_select(0, done_tensor)
                drop_counts = tracker_cur_drop_events.index_select(0, done_tensor)
                place_counts = tracker_cur_place_events.index_select(0, done_tensor)
                max_cubes = tracker_cur_max_cubes_solved.index_select(0, done_tensor)
                tracker_last_target_sum += float(last_target.sum().item())
                tracker_last_target_count += int(last_target.numel())
                tracker_grasp_events_sum += float(grasp_counts.sum().item())
                tracker_drop_events_sum += float(drop_counts.sum().item())
                tracker_place_events_sum += float(place_counts.sum().item())
                tracker_grasp_episode_count += int((grasp_counts > 0.0).sum().item())
                tracker_drop_episode_count += int((drop_counts > 0.0).sum().item())
                tracker_place_episode_count += int((place_counts > 0.0).sum().item())
                tracker_max_cubes_solved_sum += float(max_cubes.sum().item())
                tracker_max_cubes_solved_count += int(max_cubes.numel())
                tracker_cur_last_target.index_fill_(0, done_tensor, 0.0)
                tracker_cur_grasp_events.index_fill_(0, done_tensor, 0.0)
                tracker_cur_drop_events.index_fill_(0, done_tensor, 0.0)
                tracker_cur_place_events.index_fill_(0, done_tensor, 0.0)
                tracker_cur_max_cubes_solved.index_fill_(0, done_tensor, 0.0)

        obs = next_obs

    if obs_normalizer_was_training:
        obs_normalizer.train()
    if actor_backbone_was_training:
        actor_backbone.train()
    if actor_head_was_training:
        actor_head.train()

    denom = max(1, episodes_completed)
    metrics = {
        "avg_return": total_reward / denom,
        "avg_length": total_length / denom,
        "success_rate": success_count / denom,
        "lethal_rate": lethal_count / denom,
        "timeout_rate": timeout_count / denom,
    }
    if distance_count > 0:
        metrics["avg_final_distance"] = distance_sum / max(1, distance_count)
    if success_count > 0:
        metrics["avg_success_length"] = success_length_sum / max(1, success_count)
    if cubes_solved_count > 0:
        metrics["avg_cubes_solved"] = cubes_solved_sum / max(1, cubes_solved_count)
        metrics["partial_progress_rate"] = partial_progress_episode_count / max(1, cubes_solved_count)
    if cubes_total_count > 0:
        metrics["avg_cubes_total"] = cubes_total_sum / max(1, cubes_total_count)
    if cubes_solved_fraction_count > 0:
        metrics["avg_subgoal_progress"] = cubes_solved_fraction_sum / max(1, cubes_solved_fraction_count)
    if cube_max_error_count > 0:
        metrics["avg_cube_max_target_error"] = cube_max_error_sum / max(1, cube_max_error_count)
    if dense_phase_cumulative_count > 0:
        metrics["avg_dense_phase_cumulative_episode"] = (
            dense_phase_cumulative_sum / max(1, dense_phase_cumulative_count)
        )
    if tracker_last_target_count > 0:
        metrics["avg_last_target_block"] = tracker_last_target_sum / max(1, tracker_last_target_count)
        metrics["avg_episode_grasp_events"] = tracker_grasp_events_sum / max(1, tracker_last_target_count)
        metrics["avg_episode_drop_events"] = tracker_drop_events_sum / max(1, tracker_last_target_count)
        metrics["avg_episode_place_events"] = tracker_place_events_sum / max(1, tracker_last_target_count)
        metrics["grasped_episode_rate"] = tracker_grasp_episode_count / max(1, tracker_last_target_count)
        metrics["dropped_episode_rate"] = tracker_drop_episode_count / max(1, tracker_last_target_count)
        metrics["placed_episode_rate"] = tracker_place_episode_count / max(1, tracker_last_target_count)
    if tracker_max_cubes_solved_count > 0:
        metrics["avg_max_cubes_solved_tracker"] = (
            tracker_max_cubes_solved_sum / max(1, tracker_max_cubes_solved_count)
        )
    return metrics



def run_training_loop(
    args,
    device: torch.device,
    envs: OGBenchVecEnvAdapter,
    eval_envs: OGBenchVecEnvAdapter,
    wrappers,
    obs_normalizer,
    critic_obs_normalizer,
    model: ModelComponents,
    buffers: BufferComponents,
    amp: AMPComponents,
    training_logger: TrainingLogger,
    teacher_metrics: TeacherMetricsAccumulator,
    checkpoint_manager: CheckpointManager,
    record_progress,
    replay_buffer: SimpleReplayBuffer,
    demo_buffer: Optional[SimpleReplayBuffer],
    updater: FastSACUpdater,
    current_env_name: str,
    initial_obs_raw: torch.Tensor | None,
):
    """Main rollout/update loop for FastSAC OGBench training."""
    rb = replay_buffer
    pref_buffer = buffers.pref_buffer

    actor_backbone = model.actor_backbone
    actor_head = model.actor_head
    critic_backbone = model.critic_backbone
    critic_heads = model.critic_heads
    critic_target_backbone = model.critic_target_backbone
    critic_target_heads = model.critic_target_heads
    actor_optimizer = model.actor_optimizer
    q_optimizer = model.critic_optimizer
    trunk_optimizer = model.trunk_optimizer
    actor_params = model.actor_params
    critic_params = model.critic_params
    trunk_params = model.trunk_params
    log_alpha = model.log_alpha
    alpha_optimizer = model.alpha_optimizer
    target_entropy = model.target_entropy
    critic_feature_backbone = model.critic_feature_backbone

    initial_shared_backbone_state = model.initial_shared_backbone_state
    initial_critic_backbone_state = model.initial_critic_backbone_state
    initial_critic_heads_state = model.initial_critic_heads_state
    initial_target_backbone_state = model.initial_target_backbone_state
    initial_target_heads_state = model.initial_target_heads_state

    amp_enabled = amp.enabled
    amp_device_type = amp.device_type
    amp_dtype = amp.dtype
    scaler = amp.scaler

    env_switch_global_step = (
        int(max(0, args.switch_env_after_steps)) if args.switch_env_after_steps and args.switch_env_after_steps > 0 else None
    )

    def normalize_obs(x):
        return obs_normalizer(x)

    try:
        obs_raw = initial_obs_raw if initial_obs_raw is not None else envs.reset()
        obs = prepare_observation(
            obs_raw,
            device=device,
            obs_mode='state',
            pixel_shape=None,
            flatten=True,
        )
        n_obs = obs.shape[1]
        record_progress("[Init] envs.reset() returned; entering loop")

        print("[Init] Env reset complete; starting training loop", flush=True)

        total_env_steps = 0
        iteration_idx = 0
        start_time = time.time()

        cur_reward_sum = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
        cur_episode_length = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
        rewbuffer: list[float] = []
        lenbuffer: list[float] = []
        last_denied_samples = 0
        save_interval_current = args.save_interval if args.save_interval > 0 else None
        first_save_step = args.viz_first_step if (args.viz_first_step is not None and args.viz_first_step > 0) else None
        next_save_step = first_save_step if first_save_step is not None else save_interval_current
        eval_interval_current = args.eval_interval if args.eval_interval > 0 else None
        next_eval_step = eval_interval_current

        run_prefix = current_env_name.replace('-', '_')
        next_learning_starts_at = int(args.learning_starts)
        last_update_metrics = None
        did_reset_replay = False
        reward_mode_tracker = _build_cube_reward_tracker(
            args,
            num_envs=envs.num_envs,
            device=device,
            context="train_loop",
            env_name_override=current_env_name,
        )
        reward_window_sums: Dict[str, float] = {}
        reward_window_count = 0.0
        reward_window_dense_phase_episode_sum = 0.0
        reward_window_dense_phase_episode_count = 0
        dense_phase_episode_buffer: deque[float] = deque(maxlen=100)
        episode_last_target_buffer: deque[float] = deque(maxlen=100)
        episode_grasp_events_buffer: deque[float] = deque(maxlen=100)
        episode_drop_events_buffer: deque[float] = deque(maxlen=100)
        episode_place_events_buffer: deque[float] = deque(maxlen=100)
        episode_max_cubes_solved_buffer: deque[float] = deque(maxlen=100)
        episode_grasped_rate_buffer: deque[float] = deque(maxlen=100)
        episode_dropped_rate_buffer: deque[float] = deque(maxlen=100)
        episode_placed_rate_buffer: deque[float] = deque(maxlen=100)
        cur_last_target_block = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
        cur_grasp_event_count = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
        cur_drop_event_count = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
        cur_place_event_count = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
        cur_max_cubes_solved = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
        timing_enabled = bool(getattr(args, "profile_timing", False))
        timing_window = TimingWindow()
        human_reward_debug = bool(
            envs.num_envs == 1
            and str(getattr(args, "intervention_mode", "")).strip().lower() in {"human", "agent_manual_gripper"}
        )
        human_debug_step = 0
        human_debug_episode_idx = 0
        human_debug_episode_reward = 0.0
        human_debug_episode_steps = 0
        if human_reward_debug:
            print(
                "[HumanRewardDebug] enabled: episode-only reward summary "
                "(r_ep, mode_dense, mode_pre, target_err, target_dist, grasp)",
                flush=True,
            )

        while total_env_steps < args.total_timesteps:
            step_t0 = time.perf_counter()
            t_action = 0.0
            t_env = 0.0
            t_info = 0.0
            t_replay = 0.0
            t_sample = 0.0
            t_update = 0.0

            action_t0 = time.perf_counter()
            norm_obs = normalize_obs(obs)
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                pi_action, _, _ = actor_head(actor_backbone(norm_obs))
                pi_action = _apply_binary_gripper_action(
                    pi_action,
                    enabled=args.binary_gripper_actions,
                    threshold=args.binary_gripper_threshold,
                )
            t_action += time.perf_counter() - action_t0

            env_t0 = time.perf_counter()
            next_obs_raw, rewards, dones, infos = envs.step(pi_action.float())
            t_env += time.perf_counter() - env_t0

            info_t0 = time.perf_counter()
            rewards = _as_tensor_batch(rewards, device=device, dtype=torch.float32, num_envs=envs.num_envs)
            dones = _as_tensor_batch(dones, device=device, dtype=torch.bool, num_envs=envs.num_envs)
            reward_components: Dict[str, torch.Tensor] = {}
            if reward_mode_tracker is not None:
                rewards, reward_components = reward_mode_tracker.compute(base_rewards=rewards, infos=infos, dones=dones)
            actions = pi_action
            next_obs = prepare_observation(
                next_obs_raw,
                device=device,
                obs_mode='state',
                pixel_shape=None,
                flatten=True,
            )
            obs_detached_flat = obs.detach()
            next_obs_detached_flat = next_obs.detach()
            trunc_raw = infos.get('time_outs') if isinstance(infos, dict) else None
            truncations = (
                _as_tensor_batch(trunc_raw, device=device, dtype=torch.bool, num_envs=envs.num_envs)
                if trunc_raw is not None
                else torch.zeros_like(dones, device=device, dtype=torch.bool)
            )
            if truncations.shape[0] != envs.num_envs:
                # Some env adapters return timeout flags only for done envs.
                # Convert that compact form into a full per-env mask.
                done_ids = torch.nonzero(dones, as_tuple=False).flatten()
                if truncations.ndim == 1 and truncations.numel() == done_ids.numel():
                    trunc_full = torch.zeros_like(dones, device=device, dtype=torch.bool)
                    trunc_full[done_ids] = truncations.to(dtype=torch.bool)
                    truncations = trunc_full
                else:
                    record_progress(
                        "[Warn] Unexpected time_outs shape=%s for num_envs=%d; defaulting truncations to zeros"
                        % (tuple(truncations.shape), envs.num_envs)
                    )
                    truncations = torch.zeros_like(dones, device=device, dtype=torch.bool)
            applied_raw = infos.get('applied_actions', actions) if isinstance(infos, dict) else actions
            applied_actions = _as_tensor_batch(
                applied_raw, device=device, dtype=torch.float32, num_envs=envs.num_envs
            )
            student_raw = infos.get('student_actions') if isinstance(infos, dict) else None
            student_actions = (
                _as_tensor_batch(
                    student_raw, device=device, dtype=torch.float32, num_envs=envs.num_envs
                )
                if student_raw is not None
                else None
            )
            teacher_mask_raw = infos.get('teacher_intervened_mask') if isinstance(infos, dict) else None
            teacher_mask = (
                _as_tensor_batch(teacher_mask_raw, device=device, dtype=torch.bool, num_envs=envs.num_envs)
                if teacher_mask_raw is not None
                else None
            )
            if teacher_mask is None and student_actions is not None and applied_actions is not None:
                try:
                    teacher_mask = (torch.abs(applied_actions - student_actions).sum(dim=-1) > 1e-6)
                except Exception:
                    teacher_mask = None

            rewards_eff = rewards
            used_actions = applied_actions
            student_actions_for_replay = student_actions if student_actions is not None else used_actions
            teacher_intervened_for_replay = (
                teacher_mask
                if teacher_mask is not None
                else torch.zeros(envs.num_envs, dtype=torch.bool, device=device)
            )
            dones_eff = dones
            last_denied_samples = 0
            if teacher_mask is not None:
                denied_ids = torch.nonzero(teacher_mask, as_tuple=False).flatten()
                if denied_ids.numel() > 0:
                    last_denied_samples = int(denied_ids.numel())

                    pref_sampling_mode = str(getattr(args, "pref_sampling_mode", "independent")).strip().lower()
                    if (
                        pref_sampling_mode != "linked"
                        and pref_buffer is not None
                        and student_actions is not None
                        and 'teacher_actions' in infos
                    ):
                        try:
                            a_teacher_all = _as_tensor_batch(
                                infos['teacher_actions'], device=device, dtype=torch.float32, num_envs=envs.num_envs
                            )
                            pref_buffer.append(
                                obs_detached_flat[denied_ids],
                                a_teacher_all[denied_ids],
                                student_actions[denied_ids],
                            )
                        except Exception:
                            pass

            if reward_mode_tracker is not None:
                target_block = reward_components.get("target_block")
                if target_block is not None:
                    cur_last_target_block = target_block.to(device=device, dtype=torch.float32)
                grasp_event = reward_components.get("dense_phase_grasp_event")
                if grasp_event is not None:
                    cur_grasp_event_count = cur_grasp_event_count + grasp_event.to(device=device, dtype=torch.float32)
                drop_event = reward_components.get("drop_event")
                if drop_event is not None:
                    cur_drop_event_count = cur_drop_event_count + drop_event.to(device=device, dtype=torch.float32)
                place_event = reward_components.get("dense_phase_place_event")
                if place_event is not None:
                    cur_place_event_count = cur_place_event_count + place_event.to(device=device, dtype=torch.float32)
                cubes_solved_tracker = reward_components.get("cubes_solved")
                if cubes_solved_tracker is not None:
                    cur_max_cubes_solved = torch.maximum(
                        cur_max_cubes_solved,
                        cubes_solved_tracker.to(device=device, dtype=torch.float32),
                    )
                for key, tensor_value in reward_components.items():
                    reward_window_sums[key] = reward_window_sums.get(key, 0.0) + float(tensor_value.sum().item())
                reward_window_sums["final_post_intervention"] = reward_window_sums.get(
                    "final_post_intervention", 0.0
                ) + float(rewards_eff.sum().item())
                reward_window_count += float(envs.num_envs)

            if human_reward_debug:
                human_debug_step += 1
                human_debug_episode_steps += 1
                step_reward = float(rewards_eff[0].item())
                human_debug_episode_reward += step_reward

                def _dbg_component(name: str, default: float = 0.0) -> float:
                    value = reward_components.get(name)
                    if value is None:
                        return float(default)
                    try:
                        if torch.is_tensor(value):
                            if value.numel() == 0:
                                return float(default)
                            return float(value.reshape(-1)[0].item())
                        arr = np.asarray(value, dtype=np.float32).reshape(-1)
                        if arr.size == 0:
                            return float(default)
                        return float(arr[0])
                    except Exception:
                        return float(default)

                if bool(dones_eff[0].item()):
                    human_debug_episode_idx += 1
                    print(
                        "[HumanRewardEpisode] "
                        f"episode={human_debug_episode_idx} "
                        f"reward={human_debug_episode_reward:+.4f} "
                        f"steps={human_debug_episode_steps} "
                        f"mode_dense={_dbg_component('dense_phase_reward'):+.4f} "
                        f"mode_pre={_dbg_component('mode_total_pre_intervention'):+.4f} "
                        f"target_err={_dbg_component('dense_target_error'):+.4f} "
                        f"target_dist={_dbg_component('target_effector_dist'):+.4f} "
                        f"grasp={_dbg_component('target_grasp_detected'):+.1f}",
                        flush=True,
                    )
                    human_debug_episode_reward = 0.0
                    human_debug_episode_steps = 0
            t_info += time.perf_counter() - info_t0

            replay_t0 = time.perf_counter()
            transition_dict = {
                'observations': obs_detached_flat,
                'actions': used_actions.detach(),
                'student_actions': student_actions_for_replay.detach(),
                'teacher_intervened': teacher_intervened_for_replay.detach(),
                'next': {
                    'observations': next_obs_detached_flat,
                    'rewards': rewards_eff.detach(),
                    'truncations': _as_tensor_batch(
                        truncations, device=device, dtype=torch.bool, num_envs=envs.num_envs
                    ).long(),
                    'dones': _as_tensor_batch(
                        dones_eff, device=device, dtype=torch.bool, num_envs=envs.num_envs
                    ).long(),
                },
            }
            try:
                transition = TensorDict(
                    transition_dict,
                    batch_size=(envs.num_envs,),
                    device=device,
                )
            except Exception as exc:
                shape_dbg = {
                    'observations': tuple(transition_dict['observations'].shape),
                    'actions': tuple(transition_dict['actions'].shape),
                    'next.observations': tuple(transition_dict['next']['observations'].shape),
                    'next.rewards': tuple(transition_dict['next']['rewards'].shape),
                    'next.truncations': tuple(transition_dict['next']['truncations'].shape),
                    'next.dones': tuple(transition_dict['next']['dones'].shape),
                }
                raise RuntimeError(
                    f"Failed to build transition TensorDict for num_envs={envs.num_envs}; shapes={shape_dbg}"
                ) from exc
            rb.extend(transition)
            t_replay += time.perf_counter() - replay_t0

            if bool(getattr(args, "compute_q_diagnostics", False)):
                with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                    norm_obs_now = normalize_obs(obs)
                    features_now = critic_feature_backbone(norm_obs_now)
                    q_stack = torch.stack(critic_heads(features_now, used_actions), dim=0).squeeze(-1)
                    max_q = torch.max(q_stack, dim=0).values
                    min_q = torch.min(q_stack, dim=0).values
                    disagreement_step = max_q - min_q
                    qmin_step = min_q
                teacher_mask_float = (
                    teacher_mask.float() if teacher_mask is not None else torch.zeros_like(disagreement_step, device=device)
                )
                non_teacher_mask_float = 1.0 - teacher_mask_float
                teacher_metrics.update(disagreement_step, teacher_mask_float, non_teacher_mask_float, qmin_step)

            cur_reward_sum += rewards_eff
            cur_episode_length += 1
            done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                rewbuffer += cur_reward_sum[done_ids].tolist()
                lenbuffer += cur_episode_length[done_ids].tolist()
                if reward_mode_tracker is not None and "dense_phase_cumulative" in reward_components:
                    dense_phase_final = reward_components["dense_phase_cumulative"][done_ids]
                    reward_window_dense_phase_episode_sum += float(dense_phase_final.sum().item())
                    reward_window_dense_phase_episode_count += int(dense_phase_final.numel())
                    dense_phase_episode_buffer.extend([float(v.item()) for v in dense_phase_final])
                episode_last_target_buffer.extend([float(v.item()) for v in cur_last_target_block[done_ids]])
                episode_grasp_events_buffer.extend([float(v.item()) for v in cur_grasp_event_count[done_ids]])
                episode_drop_events_buffer.extend([float(v.item()) for v in cur_drop_event_count[done_ids]])
                episode_place_events_buffer.extend([float(v.item()) for v in cur_place_event_count[done_ids]])
                episode_max_cubes_solved_buffer.extend([float(v.item()) for v in cur_max_cubes_solved[done_ids]])
                episode_grasped_rate_buffer.extend(
                    [float(v.item()) for v in (cur_grasp_event_count[done_ids] > 0.0).float()]
                )
                episode_dropped_rate_buffer.extend(
                    [float(v.item()) for v in (cur_drop_event_count[done_ids] > 0.0).float()]
                )
                episode_placed_rate_buffer.extend(
                    [float(v.item()) for v in (cur_place_event_count[done_ids] > 0.0).float()]
                )
                cur_reward_sum[done_ids] = 0
                cur_episode_length[done_ids] = 0
                cur_last_target_block[done_ids] = 0
                cur_grasp_event_count[done_ids] = 0
                cur_drop_event_count[done_ids] = 0
                cur_place_event_count[done_ids] = 0
                cur_max_cubes_solved[done_ids] = 0

            iteration_idx += 1
            total_env_steps += envs.num_envs

            switched, new_env_name, switched_obs_raw, updated_wandb_run = maybe_switch_env(
                args=args,
                envs=envs,
                current_env_name=current_env_name,
                wrappers=wrappers,
                total_env_steps=total_env_steps,
                env_switch_global_step=env_switch_global_step,
                record_progress=record_progress,
                wandb_run=training_logger.wandb_run,
            )
            if switched:
                training_logger.wandb_run = updated_wandb_run
                obs = prepare_observation(
                    switched_obs_raw,
                    device=device,
                    obs_mode='state',
                    pixel_shape=None,
                    flatten=True,
                )
                current_env_name = new_env_name
                run_prefix = current_env_name.replace('-', '_')
                cur_reward_sum.zero_()
                cur_episode_length.zero_()
                reward_mode_tracker = _build_cube_reward_tracker(
                    args,
                    num_envs=envs.num_envs,
                    device=device,
                    context="env_switch",
                    env_name_override=current_env_name,
                )
                reward_window_sums.clear()
                reward_window_count = 0.0
                teacher_metrics.reset_running_stats()
                cur_last_target_block.zero_()
                cur_grasp_event_count.zero_()
                cur_drop_event_count.zero_()
                cur_place_event_count.zero_()
                cur_max_cubes_solved.zero_()
                if save_interval_current is not None and args.post_switch_viz_multiplier > 1:
                    save_interval_current = max(1, args.save_interval // args.post_switch_viz_multiplier)
                    next_save_step = total_env_steps + save_interval_current
            else:
                obs = next_obs

            if (
                args.reset_replay_on_switch
                and not did_reset_replay
                and args.reward_switch_after_steps > 0
                and total_env_steps >= args.reward_switch_after_steps
            ):
                record_progress(f"[Replay] Resetting main replay buffer at step {total_env_steps}")
                rb = create_replay_buffer(args, device, n_obs, envs.num_actions)
                did_reset_replay = True
                next_learning_starts_at = total_env_steps + int(args.learning_starts)
                record_progress(f"[Replay] Post-reset warm-up: learning resumes at env_step >= {next_learning_starts_at}")

                if args.reset_critic_on_switch:
                    record_progress(f"[Critic] Resetting critic weights/optimizer at step {total_env_steps}")
                    if args.arch_shared_trunk and initial_shared_backbone_state is not None:
                        actor_backbone.load_state_dict(initial_shared_backbone_state)
                        trunk_params = list(actor_backbone.parameters())
                        trunk_optimizer = optim.Adam(trunk_params, lr=args.critic_learning_rate)
                        critic_params = list(critic_heads.parameters())
                        q_optimizer = optim.AdamW(critic_params, lr=args.critic_learning_rate, weight_decay=1e-5)
                    elif initial_critic_backbone_state is not None and critic_backbone is not None:
                        critic_backbone.load_state_dict(initial_critic_backbone_state)
                        critic_params = list(critic_backbone.parameters()) + list(critic_heads.parameters())
                        q_optimizer = optim.AdamW(critic_params, lr=args.critic_learning_rate, weight_decay=1e-5)
                    critic_heads.load_state_dict(initial_critic_heads_state)
                    critic_target_backbone.load_state_dict(initial_target_backbone_state)
                    critic_target_heads.load_state_dict(initial_target_heads_state)
                    critic_feature_backbone = actor_backbone if args.arch_shared_trunk else critic_backbone
                    model.critic_optimizer = q_optimizer
                    model.trunk_optimizer = trunk_optimizer
                    model.trunk_params = trunk_params
                    model.critic_params = critic_params
                    model.critic_feature_backbone = critic_feature_backbone
                    updater = build_updater_from_components(
                        args=args,
                        device=device,
                        model=model,
                        buffers=buffers,
                        obs_normalizer=obs_normalizer,
                        amp=amp,
                    )
                if save_interval_current is not None and args.post_switch_viz_multiplier > 1:
                    save_interval_current = max(1, args.save_interval // args.post_switch_viz_multiplier)
                    next_save_step = total_env_steps + save_interval_current

            if next_save_step is not None and total_env_steps >= next_save_step:
                tag_name = f"step{total_env_steps}"
                ckpt_path = checkpoint_manager.save(
                    tag=tag_name,
                    step_value=total_env_steps,
                    run_prefix=run_prefix,
                    actor_backbone=actor_backbone,
                    actor_head=actor_head,
                    critic_backbone=None if args.arch_shared_trunk else critic_backbone,
                    shared_backbone=actor_backbone if args.arch_shared_trunk else None,
                    critic_heads=critic_heads,
                    critic_target_backbone=critic_target_backbone,
                    critic_target_heads=critic_target_heads,
                    obs_normalizer=obs_normalizer,
                    critic_obs_normalizer=critic_obs_normalizer,
                    log_alpha=log_alpha,
                    pixel_shape=None,
                )
                checkpoint_manager.maybe_render_policy_map(
                    tag=tag_name,
                    step_value=total_env_steps,
                    checkpoint_path=ckpt_path,
                    current_env_name=current_env_name,
                    wandb_run=training_logger.wandb_run,
                )
                if save_interval_current:
                    if first_save_step is not None and next_save_step == first_save_step:
                        next_save_step = first_save_step + save_interval_current
                    else:
                        next_save_step += save_interval_current
                else:
                    next_save_step = None

            if total_env_steps >= next_learning_starts_at and getattr(rb, 'ptr', 0) > 0:
                # Batch size is interpreted as per-update total sample count.
                base_batch = max(1, int(args.batch_size))
                pref_sampling_mode = str(getattr(args, "pref_sampling_mode", "independent")).strip().lower()
                if pref_sampling_mode == "linked":
                    b_pref = 0
                else:
                    b_pref = int(base_batch * args.pref_sample_ratio) if pref_buffer is not None else 0
                b_demo = int(base_batch * args.demo_sample_ratio) if demo_buffer is not None else 0
                main_batch = max(1, base_batch - b_pref - b_demo)

                update_t0 = time.perf_counter()
                metrics_accumulator, updates_count = updater.update(
                    replay_buffer=rb,
                    demo_buffer=demo_buffer,
                    total_env_steps=total_env_steps,
                    main_batch=main_batch,
                    base_batch=base_batch,
                    b_pref=b_pref,
                    b_demo=b_demo,
                )
                update_total_elapsed = time.perf_counter() - update_t0
                sample_elapsed = float(metrics_accumulator.get("timing_sample_s", 0.0))
                opt_elapsed = float(metrics_accumulator.get("timing_opt_s", 0.0))
                if sample_elapsed > 0.0 or opt_elapsed > 0.0:
                    t_sample += sample_elapsed
                    t_update += opt_elapsed
                    extra = max(0.0, update_total_elapsed - (sample_elapsed + opt_elapsed))
                    t_update += extra
                else:
                    t_update += update_total_elapsed
                if updates_count > 0:
                    last_update_metrics = (metrics_accumulator, updates_count)
                    # Expose latest critic diagnostics in the manual intervention gate panel.
                    if human_reward_debug:
                        denom = float(max(1, updates_count))
                        q_min_val = float(metrics_accumulator.get("q_min_pi", 0.0)) / denom
                        q_dis_val = float(metrics_accumulator.get("q_disagreement_pi", 0.0)) / denom
                        InterventionWrapper.set_manual_gate_q_stats(
                            q_min=q_min_val,
                            q_disagreement=q_dis_val,
                        )

            step_total = time.perf_counter() - step_t0
            known_time = t_action + t_env + t_info + t_replay + t_sample + t_update
            t_misc = max(0.0, step_total - known_time)
            if timing_enabled:
                timing_window.add(
                    action_s=t_action,
                    env_s=t_env,
                    info_s=t_info,
                    replay_s=t_replay,
                    sample_s=t_sample,
                    update_s=t_update,
                    misc_s=t_misc,
                    total_s=step_total,
                )

            log_requested = training_logger.should_log(total_env_steps) or total_env_steps >= args.total_timesteps
            if log_requested:
                collection_time = time.time() - start_time
                pref_sampling_mode = str(getattr(args, "pref_sampling_mode", "independent")).strip().lower()
                if pref_sampling_mode == "linked":
                    pref_size = int(rb.linked_pref_pair_count())
                    pref_capacity = int(getattr(rb, "capacity", 0))
                else:
                    pref_size = int(getattr(pref_buffer, "size", 0)) if pref_buffer is not None else 0
                    pref_capacity = int(getattr(pref_buffer, "capacity", 0)) if pref_buffer is not None else 0
                demo_size = int(getattr(demo_buffer, "size", 0)) if demo_buffer is not None else 0
                demo_capacity = int(getattr(demo_buffer, "capacity", 0)) if demo_buffer is not None else 0
                timing_summary = timing_window.summary("Perf/timing") if timing_enabled else None
                infos_for_logging = infos
                reward_mode_logs = _reward_window_to_log_tensors(
                    sums=reward_window_sums,
                    count=reward_window_count,
                    device=device,
                )
                if reward_window_dense_phase_episode_count > 0:
                    reward_mode_logs["Train/avg_dense_phase_cumulative_episode"] = torch.tensor(
                        [
                            reward_window_dense_phase_episode_sum
                            / max(1, reward_window_dense_phase_episode_count)
                        ],
                        device=device,
                        dtype=torch.float32,
                    )
                if dense_phase_episode_buffer:
                    reward_mode_logs["Train/avg_dense_phase_cumulative_episode_last100"] = torch.tensor(
                        [float(np.mean(dense_phase_episode_buffer))],
                        device=device,
                        dtype=torch.float32,
                    )
                if episode_last_target_buffer:
                    reward_mode_logs["Train/avg_last_target_block_last100"] = torch.tensor(
                        [float(np.mean(episode_last_target_buffer))], device=device, dtype=torch.float32
                    )
                if episode_grasp_events_buffer:
                    reward_mode_logs["Train/avg_episode_grasp_events_last100"] = torch.tensor(
                        [float(np.mean(episode_grasp_events_buffer))], device=device, dtype=torch.float32
                    )
                if episode_drop_events_buffer:
                    reward_mode_logs["Train/avg_episode_drop_events_last100"] = torch.tensor(
                        [float(np.mean(episode_drop_events_buffer))], device=device, dtype=torch.float32
                    )
                if episode_place_events_buffer:
                    reward_mode_logs["Train/avg_episode_place_events_last100"] = torch.tensor(
                        [float(np.mean(episode_place_events_buffer))], device=device, dtype=torch.float32
                    )
                if episode_max_cubes_solved_buffer:
                    reward_mode_logs["Train/avg_episode_max_cubes_solved_last100"] = torch.tensor(
                        [float(np.mean(episode_max_cubes_solved_buffer))], device=device, dtype=torch.float32
                    )
                if episode_grasped_rate_buffer:
                    reward_mode_logs["Train/grasped_episode_rate_last100"] = torch.tensor(
                        [float(np.mean(episode_grasped_rate_buffer))], device=device, dtype=torch.float32
                    )
                if episode_dropped_rate_buffer:
                    reward_mode_logs["Train/dropped_episode_rate_last100"] = torch.tensor(
                        [float(np.mean(episode_dropped_rate_buffer))], device=device, dtype=torch.float32
                    )
                if episode_placed_rate_buffer:
                    reward_mode_logs["Train/placed_episode_rate_last100"] = torch.tensor(
                        [float(np.mean(episode_placed_rate_buffer))], device=device, dtype=torch.float32
                    )
                if reward_mode_logs:
                    infos_for_logging = dict(infos) if isinstance(infos, dict) else {"log": {}}
                    existing_log = infos_for_logging.get("log")
                    log_payload = dict(existing_log) if isinstance(existing_log, dict) else {}
                    log_payload.update(reward_mode_logs)
                    infos_for_logging["log"] = log_payload
                reward_window_sums.clear()
                reward_window_count = 0.0
                reward_window_dense_phase_episode_sum = 0.0
                reward_window_dense_phase_episode_count = 0
                training_logger.log(
                    total_env_steps=total_env_steps,
                    total_timesteps=args.total_timesteps,
                    iteration_idx=iteration_idx,
                    collection_time=collection_time,
                    rewbuffer=rewbuffer,
                    lenbuffer=lenbuffer,
                    last_update_metrics=last_update_metrics,
                    infos=infos_for_logging,
                    log_alpha=log_alpha,
                    last_denied_samples=last_denied_samples,
                    pref_size=pref_size,
                    pref_capacity=pref_capacity,
                    replay_size=rb.size,
                    replay_capacity=rb.capacity,
                    demo_size=demo_size,
                    demo_capacity=demo_capacity,
                    timing_summary=timing_summary,
                )
                last_update_metrics = None
                if timing_enabled:
                    timing_window.reset()

            if next_eval_step is not None and total_env_steps >= next_eval_step:
                eval_metrics = run_eval_metrics(
                    args=args,
                    device=device,
                    eval_envs=eval_envs,
                    actor_backbone=actor_backbone,
                    actor_head=actor_head,
                    obs_normalizer=obs_normalizer,
                    amp=amp,
                )
                training_logger.log_eval(total_env_steps=total_env_steps, metrics=eval_metrics)
                next_eval_step += eval_interval_current

        final_ckpt = checkpoint_manager.save(
            tag='final',
            step_value=total_env_steps,
            run_prefix=run_prefix,
            actor_backbone=actor_backbone,
            actor_head=actor_head,
            critic_backbone=None if args.arch_shared_trunk else critic_backbone,
            shared_backbone=actor_backbone if args.arch_shared_trunk else None,
            critic_heads=critic_heads,
            critic_target_backbone=critic_target_backbone,
            critic_target_heads=critic_target_heads,
            obs_normalizer=obs_normalizer,
            critic_obs_normalizer=critic_obs_normalizer,
            log_alpha=log_alpha,
            pixel_shape=None,
        )
        checkpoint_manager.maybe_render_policy_map(
            tag='final',
            step_value=total_env_steps,
            checkpoint_path=final_ckpt,
            current_env_name=current_env_name,
            wandb_run=training_logger.wandb_run,
        )

        total_time = time.time() - start_time
        summary_line = (
            "✅ FastSAC training complete"
            f" env_steps={total_env_steps}"
            f" iterations={iteration_idx}"
            f" duration_sec={total_time:.1f}"
            f" models_dir={checkpoint_manager.run_model_dir}"
        )
        print("=" * 80)
        print(summary_line)
        record_progress(summary_line)
        return total_env_steps, iteration_idx, total_time
    finally:
        training_logger.finish()

"""Faithful TD3-style PVP orchestration for manipulation OGBench tasks."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if "fasttd3/fast_sac" not in sys.path:
    sys.path.append("fasttd3/fast_sac")

from .env_wrappers_manip import canonicalize_cube_reward_mode
from .fastsac_ogbench_env import _apply_binary_gripper_action, _as_tensor_batch, _build_cube_reward_tracker
from .fastsac_ogbench_manip_env import build_manip_environment, build_manip_eval_environment
from .fastsac_ogbench_setup import build_teacher_metrics, select_device
from .hgdagger_ogbench_manip_train import _default_wandb_mode
from .logging import TrainingLogger
from .obs import prepare_observation


class TD3Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class TD3Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = TD3Critic(obs_dim, act_dim, hidden_dim)
        self.q2 = TD3Critic(obs_dim, act_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, act), self.q2(obs, act)

    def min_q(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, act)
        return torch.minimum(q1, q2)


@dataclass
class PVPBatch:
    states: torch.Tensor
    actions_behavior: torch.Tensor
    actions_novice: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    interventions: torch.Tensor
    intervention_starts: torch.Tensor


class PVPReplayBuffer:
    def __init__(self, *, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(max(1, capacity))
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.states = torch.empty((self.capacity, self.obs_dim), dtype=torch.float32)
        self.actions_behavior = torch.empty((self.capacity, self.act_dim), dtype=torch.float32)
        self.actions_novice = torch.empty((self.capacity, self.act_dim), dtype=torch.float32)
        self.next_states = torch.empty((self.capacity, self.obs_dim), dtype=torch.float32)
        self.rewards = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.dones = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.interventions = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.intervention_starts = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.ptr = 0
        self.size = 0

    def append(
        self,
        *,
        state: torch.Tensor,
        action_behavior: torch.Tensor,
        action_novice: torch.Tensor,
        next_state: torch.Tensor,
        reward: float,
        done: bool,
        intervened: bool,
        intervention_start: bool,
    ) -> None:
        idx = self.ptr
        self.states[idx].copy_(state.detach().to("cpu", torch.float32))
        self.actions_behavior[idx].copy_(action_behavior.detach().to("cpu", torch.float32))
        self.actions_novice[idx].copy_(action_novice.detach().to("cpu", torch.float32))
        self.next_states[idx].copy_(next_state.detach().to("cpu", torch.float32))
        self.rewards[idx, 0] = float(reward)
        self.dones[idx, 0] = 1.0 if bool(done) else 0.0
        self.interventions[idx, 0] = 1.0 if bool(intervened) else 0.0
        self.intervention_starts[idx, 0] = 1.0 if bool(intervention_start) else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Optional[PVPBatch]:
        if self.size < int(batch_size):
            return None
        idx = torch.randint(0, self.size, (int(batch_size),), device=torch.device("cpu"))
        return PVPBatch(
            states=self.states[idx].to(device, non_blocking=True),
            actions_behavior=self.actions_behavior[idx].to(device, non_blocking=True),
            actions_novice=self.actions_novice[idx].to(device, non_blocking=True),
            next_states=self.next_states[idx].to(device, non_blocking=True),
            rewards=self.rewards[idx].to(device, non_blocking=True),
            dones=self.dones[idx].to(device, non_blocking=True),
            interventions=self.interventions[idx].to(device, non_blocking=True),
            intervention_starts=self.intervention_starts[idx].to(device, non_blocking=True),
        )


def _concat_pvp_batches(a: PVPBatch, b: PVPBatch) -> PVPBatch:
    return PVPBatch(
        states=torch.cat([a.states, b.states], dim=0),
        actions_behavior=torch.cat([a.actions_behavior, b.actions_behavior], dim=0),
        actions_novice=torch.cat([a.actions_novice, b.actions_novice], dim=0),
        next_states=torch.cat([a.next_states, b.next_states], dim=0),
        rewards=torch.cat([a.rewards, b.rewards], dim=0),
        dones=torch.cat([a.dones, b.dones], dim=0),
        interventions=torch.cat([a.interventions, b.interventions], dim=0),
        intervention_starts=torch.cat([a.intervention_starts, b.intervention_starts], dim=0),
    )


def _prepare_run_dirs(args):
    if not getattr(args, "exp_name", None):
        env_tag = str(args.env_name).replace("-v0", "").replace("-", "_")
        args.exp_name = f"{env_tag}_pvp_td3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logs_root = Path("logs") / "pvp_td3"
    models_root = Path("models") / "pvp_td3"
    logs_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    run_log_dir = logs_root / args.exp_name
    run_model_dir = models_root / args.exp_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = run_log_dir / "training.log"
    progress_file = open(log_file_path, "a", encoding="utf-8")
    progress_file.write(f"# logging started {datetime.now().isoformat()}\n")
    progress_file.flush()

    def record_progress(message: str) -> None:
        progress_file.write(f"{datetime.now().isoformat()} {message}\n")
        progress_file.flush()

    config_path = run_log_dir / "args.json"
    with open(config_path, "w", encoding="utf-8") as cfg_file:
        json.dump(vars(args), cfg_file, indent=2)
    print(f"Saved run config: {config_path}", flush=True)
    return run_log_dir, run_model_dir, record_progress, progress_file


def _save_pvp_checkpoint(
    *,
    run_model_dir: Path,
    tag: str,
    step_value: int,
    run_prefix: str,
    actor: TD3Actor,
    critic: TwinCritic,
    actor_target: TD3Actor,
    critic_target: TwinCritic,
    obs_normalizer,
    args,
) -> Path:
    save_path = run_model_dir / f"{run_prefix}_{tag}.pt"
    checkpoint = {
        "step": int(step_value),
        "actor_state": actor.state_dict(),
        "critic_state": critic.state_dict(),
        "actor_target_state": actor_target.state_dict(),
        "critic_target_state": critic_target.state_dict(),
        "obs_normalizer_state": obs_normalizer.state_dict() if hasattr(obs_normalizer, "state_dict") else None,
        "args": vars(args),
    }
    torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
    return save_path


def _run_pvp_eval(
    *,
    args,
    device: torch.device,
    eval_envs,
    actor: TD3Actor,
    obs_normalizer,
) -> Dict[str, float]:
    num_eval_episodes = max(1, int(args.num_eval_episodes))
    obs_normalizer_was_training = obs_normalizer.training
    actor_was_training = actor.training
    obs_normalizer.eval()
    actor.eval()

    obs_raw = eval_envs.reset()
    obs = prepare_observation(obs_raw, device=device, obs_mode="state", pixel_shape=None, flatten=True)
    eval_reward_tracker = _build_cube_reward_tracker(args, num_envs=eval_envs.num_envs, device=device, context="eval_metrics")
    episode_returns = torch.zeros(eval_envs.num_envs, device=device)
    episode_lengths = torch.zeros(eval_envs.num_envs, device=device)
    total_reward = 0.0
    total_length = 0.0
    success_count = 0
    success_length_sum = 0.0
    timeout_count = 0
    lethal_count = 0
    distance_sum = 0.0
    distance_count = 0
    cubes_solved_sum = 0.0
    cubes_solved_count = 0
    partial_progress_episode_count = 0
    cubes_total_sum = 0.0
    cubes_total_count = 0
    cubes_solved_fraction_sum = 0.0
    cubes_solved_fraction_count = 0
    cube_max_error_sum = 0.0
    cube_max_error_count = 0
    episodes_completed = 0

    while episodes_completed < num_eval_episodes:
        with torch.no_grad():
            norm_obs = obs_normalizer(obs)
            actions = actor(norm_obs)
            actions = _apply_binary_gripper_action(
                actions,
                enabled=args.binary_gripper_actions,
                threshold=args.binary_gripper_threshold,
            )
        next_obs_raw, rewards, dones, infos = eval_envs.step(actions.float())
        rewards_t = _as_tensor_batch(rewards, device=device, dtype=torch.float32, num_envs=eval_envs.num_envs)
        dones_t = _as_tensor_batch(dones, device=device, dtype=torch.bool, num_envs=eval_envs.num_envs)
        if eval_reward_tracker is not None:
            rewards_t, _ = eval_reward_tracker.compute(base_rewards=rewards_t, infos=infos, dones=dones_t)
        obs = prepare_observation(next_obs_raw, device=device, obs_mode="state", pixel_shape=None, flatten=True)

        episode_returns += rewards_t
        episode_lengths += 1
        done_indices = torch.nonzero(dones_t).flatten().tolist()
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
            timeout_count += sum(1 for val in timeouts[: len(completed_lengths)] if float(val) > 0.0)
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

    if obs_normalizer_was_training:
        obs_normalizer.train()
    if actor_was_training:
        actor.train()
    denom = max(1, episodes_completed)
    metrics = {
        "avg_return": total_reward / denom,
        "avg_length": total_length / denom,
        "success_rate": success_count / denom,
        "lethal_rate": lethal_count / denom,
        "timeout_rate": timeout_count / denom,
    }
    if success_count > 0:
        metrics["avg_success_length"] = success_length_sum / max(1, success_count)
    if distance_count > 0:
        metrics["avg_final_distance"] = distance_sum / max(1, distance_count)
    if cubes_solved_count > 0:
        metrics["avg_cubes_solved"] = cubes_solved_sum / max(1, cubes_solved_count)
        metrics["partial_progress_rate"] = partial_progress_episode_count / max(1, cubes_solved_count)
    if cubes_total_count > 0:
        metrics["avg_cubes_total"] = cubes_total_sum / max(1, cubes_total_count)
    if cubes_solved_fraction_count > 0:
        metrics["avg_subgoal_progress"] = cubes_solved_fraction_sum / max(1, cubes_solved_fraction_count)
    if cube_max_error_count > 0:
        metrics["avg_cube_max_target_error"] = cube_max_error_sum / max(1, cube_max_error_count)
    return metrics


def _sample_balanced_batch(
    *,
    novice_buffer: PVPReplayBuffer,
    human_buffer: PVPReplayBuffer,
    batch_size: int,
    device: torch.device,
    use_balance_sample: bool,
) -> Optional[PVPBatch]:
    if use_balance_sample:
        half = int(batch_size // 2)
        if novice_buffer.size >= half and human_buffer.size >= (batch_size - half):
            novice = novice_buffer.sample(half, device)
            human = human_buffer.sample(batch_size - half, device)
            if novice is None or human is None:
                return None
            return _concat_pvp_batches(novice, human)
    if human_buffer.size >= batch_size:
        return human_buffer.sample(batch_size, device)
    if novice_buffer.size >= batch_size:
        return novice_buffer.sample(batch_size, device)
    return None


def run_pvp_td3_ogbench_manip(args) -> None:
    os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
    os.environ.setdefault("WANDB_MODE", _default_wandb_mode())
    os.environ.setdefault("WANDB_CONSOLE", "off")
    os.environ.setdefault("WANDB_SILENT", "true")

    args.algo_variant = "pvp_td3"
    args.cube_reward_mode = canonicalize_cube_reward_mode(str(getattr(args, "cube_reward_mode", "dense")))
    device = select_device(args)
    run_log_dir, run_model_dir, record_progress, progress_file = _prepare_run_dirs(args)
    print(f"Faithful PVP-TD3 OGBench (manip) on {args.env_name} device={device}", flush=True)
    print(f"Log directory: {run_log_dir}", flush=True)
    print(f"Model directory: {run_model_dir}", flush=True)

    envs, _, obs_normalizer, _, n_obs, n_act, initial_obs_raw = build_manip_environment(args, device, record_progress)
    eval_envs = build_manip_eval_environment(args, device)
    teacher_metrics = build_teacher_metrics(args, device)
    training_logger = TrainingLogger(args=args, record_progress=record_progress, teacher_metrics=teacher_metrics)

    actor = TD3Actor(n_obs, n_act, hidden_dim=int(getattr(args, "actor_hidden_dim", 256))).to(device)
    actor_target = copy.deepcopy(actor).to(device)
    critic = TwinCritic(n_obs, n_act, hidden_dim=int(getattr(args, "critic_hidden_dim", 256))).to(device)
    critic_target = copy.deepcopy(critic).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(args.actor_learning_rate))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(args.critic_learning_rate))

    novice_buffer = PVPReplayBuffer(capacity=int(max(1, args.buffer_size)), obs_dim=n_obs, act_dim=n_act)
    human_buffer = PVPReplayBuffer(capacity=int(max(1, args.demo_buffer_capacity)), obs_dim=n_obs, act_dim=n_act)

    obs = prepare_observation(
        initial_obs_raw if initial_obs_raw is not None else envs.reset(),
        device=device,
        obs_mode="state",
        pixel_shape=None,
        flatten=True,
    )
    reward_tracker = _build_cube_reward_tracker(args, num_envs=envs.num_envs, device=device, context="train_loop")

    total_env_steps = 0
    iteration_idx = 0
    start_time = time.time()
    cur_reward_sum = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
    cur_episode_length = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
    rewbuffer: list[float] = []
    lenbuffer: list[float] = []
    total_interventions = 0
    last_teacher_mask = torch.zeros(envs.num_envs, dtype=torch.bool, device=device)
    next_eval_step = int(args.eval_interval) if int(args.eval_interval) > 0 else None
    next_save_step = int(args.save_interval) if int(args.save_interval) > 0 else None
    run_prefix = args.env_name.replace("-", "_")
    critic_loss_value = 0.0
    actor_loss_value = 0.0
    q_min_data_mean = 0.0
    q_min_teacher_mean = 0.0
    q_min_non_teacher_mean = 0.0
    target_q_mean = 0.0
    q_disagreement_mean = 0.0
    proxy_teacher_loss_value = 0.0
    proxy_student_loss_value = 0.0
    pvp_intervened_batch_fraction = 0.0
    pvp_td_reward_mean = 0.0
    actor_update_count = 0

    try:
        while total_env_steps < int(args.total_timesteps):
            norm_obs = obs_normalizer(obs)
            with torch.no_grad():
                novice_actions = actor(norm_obs)
                novice_actions = _apply_binary_gripper_action(
                    novice_actions,
                    enabled=args.binary_gripper_actions,
                    threshold=args.binary_gripper_threshold,
                )
            next_obs_raw, rewards, dones, infos = envs.step(novice_actions.float())
            rewards_t = _as_tensor_batch(rewards, device=device, dtype=torch.float32, num_envs=envs.num_envs)
            dones_t = _as_tensor_batch(dones, device=device, dtype=torch.bool, num_envs=envs.num_envs)
            if reward_tracker is not None:
                rewards_t, _ = reward_tracker.compute(base_rewards=rewards_t, infos=infos, dones=dones_t)
            applied_actions = _as_tensor_batch(
                infos.get("applied_actions", novice_actions),
                device=device,
                dtype=torch.float32,
                num_envs=envs.num_envs,
            )
            teacher_mask = torch.abs(applied_actions - novice_actions).sum(dim=-1) > 1e-6
            intervention_starts = teacher_mask & (~last_teacher_mask)
            last_teacher_mask = teacher_mask
            next_obs = prepare_observation(next_obs_raw, device=device, obs_mode="state", pixel_shape=None, flatten=True)

            for env_idx in range(envs.num_envs):
                reward_value = float(rewards_t[env_idx].item()) if bool(getattr(args, "pvp_include_env_reward_in_td", False)) else 0.0
                target_buffer = human_buffer if bool(teacher_mask[env_idx].item()) else novice_buffer
                target_buffer.append(
                    state=obs[env_idx],
                    action_behavior=applied_actions[env_idx],
                    action_novice=novice_actions[env_idx],
                    next_state=next_obs[env_idx],
                    reward=reward_value,
                    done=bool(dones_t[env_idx].item()),
                    intervened=bool(teacher_mask[env_idx].item()),
                    intervention_start=bool(intervention_starts[env_idx].item()),
                )
            total_interventions += int(teacher_mask.to(torch.int64).sum().item())

            cur_reward_sum += rewards_t
            cur_episode_length += 1
            done_ids = torch.nonzero(dones_t).flatten()
            if done_ids.numel() > 0:
                rewbuffer += cur_reward_sum[done_ids].tolist()
                lenbuffer += cur_episode_length[done_ids].tolist()
                cur_reward_sum[done_ids] = 0
                cur_episode_length[done_ids] = 0

            if total_env_steps >= int(args.learning_starts):
                actor_update_count = 0
                for update_idx in range(int(max(1, args.num_updates))):
                    batch = _sample_balanced_batch(
                        novice_buffer=novice_buffer,
                        human_buffer=human_buffer,
                        batch_size=int(args.batch_size),
                        device=device,
                        use_balance_sample=bool(getattr(args, "pvp_balance_sample", True)),
                    )
                    if batch is None:
                        break

                    batch_obs = obs_normalizer(batch.states)
                    batch_next_obs = obs_normalizer(batch.next_states)

                    with torch.no_grad():
                        noise = torch.randn_like(batch.actions_behavior) * float(getattr(args, "pvp_target_policy_noise", 0.2))
                        noise = noise.clamp(
                            -float(getattr(args, "pvp_target_noise_clip", 0.5)),
                            float(getattr(args, "pvp_target_noise_clip", 0.5)),
                        )
                        next_actions = (actor_target(batch_next_obs) + noise).clamp(-1.0, 1.0)
                        target_q = critic_target.min_q(batch_next_obs, next_actions)
                        target_q = batch.rewards + (1.0 - batch.dones) * float(args.gamma) * target_q

                    q1_behavior, q2_behavior = critic(batch_obs, batch.actions_behavior)
                    q1_novice, q2_novice = critic(batch_obs, batch.actions_novice)

                    td_mask = torch.ones_like(batch.interventions)
                    if bool(getattr(args, "pvp_stop_td_on_intervention_start", True)):
                        td_mask = 1.0 - batch.intervention_starts

                    q_value_bound = float(getattr(args, "pvp_proxy_value_bound", 1.0))
                    cql_coeff = float(getattr(args, "pvp_cql_coefficient", 1.0))

                    critic_loss = 0.5 * F.mse_loss(td_mask * q1_behavior, td_mask * target_q)
                    critic_loss = critic_loss + 0.5 * F.mse_loss(td_mask * q2_behavior, td_mask * target_q)
                    proxy_teacher_loss = (
                        batch.interventions * cql_coeff * F.mse_loss(
                            q1_behavior, q_value_bound * torch.ones_like(q1_behavior), reduction="none"
                        )
                    ).mean()
                    proxy_teacher_loss = proxy_teacher_loss + (
                        batch.interventions * cql_coeff * F.mse_loss(
                            q2_behavior, q_value_bound * torch.ones_like(q2_behavior), reduction="none"
                        )
                    ).mean()
                    proxy_student_loss = (
                        batch.interventions * cql_coeff * F.mse_loss(
                            q1_novice, -q_value_bound * torch.ones_like(q1_novice), reduction="none"
                        )
                    ).mean()
                    proxy_student_loss = proxy_student_loss + (
                        batch.interventions * cql_coeff * F.mse_loss(
                            q2_novice, -q_value_bound * torch.ones_like(q2_novice), reduction="none"
                        )
                    ).mean()
                    critic_loss_total = critic_loss + proxy_teacher_loss + proxy_student_loss

                    critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"))
                    critic_optimizer.step()

                    if update_idx % int(max(1, getattr(args, "pvp_policy_delay", 2))) == 0:
                        actor_actions = actor(batch_obs)
                        actor_loss = -critic.min_q(batch_obs, actor_actions).mean()
                        actor_optimizer.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"))
                        actor_optimizer.step()
                        actor_update_count += 1

                        tau = float(args.tau)
                        for src_param, tgt_param in zip(actor.parameters(), actor_target.parameters()):
                            tgt_param.data.mul_(1.0 - tau).add_(tau * src_param.data)
                        for src_param, tgt_param in zip(critic.parameters(), critic_target.parameters()):
                            tgt_param.data.mul_(1.0 - tau).add_(tau * src_param.data)
                        actor_loss_value = float(actor_loss.detach().cpu().item())

                    with torch.no_grad():
                        qmin_data = torch.minimum(q1_behavior, q2_behavior)
                        qdis_data = torch.abs(q1_behavior - q2_behavior)
                        teacher_rows = batch.interventions.squeeze(-1) > 0.5
                        non_teacher_rows = ~teacher_rows
                        q_min_data_mean = float(qmin_data.mean().detach().cpu().item())
                        q_disagreement_mean = float(qdis_data.mean().detach().cpu().item())
                        q_min_teacher_mean = float(qmin_data[teacher_rows].mean().detach().cpu().item()) if bool(teacher_rows.any().item()) else 0.0
                        q_min_non_teacher_mean = float(qmin_data[non_teacher_rows].mean().detach().cpu().item()) if bool(non_teacher_rows.any().item()) else 0.0
                        target_q_mean = float(target_q.mean().detach().cpu().item())
                        pvp_intervened_batch_fraction = float(batch.interventions.mean().detach().cpu().item())
                        pvp_td_reward_mean = float(batch.rewards.mean().detach().cpu().item())
                    critic_loss_value = float(critic_loss_total.detach().cpu().item())
                    proxy_teacher_loss_value = float(proxy_teacher_loss.detach().cpu().item())
                    proxy_student_loss_value = float(proxy_student_loss.detach().cpu().item())

            total_env_steps += envs.num_envs
            iteration_idx += 1
            obs = next_obs

            if training_logger.should_log(total_env_steps):
                elapsed = max(1e-6, time.time() - start_time)
                logs = {
                    "Perf/total_fps": int(total_env_steps / elapsed),
                    "Perf/collection_time_sec": elapsed,
                    "Perf/env_steps": float(total_env_steps),
                    "Perf/iterations": float(iteration_idx),
                    "Train/mean_reward": float(np.mean(rewbuffer[-100:])) if rewbuffer else 0.0,
                    "Train/mean_episode_length": float(np.mean(lenbuffer[-100:])) if lenbuffer else 0.0,
                    "Train/actor_loss": float(actor_loss_value),
                    "Train/critic_loss": float(critic_loss_value),
                    "Train/critic_loss_total": float(critic_loss_value),
                    "Train/target_q_mean": float(target_q_mean),
                    "Train/q_min_data_mean": float(q_min_data_mean),
                    "Train/q_min_teacher_action_mean": float(q_min_teacher_mean),
                    "Train/q_min_non_teacher_action_mean": float(q_min_non_teacher_mean),
                    "Train/q_disagreement_data_mean": float(q_disagreement_mean),
                    "Train/total_interventions": float(total_interventions),
                    "Train/pvp_proxy_teacher_loss": float(proxy_teacher_loss_value),
                    "Train/pvp_proxy_student_loss": float(proxy_student_loss_value),
                    "Train/pvp_intervened_batch_fraction": float(pvp_intervened_batch_fraction),
                    "Train/pvp_td_reward_mean": float(pvp_td_reward_mean),
                    "Train/pvp_proxy_value_bound": float(getattr(args, "pvp_proxy_value_bound", 1.0)),
                    "Train/pvp_include_env_reward_in_td": 1.0 if bool(getattr(args, "pvp_include_env_reward_in_td", False)) else 0.0,
                    "Train/buffer_replay_size": float(novice_buffer.size),
                    "Train/buffer_demo_size": float(human_buffer.size),
                    "Train/buffer_novice_size": float(novice_buffer.size),
                    "Train/buffer_human_size": float(human_buffer.size),
                    "/Buffers/replay_size": float(novice_buffer.size),
                    "/Buffers/demo_size": float(human_buffer.size),
                    "/Buffers/novice_size": float(novice_buffer.size),
                    "/Buffers/human_size": float(human_buffer.size),
                    "/Buffers/replay_capacity": float(novice_buffer.capacity),
                    "/Buffers/demo_capacity": float(human_buffer.capacity),
                    "Train/actor_updates_per_iter": float(actor_update_count),
                    "Train/updates_per_iter": float(args.num_updates),
                }
                if "log" in infos and isinstance(infos["log"], dict):
                    for key, value in infos["log"].items():
                        try:
                            logs[key] = float(value.float().mean().item())
                        except Exception:
                            pass
                console_line = (
                    f"[PVP-TD3] env_steps {total_env_steps}/{args.total_timesteps} | "
                    f"iter {iteration_idx} | fps {logs['Perf/total_fps']:.0f} | "
                    f"qloss {logs['Train/critic_loss']:.3f} | "
                    f"teacher_frac {logs.get('/Teacher/teacher_fraction_steps', 0.0):.2f} | "
                    f"novice {novice_buffer.size} | human {human_buffer.size}"
                )
                print(console_line, flush=True)
                record_progress(console_line)
                wandb_run = training_logger.ensure_wandb_run()
                if wandb_run is not None:
                    wandb_run.log(logs, step=total_env_steps)

            if next_eval_step is not None and total_env_steps >= next_eval_step:
                eval_metrics = _run_pvp_eval(args=args, device=device, eval_envs=eval_envs, actor=actor, obs_normalizer=obs_normalizer)
                training_logger.log_eval(total_env_steps=total_env_steps, metrics=eval_metrics)
                next_eval_step += int(args.eval_interval)

            if next_save_step is not None and total_env_steps >= next_save_step:
                save_path = _save_pvp_checkpoint(
                    run_model_dir=run_model_dir,
                    tag=f"step{total_env_steps}",
                    step_value=total_env_steps,
                    run_prefix=run_prefix,
                    actor=actor,
                    critic=critic,
                    actor_target=actor_target,
                    critic_target=critic_target,
                    obs_normalizer=obs_normalizer,
                    args=args,
                )
                record_progress(f"[Checkpoint] saved {save_path}")
                next_save_step += int(args.save_interval)

        final_path = _save_pvp_checkpoint(
            run_model_dir=run_model_dir,
            tag="final",
            step_value=total_env_steps,
            run_prefix=run_prefix,
            actor=actor,
            critic=critic,
            actor_target=actor_target,
            critic_target=critic_target,
            obs_normalizer=obs_normalizer,
            args=args,
        )
        record_progress(f"[Checkpoint] saved {final_path}")
        print(
            f"================================================================================\n"
            f"✅ Faithful PVP-TD3 training complete env_steps={total_env_steps} iterations={iteration_idx} "
            f"duration_sec={time.time() - start_time:.1f} models_dir={run_model_dir}",
            flush=True,
        )
    finally:
        try:
            envs.close()
        except Exception:
            pass
        try:
            eval_envs.close()
        except Exception:
            pass
        training_logger.finish()
        progress_file.close()

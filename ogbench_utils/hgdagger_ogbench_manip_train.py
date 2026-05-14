"""Manipulation-specific HG-DAgger orchestration."""

from __future__ import annotations

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
import torch.nn.functional as F


def _default_wandb_mode() -> str:
    explicit = str(os.environ.get("WANDB_MODE", "")).strip()
    if explicit:
        return explicit
    cluster_markers = ("SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_JOB_NODELIST")
    on_cluster = any(str(os.environ.get(key, "")).strip() for key in cluster_markers)
    return "offline" if on_cluster else "online"


os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
os.environ.setdefault("WANDB_MODE", _default_wandb_mode())
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")

if "fasttd3/fast_sac" not in sys.path:
    sys.path.append("fasttd3/fast_sac")

from .env_wrappers_manip import canonicalize_cube_reward_mode
from .fastsac_ogbench_env import _apply_binary_gripper_action, _as_tensor_batch, _build_cube_reward_tracker
from .fastsac_ogbench_manip_env import build_manip_environment, build_manip_eval_environment
from .fastsac_ogbench_setup import build_teacher_metrics, select_device
from .logging import TrainingLogger
from .obs import prepare_observation
from .policy import GaussianPolicyHead, MLPBackbone
from .repro import seed_everything


@dataclass
class ExpertBatch:
    states: torch.Tensor
    actions: torch.Tensor


class ExpertDatasetBuffer:
    def __init__(self, *, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(max(1, capacity))
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.states = torch.empty((self.capacity, self.obs_dim), dtype=torch.float32)
        self.actions = torch.empty((self.capacity, self.act_dim), dtype=torch.float32)
        self.ptr = 0
        self.size = 0

    def append(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        if states is None or actions is None:
            return
        if states.numel() == 0 or actions.numel() == 0:
            return
        states_cpu = states.detach().to("cpu", non_blocking=False).to(torch.float32)
        actions_cpu = actions.detach().to("cpu", non_blocking=False).to(torch.float32)
        batch = int(states_cpu.shape[0])
        if batch > self.capacity:
            states_cpu = states_cpu[-self.capacity :]
            actions_cpu = actions_cpu[-self.capacity :]
            batch = self.capacity
        end = self.ptr + batch
        if end <= self.capacity:
            self.states[self.ptr:end].copy_(states_cpu)
            self.actions[self.ptr:end].copy_(actions_cpu)
        else:
            first = self.capacity - self.ptr
            self.states[self.ptr:].copy_(states_cpu[:first])
            self.actions[self.ptr:].copy_(actions_cpu[:first])
            remain = batch - first
            self.states[:remain].copy_(states_cpu[first:])
            self.actions[:remain].copy_(actions_cpu[first:])
        self.ptr = (self.ptr + batch) % self.capacity
        self.size = min(self.size + batch, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Optional[ExpertBatch]:
        if self.size <= 0:
            return None
        idx = torch.randint(0, self.size, (int(batch_size),), device=torch.device("cpu"))
        states = self.states[idx].to(device, non_blocking=True)
        actions = self.actions[idx].to(device, non_blocking=True)
        return ExpertBatch(states=states, actions=actions)


class HGDaggerEnsemble(torch.nn.Module):
    def __init__(self, *, obs_dim: int, act_dim: int, args, device: torch.device):
        super().__init__()
        self.backbones = torch.nn.ModuleList(
            [
                MLPBackbone(
                    obs_dim,
                    args.actor_hidden_dim,
                    use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
                    layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
                )
                for _ in range(int(getattr(args, "hg_ensemble_size", 5)))
            ]
        )
        self.heads = torch.nn.ModuleList(
            [
                GaussianPolicyHead(
                    self.backbones[idx].output_dim,
                    act_dim,
                    args.actor_hidden_dim,
                    args.init_scale,
                    use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
                    layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
                )
                for idx in range(int(getattr(args, "hg_ensemble_size", 5)))
            ]
        )
        self.to(device)

    def mean_actions(self, obs_flat: torch.Tensor) -> torch.Tensor:
        means = []
        for backbone, head in zip(self.backbones, self.heads):
            _, _, mean = head(backbone(obs_flat))
            means.append(mean)
        return torch.stack(means, dim=0)


def _prepare_run_dirs(args):
    if not getattr(args, "exp_name", None):
        env_tag = str(args.env_name).replace("-v0", "").replace("-", "_")
        args.exp_name = f"{env_tag}_hgdagger_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logs_root = Path("logs") / "hg_dagger"
    models_root = Path("models") / "hg_dagger"
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


def _info_teacher_mask(infos, *, actions: torch.Tensor, device: torch.device, num_envs: int) -> torch.Tensor:
    teacher_mask_raw = infos.get("teacher_intervened_mask") if isinstance(infos, dict) else None
    if teacher_mask_raw is not None:
        return _as_tensor_batch(teacher_mask_raw, device=device, dtype=torch.bool, num_envs=num_envs)
    applied_raw = infos.get("applied_actions", actions) if isinstance(infos, dict) else actions
    applied_actions = _as_tensor_batch(applied_raw, device=device, dtype=torch.float32, num_envs=num_envs)
    return torch.abs(applied_actions - actions).sum(dim=-1) > 1e-6


def _collect_prefill_demos(
    *,
    args,
    device: torch.device,
    buffer: ExpertDatasetBuffer,
    obs_dim: int,
    act_dim: int,
    training_logger: TrainingLogger,
    record_progress,
) -> None:
    target_episodes = int(max(0, getattr(args, "demo_prefill_episodes", 0)))
    if target_episodes <= 0:
        return
    demo_args = type(args)(**vars(args))
    demo_num_envs = int(getattr(args, "demo_prefill_num_envs", 0))
    if demo_num_envs > 0:
        demo_args.num_envs = demo_num_envs
    demo_args.use_intervention = True
    demo_args.intervention_mode = "agent_always"
    demo_args.intervention_episode_prob = 1.0
    demo_args.intervention_episode_prob_min = 1.0
    demo_args.intervention_episode_prob_decay_steps = 0
    demo_args.train_render_mode = "none"

    envs, _, _, _, _, _, initial_obs_raw = build_manip_environment(demo_args, device, record_progress)
    pending_states = [[] for _ in range(envs.num_envs)]
    pending_actions = [[] for _ in range(envs.num_envs)]
    obs = prepare_observation(initial_obs_raw, device=device, obs_mode="state", pixel_shape=None, flatten=True)
    zero_actions = torch.zeros(envs.num_envs, act_dim, dtype=torch.float32, device=device)
    total_episodes = 0
    total_steps = 0
    try:
        while total_episodes < target_episodes:
            next_obs_raw, _, dones, infos = envs.step(zero_actions)
            dones_t = _as_tensor_batch(dones, device=device, dtype=torch.bool, num_envs=envs.num_envs)
            applied_actions = _as_tensor_batch(
                infos.get("applied_actions", zero_actions),
                device=device,
                dtype=torch.float32,
                num_envs=envs.num_envs,
            )
            for env_idx in range(envs.num_envs):
                pending_states[env_idx].append(obs[env_idx].detach().cpu())
                pending_actions[env_idx].append(applied_actions[env_idx].detach().cpu())
                total_steps += 1
                if bool(dones_t[env_idx].item()):
                    if total_episodes < target_episodes:
                        if pending_states[env_idx]:
                            buffer.append(
                                torch.stack(pending_states[env_idx], dim=0),
                                torch.stack(pending_actions[env_idx], dim=0),
                            )
                        total_episodes += 1
                    pending_states[env_idx].clear()
                    pending_actions[env_idx].clear()
            obs = prepare_observation(next_obs_raw, device=device, obs_mode="state", pixel_shape=None, flatten=True)
    finally:
        envs.close()
        payload = {
            "DemoPrefill/env_steps": float(total_steps),
            "DemoPrefill/episodes": float(total_episodes),
            "DemoPrefill/episodes_target": float(target_episodes),
            "DemoPrefill/episodes_progress": float(total_episodes / max(1, target_episodes)),
            "DemoPrefill/target_buffer_size": float(buffer.size),
            "DemoPrefill/completed": 1.0,
        }
        training_logger.log_prefill(pseudo_step=-999999 + total_steps, payload=payload)
        record_progress(f"[HGDagger] prefill complete episodes={total_episodes} steps={total_steps} dataset={buffer.size}")


def _run_hg_eval(
    *,
    args,
    device: torch.device,
    eval_envs,
    ensemble: HGDaggerEnsemble,
    obs_normalizer,
) -> Dict[str, float]:
    num_eval_episodes = max(1, int(args.num_eval_episodes))
    obs_normalizer_was_training = obs_normalizer.training
    ensemble_was_training = ensemble.training
    obs_normalizer.eval()
    ensemble.eval()

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
            mean_actions = ensemble.mean_actions(norm_obs).mean(dim=0)
            mean_actions = _apply_binary_gripper_action(
                mean_actions,
                enabled=args.binary_gripper_actions,
                threshold=args.binary_gripper_threshold,
            )
        next_obs_raw, rewards, dones, infos = eval_envs.step(mean_actions.float())
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
    if ensemble_was_training:
        ensemble.train()
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


def _save_hg_checkpoint(
    *,
    run_model_dir: Path,
    tag: str,
    step_value: int,
    run_prefix: str,
    ensemble: HGDaggerEnsemble,
    obs_normalizer,
    args,
    tau_estimate: float | None,
) -> Path:
    save_path = run_model_dir / f"{run_prefix}_{tag}.pt"
    checkpoint = {
        "step": int(step_value),
        "ensemble_backbones": [module.state_dict() for module in ensemble.backbones],
        "ensemble_heads": [module.state_dict() for module in ensemble.heads],
        "obs_normalizer_state": obs_normalizer.state_dict() if hasattr(obs_normalizer, "state_dict") else None,
        "hg_tau_estimate": None if tau_estimate is None else float(tau_estimate),
        "args": vars(args),
    }
    torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
    return save_path


def _tau_from_doubts(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    cutoff = float(np.percentile(arr, float(percentile)))
    tail = arr[arr >= cutoff]
    if tail.size == 0:
        return None
    return float(tail.mean())


def run_hgdagger_ogbench_manip(args) -> None:
    args.algo_variant = "hg_dagger"
    args.cube_reward_mode = canonicalize_cube_reward_mode(str(getattr(args, "cube_reward_mode", "dense")))
    device = select_device(args)
    seed_everything(int(getattr(args, "seed", 42)))
    run_log_dir, run_model_dir, record_progress, progress_file = _prepare_run_dirs(args)
    print(f"HG-DAgger OGBench (manip) on {args.env_name} device={device}", flush=True)
    print(f"Log directory: {run_log_dir}", flush=True)
    print(f"Model directory: {run_model_dir}", flush=True)

    envs, wrappers, obs_normalizer, _, n_obs, n_act, initial_obs_raw = build_manip_environment(args, device, record_progress)
    eval_envs = build_manip_eval_environment(args, device)
    teacher_metrics = build_teacher_metrics(args, device)
    training_logger = TrainingLogger(args=args, record_progress=record_progress, teacher_metrics=teacher_metrics)
    ensemble = HGDaggerEnsemble(obs_dim=n_obs, act_dim=n_act, args=args, device=device)
    optimizer = torch.optim.AdamW(ensemble.parameters(), lr=args.actor_learning_rate, weight_decay=1e-5)
    expert_buffer = ExpertDatasetBuffer(capacity=int(max(1, args.demo_buffer_capacity)), obs_dim=n_obs, act_dim=n_act)
    _collect_prefill_demos(
        args=args,
        device=device,
        buffer=expert_buffer,
        obs_dim=n_obs,
        act_dim=n_act,
        training_logger=training_logger,
        record_progress=record_progress,
    )

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
    doubt_values: list[float] = []
    last_teacher_mask = torch.zeros(envs.num_envs, dtype=torch.bool, device=device)
    total_interventions = 0
    last_bc_loss = 0.0
    next_eval_step = int(args.eval_interval) if int(args.eval_interval) > 0 else None
    next_save_step = int(args.save_interval) if int(args.save_interval) > 0 else None
    run_prefix = args.env_name.replace("-", "_")

    try:
        while total_env_steps < int(args.total_timesteps):
            norm_obs = obs_normalizer(obs)
            with torch.no_grad():
                member_means = ensemble.mean_actions(norm_obs)
                mean_actions = member_means.mean(dim=0)
                doubt = torch.linalg.vector_norm(torch.var(member_means, dim=0, unbiased=False), dim=-1)
                mean_actions = _apply_binary_gripper_action(
                    mean_actions,
                    enabled=args.binary_gripper_actions,
                    threshold=args.binary_gripper_threshold,
                )
            next_obs_raw, rewards, dones, infos = envs.step(mean_actions.float())
            rewards_t = _as_tensor_batch(rewards, device=device, dtype=torch.float32, num_envs=envs.num_envs)
            dones_t = _as_tensor_batch(dones, device=device, dtype=torch.bool, num_envs=envs.num_envs)
            if reward_tracker is not None:
                rewards_t, _ = reward_tracker.compute(base_rewards=rewards_t, infos=infos, dones=dones_t)
            applied_actions = _as_tensor_batch(
                infos.get("applied_actions", mean_actions),
                device=device,
                dtype=torch.float32,
                num_envs=envs.num_envs,
            )
            teacher_mask = _info_teacher_mask(infos, actions=mean_actions, device=device, num_envs=envs.num_envs)
            takeover_starts = teacher_mask & (~last_teacher_mask)
            if bool(takeover_starts.any().item()):
                doubt_values.extend([float(v) for v in doubt[takeover_starts].detach().cpu().tolist()])
            last_teacher_mask = teacher_mask

            if bool(teacher_mask.any().item()):
                expert_buffer.append(obs[teacher_mask], applied_actions[teacher_mask])
            total_interventions += int(teacher_mask.to(torch.int64).sum().item())

            next_obs = prepare_observation(next_obs_raw, device=device, obs_mode="state", pixel_shape=None, flatten=True)

            cur_reward_sum += rewards_t
            cur_episode_length += 1
            done_ids = torch.nonzero(dones_t).flatten()
            if done_ids.numel() > 0:
                rewbuffer += cur_reward_sum[done_ids].tolist()
                lenbuffer += cur_episode_length[done_ids].tolist()
                cur_reward_sum[done_ids] = 0
                cur_episode_length[done_ids] = 0

            if expert_buffer.size >= int(max(1, args.batch_size)) and total_env_steps >= int(args.learning_starts):
                ensemble.train()
                total_loss = 0.0
                updates_done = 0
                for _ in range(int(max(1, args.num_updates))):
                    batch = expert_buffer.sample(int(max(1, args.batch_size)), device)
                    if batch is None:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    batch_obs = obs_normalizer(batch.states)
                    member_means = ensemble.mean_actions(batch_obs)
                    loss = 0.0
                    for member_idx in range(member_means.shape[0]):
                        loss = loss + F.mse_loss(member_means[member_idx], batch.actions)
                    loss = loss / float(member_means.shape[0])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ensemble.parameters(), max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"))
                    optimizer.step()
                    total_loss += float(loss.detach().cpu().item())
                    updates_done += 1
                if updates_done > 0:
                    last_bc_loss = total_loss / float(updates_done)

            total_env_steps += envs.num_envs
            iteration_idx += 1
            obs = next_obs

            if training_logger.should_log(total_env_steps):
                elapsed = max(1e-6, time.time() - start_time)
                tau_estimate = _tau_from_doubts(doubt_values, float(getattr(args, "hg_doubt_percentile", 75.0)))
                logs = {
                    "Perf/total_fps": int(total_env_steps / elapsed),
                    "Perf/collection_time_sec": elapsed,
                    "Perf/env_steps": float(total_env_steps),
                    "Perf/iterations": float(iteration_idx),
                    "Train/mean_reward": float(np.mean(rewbuffer[-100:])) if rewbuffer else 0.0,
                    "Train/mean_episode_length": float(np.mean(lenbuffer[-100:])) if lenbuffer else 0.0,
                    "Train/bc_loss": float(last_bc_loss),
                    "Train/expert_buffer_size": float(expert_buffer.size),
                    "Train/total_interventions": float(total_interventions),
                    "Train/hg_doubt_intervention_mean": float(np.mean(doubt_values)) if doubt_values else 0.0,
                    "Train/hg_doubt_tau_estimate": float(tau_estimate) if tau_estimate is not None else 0.0,
                    "Train/hg_doubt_count": float(len(doubt_values)),
                    "Train/hg_ensemble_size": float(getattr(args, "hg_ensemble_size", 5)),
                }
                if "log" in infos and isinstance(infos["log"], dict):
                    for key, value in infos["log"].items():
                        try:
                            logs[key] = float(value.float().mean().item())
                        except Exception:
                            pass
                console_line = (
                    f"[HGDagger] env_steps {total_env_steps}/{args.total_timesteps} | "
                    f"iter {iteration_idx} | fps {logs['Perf/total_fps']:.0f} | "
                    f"bc_loss {logs['Train/bc_loss']:.3f} | "
                    f"teacher_frac {logs.get('/Teacher/teacher_fraction_steps', 0.0):.2f} | "
                    f"dataset {expert_buffer.size}"
                )
                print(console_line, flush=True)
                record_progress(console_line)
                wandb_run = training_logger.ensure_wandb_run()
                if wandb_run is not None:
                    wandb_run.log(logs, step=total_env_steps)

            if next_eval_step is not None and total_env_steps >= next_eval_step:
                eval_metrics = _run_hg_eval(args=args, device=device, eval_envs=eval_envs, ensemble=ensemble, obs_normalizer=obs_normalizer)
                training_logger.log_eval(total_env_steps=total_env_steps, metrics=eval_metrics)
                next_eval_step += int(args.eval_interval)

            if next_save_step is not None and total_env_steps >= next_save_step:
                tau_estimate = _tau_from_doubts(doubt_values, float(getattr(args, "hg_doubt_percentile", 75.0)))
                save_path = _save_hg_checkpoint(
                    run_model_dir=run_model_dir,
                    tag=f"step{total_env_steps}",
                    step_value=total_env_steps,
                    run_prefix=run_prefix,
                    ensemble=ensemble,
                    obs_normalizer=obs_normalizer,
                    args=args,
                    tau_estimate=tau_estimate,
                )
                record_progress(f"[Checkpoint] saved {save_path}")
                next_save_step += int(args.save_interval)

        tau_estimate = _tau_from_doubts(doubt_values, float(getattr(args, "hg_doubt_percentile", 75.0)))
        final_path = _save_hg_checkpoint(
            run_model_dir=run_model_dir,
            tag="final",
            step_value=total_env_steps,
            run_prefix=run_prefix,
            ensemble=ensemble,
            obs_normalizer=obs_normalizer,
            args=args,
            tau_estimate=tau_estimate,
        )
        record_progress(f"[Checkpoint] saved {final_path}")
        print(
            f"================================================================================\n"
            f"[Done] HG-DAgger training complete env_steps={total_env_steps} iterations={iteration_idx} "
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

"""Manipulation-specific behavior cloning orchestration."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

if "fasttd3/fast_sac" not in sys.path:
    sys.path.append("fasttd3/fast_sac")

from .env_wrappers_manip import canonicalize_cube_reward_mode
from .fastsac_ogbench_env import _apply_binary_gripper_action, _as_tensor_batch, _build_cube_reward_tracker
from .fastsac_ogbench_manip_env import build_manip_environment, build_manip_eval_environment
from .fastsac_ogbench_setup import build_teacher_metrics, select_device
from .hgdagger_ogbench_manip_train import (
    ExpertDatasetBuffer,
    _collect_prefill_demos,
    _default_wandb_mode,
    _info_teacher_mask,
    _prepare_run_dirs,
)
from .logging import TrainingLogger
from .obs import prepare_observation
from .policy import GaussianPolicyHead, MLPBackbone


class BehaviorCloningPolicy(torch.nn.Module):
    def __init__(self, *, obs_dim: int, act_dim: int, args, device: torch.device):
        super().__init__()
        self.backbone = MLPBackbone(
            obs_dim,
            args.actor_hidden_dim,
            use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
            layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
        )
        self.head = GaussianPolicyHead(
            self.backbone.output_dim,
            act_dim,
            args.actor_hidden_dim,
            args.init_scale,
            use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
            layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
        )
        self.to(device)

    def mean_actions(self, obs_flat: torch.Tensor) -> torch.Tensor:
        _, _, mean = self.head(self.backbone(obs_flat))
        return mean


def _prepare_bc_run_dirs(args):
    if not getattr(args, "exp_name", None):
        env_tag = str(args.env_name).replace("-v0", "").replace("-", "_")
        args.exp_name = f"{env_tag}_bc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logs_root = Path("logs") / "behavior_cloning"
    models_root = Path("models") / "behavior_cloning"
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


def _run_bc_eval(
    *,
    args,
    device: torch.device,
    eval_envs,
    policy: BehaviorCloningPolicy,
    obs_normalizer,
) -> Dict[str, float]:
    num_eval_episodes = max(1, int(args.num_eval_episodes))
    obs_normalizer_was_training = obs_normalizer.training
    policy_was_training = policy.training
    obs_normalizer.eval()
    policy.eval()

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
            mean_actions = policy.mean_actions(norm_obs)
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
    if policy_was_training:
        policy.train()
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


def _save_bc_checkpoint(
    *,
    run_model_dir: Path,
    tag: str,
    step_value: int,
    run_prefix: str,
    policy: BehaviorCloningPolicy,
    obs_normalizer,
    args,
) -> Path:
    save_path = run_model_dir / f"{run_prefix}_{tag}.pt"
    checkpoint = {
        "step": int(step_value),
        "actor_backbone_state": policy.backbone.state_dict(),
        "actor_head_state": policy.head.state_dict(),
        "obs_normalizer_state": obs_normalizer.state_dict() if hasattr(obs_normalizer, "state_dict") else None,
        "args": vars(args),
    }
    torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
    return save_path


def run_bc_ogbench_manip(args) -> None:
    os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
    os.environ.setdefault("WANDB_MODE", _default_wandb_mode())
    os.environ.setdefault("WANDB_CONSOLE", "off")
    os.environ.setdefault("WANDB_SILENT", "true")

    args.algo_variant = "behavior_cloning"
    args.cube_reward_mode = canonicalize_cube_reward_mode(str(getattr(args, "cube_reward_mode", "dense")))
    device = select_device(args)
    run_log_dir, run_model_dir, record_progress, progress_file = _prepare_bc_run_dirs(args)
    print(f"Behavior Cloning OGBench (manip) on {args.env_name} device={device}", flush=True)
    print(f"Log directory: {run_log_dir}", flush=True)
    print(f"Model directory: {run_model_dir}", flush=True)

    teacher_metrics = build_teacher_metrics(args, device)
    training_logger = TrainingLogger(args=args, record_progress=record_progress, teacher_metrics=teacher_metrics)
    obs_dim_probe, act_dim_probe = 0, 0
    probe_args = type(args)(**vars(args))
    probe_args.num_envs = 1
    probe_envs, _, _, _, obs_dim_probe, act_dim_probe, _ = build_manip_environment(probe_args, device, record_progress)
    try:
        pass
    finally:
        probe_envs.close()

    expert_buffer = ExpertDatasetBuffer(
        capacity=int(max(1, args.demo_buffer_capacity)),
        obs_dim=obs_dim_probe,
        act_dim=act_dim_probe,
    )
    _collect_prefill_demos(
        args=args,
        device=device,
        buffer=expert_buffer,
        obs_dim=obs_dim_probe,
        act_dim=act_dim_probe,
        training_logger=training_logger,
        record_progress=record_progress,
    )

    train_args = type(args)(**vars(args))
    if not bool(getattr(args, "bc_collect_online_teacher_data", False)):
        train_args.use_intervention = False
        train_args.intervention_mode = "agent"
        train_args.intervention_episode_prob = 0.0
        train_args.intervention_episode_prob_min = 0.0
        train_args.intervention_episode_prob_decay_steps = 0
        record_progress("[BehaviorCloning] using demo-only baseline (no online teacher aggregation).")
    else:
        record_progress("[BehaviorCloning] online teacher aggregation enabled.")

    envs, _, obs_normalizer, _, n_obs, n_act, initial_obs_raw = build_manip_environment(train_args, device, record_progress)
    eval_envs = build_manip_eval_environment(train_args, device)
    policy = BehaviorCloningPolicy(obs_dim=n_obs, act_dim=n_act, args=args, device=device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.actor_learning_rate, weight_decay=1e-5)

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
    last_bc_loss = 0.0
    next_eval_step = int(args.eval_interval) if int(args.eval_interval) > 0 else None
    next_save_step = int(args.save_interval) if int(args.save_interval) > 0 else None
    run_prefix = args.env_name.replace("-", "_")

    try:
        while total_env_steps < int(args.total_timesteps):
            norm_obs = obs_normalizer(obs)
            with torch.no_grad():
                mean_actions = policy.mean_actions(norm_obs)
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
            if bool(getattr(args, "bc_collect_online_teacher_data", False)) and bool(teacher_mask.any().item()):
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
                policy.train()
                total_loss = 0.0
                updates_done = 0
                for _ in range(int(max(1, args.num_updates))):
                    batch = expert_buffer.sample(int(max(1, args.batch_size)), device)
                    if batch is None:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    batch_obs = obs_normalizer(batch.states)
                    pred_actions = policy.mean_actions(batch_obs)
                    loss = F.mse_loss(pred_actions, batch.actions)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(),
                        max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
                    )
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
                }
                if "log" in infos and isinstance(infos["log"], dict):
                    for key, value in infos["log"].items():
                        try:
                            logs[key] = float(value.float().mean().item())
                        except Exception:
                            pass
                console_line = (
                    f"[BehaviorCloning] env_steps {total_env_steps}/{args.total_timesteps} | "
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
                eval_metrics = _run_bc_eval(args=args, device=device, eval_envs=eval_envs, policy=policy, obs_normalizer=obs_normalizer)
                training_logger.log_eval(total_env_steps=total_env_steps, metrics=eval_metrics)
                next_eval_step += int(args.eval_interval)

            if next_save_step is not None and total_env_steps >= next_save_step:
                save_path = _save_bc_checkpoint(
                    run_model_dir=run_model_dir,
                    tag=f"step{total_env_steps}",
                    step_value=total_env_steps,
                    run_prefix=run_prefix,
                    policy=policy,
                    obs_normalizer=obs_normalizer,
                    args=args,
                )
                record_progress(f"[Checkpoint] saved {save_path}")
                next_save_step += int(args.save_interval)

        final_path = _save_bc_checkpoint(
            run_model_dir=run_model_dir,
            tag="final",
            step_value=total_env_steps,
            run_prefix=run_prefix,
            policy=policy,
            obs_normalizer=obs_normalizer,
            args=args,
        )
        record_progress(f"[Checkpoint] saved {final_path}")
        print(
            f"================================================================================\n"
            f"[Done] Behavior cloning training complete env_steps={total_env_steps} iterations={iteration_idx} "
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

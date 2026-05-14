"""Maze-specific HG-DAgger orchestration."""

from __future__ import annotations

import os
import sys
import time
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

from .fastsac_ogbench_env import _as_tensor_batch
from .fastsac_ogbench_maze_env import build_maze_environment, build_maze_eval_environment
from .fastsac_ogbench_setup import build_teacher_metrics, select_device
from .hgdagger_ogbench_manip_train import (
    ExpertDatasetBuffer,
    HGDaggerEnsemble,
    _prepare_run_dirs,
    _save_hg_checkpoint,
    _tau_from_doubts,
)
from .logging import TrainingLogger
from .maze_eval_artifacts import MazeEvalArtifactManager
from .obs import prepare_observation


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
    envs, _, _, _, _, act_dim, initial_obs_raw = build_maze_environment(demo_args, device, record_progress)
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
                    if total_episodes < target_episodes and pending_states[env_idx]:
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
        record_progress(f"[HGDaggerMaze] prefill complete episodes={total_episodes} steps={total_steps} dataset={buffer.size}")


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
    episodes_completed = 0

    while episodes_completed < num_eval_episodes:
        with torch.no_grad():
            norm_obs = obs_normalizer(obs)
            member_means = ensemble.mean_actions(norm_obs)
            mean_actions = member_means.mean(dim=0)
        next_obs_raw, rewards, dones, infos = eval_envs.step(mean_actions.float())
        rewards_t = _as_tensor_batch(rewards, device=device, dtype=torch.float32, num_envs=eval_envs.num_envs)
        dones_t = _as_tensor_batch(dones, device=device, dtype=torch.bool, num_envs=eval_envs.num_envs)
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
    return metrics


def _maybe_render_policy_map(
    *,
    args,
    generate_policy_map,
    checkpoint_path: Path,
    run_log_dir: Path,
    step_value: int,
    tag: str,
    training_logger: TrainingLogger,
    record_progress,
) -> None:
    if not bool(getattr(args, "viz_on_checkpoint", False)) or generate_policy_map is None:
        return
    try:
        viz_dir = run_log_dir / "policy_maps"
        png_path, _, _ = generate_policy_map(
            model_path=checkpoint_path,
            output_dir=viz_dir,
            tag=tag,
            env_name=str(args.env_name),
            grid_resolution=int(getattr(args, "viz_grid_resolution", 32)),
            quiver_stride=int(getattr(args, "viz_quiver_stride", 2)),
            device=str(getattr(args, "viz_device", "cpu")),
            seed=int(getattr(args, "viz_seed", 0)),
            cache_path=run_log_dir / "policy_map_goal.json",
        )
        record_progress(f"[Viz] generated {png_path}")
        wandb_run = training_logger.ensure_wandb_run()
        if wandb_run is not None:
            import wandb

            wandb_run.log({"viz/policy_map": wandb.Image(str(png_path), caption=tag)}, step=step_value)
    except Exception as exc:
        record_progress(f"[Viz] failed at step {step_value}: {exc}")


def run_hgdagger_ogbench_maze(args, generate_policy_map=None) -> None:
    args.algo_variant = "hg_dagger"
    device = select_device(args)
    run_log_dir, run_model_dir, record_progress, progress_file = _prepare_run_dirs(args)
    print(f"HG-DAgger OGBench (maze) on {args.env_name} device={device}", flush=True)
    print(f"Log directory: {run_log_dir}", flush=True)
    print(f"Model directory: {run_model_dir}", flush=True)

    envs, _, obs_normalizer, _, n_obs, n_act, initial_obs_raw = build_maze_environment(args, device, record_progress)
    eval_envs = build_maze_eval_environment(args, device)
    teacher_metrics = build_teacher_metrics(args, device)
    training_logger = TrainingLogger(args=args, record_progress=record_progress, teacher_metrics=teacher_metrics)
    maze_eval_artifacts = MazeEvalArtifactManager(
        args=args,
        output_dir=run_log_dir / "eval_trajectories",
        record_progress=record_progress,
    )
    ensemble = HGDaggerEnsemble(obs_dim=n_obs, act_dim=n_act, args=args, device=device)
    optimizer = torch.optim.AdamW(ensemble.parameters(), lr=args.actor_learning_rate, weight_decay=1e-5)
    expert_buffer = ExpertDatasetBuffer(capacity=int(max(1, args.demo_buffer_capacity)), obs_dim=n_obs, act_dim=n_act)
    _collect_prefill_demos(
        args=args,
        device=device,
        buffer=expert_buffer,
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
    first_success_env_steps: int | None = None
    first_success_wallclock_sec: float | None = None
    first_success_intervention_steps: int | None = None
    completed_episodes_total = 0
    success_episodes_total = 0

    try:
        while total_env_steps < int(args.total_timesteps):
            norm_obs = obs_normalizer(obs)
            with torch.no_grad():
                member_means = ensemble.mean_actions(norm_obs)
                mean_actions = member_means.mean(dim=0)
                doubt = torch.linalg.vector_norm(torch.var(member_means, dim=0, unbiased=False), dim=-1)
            next_obs_raw, rewards, dones, infos = envs.step(mean_actions.float())
            rewards_t = _as_tensor_batch(rewards, device=device, dtype=torch.float32, num_envs=envs.num_envs)
            dones_t = _as_tensor_batch(dones, device=device, dtype=torch.bool, num_envs=envs.num_envs)
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
                completed_episodes_total += int(done_ids.numel())
                goals = infos.get("goals_reached") or []
                success_this_step = sum(1 for idx in range(min(len(goals), int(done_ids.numel()))) if float(goals[idx]) > 0.0)
                if success_this_step > 0:
                    success_episodes_total += int(success_this_step)
                    if first_success_env_steps is None:
                        first_success_env_steps = int(total_env_steps + envs.num_envs)
                        first_success_wallclock_sec = float(time.time() - start_time)
                        first_success_intervention_steps = int(total_interventions)
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
                    loss = torch.tensor(0.0, device=device)
                    for member_idx in range(member_means.shape[0]):
                        loss = loss + F.mse_loss(member_means[member_idx], batch.actions)
                    loss = loss / float(member_means.shape[0])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        ensemble.parameters(),
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
                    "Train/cumulative_intervention_fraction": float(total_interventions) / max(1.0, float(total_env_steps)),
                    "Train/completed_episodes": float(completed_episodes_total),
                    "Train/successful_episodes": float(success_episodes_total),
                    "Train/hg_doubt_intervention_mean": float(np.mean(doubt_values)) if doubt_values else 0.0,
                    "Train/hg_doubt_tau_estimate": float(tau_estimate) if tau_estimate is not None else 0.0,
                    "Train/hg_doubt_count": float(len(doubt_values)),
                    "Train/hg_ensemble_size": float(getattr(args, "hg_ensemble_size", 5)),
                }
                if first_success_env_steps is not None:
                    logs["Train/first_success_env_steps"] = float(first_success_env_steps)
                if first_success_wallclock_sec is not None:
                    logs["Train/first_success_wallclock_sec"] = float(first_success_wallclock_sec)
                if first_success_intervention_steps is not None:
                    logs["Train/first_success_intervention_steps"] = float(first_success_intervention_steps)
                if "log" in infos and isinstance(infos["log"], dict):
                    for key, value in infos["log"].items():
                        try:
                            logs[key] = float(value.float().mean().item())
                        except Exception:
                            pass
                console_line = (
                    f"[HGDaggerMaze] env_steps {total_env_steps}/{args.total_timesteps} | "
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
                eval_metrics = _run_hg_eval(
                    args=args,
                    device=device,
                    eval_envs=eval_envs,
                    ensemble=ensemble,
                    obs_normalizer=obs_normalizer,
                )
                training_logger.log_eval(total_env_steps=total_env_steps, metrics=eval_metrics)
                if maze_eval_artifacts is not None:
                    def _policy_step_fn(obs_tensor: torch.Tensor) -> torch.Tensor:
                        norm_obs = obs_normalizer(obs_tensor)
                        with torch.no_grad():
                            return ensemble.mean_actions(norm_obs).mean(dim=0)

                    maze_eval_artifacts.maybe_save_trajectory_plot(
                        device=device,
                        tag=f"eval_step{total_env_steps}",
                        step_value=total_env_steps,
                        policy_step_fn=_policy_step_fn,
                        wandb_run=training_logger.wandb_run,
                    )
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
                _maybe_render_policy_map(
                    args=args,
                    generate_policy_map=generate_policy_map,
                    checkpoint_path=save_path,
                    run_log_dir=run_log_dir,
                    step_value=total_env_steps,
                    tag=f"step{total_env_steps}",
                    training_logger=training_logger,
                    record_progress=record_progress,
                )
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
        _maybe_render_policy_map(
            args=args,
            generate_policy_map=generate_policy_map,
            checkpoint_path=final_path,
            run_log_dir=run_log_dir,
            step_value=total_env_steps,
            tag="final",
            training_logger=training_logger,
            record_progress=record_progress,
        )
        print(
            f"================================================================================\n"
            f"[Done] HG-DAgger maze training complete env_steps={total_env_steps} iterations={iteration_idx} "
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

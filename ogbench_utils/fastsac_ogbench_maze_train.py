"""Maze-specific FastSAC OGBench training orchestration."""

from __future__ import annotations

import os
import sys

import torch


def _default_wandb_mode() -> str:
    explicit = str(os.environ.get("WANDB_MODE", "")).strip()
    if explicit:
        return explicit
    cluster_markers = ("SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_JOB_NODELIST")
    on_cluster = any(str(os.environ.get(key, "")).strip() for key in cluster_markers)
    return "offline" if on_cluster else "online"


# Ensure EGL is the default MuJoCo backend unless users override it explicitly.
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

# Default WANDB to online for local runs, but keep SLURM/cluster runs offline unless explicitly overridden.
os.environ.setdefault("WANDB_MODE", _default_wandb_mode())
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")

# Add FastSAC path for fast_sac / fast_sac_utils imports.
if "fasttd3/fast_sac" not in sys.path:
    sys.path.append("fasttd3/fast_sac")

from .fastsac_ogbench_loop import prefill_replay_buffer_with_demos, run_training_loop
from .fastsac_ogbench_maze_env import build_maze_environment, build_maze_eval_environment
from .fastsac_ogbench_setup import (
    build_teacher_metrics,
    build_updater_from_components,
    create_replay_buffer,
    ensure_experiment_name,
    initialize_amp,
    initialize_buffers,
    initialize_logging_components,
    initialize_models,
    prepare_run_dirs,
    select_device,
)


def run_fastsac_ogbench_maze(args, generate_policy_map=None) -> None:
    if getattr(args, "obs_mode", "state") != "state":
        raise ValueError(
            "FastSAC OGBench maze path no longer supports pixel observations. "
            "Use obs_mode='state' or a DRQ-v2 entrypoint for pixel training."
        )

    device = select_device(args)
    ensure_experiment_name(args)

    run_log_dir, run_model_dir, viz_output_dir, viz_cache_path, record_progress, progress_file = prepare_run_dirs(args)
    print(f"FastSAC OGBench (maze) on {args.env_name} device={device}")
    print(f"Log directory: {run_log_dir}")
    print(f"Model directory: {run_model_dir}")

    (
        envs,
        wrappers,
        obs_normalizer,
        critic_obs_normalizer,
        n_obs,
        n_act,
        initial_obs_raw,
    ) = build_maze_environment(args, device, record_progress)
    eval_envs = build_maze_eval_environment(args, device)

    model = initialize_models(args, device, n_obs, n_act, record_progress)
    amp = initialize_amp(args, device)

    if args.compile:
        model.actor_backbone = torch.compile(model.actor_backbone)
        model.actor_head = torch.compile(model.actor_head)
        if model.critic_backbone is not None and not args.arch_shared_trunk:
            model.critic_backbone = torch.compile(model.critic_backbone)
        model.critic_heads = torch.compile(model.critic_heads)
        model.critic_target_backbone = torch.compile(model.critic_target_backbone)
        model.critic_target_heads = torch.compile(model.critic_target_heads)
        obs_normalizer = torch.compile(obs_normalizer)
        critic_obs_normalizer = torch.compile(critic_obs_normalizer)

    buffers = initialize_buffers(args, device, n_obs, n_act, obs_normalizer)
    replay_buffer = create_replay_buffer(args, device, n_obs, n_act)
    demo_buffer = (
        create_replay_buffer(
            args,
            device,
            n_obs,
            n_act,
            buffer_size=args.demo_buffer_capacity,
            n_env_override=1,
        )
        if args.demo_buffer_enable
        else None
    )
    updater = build_updater_from_components(
        args=args,
        device=device,
        model=model,
        buffers=buffers,
        obs_normalizer=obs_normalizer,
        amp=amp,
    )

    teacher_metrics = build_teacher_metrics(args, device)
    logging_components = initialize_logging_components(
        args=args,
        record_progress=record_progress,
        run_model_dir=run_model_dir,
        viz_output_dir=viz_output_dir,
        viz_cache_path=viz_cache_path,
        teacher_metrics=teacher_metrics,
        generate_policy_map=generate_policy_map,
    )

    prefill_replay_buffer_with_demos(
        args=args,
        device=device,
        env_family="maze",
        replay_buffer=replay_buffer,
        demo_buffer=demo_buffer,
        obs_normalizer=obs_normalizer,
        model=model,
        record_progress=record_progress,
    )

    try:
        run_training_loop(
            args=args,
            device=device,
            envs=envs,
            eval_envs=eval_envs,
            wrappers=wrappers,
            obs_normalizer=obs_normalizer,
            critic_obs_normalizer=critic_obs_normalizer,
            model=model,
            buffers=buffers,
            amp=amp,
            training_logger=logging_components.training_logger,
            teacher_metrics=logging_components.teacher_metrics,
            checkpoint_manager=logging_components.checkpoint_manager,
            record_progress=record_progress,
            replay_buffer=replay_buffer,
            demo_buffer=demo_buffer,
            updater=updater,
            current_env_name=args.env_name,
            initial_obs_raw=initial_obs_raw,
        )
    finally:
        try:
            progress_file.close()
        except Exception:
            pass
        try:
            envs.close()
        except Exception:
            pass
        try:
            eval_envs.close()
        except Exception:
            pass

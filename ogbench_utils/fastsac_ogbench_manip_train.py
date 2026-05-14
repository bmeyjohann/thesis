"""Manipulation-specific FastSAC OGBench training orchestration."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

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

from .env_wrappers_manip import canonicalize_cube_reward_mode
from .fastsac_ogbench_loop import prefill_replay_buffer_with_demos, run_training_loop
from .fastsac_ogbench_manip_env import build_manip_environment, build_manip_eval_environment
from .manip_dataset_io import DEFAULT_MANIP_DATASET_DIR, extend_buffer_from_dataset, find_latest_transition_dataset
from .repro import seed_everything
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
from .vr_mapping_web import create_vr_source
from .vr_teleop import VRTeleopInterface, VRManipActionMapper, VRManipMappingConfig, apply_vr_mapping_profile
from .vr_teleop import wait_for_fresh_gate_press


def _maybe_build_train_vr_teleop(args) -> tuple[Optional[VRTeleopInterface], Optional[object]]:
    if not bool(getattr(args, "use_intervention", False)):
        return None, None
    if str(getattr(args, "intervention_mode", "")).lower() != "human":
        return None, None
    if str(getattr(args, "human_input_device", "keyboard")).lower() != "vr":
        return None, None

    source = create_vr_source(
        vr_mode=str(getattr(args, "vr_mode", "connect")),
        vr_host=str(getattr(args, "vr_host", "")),
        vr_port=int(getattr(args, "vr_port", 0)),
        cache_path=Path(str(getattr(args, "vr_cache_path", ""))),
        reconnect_seconds=float(getattr(args, "vr_reconnect_seconds", 2.0)),
    )
    print(source.banner_text(), flush=True)

    mapping_path = Path(str(getattr(args, "vr_mapping_path", "")))
    mapping_config = VRManipMappingConfig(
        hand=str(getattr(args, "vr_hand", "right")),
        require_gate=bool(getattr(args, "vr_require_gate", True)),
        gate_button=str(getattr(args, "vr_gate_button", "grip")),
        gripper_mirror_toggle_button=str(getattr(args, "vr_gripper_mirror_toggle_button", "none")),
    )
    if bool(getattr(args, "vr_use_saved_mapping", True)):
        mapping_config, loaded = apply_vr_mapping_profile(mapping_config, mapping_path)
        if loaded:
            print(f"[VRTrain] loaded vr mapping profile from {mapping_path}", flush=True)
    teleop = VRTeleopInterface(
        source,
        VRManipActionMapper(action_dim=(4 if bool(getattr(args, "disable_rotation", False)) else 5), config=mapping_config),
        return_none_when_idle=True,
        idle_threshold=1e-6,
        mapping_path=mapping_path,
    )
    if float(getattr(args, "human_intervention_threshold", 0.1)) == 0.1:
        args.human_intervention_threshold = -1e-6
        print("[VRTrain] human_intervention_threshold default overridden to -1e-6 for VR teleop gating", flush=True)
    print(
        f"[VRTrain] enabled persistent VR human intervention "
        f"transport={getattr(args, 'vr_mode', 'connect')} mapping={mapping_path}",
        flush=True,
    )
    return teleop, source


def run_fastsac_ogbench_manip(args, generate_policy_map=None) -> None:
    algo_variant = str(getattr(args, "algo_variant", "own") or "own").strip().lower()
    if algo_variant not in {"own", "pvp", "eil"}:
        raise ValueError(f"Unsupported manip algo_variant={algo_variant!r}.")
    args.algo_variant = algo_variant
    if algo_variant == "pvp":
        if not bool(getattr(args, "demo_buffer_enable", False)):
            raise ValueError("--algo_variant pvp requires --demo_buffer_enable (used as the human/intervention buffer).")
        if float(getattr(args, "demo_sample_ratio", 0.0)) <= 0.0:
            raise ValueError("--algo_variant pvp requires --demo_sample_ratio > 0 for balanced novice/human sampling.")
        if bool(getattr(args, "store_intervened_in_demo_buffer", False)):
            raise ValueError("--store_intervened_in_demo_buffer is redundant with --algo_variant pvp buffer routing.")
        args.pref_buffer_enable = False
        args.pref_rank_weight = 0.0
        args.pref_sample_ratio = 0.0
    elif algo_variant == "eil":
        args.pref_buffer_enable = False
        args.pref_rank_weight = 0.0
        args.pref_sample_ratio = 0.0
        if float(getattr(args, "fixed_alpha", -1.0)) < 0.0:
            args.fixed_alpha = 0.0
        args.alpha_init = 0.0
        args.alpha_min = 0.0
        args.alpha_max = 0.0
        args.alpha_freeze_steps = 0
    canonical_reward_mode = canonicalize_cube_reward_mode(str(getattr(args, "cube_reward_mode", "dense")))
    args.cube_reward_mode = canonical_reward_mode
    if str(getattr(args, "train_render_mode", "none")).lower() == "human":
        if os.environ.get("MUJOCO_GL", "").lower() in {"", "egl"}:
            os.environ["MUJOCO_GL"] = "glfw"
            print("[Render] train_render_mode=human -> forcing MUJOCO_GL=glfw")
    if getattr(args, "obs_mode", "state") != "state":
        raise ValueError(
            "FastSAC OGBench manipulation path no longer supports pixel observations. "
            "Use obs_mode='state' or a DRQ-v2 entrypoint for pixel training."
        )

    device = select_device(args)
    seed_everything(int(getattr(args, "seed", 42)))
    ensure_experiment_name(args)

    run_log_dir, run_model_dir, viz_output_dir, viz_cache_path, record_progress, progress_file = prepare_run_dirs(args)
    print(f"FastSAC OGBench (manip) on {args.env_name} device={device}")
    print(f"Log directory: {run_log_dir}")
    print(f"Model directory: {run_model_dir}")

    teleop_interface, vr_source = _maybe_build_train_vr_teleop(args)

    (
        envs,
        wrappers,
        obs_normalizer,
        critic_obs_normalizer,
        n_obs,
        n_act,
        initial_obs_raw,
    ) = build_manip_environment(args, device, record_progress, teleop_interface=teleop_interface)
    eval_envs = build_manip_eval_environment(args, device)

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
    replay_buffer = create_replay_buffer(
        args,
        device,
        n_obs,
        n_act,
        n_env_override=(1 if algo_variant == "pvp" else None),
    )
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

    demo_dataset_path = str(getattr(args, "demo_dataset_path", "") or "").strip()
    if not demo_dataset_path and bool(getattr(args, "demo_dataset_auto_load", False)):
        dataset_root = str(getattr(args, "demo_dataset_dir", "") or "").strip() or str(DEFAULT_MANIP_DATASET_DIR)
        found = find_latest_transition_dataset(env_name=args.env_name, dataset_dir=dataset_root)
        if found is not None:
            demo_dataset_path = str(found)
            print(f"[Dataset] auto-selected manipulation dataset {demo_dataset_path}", flush=True)
    if demo_dataset_path:
        dataset_target = str(getattr(args, "demo_dataset_target", "demo")).strip().lower()
        if dataset_target == "demo":
            if demo_buffer is None:
                raise ValueError("--demo_dataset_target demo requires --demo_buffer_enable.")
            target_buffer = demo_buffer
        else:
            target_buffer = replay_buffer
        load_stats = extend_buffer_from_dataset(
            buffer=target_buffer,
            dataset_path=demo_dataset_path,
            device=device,
            max_rows=int(getattr(args, "demo_dataset_max_rows", 0)),
            expected_obs_dim=n_obs,
            expected_act_dim=n_act,
        )
        print(
            f"[Dataset] loaded {load_stats['rows_loaded']} rows from {load_stats['path']} "
            f"into {dataset_target} buffer",
            flush=True,
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
        env_family="manip",
        replay_buffer=replay_buffer,
        demo_buffer=demo_buffer,
        obs_normalizer=obs_normalizer,
        model=model,
        record_progress=record_progress,
        training_logger=logging_components.training_logger,
    )

    if bool(getattr(args, "wait_for_human_start", False)) and teleop_interface is not None:
        wait_for_fresh_gate_press(
            teleop_interface,
            label="VRTrainStart",
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
        if vr_source is not None:
            try:
                vr_source.close()
            except Exception:
                pass

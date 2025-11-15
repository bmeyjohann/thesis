#!/usr/bin/env python3
"""
FastSAC training for OGBench environments using VectorizedOGBenchEnv and teacher interventions.

This entry-point keeps the CLI orchestration minimal and delegates the heavy lifting to helpers in
``ogbench_utils`` so individual portions of the training pipeline are easier to follow and reuse.
"""

import os
import sys
import json
import copy
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tensordict import TensorDict

# Ensure EGL is the default MuJoCo backend unless users override it explicitly.
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

# Defer WANDB mode selection until after args are parsed; default to offline unless explicitly enabled.
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")

# Add FastSAC path
sys.path.append('fasttd3/fast_sac')

from fast_sac import Actor, Critic  # noqa: E402
from fast_sac_utils import (  # noqa: E402
    EmpiricalNormalization,
    SimpleReplayBuffer,
)

import ogbench  # noqa: F401  # Registers environments.
from fasttd3.fast_sac.environments.ogbench_env import OGBenchVecEnvAdapter  # noqa: E402

from ogbench_utils import (  # noqa: E402
    CriticEnsemble,
    GaussianPolicyHead,
    IdentityNormalizer,
    MLPBackbone,
    PixelBackbone,
    build_ogbench_wrapper,
    build_train_parser,
    prepare_observation,
    reshape_observation,
    infer_pixel_shape,
    CounterfactualBuffer,
    PreferencePairBuffer,
    PreferenceTDBuffer,
    FastSACUpdater,
    TeacherMetricsAccumulator,
    TrainingLogger,
    CheckpointManager,
    maybe_switch_env,
)

TOOLS_PATH = Path(__file__).resolve().parent / "tools"
if TOOLS_PATH.exists():
    sys.path.append(str(TOOLS_PATH))
try:
    from visualize_policy_map import generate_policy_map  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - optional dependency for viz
    generate_policy_map = None


def parse_args():
    return build_train_parser().parse_args()


def make_wrappers(args):
    reward_switch = args.reward_switch_after_steps // max(1, args.num_envs)
    intervention_mode = args.intervention_mode if args.use_intervention else "none"
    wrapper = build_ogbench_wrapper(
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
        hard_block_lethal=args.hard_block_lethal,
        intervention_enable_after_steps=args.intervention_enable_after_steps,
    )
    return [wrapper]


@dataclass
class ModelComponents:
    actor_backbone: nn.Module
    actor_head: nn.Module
    critic_backbone: Optional[nn.Module]
    critic_heads: nn.Module
    critic_target_backbone: nn.Module
    critic_target_heads: nn.Module
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    trunk_optimizer: Optional[optim.Optimizer]
    actor_params: list
    critic_params: list
    trunk_params: Optional[list]
    log_alpha: torch.Tensor
    alpha_optimizer: optim.Optimizer
    target_entropy: float
    critic_feature_backbone: nn.Module
    initial_shared_backbone_state: Optional[dict]
    initial_critic_backbone_state: Optional[dict]
    initial_critic_heads_state: dict
    initial_target_backbone_state: dict
    initial_target_heads_state: dict


@dataclass
class BufferComponents:
    cf_buffer: Optional[CounterfactualBuffer]
    pref_buffer: Optional[PreferencePairBuffer]
    pref_td_buffer: Optional[PreferenceTDBuffer]


@dataclass
class AMPComponents:
    enabled: bool
    device_type: str
    dtype: torch.dtype
    scaler: GradScaler


@dataclass
class LoggingComponents:
    teacher_metrics: TeacherMetricsAccumulator
    training_logger: TrainingLogger
    checkpoint_manager: CheckpointManager


def select_device(args) -> torch.device:
    if args.device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(args.device)


def parse_pixel_backbone_args(args) -> None:
    default_channels = (32, 64, 64)
    try:
        parsed_channels = tuple(int(ch.strip()) for ch in args.pixel_conv_channels.split(',') if ch.strip())
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Invalid --pixel_conv_channels value: {args.pixel_conv_channels}") from exc
    if not parsed_channels:
        parsed_channels = default_channels
    setattr(args, 'pixel_conv_channels_parsed', parsed_channels)

    def _parse_int_list(raw: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
        try:
            values = tuple(int(x.strip()) for x in raw.split(',') if x.strip())
            return values or default
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Invalid integer list value: {raw}") from exc

    kernel_default = (8, 4, 3)
    stride_default = (4, 2, 1)
    setattr(args, 'pixel_kernel_sizes_parsed', _parse_int_list(args.pixel_kernel_sizes, kernel_default))
    setattr(args, 'pixel_strides_parsed', _parse_int_list(args.pixel_strides, stride_default))
    setattr(args, 'pixel_final_pool_parsed', int(max(0, args.pixel_final_pool)))


def ensure_experiment_name(args) -> None:
    if args.exp_name:
        return
    default_buffer_size = 1_000_000
    env_tag = args.env_name.replace('-v0', '').replace('-', '_')
    components = [env_tag, args.reward_type]
    if args.use_intervention:
        components.append(f"teacher_tol{int(args.tolerance_value)}")
        if args.intervention_enable_after_steps > 0:
            components.append(f"warmup{args.intervention_enable_after_steps}")
    else:
        components.append('student')
    if args.buffer_size != default_buffer_size:
        components.append(f"buf{args.buffer_size}")
    if args.store_denied_actions:
        components.append('denied')
    components.append(f"nenv{args.num_envs}")
    components.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
    args.exp_name = '_'.join(components)


def prepare_run_dirs(args) -> Tuple[Path, Path, Path, Path, callable, Any]:
    logs_root = Path('logs') / 'fast_sac'
    models_root = Path('models') / 'fast_sac'
    logs_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    run_log_dir = logs_root / args.exp_name
    run_model_dir = models_root / args.exp_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)
    viz_output_dir = run_log_dir / 'policy_maps'
    viz_cache_path = run_log_dir / 'policy_map_goal.json'

    log_file_path = run_log_dir / 'training.log'
    progress_file = open(log_file_path, 'a', encoding='utf-8')
    progress_file.write(f"# logging started {datetime.now().isoformat()}\n")
    progress_file.flush()

    def record_progress(message: str) -> None:
        progress_file.write(f"{datetime.now().isoformat()} {message}\n")
        progress_file.flush()

    config_path = run_log_dir / 'args.json'
    with open(config_path, 'w', encoding='utf-8') as cfg_file:
        json.dump(vars(args), cfg_file, indent=2)
    print(f"Saved run config: {config_path}")

    return run_log_dir, run_model_dir, viz_output_dir, viz_cache_path, record_progress, progress_file


def build_environment(
    args,
    device: torch.device,
    record_progress,
) -> Tuple[
    OGBenchVecEnvAdapter,
    list,
    Optional[Tuple[int, int, int]],
    Any,
    Any,
    int,
    int,
    torch.Tensor | None,
]:
    wrappers = make_wrappers(args)
    env_kwargs: Dict[str, Any] = {}
    if args.obs_mode == 'pixels':
        env_kwargs['render_mode'] = 'rgb_array'
        if args.pixel_width:
            env_kwargs['width'] = int(args.pixel_width)
        if args.pixel_height:
            env_kwargs['height'] = int(args.pixel_height)
        if args.pixel_camera:
            env_kwargs['camera_name'] = args.pixel_camera
        else:
            env_kwargs['pixel_camera_mode'] = args.pixel_camera_mode
            env_kwargs['pixel_local_view_size'] = args.pixel_local_view_size
            env_kwargs['pixel_local_camera_height'] = args.pixel_local_camera_height
            env_kwargs['pixel_first_person_distance'] = args.pixel_first_person_distance
            env_kwargs['pixel_first_person_height'] = args.pixel_first_person_height
            env_kwargs['pixel_first_person_lookahead'] = args.pixel_first_person_lookahead
            env_kwargs['pixel_first_person_pitch'] = args.pixel_first_person_pitch
    record_progress("[Init] constructing vector env adapter")
    envs = OGBenchVecEnvAdapter(
        env_name=args.env_name,
        num_envs=args.num_envs,
        device=device,
        wrappers=wrappers,
        clip_actions=1.0,
        **env_kwargs,
    )
    record_progress("[Init] env adapter constructed")
    print("[Init] Env adapter constructed", flush=True)
    initial_obs_raw = envs.reset()
    record_progress("[Init] env reset for initial observation sample")

    pixel_shape = None
    if args.obs_mode == 'pixels':
        pixel_shape = infer_pixel_shape(initial_obs_raw)
        record_progress(
            "[Init] pixel obs shape=%s, conv_channels=%s, kernel_sizes=%s, strides=%s, pool=%s"
            % (
                pixel_shape,
                args.pixel_conv_channels_parsed,
                args.pixel_kernel_sizes_parsed,
                args.pixel_strides_parsed,
                args.pixel_final_pool_parsed if args.pixel_final_pool_parsed > 0 else None,
            )
        )
        obs_normalizer = IdentityNormalizer().to(device)
        critic_obs_normalizer = IdentityNormalizer().to(device)
        obs_dim = int(np.prod(pixel_shape)) if pixel_shape is not None else envs.num_obs
    else:
        obs_dim = envs.num_obs
        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)

    return (
        envs,
        wrappers,
        pixel_shape,
        obs_normalizer,
        critic_obs_normalizer,
        obs_dim,
        envs.num_actions,
        initial_obs_raw,
    )


def initialize_models(
    args,
    device: torch.device,
    n_obs: int,
    n_act: int,
    pixel_shape: Optional[Tuple[int, int, int]],
    record_progress,
) -> ModelComponents:
    def build_backbone(mode: str, hidden_dim: int) -> nn.Module:
        if mode == 'pixels':
            if pixel_shape is None:
                raise ValueError("pixel_shape must be provided when obs_mode='pixels'")
            c, h, w = pixel_shape
            return PixelBackbone(
                (c, h, w),
                hidden_dim,
                conv_channels=args.pixel_conv_channels_parsed,
                kernel_sizes=args.pixel_kernel_sizes_parsed,
                strides=args.pixel_strides_parsed,
                final_pool=args.pixel_final_pool_parsed if args.pixel_final_pool_parsed > 0 else None,
            ).to(device)
        return MLPBackbone(n_obs, hidden_dim).to(device)

    if args.arch_shared_trunk:
        shared_backbone = build_backbone(args.obs_mode, args.shared_hidden_dim)
        if args.obs_mode == 'pixels' and hasattr(shared_backbone, 'fc'):
            record_progress(f'[Init] shared trunk fc weight {shared_backbone.fc[0].weight.shape}')
        actor_backbone = shared_backbone
        actor_head = GaussianPolicyHead(shared_backbone.output_dim, n_act, args.actor_hidden_dim, args.init_scale).to(device)
        critic_backbone = shared_backbone
        critic_heads = CriticEnsemble(shared_backbone.output_dim, n_act, args.critic_hidden_dim, args.num_critics).to(device)
        trunk_params = list(shared_backbone.parameters())
        trunk_optimizer = optim.Adam(trunk_params, lr=args.critic_learning_rate)
        actor_params = list(actor_head.parameters())
        critic_params = list(critic_heads.parameters())
        initial_shared_backbone_state = copy.deepcopy(shared_backbone.state_dict())
        initial_critic_backbone_state = None
    else:
        actor_backbone = build_backbone(args.obs_mode, args.actor_hidden_dim)
        actor_head = GaussianPolicyHead(actor_backbone.output_dim, n_act, args.actor_hidden_dim, args.init_scale).to(device)
        critic_backbone = build_backbone(args.obs_mode, args.critic_hidden_dim)
        critic_heads = CriticEnsemble(critic_backbone.output_dim, n_act, args.critic_hidden_dim, args.num_critics).to(device)
        trunk_params = None
        trunk_optimizer = None
        actor_params = list(actor_backbone.parameters()) + list(actor_head.parameters())
        critic_params = list(critic_backbone.parameters()) + list(critic_heads.parameters())
        initial_shared_backbone_state = None
        initial_critic_backbone_state = copy.deepcopy(critic_backbone.state_dict())

    critic_feature_backbone = actor_backbone if args.arch_shared_trunk else critic_backbone
    critic_target_backbone = copy.deepcopy(critic_backbone)
    critic_target_heads = copy.deepcopy(critic_heads)

    critic_optimizer = optim.AdamW(critic_params, lr=args.critic_learning_rate, weight_decay=1e-5)
    actor_optimizer = optim.AdamW(actor_params, lr=args.actor_learning_rate, weight_decay=1e-5)

    initial_critic_heads_state = copy.deepcopy(critic_heads.state_dict())
    initial_target_backbone_state = copy.deepcopy(critic_target_backbone.state_dict())
    initial_target_heads_state = copy.deepcopy(critic_target_heads.state_dict())

    target_entropy = -float(n_act)
    log_alpha = torch.ones(1, requires_grad=True, device=device)
    log_alpha.data.copy_(torch.tensor([np.log(0.001)], device=device))
    alpha_optimizer = optim.Adam([log_alpha], lr=args.critic_learning_rate)

    return ModelComponents(
        actor_backbone=actor_backbone,
        actor_head=actor_head,
        critic_backbone=critic_backbone,
        critic_heads=critic_heads,
        critic_target_backbone=critic_target_backbone,
        critic_target_heads=critic_target_heads,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        trunk_optimizer=trunk_optimizer,
        actor_params=actor_params,
        critic_params=critic_params,
        trunk_params=trunk_params,
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
        target_entropy=target_entropy,
        critic_feature_backbone=critic_feature_backbone,
        initial_shared_backbone_state=initial_shared_backbone_state,
        initial_critic_backbone_state=initial_critic_backbone_state,
        initial_critic_heads_state=initial_critic_heads_state,
        initial_target_backbone_state=initial_target_backbone_state,
        initial_target_heads_state=initial_target_heads_state,
    )


def initialize_buffers(
    args,
    device: torch.device,
    n_obs: int,
    n_act: int,
    obs_normalizer,
) -> BufferComponents:
    def normalize_obs(x):
        return obs_normalizer(x)

    cf_buffer = (
        CounterfactualBuffer(
            capacity=int(max(1, args.cf_capacity)),
            obs_dim=n_obs,
            act_dim=n_act,
            device=device,
            normalize_fn=normalize_obs,
        )
        if args.cf_buffer_enable
        else None
    )

    pref_buffer = (
        PreferencePairBuffer(
            capacity=int(max(1, args.pref_capacity)),
            obs_dim=n_obs,
            act_dim=n_act,
            device=device,
            normalize_fn=normalize_obs,
        )
        if args.pref_buffer_enable
        else None
    )

    pref_td_buffer = (
        PreferenceTDBuffer(
            capacity=int(max(1, args.pref_td_capacity)),
            obs_dim=n_obs,
            act_dim=n_act,
            device=device,
            normalize_fn=normalize_obs,
        )
        if args.pref_td_buffer_enable
        else None
    )

    return BufferComponents(cf_buffer=cf_buffer, pref_buffer=pref_buffer, pref_td_buffer=pref_td_buffer)


def initialize_amp(args, device: torch.device) -> AMPComponents:
    amp_enabled = args.amp and device.type == 'cuda'
    amp_device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    return AMPComponents(enabled=amp_enabled, device_type=amp_device_type, dtype=amp_dtype, scaler=scaler)


def build_teacher_metrics(args, device: torch.device) -> TeacherMetricsAccumulator:
    hist_edges_tensor = None
    hist_labels: list[str] = []
    if args.disagreement_hist_edges:
        try:
            parsed_edges = [float(edge.strip()) for edge in args.disagreement_hist_edges.split(',') if edge.strip()]
            parsed_edges = sorted(edge for edge in parsed_edges if edge > 0.0)
        except Exception:
            parsed_edges = []
        if parsed_edges:
            hist_edges_tensor = torch.tensor(parsed_edges, dtype=torch.float32, device=device)
            num_bins = hist_edges_tensor.numel() + 1
            for idx in range(num_bins):
                if idx == 0:
                    hist_labels.append(f"<= {parsed_edges[0]:.2f}")
                elif idx == num_bins - 1:
                    hist_labels.append(f">= {parsed_edges[-1]:.2f}")
                else:
                    hist_labels.append(f"({parsed_edges[idx-1]:.2f}, {parsed_edges[idx]:.2f}]")

    if args.disagreement_thresholds:
        try:
            thresh_vals = [float(x.strip()) for x in args.disagreement_thresholds.split(',') if x.strip()]
            thresh_vals = sorted(t for t in thresh_vals if t > 0.0)
        except Exception:
            thresh_vals = []
    else:
        thresh_vals = []
    thresh_tensor = torch.tensor(thresh_vals, dtype=torch.float32, device=device) if thresh_vals else None
    return TeacherMetricsAccumulator(
        device=device,
        hist_edges_tensor=hist_edges_tensor,
        hist_labels=hist_labels,
        thresh_tensor=thresh_tensor,
        thresh_values=thresh_vals,
    )


def initialize_logging_components(
    args,
    record_progress,
    run_model_dir: Path,
    viz_output_dir: Path,
    viz_cache_path: Path,
    teacher_metrics: TeacherMetricsAccumulator,
) -> LoggingComponents:
    training_logger = TrainingLogger(
        args=args,
        record_progress=record_progress,
        teacher_metrics=teacher_metrics,
    )
    checkpoint_manager = CheckpointManager(
        run_model_dir=run_model_dir,
        viz_output_dir=viz_output_dir,
        viz_cache_path=viz_cache_path,
        record_progress=record_progress,
        args=args,
        generate_policy_map=generate_policy_map,
    )
    return LoggingComponents(
        teacher_metrics=teacher_metrics,
        training_logger=training_logger,
        checkpoint_manager=checkpoint_manager,
    )


def create_replay_buffer(
    args,
    device: torch.device,
    n_obs: int,
    n_act: int,
    pixel_shape: Optional[Tuple[int, int, int]],
) -> SimpleReplayBuffer:
    return SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_obs,
        asymmetric_obs=False,
        playground_mode=False,
        n_steps=1,
        gamma=args.gamma,
        device=device,
        pixel_shape=pixel_shape if args.obs_mode == 'pixels' else None,
    )


def build_updater_from_components(
    args,
    device: torch.device,
    model: ModelComponents,
    buffers: BufferComponents,
    obs_normalizer,
    pixel_shape: Optional[Tuple[int, int, int]],
    amp: AMPComponents,
) -> FastSACUpdater:
    return FastSACUpdater(
        args=args,
        actor_backbone=model.actor_backbone,
        actor_head=model.actor_head,
        critic_backbone=model.critic_backbone,
        critic_heads=model.critic_heads,
        critic_target_backbone=model.critic_target_backbone,
        critic_target_heads=model.critic_target_heads,
        actor_optimizer=model.actor_optimizer,
        critic_optimizer=model.critic_optimizer,
        trunk_optimizer=model.trunk_optimizer,
        alpha_optimizer=model.alpha_optimizer,
        actor_params=model.actor_params,
        critic_params=model.critic_params,
        trunk_params=model.trunk_params,
        log_alpha=model.log_alpha,
        reshape_obs_fn=lambda x: reshape_observation(x, obs_mode=args.obs_mode, pixel_shape=pixel_shape),
        normalize_obs_fn=obs_normalizer,
        device=device,
        amp_enabled=amp.enabled,
        amp_dtype=amp.dtype,
        amp_device_type=amp.device_type,
        scaler=amp.scaler,
        target_entropy=model.target_entropy,
        cf_buffer=buffers.cf_buffer,
        pref_buffer=buffers.pref_buffer,
        pref_td_buffer=buffers.pref_td_buffer,
        pixel_shape=pixel_shape,
        pixel_random_shift_pad=args.pixel_random_shift_pad,
    )


def run_training_loop(
    args,
    device: torch.device,
    envs: OGBenchVecEnvAdapter,
    wrappers,
    pixel_shape,
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
    updater: FastSACUpdater,
    current_env_name: str,
    initial_obs_raw: torch.Tensor | None,
):
    rb = replay_buffer
    cf_buffer = buffers.cf_buffer
    pref_buffer = buffers.pref_buffer
    pref_td_buffer = buffers.pref_td_buffer

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
        if args.obs_mode == 'pixels':
            pixel_shape_local = infer_pixel_shape(obs_raw)
            if pixel_shape != pixel_shape_local:
                pixel_shape = pixel_shape_local
                record_progress(f'[Pixels] inferred pixel shape {pixel_shape}')
        obs = prepare_observation(
            obs_raw,
            device=device,
            obs_mode=args.obs_mode,
            pixel_shape=pixel_shape,
            flatten=True,
        )
        n_obs = obs.shape[1]
        record_progress("[Init] envs.reset() returned; entering loop")

        if args.debug_pixel_dump:
            raw_tensor = obs_raw if torch.is_tensor(obs_raw) else torch.as_tensor(obs_raw)
            raw_stats = raw_tensor.float()
            norm_tensor = obs.float()
            record_progress(
                "[Debug] raw_obs shape=%s mean=%.6f std=%.6f min=%.6f max=%.6f"
                % (
                    tuple(raw_tensor.shape),
                    raw_stats.mean().item(),
                    raw_stats.std().item(),
                    raw_stats.min().item(),
                    raw_stats.max().item(),
                )
            )
            record_progress(
                "[Debug] normalized_obs mean=%.6f std=%.6f min=%.6f max=%.6f"
                % (
                    norm_tensor.mean().item(),
                    norm_tensor.std().item(),
                    norm_tensor.min().item(),
                    norm_tensor.max().item(),
                )
            )
            with torch.no_grad():
                single_obs = reshape_observation(obs[:1], obs_mode=args.obs_mode, pixel_shape=pixel_shape)
                features = actor_backbone(single_obs)
                act_sample, log_pi_sample, mean_sample = actor_head(features)
            record_progress(
                "[Debug] initial_action mean=%s log_pi=%.6f action_l2=%.6f"
                % (
                    np.array2string(mean_sample.detach().cpu().numpy(), precision=4),
                    float(log_pi_sample.detach().cpu().item()),
                    float(act_sample.detach().norm().cpu().item()),
                )
            )

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

        run_prefix = current_env_name.replace('-', '_')
        next_learning_starts_at = int(args.learning_starts)
        last_update_metrics = None
        did_reset_replay = False

        while total_env_steps < args.total_timesteps:
            norm_obs = normalize_obs(obs)
            obs_actor_input = reshape_observation(norm_obs, obs_mode=args.obs_mode, pixel_shape=pixel_shape)
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                pi_action, _, _ = actor_head(actor_backbone(obs_actor_input))
            next_obs_raw, rewards, dones, infos = envs.step(pi_action.float())
            actions = pi_action
            next_obs = prepare_observation(
                next_obs_raw,
                device=device,
                obs_mode=args.obs_mode,
                pixel_shape=pixel_shape,
                flatten=True,
            )
            obs_detached_flat = obs.detach()
            next_obs_detached_flat = next_obs.detach()
            truncations = infos.get('time_outs', torch.zeros_like(dones, device=device))
            applied_actions = infos.get('applied_actions', actions)
            student_actions = infos.get('student_actions')
            teacher_mask = infos.get('teacher_intervened_mask')
            if teacher_mask is None and student_actions is not None and applied_actions is not None:
                try:
                    teacher_mask = (torch.abs(applied_actions - student_actions).sum(dim=-1) > 1e-6)
                except Exception:
                    teacher_mask = None

            rewards_eff = rewards
            used_actions = applied_actions
            dones_eff = dones
            last_denied_samples = 0
            if teacher_mask is not None:
                denied_ids = torch.nonzero(teacher_mask, as_tuple=False).flatten()
                if denied_ids.numel() > 0:
                    last_denied_samples = int(denied_ids.numel())
                    mode = args.intervention_reward_mode
                    val = float(args.intervention_reward_value)
                    if mode == 'penalty_student':
                        if student_actions is not None:
                            used_actions = used_actions.clone()
                            used_actions[denied_ids] = student_actions[denied_ids]
                        if val != 0.0:
                            rewards_eff = rewards_eff.clone()
                            rewards_eff[denied_ids] = rewards_eff[denied_ids] - abs(val)
                        dones_eff = dones_eff.clone()
                        dones_eff[denied_ids] = 1
                    elif mode == 'bonus_teacher' and val != 0.0:
                        rewards_eff = rewards_eff.clone()
                        rewards_eff[denied_ids] = rewards_eff[denied_ids] + abs(val)
                    bonus_val = float(args.bonus_teacher_value)
                    if bonus_val != 0.0:
                        if rewards_eff is rewards:
                            rewards_eff = rewards_eff.clone()
                        rewards_eff[denied_ids] = rewards_eff[denied_ids] + bonus_val

                    if pref_buffer is not None and student_actions is not None and 'teacher_actions' in infos:
                        try:
                            a_teacher_all = infos['teacher_actions']
                            pref_buffer.append(
                                obs_detached_flat[denied_ids],
                                a_teacher_all[denied_ids],
                                student_actions[denied_ids],
                            )
                        except Exception:
                            pass

                    if pref_td_buffer is not None and student_actions is not None and 'teacher_actions' in infos:
                        try:
                            a_teacher_all = infos['teacher_actions']
                            s_now = obs_detached_flat[denied_ids]
                            s_next = next_obs_detached_flat[denied_ids]
                            r_teacher = rewards[denied_ids].clone().view(-1, 1)
                            if float(args.pref_td_teacher_bonus_value) != 0.0:
                                r_teacher = r_teacher + float(args.pref_td_teacher_bonus_value)
                            d_teacher = dones[denied_ids].clone().view(-1, 1).float()
                            pref_td_buffer.append_teacher(
                                s_now,
                                a_teacher_all[denied_ids],
                                r_teacher.squeeze(1),
                                s_next,
                                d_teacher.squeeze(1),
                            )
                            r_student = -abs(float(args.pref_td_penalty_value)) * torch.ones((denied_ids.numel(), 1), device=device, dtype=torch.float32)
                            pref_td_buffer.append_student(
                                s_now,
                                student_actions[denied_ids],
                                r_student.squeeze(1),
                            )
                        except Exception:
                            pass

            transition = TensorDict(
                {
                    'observations': obs_detached_flat,
                    'actions': used_actions.detach(),
                    'next': {
                        'observations': next_obs_detached_flat,
                        'rewards': rewards_eff.detach(),
                        'truncations': truncations.long(),
                        'dones': dones_eff.long(),
                    },
                },
                batch_size=(envs.num_envs,),
                device=device,
            )
            rb.extend(transition)

            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                norm_obs_now = normalize_obs(obs)
                features_now = critic_feature_backbone(
                    reshape_observation(norm_obs_now, obs_mode=args.obs_mode, pixel_shape=pixel_shape)
                )
                q_stack = torch.stack(critic_heads(features_now, used_actions), dim=0).squeeze(-1)
                max_q = torch.max(q_stack, dim=0).values
                min_q = torch.min(q_stack, dim=0).values
                disagreement_step = max_q - min_q
                qmin_step = min_q
            teacher_mask_float = teacher_mask.float() if teacher_mask is not None else torch.zeros_like(disagreement_step, device=device)
            non_teacher_mask_float = 1.0 - teacher_mask_float
            teacher_metrics.update(disagreement_step, teacher_mask_float, non_teacher_mask_float, qmin_step)

            if cf_buffer is not None and teacher_mask is not None and student_actions is not None:
                denied_ids = torch.nonzero(teacher_mask, as_tuple=False).flatten()
                if denied_ids.numel() > 0:
                    cf_buffer.append(obs_detached_flat[denied_ids], student_actions[denied_ids])

            cur_reward_sum += rewards
            cur_episode_length += 1
            done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                rewbuffer += cur_reward_sum[done_ids].tolist()
                lenbuffer += cur_episode_length[done_ids].tolist()
                cur_reward_sum[done_ids] = 0
                cur_episode_length[done_ids] = 0

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
                if args.obs_mode == 'pixels':
                    pixel_shape = infer_pixel_shape(switched_obs_raw)
                obs = prepare_observation(
                    switched_obs_raw,
                    device=device,
                    obs_mode=args.obs_mode,
                    pixel_shape=pixel_shape,
                    flatten=True,
                )
                current_env_name = new_env_name
                run_prefix = current_env_name.replace('-', '_')
                cur_reward_sum.zero_()
                cur_episode_length.zero_()
                teacher_metrics.reset_running_stats()
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
                rb = create_replay_buffer(args, device, n_obs, envs.num_actions, pixel_shape)
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
                        pixel_shape=pixel_shape,
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
                    pixel_shape=pixel_shape,
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
                base_batch = args.batch_size // max(1, args.num_envs)
                b_pref = int(base_batch * args.pref_sample_ratio) if pref_buffer is not None else 0
                b_pref_td = int(base_batch * args.pref_td_sample_ratio) if pref_td_buffer is not None else 0
                main_batch = max(1, base_batch - b_pref - b_pref_td)

                metrics_accumulator, updates_count = updater.update(
                    replay_buffer=rb,
                    total_env_steps=total_env_steps,
                    main_batch=main_batch,
                    base_batch=base_batch,
                    b_pref=b_pref,
                    b_pref_td=b_pref_td,
                )
                if updates_count > 0:
                    last_update_metrics = (metrics_accumulator, updates_count)

            log_requested = training_logger.should_log(total_env_steps) or total_env_steps >= args.total_timesteps
            if log_requested:
                collection_time = time.time() - start_time
                pref_size = pref_buffer.size if pref_buffer is not None else -1
                pref_td_teacher_size = pref_td_buffer.teacher_size if pref_td_buffer is not None else -1
                pref_td_student_size = pref_td_buffer.student_size if pref_td_buffer is not None else -1
                training_logger.log(
                    total_env_steps=total_env_steps,
                    total_timesteps=args.total_timesteps,
                    iteration_idx=iteration_idx,
                    collection_time=collection_time,
                    rewbuffer=rewbuffer,
                    lenbuffer=lenbuffer,
                    last_update_metrics=last_update_metrics,
                    infos=infos,
                    log_alpha=log_alpha,
                    last_denied_samples=last_denied_samples,
                    pref_size=pref_size,
                    pref_td_teacher_size=pref_td_teacher_size,
                    pref_td_student_size=pref_td_student_size,
                )
                last_update_metrics = None

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
            pixel_shape=pixel_shape,
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


def main():
    args = parse_args()
    device = select_device(args)
    parse_pixel_backbone_args(args)
    args.cf_penalty_target = -abs(float(args.cf_penalty))
    ensure_experiment_name(args)

    run_log_dir, run_model_dir, viz_output_dir, viz_cache_path, record_progress, progress_file = prepare_run_dirs(args)
    print(f"FastSAC OGBench on {args.env_name} device={device}")
    print(f"Log directory: {run_log_dir}")
    print(f"Model directory: {run_model_dir}")

    (
        envs,
        wrappers,
        pixel_shape,
        obs_normalizer,
        critic_obs_normalizer,
        n_obs,
        n_act,
        initial_obs_raw,
    ) = build_environment(
        args,
        device,
        record_progress,
    )

    model = initialize_models(args, device, n_obs, n_act, pixel_shape, record_progress)
    amp = initialize_amp(args, device)

    if args.compile:
        model.actor_backbone = torch.compile(model.actor_backbone)
        model.actor_head = torch.compile(model.actor_head)
        if model.critic_backbone is not None and not args.arch_shared_trunk:
            model.critic_backbone = torch.compile(model.critic_backbone)
        model.critic_heads = torch.compile(model.critic_heads)
        model.critic_target_backbone = torch.compile(model.critic_target_backbone)
        model.critic_target_heads = torch.compile(model.critic_target_heads)
        obs_normalizer_compiled = torch.compile(obs_normalizer)
        critic_obs_normalizer_compiled = torch.compile(critic_obs_normalizer)
        obs_normalizer = obs_normalizer_compiled
        critic_obs_normalizer = critic_obs_normalizer_compiled

    buffers = initialize_buffers(args, device, n_obs, n_act, obs_normalizer)
    replay_buffer = create_replay_buffer(args, device, n_obs, n_act, pixel_shape)
    updater = build_updater_from_components(
        args=args,
        device=device,
        model=model,
        buffers=buffers,
        obs_normalizer=obs_normalizer,
        pixel_shape=pixel_shape,
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
    )

    try:
        run_training_loop(
            args=args,
            device=device,
            envs=envs,
            wrappers=wrappers,
            pixel_shape=pixel_shape,
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


if __name__ == '__main__':
    main()

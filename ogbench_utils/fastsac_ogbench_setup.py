"""Model/buffer/logger setup utilities for FastSAC OGBench."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from fast_sac_utils import SimpleReplayBuffer

from .buffers import PreferencePairBuffer
from .fastsac_ogbench_types import AMPComponents, BufferComponents, LoggingComponents, ModelComponents
from .logging import CheckpointManager, TeacherMetricsAccumulator, TrainingLogger
from .policy import CriticEnsemble, GaussianPolicyHead, MLPBackbone
from .update import FastSACUpdater

def select_device(args) -> torch.device:
    if args.device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(args.device)


def ensure_experiment_name(args) -> None:
    if args.exp_name:
        return
    default_buffer_size = 1_000_000
    env_tag = args.env_name.replace('-v0', '').replace('-', '_')
    components = [env_tag, args.reward_type]
    if getattr(args, "cube_reward_mode", "dense") != "dense":
        components.append(str(args.cube_reward_mode))
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


def initialize_models(
    args,
    device: torch.device,
    n_obs: int,
    n_act: int,
    record_progress,
) -> ModelComponents:
    """Initialize actor/critic modules and optimizers for FastSAC OGBench."""
    def build_backbone(hidden_dim: int) -> nn.Module:
        return MLPBackbone(
            n_obs,
            hidden_dim,
            use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
            layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
        ).to(device)

    if args.arch_shared_trunk:
        shared_backbone = build_backbone(args.shared_hidden_dim)
        actor_backbone = shared_backbone
        actor_head = GaussianPolicyHead(
            shared_backbone.output_dim,
            n_act,
            args.actor_hidden_dim,
            args.init_scale,
            use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
            layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
        ).to(device)
        critic_backbone = shared_backbone
        critic_heads = CriticEnsemble(
            shared_backbone.output_dim,
            n_act,
            args.critic_hidden_dim,
            args.num_critics,
            use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
            layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
        ).to(device)
        trunk_params = list(shared_backbone.parameters())
        trunk_optimizer = optim.Adam(trunk_params, lr=args.critic_learning_rate)
        actor_params = list(actor_head.parameters())
        critic_params = list(critic_heads.parameters())
        initial_shared_backbone_state = copy.deepcopy(shared_backbone.state_dict())
        initial_critic_backbone_state = None
    else:
        actor_backbone = build_backbone(args.actor_hidden_dim)
        actor_head = GaussianPolicyHead(
            actor_backbone.output_dim,
            n_act,
            args.actor_hidden_dim,
            args.init_scale,
            use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
            layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
        ).to(device)
        critic_backbone = build_backbone(args.critic_hidden_dim)
        critic_heads = CriticEnsemble(
            critic_backbone.output_dim,
            n_act,
            args.critic_hidden_dim,
            args.num_critics,
            use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
            layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
        ).to(device)
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
    """Initialize auxiliary training buffers (currently preference pairs only)."""
    def normalize_obs(x):
        return obs_normalizer(x)

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

    return BufferComponents(pref_buffer=pref_buffer)


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
    generate_policy_map,
) -> LoggingComponents:
    """Create logger/checkpoint components used by the training loop."""
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
    buffer_size: Optional[int] = None,
    n_env_override: Optional[int] = None,
) -> SimpleReplayBuffer:
    """Create replay buffer compatible with FastSAC state observations."""
    return SimpleReplayBuffer(
        n_env=int(n_env_override if n_env_override is not None else args.num_envs),
        buffer_size=int(buffer_size if buffer_size is not None else args.buffer_size),
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_obs,
        asymmetric_obs=False,
        playground_mode=False,
        n_steps=1,
        gamma=args.gamma,
        device=device,
        pixel_shape=None,
    )


def build_updater_from_components(
    args,
    device: torch.device,
    model: ModelComponents,
    buffers: BufferComponents,
    obs_normalizer,
    amp: AMPComponents,
) -> FastSACUpdater:
    """Wire model/buffer/AMP state into a FastSACUpdater instance."""
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
        reshape_obs_fn=lambda x: x,
        normalize_obs_fn=obs_normalizer,
        device=device,
        amp_enabled=amp.enabled,
        amp_dtype=amp.dtype,
        amp_device_type=amp.device_type,
        scaler=amp.scaler,
        target_entropy=model.target_entropy,
        pref_buffer=buffers.pref_buffer,
    )

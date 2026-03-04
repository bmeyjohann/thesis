"""Shared typed containers for FastSAC OGBench orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from .buffers import PreferencePairBuffer
from .logging import CheckpointManager, TeacherMetricsAccumulator, TrainingLogger


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
    pref_buffer: Optional[PreferencePairBuffer]


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

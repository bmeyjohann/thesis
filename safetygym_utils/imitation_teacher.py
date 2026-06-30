from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class HumanInterventionImitationPolicy(nn.Module):
    """Two-head policy: continuous human action plus intervention logit."""

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(obs_dim)
        for _ in range(int(max(1, num_layers))):
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(in_dim, int(act_dim))
        self.intervention_head = nn.Linear(in_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        action_unit = torch.tanh(self.action_head(h))
        intervention_logit = self.intervention_head(h).squeeze(-1)
        return action_unit, intervention_logit


class TemporalConvHumanInterventionPolicy(nn.Module):
    """Two-head policy with temporal convolutions over stacked transition context."""

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        context_len: int,
        segment_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.context_len = int(max(1, context_len))
        self.segment_dim = int(max(1, segment_dim))
        expected_dim = self.context_len * self.segment_dim
        if int(obs_dim) != expected_dim:
            raise ValueError(
                f"temporal_cnn obs_dim={obs_dim} must equal context_len*segment_dim={expected_dim}"
            )
        layers: list[nn.Module] = []
        in_dim = self.segment_dim
        for _ in range(int(max(1, num_layers))):
            layers.append(nn.Conv1d(in_dim, int(hidden_dim), kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        self.temporal = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(int(hidden_dim), int(act_dim))
        self.intervention_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = obs.view(obs.shape[0], self.context_len, self.segment_dim).transpose(1, 2)
        h_seq = self.temporal(seq)
        h = self.head(h_seq[:, :, -1])
        action_unit = torch.tanh(self.action_head(h))
        intervention_logit = self.intervention_head(h).squeeze(-1)
        return action_unit, intervention_logit


class AttentionHumanInterventionPolicy(nn.Module):
    """Two-head policy with a Transformer encoder over stacked transition context."""

    def __init__(
        self,
        *,
        obs_dim: int,
        act_dim: int,
        context_len: int,
        segment_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        num_heads: int = 4,
    ):
        super().__init__()
        self.context_len = int(max(1, context_len))
        self.segment_dim = int(max(1, segment_dim))
        expected_dim = self.context_len * self.segment_dim
        if int(obs_dim) != expected_dim:
            raise ValueError(
                f"attention obs_dim={obs_dim} must equal context_len*segment_dim={expected_dim}"
            )
        hidden = int(hidden_dim)
        heads = int(max(1, num_heads))
        while hidden % heads != 0 and heads > 1:
            heads -= 1
        self.input_proj = nn.Linear(self.segment_dim, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.context_len, hidden))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=max(hidden * 4, hidden),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(max(1, num_layers)))
        self.norm = nn.LayerNorm(hidden)
        self.action_head = nn.Linear(hidden, int(act_dim))
        self.intervention_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = obs.view(obs.shape[0], self.context_len, self.segment_dim)
        h = self.input_proj(seq) + self.pos_embed[:, : self.context_len]
        h = self.encoder(h)
        h = self.norm(h[:, -1])
        action_unit = torch.tanh(self.action_head(h))
        intervention_logit = self.intervention_head(h).squeeze(-1)
        return action_unit, intervention_logit


def make_imitation_policy(
    *,
    architecture: str,
    obs_dim: int,
    act_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.0,
    context_len: int = 1,
    segment_dim: int | None = None,
    num_heads: int = 4,
) -> nn.Module:
    arch = str(architecture or "mlp").strip().lower()
    if arch in {"mlp", "frame_mlp", "framestack_mlp"}:
        return HumanInterventionImitationPolicy(
            obs_dim=int(obs_dim),
            act_dim=int(act_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
    seg = int(segment_dim or 0)
    if seg <= 0:
        raise ValueError(f"architecture={architecture} requires segment_dim")
    if arch in {"temporal_cnn", "cnn", "conv1d"}:
        return TemporalConvHumanInterventionPolicy(
            obs_dim=int(obs_dim),
            act_dim=int(act_dim),
            context_len=int(context_len),
            segment_dim=seg,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
    if arch in {"attention", "transformer", "temporal_attention"}:
        return AttentionHumanInterventionPolicy(
            obs_dim=int(obs_dim),
            act_dim=int(act_dim),
            context_len=int(context_len),
            segment_dim=seg,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            num_heads=int(num_heads),
        )
    raise ValueError(f"Unsupported imitation architecture: {architecture}")


def normalize_observations(obs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (obs - mean) / torch.clamp(std, min=1e-6)


def scale_unit_action(action_unit: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor) -> torch.Tensor:
    center = 0.5 * (action_high + action_low)
    half = 0.5 * (action_high - action_low)
    return center + action_unit * half


def action_to_unit_np(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    half = np.maximum(0.5 * (high - low), 1e-6)
    center = 0.5 * (high + low)
    return np.clip((np.asarray(action, dtype=np.float32) - center) / half, -1.0, 1.0).astype(np.float32)


def build_window_features(
    *,
    observations: np.ndarray,
    student_actions: np.ndarray,
    teacher_intervened: np.ndarray,
    episode_ids: np.ndarray,
    context_len: int,
) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    student = np.asarray(student_actions, dtype=np.float32)
    mask_raw = np.asarray(teacher_intervened, dtype=np.float32).reshape(-1)
    eps = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    n_rows, obs_dim = obs.shape
    act_dim = student.shape[1]
    segment_dim = obs_dim + act_dim + 1
    ctx = int(max(1, context_len))
    features = np.zeros((n_rows, ctx * segment_dim), dtype=np.float32)
    prev_mask = np.zeros((n_rows, 1), dtype=np.float32)
    for row in range(1, n_rows):
        if int(eps[row]) == int(eps[row - 1]):
            prev_mask[row, 0] = float(mask_raw[row - 1])
    segments = np.concatenate([obs, student, prev_mask], axis=1).astype(np.float32, copy=False)
    for row in range(n_rows):
        out_start = (ctx - 1) * segment_dim
        for src in range(row, max(-1, row - ctx), -1):
            if src < 0 or int(eps[src]) != int(eps[row]):
                break
            features[row, out_start : out_start + segment_dim] = segments[src]
            out_start -= segment_dim
            if out_start < 0:
                break
    return features


def save_imitation_checkpoint(
    *,
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    obs_mean: torch.Tensor,
    obs_std: torch.Tensor,
    action_low: np.ndarray,
    action_high: np.ndarray,
    metadata: dict[str, Any],
    step: int,
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "obs_mean": obs_mean.detach().cpu(),
        "obs_std": obs_std.detach().cpu(),
        "action_low": torch.as_tensor(action_low, dtype=torch.float32).detach().cpu(),
        "action_high": torch.as_tensor(action_high, dtype=torch.float32).detach().cpu(),
        "metadata": dict(metadata),
        "global_step": int(step),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, target)
    sidecar = target.with_suffix(".json")
    sidecar.write_text(json.dumps(dict(metadata, checkpoint=str(target), global_step=int(step)), indent=2, sort_keys=True) + "\n")
    return target


def load_imitation_checkpoint(
    checkpoint_path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Imitation teacher checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = dict(checkpoint.get("metadata") or {})
    obs_dim = int(metadata.get("obs_dim") or torch.as_tensor(checkpoint["obs_mean"]).numel())
    act_dim = int(metadata.get("act_dim") or torch.as_tensor(checkpoint["action_low"]).numel())
    model = make_imitation_policy(
        architecture=str(metadata.get("architecture", "mlp")),
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=int(metadata.get("hidden_dim", 256)),
        num_layers=int(metadata.get("num_layers", 3)),
        dropout=float(metadata.get("dropout", 0.0)),
        context_len=int(metadata.get("context_len", 1)),
        segment_dim=int(metadata.get("segment_dim", 0) or 0),
        num_heads=int(metadata.get("num_heads", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    state = {
        "path": str(path),
        "metadata": metadata,
        "obs_mean": torch.as_tensor(checkpoint["obs_mean"], device=device, dtype=torch.float32).view(1, -1),
        "obs_std": torch.as_tensor(checkpoint["obs_std"], device=device, dtype=torch.float32).view(1, -1),
        "action_low": torch.as_tensor(checkpoint["action_low"], device=device, dtype=torch.float32).view(1, -1),
        "action_high": torch.as_tensor(checkpoint["action_high"], device=device, dtype=torch.float32).view(1, -1),
        "global_step": int(checkpoint.get("global_step", 0)),
    }
    return model, state

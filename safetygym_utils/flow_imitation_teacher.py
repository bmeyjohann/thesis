from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .imitation_teacher import normalize_observations, scale_unit_action


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = int(in_dim)
    for _ in range(int(max(1, num_layers))):
        layers.append(nn.Linear(last, int(hidden_dim)))
        layers.append(nn.SiLU())
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        last = int(hidden_dim)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


class FlowInterventionImitationPolicy(nn.Module):
    """Intervention classifier plus rectified-flow action model."""

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
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.gate = _mlp(
            self.obs_dim,
            1,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        self.flow = _mlp(
            self.obs_dim + self.act_dim + 1,
            self.act_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )

    def gate_logit(self, obs: torch.Tensor) -> torch.Tensor:
        return self.gate(obs).squeeze(-1)

    def velocity(self, obs: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.flow(torch.cat([obs, xt, t], dim=-1))

    @torch.no_grad()
    def sample_unit_actions(
        self,
        obs: torch.Tensor,
        *,
        sample_steps: int,
        num_samples: int = 1,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]
        obs = obs.repeat_interleave(int(max(1, num_samples)), dim=0)
        x = torch.randn((obs.shape[0], self.act_dim), device=obs.device, dtype=obs.dtype) * float(noise_scale)
        steps = int(max(1, sample_steps))
        dt = 1.0 / float(steps)
        for idx in range(steps):
            t = torch.full((obs.shape[0], 1), float(idx) / float(steps), device=obs.device, dtype=obs.dtype)
            x = x + dt * self.velocity(obs, x, t)
        x = torch.tanh(x)
        if single:
            x = x.view(int(max(1, num_samples)), self.act_dim)
        return x


def select_flow_sample(samples: torch.Tensor, mode: str) -> torch.Tensor:
    """Select one unit-space action from `[num_samples, act_dim]` flow samples."""
    selector = str(mode or "first").strip().lower()
    if samples.ndim != 2:
        raise ValueError(f"Expected samples with shape [num_samples, act_dim], got {tuple(samples.shape)}")
    if selector == "mean":
        return torch.clamp(samples.mean(dim=0), -1.0, 1.0)
    if selector in {"max_norm", "max_abs", "saturated"}:
        idx = torch.argmax(torch.linalg.norm(samples, dim=-1))
        return samples[idx]
    if selector in {"max_turn", "turn"} and samples.shape[-1] >= 2:
        idx = torch.argmax(torch.abs(samples[:, 0] - samples[:, 1]))
        return samples[idx]
    if selector != "first":
        raise ValueError(f"Unsupported flow sample selector: {mode}")
    return samples[0]


def save_flow_imitation_checkpoint(
    *,
    path: Path | str,
    model: FlowInterventionImitationPolicy,
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
        "format": "safetygym_flow_intervention_imitation",
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "obs_mean": obs_mean.detach().cpu(),
        "obs_std": obs_std.detach().cpu(),
        "action_low": np.asarray(action_low, dtype=np.float32),
        "action_high": np.asarray(action_high, dtype=np.float32),
        "metadata": dict(metadata),
        "step": int(step),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, target)
    with target.with_suffix(".json").open("w", encoding="utf-8") as f:
        import json

        json.dump({"path": str(target), "metadata": dict(metadata), "step": int(step)}, f, indent=2, sort_keys=True)
    return target


def load_flow_imitation_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str,
) -> tuple[FlowInterventionImitationPolicy, dict[str, Any]]:
    checkpoint = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    if checkpoint.get("format") != "safetygym_flow_intervention_imitation":
        raise ValueError(f"Not a flow intervention imitation checkpoint: {path}")
    meta = dict(checkpoint.get("metadata") or {})
    model = FlowInterventionImitationPolicy(
        obs_dim=int(meta.get("obs_dim", checkpoint["obs_mean"].shape[-1])),
        act_dim=int(meta.get("act_dim", np.asarray(checkpoint["action_low"]).reshape(-1).shape[0])),
        hidden_dim=int(meta.get("hidden_dim", 256)),
        num_layers=int(meta.get("num_layers", 3)),
        dropout=float(meta.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    state = dict(checkpoint)
    state["obs_mean"] = torch.as_tensor(checkpoint["obs_mean"], device=device, dtype=torch.float32)
    state["obs_std"] = torch.as_tensor(checkpoint["obs_std"], device=device, dtype=torch.float32)
    state["action_low"] = torch.as_tensor(checkpoint["action_low"], device=device, dtype=torch.float32)
    state["action_high"] = torch.as_tensor(checkpoint["action_high"], device=device, dtype=torch.float32)
    return model, state


def flow_action_to_env(
    *,
    model: FlowInterventionImitationPolicy,
    state: dict[str, Any],
    obs_features: torch.Tensor,
    sample_steps: int,
    num_samples: int,
    noise_scale: float,
    selector: str,
) -> torch.Tensor:
    obs_norm = normalize_observations(obs_features, state["obs_mean"], state["obs_std"])
    unit_samples = model.sample_unit_actions(
        obs_norm.reshape(-1),
        sample_steps=int(sample_steps),
        num_samples=int(num_samples),
        noise_scale=float(noise_scale),
    )
    unit = select_flow_sample(unit_samples, selector)
    return scale_unit_action(unit, state["action_low"], state["action_high"])

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class ManeuverBCConfig:
    obs_dim: int
    hidden_dims: tuple[int, ...] = (512, 512, 256)
    activation: str = "tanh"
    obs_mean: tuple[float, ...] | None = None
    obs_std: tuple[float, ...] | None = None
    forward_throttle: float = 0.8


def maneuver_labels_from_actions(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Maneuver labels require actions with shape [N, >=2].")
    labels = np.full((arr.shape[0],), 2, dtype=np.int64)
    turn = arr[:, 1]
    throttle = arr[:, 0]
    labels[(throttle < 0.2) & (turn < -0.5)] = 0
    labels[(throttle < 0.2) & (turn > 0.5)] = 1
    return labels


def _activation(name: str) -> type[nn.Module]:
    key = str(name).strip().lower()
    if key == "relu":
        return nn.ReLU
    if key == "elu":
        return nn.ELU
    return nn.Tanh


class ManeuverBCPolicy(nn.Module):
    """Classify teacher maneuvers and regress only the forward steering trim."""

    def __init__(self, config: ManeuverBCConfig):
        super().__init__()
        self.config = config
        act = _activation(config.activation)
        layers: list[nn.Module] = []
        prev = int(config.obs_dim)
        for width in config.hidden_dims:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(act())
            prev = int(width)
        self.net = nn.Sequential(*layers)
        self.logits = nn.Linear(prev, 3)
        self.forward_turn = nn.Linear(prev, 1)

        obs_mean = np.zeros((int(config.obs_dim),), dtype=np.float32)
        obs_std = np.ones((int(config.obs_dim),), dtype=np.float32)
        if config.obs_mean is not None:
            obs_mean = np.asarray(config.obs_mean, dtype=np.float32).reshape(-1)
        if config.obs_std is not None:
            obs_std = np.asarray(config.obs_std, dtype=np.float32).reshape(-1)
        self.register_buffer("obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_std", torch.as_tensor(np.maximum(obs_std, 1e-6), dtype=torch.float32))

    def normalized_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / self.obs_std

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(self.normalized_obs(obs))
        return self.logits(h), torch.tanh(self.forward_turn(h)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray | torch.Tensor, *, device: torch.device | str | None = None) -> np.ndarray:
        was_training = self.training
        self.eval()
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), dtype=torch.float32)
        else:
            obs_t = obs.detach().to(dtype=torch.float32)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
        if device is not None:
            obs_t = obs_t.to(device)
        logits, forward_turn = self(obs_t)
        cls = torch.argmax(logits, dim=-1)
        action = torch.zeros((obs_t.shape[0], 2), device=obs_t.device, dtype=torch.float32)
        action[cls == 0, 1] = -1.0
        action[cls == 1, 1] = 1.0
        fwd = cls == 2
        action[fwd, 0] = float(self.config.forward_throttle)
        action[fwd, 1] = forward_turn[fwd].clamp(-1.0, 1.0)
        if was_training:
            self.train()
        return action.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)


def save_maneuver_policy(path: Path | str, policy: ManeuverBCPolicy, metadata: dict[str, Any] | None = None) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    cfg = policy.config
    payload = {
        "format": "maneuver_bc",
        "config": {
            "obs_dim": int(cfg.obs_dim),
            "hidden_dims": list(cfg.hidden_dims),
            "activation": str(cfg.activation),
            "obs_mean": policy.obs_mean.detach().cpu().numpy().astype(float).tolist(),
            "obs_std": policy.obs_std.detach().cpu().numpy().astype(float).tolist(),
            "forward_throttle": float(cfg.forward_throttle),
        },
        "state_dict": policy.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, target)
    return target


def load_maneuver_policy(path: Path | str, *, device: torch.device | str = "cpu") -> ManeuverBCPolicy:
    checkpoint = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    if checkpoint.get("format") != "maneuver_bc":
        raise ValueError(f"Not a maneuver_bc checkpoint: {path}")
    raw = checkpoint["config"]
    cfg = ManeuverBCConfig(
        obs_dim=int(raw["obs_dim"]),
        hidden_dims=tuple(int(x) for x in raw.get("hidden_dims", (512, 512, 256))),
        activation=str(raw.get("activation", "tanh")),
        obs_mean=tuple(float(x) for x in raw.get("obs_mean", [])),
        obs_std=tuple(float(x) for x in raw.get("obs_std", [])),
        forward_throttle=float(raw.get("forward_throttle", 0.8)),
    )
    policy = ManeuverBCPolicy(cfg).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy

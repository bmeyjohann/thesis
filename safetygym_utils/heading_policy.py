from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from safetygym_utils.controllers import ScriptedLidarTeacherController


@dataclass
class HeadingBCConfig:
    obs_dim: int
    hidden_dims: tuple[int, ...] = (512, 512, 256)
    activation: str = "tanh"
    obs_mean: tuple[float, ...] | None = None
    obs_std: tuple[float, ...] | None = None
    heading_tolerance: float = 0.20
    forward_throttle: float = 0.8


def _activation(name: str) -> type[nn.Module]:
    key = str(name).strip().lower()
    if key == "relu":
        return nn.ReLU
    if key == "elu":
        return nn.ELU
    return nn.Tanh


def _iter_wrappers(env):
    cur = env
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = getattr(cur, "env", None)


def _adapt_raw_wheels_to_env(env, wheel_action: np.ndarray) -> np.ndarray:
    action = np.asarray(wheel_action, dtype=np.float32).reshape(-1)
    for wrapper in _iter_wrappers(env):
        if hasattr(wrapper, "reverse_action") and getattr(wrapper, "action_mode", "raw_wheels") in {
            "throttle_turn",
            "cardinal",
        }:
            action = np.asarray(wrapper.reverse_action(action), dtype=np.float32).reshape(-1)
            break
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    if action.shape[0] < low.shape[0]:
        action = np.pad(action, (0, low.shape[0] - action.shape[0]), mode="constant")
    elif action.shape[0] > low.shape[0]:
        action = action[: low.shape[0]]
    return np.clip(action, low, high).astype(np.float32, copy=False)


class HeadingBCPolicy(nn.Module):
    """Predict the teacher's desired local heading, then use deterministic car control."""

    def __init__(self, config: HeadingBCConfig):
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
        self.head = nn.Linear(prev, 2)

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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.net(self.normalized_obs(obs)))
        return raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    @torch.no_grad()
    def predict_angle(self, obs: np.ndarray | torch.Tensor, *, device: torch.device | str | None = None) -> float:
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
        vec = self(obs_t)[0].detach().cpu().numpy()
        if was_training:
            self.train()
        return float(np.arctan2(float(vec[0]), float(vec[1])))

    @torch.no_grad()
    def act(self, obs: np.ndarray | torch.Tensor, *, env, device: torch.device | str | None = None) -> np.ndarray:
        angle = self.predict_angle(obs, device=device)
        lim = ScriptedLidarTeacherController._wheel_limit(
            np.asarray(env.action_space.low, dtype=np.float32),
            np.asarray(env.action_space.high, dtype=np.float32),
        )
        tol = float(max(1e-3, self.config.heading_tolerance))
        if abs(angle) > tol:
            wheel = np.asarray([lim, -lim], dtype=np.float32) if angle > 0.0 else np.asarray([-lim, lim], dtype=np.float32)
        else:
            turn = float(np.clip(-angle / tol, -1.0, 1.0))
            throttle = float(self.config.forward_throttle)
            wheel = np.asarray([throttle - turn, throttle + turn], dtype=np.float32) * lim
        return _adapt_raw_wheels_to_env(env, wheel)


def save_heading_policy(path: Path | str, policy: HeadingBCPolicy, metadata: dict[str, Any] | None = None) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    cfg = policy.config
    payload = {
        "format": "heading_bc",
        "config": {
            "obs_dim": int(cfg.obs_dim),
            "hidden_dims": list(cfg.hidden_dims),
            "activation": str(cfg.activation),
            "obs_mean": policy.obs_mean.detach().cpu().numpy().astype(float).tolist(),
            "obs_std": policy.obs_std.detach().cpu().numpy().astype(float).tolist(),
            "heading_tolerance": float(cfg.heading_tolerance),
            "forward_throttle": float(cfg.forward_throttle),
        },
        "state_dict": policy.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, target)
    return target


def load_heading_policy(path: Path | str, *, device: torch.device | str = "cpu") -> HeadingBCPolicy:
    checkpoint = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    if checkpoint.get("format") != "heading_bc":
        raise ValueError(f"Not a heading_bc checkpoint: {path}")
    raw = checkpoint["config"]
    cfg = HeadingBCConfig(
        obs_dim=int(raw["obs_dim"]),
        hidden_dims=tuple(int(x) for x in raw.get("hidden_dims", (512, 512, 256))),
        activation=str(raw.get("activation", "tanh")),
        obs_mean=tuple(float(x) for x in raw.get("obs_mean", [])),
        obs_std=tuple(float(x) for x in raw.get("obs_std", [])),
        heading_tolerance=float(raw.get("heading_tolerance", 0.20)),
        forward_throttle=float(raw.get("forward_throttle", 0.8)),
    )
    policy = HeadingBCPolicy(cfg).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy

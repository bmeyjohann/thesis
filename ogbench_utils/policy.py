from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class PixelNormalizer(nn.Module):
    """Normalize pixel observations to the [0, 1] range."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            x = x.float()
        if torch.numel(x) == 0:
            return x
        # Some envs already provide [0,1] floats, others give uint8 0..255
        if torch.max(x) > 1.0:
            x = x / 255.0
        return x


class IdentityNormalizer(nn.Module):
    """Pass-through normalizer for already-normalized inputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MLPBackbone(nn.Module):
    """Two-layer MLP feature extractor."""

    def __init__(self, input_dim: int, hidden_dim: int, use_layer_norm: bool = False, layer_norm_eps: float = 1e-5):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim, eps=float(layer_norm_eps)))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim, eps=float(layer_norm_eps)))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(
            *layers,
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _expand_sequence(values: Sequence[int] | None, target_len: int, fallback: Iterable[int]) -> list[int]:
    if values is None:
        return list(fallback)
    result = list(int(v) for v in values)
    if not result:
        return list(fallback)
    if len(result) < target_len:
        result.extend([result[-1]] * (target_len - len(result)))
    return result[:target_len]


class PixelBackbone(nn.Module):
    """CNN feature extractor for pixel observations."""

    def __init__(
        self,
        input_shape: Sequence[int],
        feature_dim: int,
        conv_channels: Sequence[int] | None = None,
        kernel_sizes: Sequence[int] | None = None,
        strides: Sequence[int] | None = None,
        final_pool: int | None = None,
    ):
        super().__init__()
        c, h, w = (int(v) for v in input_shape)
        channels = _expand_sequence(conv_channels, target_len=len(conv_channels or (32, 64, 64)), fallback=(32, 64, 64))
        kernels = _expand_sequence(kernel_sizes, target_len=len(channels), fallback=(8, 4, 3))
        stride_vals = _expand_sequence(strides, target_len=len(channels), fallback=(4, 2, 1))

        layers: list[nn.Module] = []
        in_channels = c
        for idx, out_channels in enumerate(channels):
            k = int(kernels[idx])
            s = int(stride_vals[idx])
            padding = k // 2 if s == 1 else 0
            layers.append(nn.Conv2d(in_channels, int(out_channels), kernel_size=k, stride=s, padding=padding))
            layers.append(nn.ReLU())
            in_channels = int(out_channels)
        if final_pool:
            layers.append(nn.AdaptiveAvgPool2d(final_pool))
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, feature_dim),
            nn.ReLU(),
        )
        self.output_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] not in (1, 3, 4) and x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class GaussianPolicyHead(nn.Module):
    """Diagonal Gaussian policy head with Tanh squashing."""

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        init_scale: float,
        use_layer_norm: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        hidden_dim_2 = max(1, hidden_dim // 2)
        layers: list[nn.Module] = [nn.Linear(feature_dim, hidden_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim, eps=float(layer_norm_eps)))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim_2))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim_2, eps=float(layer_norm_eps)))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(
            *layers,
        )
        self.fc_mu = nn.Linear(hidden_dim_2, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim_2, action_dim)
        nn.init.normal_(self.fc_mu.weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu.bias, 0.0)

    def forward(self, features: torch.Tensor):
        x = self.net(features)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class CriticHead(nn.Module):
    """Single Q-value head."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int, use_layer_norm: bool = False, layer_norm_eps: float = 1e-5):
        super().__init__()
        hidden_dim_2 = max(1, hidden_dim // 2)
        layers: list[nn.Module] = [nn.Linear(feature_dim + action_dim, hidden_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim, eps=float(layer_norm_eps)))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim_2))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim_2, eps=float(layer_norm_eps)))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim_2, 1))
        self.net = nn.Sequential(
            *layers,
        )

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([features, actions], dim=-1)
        return self.net(x)


class CriticEnsemble(nn.Module):
    """Ensemble of critic heads."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_heads: int,
        use_layer_norm: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            CriticHead(
                feature_dim,
                action_dim,
                hidden_dim,
                use_layer_norm=use_layer_norm,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_heads)
        ])

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> list[torch.Tensor]:
        return [head(features, actions) for head in self.heads]

    def min_q(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        qs = self.forward(features, actions)
        stacked = torch.stack(qs, dim=0)
        return torch.min(stacked, dim=0).values

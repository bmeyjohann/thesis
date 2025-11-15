#!/usr/bin/env python3
"""Visualize FastSAC actor/critic over a 2-D grid for pointmaze environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Tuple

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib import colors
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAST_SAC_PATH = PROJECT_ROOT / "fasttd3" / "fast_sac"
if str(FAST_SAC_PATH) not in sys.path:
    sys.path.append(str(FAST_SAC_PATH))

from fast_sac import Actor, Critic  # type: ignore  # noqa: E402
from fast_sac_utils import EmpiricalNormalization  # type: ignore  # noqa: E402
from drqv2.drqv2 import Encoder as DrQEncoder, Actor as DrQActor, Critic as DrQCritic  # type: ignore  # noqa: E402
from ogbench.wrappers import FlexibleObsWrapper  # type: ignore  # noqa: E402


def _parse_int_sequence(value, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, (list, tuple)):
        seq = [int(v) for v in value]
        return tuple(seq if seq else default)
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(',')]
        seq = [int(p) for p in parts if p]
        return tuple(seq if seq else default)
    return tuple(default)


class IdentityNormalizer(nn.Module):
    def forward(self, x):
        return x


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.net(x)


class PixelBackbone(nn.Module):
    def __init__(
        self,
        input_shape,
        feature_dim: int,
        conv_channels: tuple[int, ...],
        kernel_sizes: tuple[int, ...],
        strides: tuple[int, ...],
        final_pool: int | None = None,
    ):
        super().__init__()
        c, h, w = input_shape
        channels = list(conv_channels)
        kernels = list(kernel_sizes)
        strides_list = list(strides)
        if len(kernels) < len(channels):
            kernels.extend([kernels[-1]] * (len(channels) - len(kernels)))
        if len(strides_list) < len(channels):
            strides_list.extend([strides_list[-1]] * (len(channels) - len(strides_list)))
        layers: list[nn.Module] = []
        in_ch = c
        for idx, out_ch in enumerate(channels):
            k = int(kernels[idx])
            s = int(strides_list[idx])
            padding = k // 2 if s == 1 else 0
            layers.append(nn.Conv2d(in_ch, int(out_ch), kernel_size=k, stride=s, padding=padding))
            layers.append(nn.ReLU())
            in_ch = int(out_ch)
        if final_pool is not None and final_pool > 0:
            layers.append(nn.AdaptiveAvgPool2d(final_pool))
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = self.conv(dummy).reshape(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, feature_dim),
            nn.ReLU(),
        )
        self.output_dim = feature_dim

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] not in (1, 3, 4):
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class PixelNormalizer(nn.Module):
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        if torch.max(x) > 1.0:
            x = x / 255.0
        return x


class DrQPixelPreprocessor(nn.Module):
    def __init__(self, target_channels: int):
        super().__init__()
        self.target_channels = int(target_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"Expected 3D/4D tensor for pixel observations, got shape={tuple(x.shape)}")
        valid_channels = {self.target_channels, 3, 4, 1}
        if x.shape[1] not in valid_channels:
            # Try to locate a channel axis among spatial dims (2 or 3)
            promoted = False
            for candidate in (self.target_channels, 3, 4, 1):
                for axis in (1, 2, 3):
                    if x.shape[axis] == candidate:
                        if axis != 1:
                            axes = [0, axis] + [i for i in range(1, 4) if i != axis]
                            x = x.permute(*axes)
                        promoted = True
                        break
                if promoted:
                    break
            if not promoted:
                x = x.permute(0, 3, 1, 2)
        if x.dtype != torch.float32:
            x = x.float()
        if torch.max(x) <= 1.0:
            x = x * 255.0
        if x.shape[1] != self.target_channels:
            if self.target_channels % x.shape[1] != 0:
                raise ValueError(f"Cannot broadcast pixels with {x.shape[1]} channels to {self.target_channels} (tensor shape={tuple(x.shape)})")
            repeat = self.target_channels // x.shape[1]
            x = x.repeat(1, repeat, 1, 1)
        return x


class DrQActorWrapper(nn.Module):
    def __init__(self, encoder: DrQEncoder, actor: DrQActor, eval_std: float, preproc: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.eval_std = float(eval_std)
        self.preproc = preproc

    def eval(self):
        self.encoder.eval()
        self.actor.eval()

    def forward(self, obs: torch.Tensor):
        obs_proc = self.preproc(obs)
        repr_obs = self.encoder(obs_proc)
        std = torch.as_tensor(self.eval_std, device=obs_proc.device)
        dist = self.actor(repr_obs, std)
        mean = dist.mean
        dummy_log = torch.zeros(mean.shape[0], 1, device=mean.device)
        return mean, dummy_log, mean


class DrQCriticWrapper(nn.Module):
    def __init__(self, encoder: DrQEncoder, critic: DrQCritic, preproc: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.critic = critic
        self.preproc = preproc

    def eval(self):
        self.encoder.eval()
        self.critic.eval()

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        obs_proc = self.preproc(obs)
        repr_obs = self.encoder(obs_proc)
        q1, q2 = self.critic(repr_obs, actions)
        return [q1, q2]


class GaussianPolicyHead(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int, init_scale: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim // 2, action_dim)
        nn.init.normal_(self.fc_mu.weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu.bias, 0.0)

    def forward(self, features):
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
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features, actions):
        x = torch.cat([features, actions], dim=-1)
        return self.net(x)


class CriticEnsemble(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([
            CriticHead(feature_dim, action_dim, hidden_dim) for _ in range(num_heads)
        ])

    def forward(self, features, actions):
        return [head(features, actions) for head in self.heads]


class ActorWrapper:
    def __init__(self, backbone: nn.Module, head: nn.Module):
        self.backbone = backbone
        self.head = head

    def eval(self):
        self.backbone.eval()
        self.head.eval()

    def __call__(self, obs):
        return self.head(self.backbone(obs))


class CriticWrapper:
    def __init__(self, backbone: nn.Module, ensemble: CriticEnsemble):
        self.backbone = backbone
        self.ensemble = ensemble

    def eval(self):
        self.backbone.eval()
        self.ensemble.eval()

    def __call__(self, obs, actions):
        features = self.backbone(obs)
        return self.ensemble(features, actions)

WALL_TILE_IDS = {1}
DANGEROUS_TILE_ID = 2


class PixelObservationSampler:
    def __init__(
        self,
        env_id: str,
        width: int,
        height: int,
        camera_name: str | None,
        seed: int,
        device: torch.device,
        pixel_camera_mode: str = 'global',
        pixel_local_view_size: float = 12.0,
        pixel_local_camera_height: Optional[float] = None,
        pixel_first_person_distance: float = 3.0,
        pixel_first_person_height: float = 1.0,
        pixel_first_person_lookahead: float = 2.0,
        pixel_first_person_pitch: float = -15.0,
    ):
        env_kwargs = dict(
            render_mode="rgb_array",
            width=width,
            height=height,
            camera_name=camera_name,
        )
        if camera_name is None:
            env_kwargs.update(
                pixel_camera_mode=pixel_camera_mode,
                pixel_local_view_size=pixel_local_view_size,
                pixel_local_camera_height=pixel_local_camera_height,
                pixel_first_person_distance=pixel_first_person_distance,
                pixel_first_person_height=pixel_first_person_height,
                pixel_first_person_lookahead=pixel_first_person_lookahead,
                pixel_first_person_pitch=pixel_first_person_pitch,
            )
        self.env = gym.make(env_id, **env_kwargs)
        self.device = device
        self.env.reset(seed=seed)

    def prepare(self, goal_xy: np.ndarray) -> None:
        self.env.unwrapped.set_goal(goal_xy=goal_xy)

    def sample(self, points: np.ndarray) -> torch.Tensor:
        frames = []
        for xy in points:
            self.env.unwrapped.set_xy(np.array(xy, dtype=np.float32))
            frame = self.env.render()
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            frame_chw = np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0
            frames.append(frame_chw)
        tensor = torch.from_numpy(np.stack(frames, axis=0)).to(self.device)
        return tensor

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate actor/critic on a position grid.")
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Path to checkpoint (e.g. models/..._final.pt)")
    parser.add_argument("--env_name", type=str, default=None,
                        help="Gym env name; default reads from checkpoint")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Evaluation device (cpu or cuda)")
    parser.add_argument("--grid_resolution", type=int, default=64,
                        help="Samples per axis for grid evaluation")
    parser.add_argument("--x_range", type=float, nargs=2, default=None,
                        help="Optional agent x bounds [min max]; defaults to maze extent")
    parser.add_argument("--y_range", type=float, nargs=2, default=None,
                        help="Optional agent y bounds [min max]; defaults to maze extent")
    parser.add_argument("--goal", type=float, nargs=2, default=None,
                        help="Optional goal override (x y)")
    parser.add_argument("--quiver_stride", type=int, default=2,
                        help="Subsampling stride for quiver arrows (>=1)")
    parser.add_argument("--output_dir", type=Path, default=Path("visualizations"),
                        help="Directory for generated figures + metadata")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional suffix for file names")
    parser.add_argument("--seed", type=int, default=0,
                        help="Environment reset seed (for goal sampling)")
    parser.add_argument("--goal_cache", type=Path, default=None,
                        help="Optional path to cache goal/range info for consistency across runs")
    return parser.parse_args()


def load_checkpoint(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    if 'drq_encoder' in ckpt:
        return ckpt, 'drqv2'
    if 'actor_backbone' in ckpt:
        return ckpt, 'fastsac_v2'
    required = {"actor_state_dict", "qnet_state_dict", "obs_normalizer_state", "args"}
    missing = required - ckpt.keys()
    if missing:
        raise KeyError(f"Checkpoint {model_path} missing keys: {missing}")
    return ckpt, 'fastsac'


def build_networks(ckpt: dict, device: torch.device, policy_type: str):
    args = ckpt.get('args', {}) or {}
    if policy_type == 'fastsac':
        obs_dim = ckpt['obs_normalizer_state']['_mean'].shape[1]
        act_dim = ckpt['actor_state_dict']['fc_mu.weight'].shape[0]

        actor = Actor(
            n_obs=obs_dim,
            n_act=act_dim,
            num_envs=args.get('num_envs', 1),
            init_scale=args.get('init_scale', 0.01),
            hidden_dim=args.get('actor_hidden_dim', 512),
            device=device,
        )
        actor.load_state_dict(ckpt['actor_state_dict'])
        actor.eval()

        critic = Critic(
            n_obs=obs_dim,
            n_act=act_dim,
            hidden_dim=args.get('critic_hidden_dim', 1024),
            device=device,
        )
        critic.load_state_dict(ckpt['qnet_state_dict'])
        critic.eval()

        obs_norm = EmpiricalNormalization(shape=obs_dim, device=device)
        obs_norm.load_state_dict(ckpt['obs_normalizer_state'])
        obs_norm.eval()

        return actor, critic, obs_norm, args

    if policy_type == 'drqv2':
        pixel_shape = ckpt.get('pixel_shape') or args.get('pixel_shape')
        if pixel_shape is None:
            raise ValueError("DrQ checkpoint missing pixel_shape metadata")
        pixel_shape = tuple(int(x) for x in pixel_shape)
        if len(pixel_shape) != 3:
            raise ValueError(f"Unexpected pixel_shape for DrQ checkpoint: {pixel_shape}")
        action_shape = ckpt.get('action_shape') or args.get('action_shape')
        if action_shape is None:
            raise ValueError("DrQ checkpoint missing action_shape metadata")
        action_shape = tuple(int(x) for x in action_shape)
        feature_dim = args.get('drq_feature_dim', 50)
        hidden_dim = args.get('drq_hidden_dim', 1024)
        eval_std = float(args.get('viz_eval_std', 0.0))

        encoder = DrQEncoder(pixel_shape).to(device)
        encoder.load_state_dict(ckpt['drq_encoder'])
        encoder.eval()

        actor = DrQActor(encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        actor.load_state_dict(ckpt['drq_actor'])
        actor.eval()

        critic = DrQCritic(encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        critic.load_state_dict(ckpt['drq_critic'])
        critic.eval()

        preproc = DrQPixelPreprocessor(pixel_shape[0]).to(device)
        actor_wrapper = DrQActorWrapper(encoder, actor, eval_std, preproc)
        critic_wrapper = DrQCriticWrapper(encoder, critic, preproc)
        return actor_wrapper, critic_wrapper, preproc, args

    obs_mode = args.get('obs_mode', 'state')
    if obs_mode == 'pixels':
        pixel_shape = ckpt.get('pixel_shape') or args.get('pixel_shape')
        if pixel_shape is None:
            raise ValueError('Pixel checkpoint missing pixel_shape metadata')
        pixel_shape = tuple(int(x) for x in pixel_shape)
        if len(pixel_shape) != 3:
            raise ValueError(f"Unexpected pixel_shape: {pixel_shape}")

        conv_channels = _parse_int_sequence(
            args.get('pixel_conv_channels_parsed', args.get('pixel_conv_channels')), (32, 64, 64)
        )
        kernel_sizes = _parse_int_sequence(
            args.get('pixel_kernel_sizes_parsed', args.get('pixel_kernel_sizes')), (8, 4, 3)
        )
        strides = _parse_int_sequence(
            args.get('pixel_strides_parsed', args.get('pixel_strides')), (4, 2, 1)
        )
        final_pool = args.get('pixel_final_pool_parsed', args.get('pixel_final_pool', 0))
        if isinstance(final_pool, (list, tuple)):
            final_pool = final_pool[0] if final_pool else 0
        final_pool_val = int(final_pool) if final_pool else None

        actor_backbone_state = ckpt['actor_backbone']
        actor_hidden_dim = args.get('actor_hidden_dim', 512)
        actor_backbone = PixelBackbone(
            pixel_shape,
            actor_hidden_dim,
            conv_channels,
            kernel_sizes,
            strides,
            final_pool_val,
        ).to(device)
        actor_backbone.load_state_dict(actor_backbone_state)
        actor_backbone.eval()

        actor_head_state = ckpt['actor_head']
        act_dim = actor_head_state['fc_mu.weight'].shape[0]
        actor_head = GaussianPolicyHead(actor_backbone.output_dim, act_dim, actor_hidden_dim, args.get('init_scale', 0.01)).to(device)
        actor_head.load_state_dict(actor_head_state)
        actor_head.eval()
        actor = ActorWrapper(actor_backbone, actor_head)

        critic_backbone_state = ckpt.get('critic_backbone') or ckpt.get('shared_backbone')
        critic_hidden_dim = args.get('critic_hidden_dim', 1024)
        if critic_backbone_state is None:
            critic_backbone = actor_backbone
        else:
            critic_backbone = PixelBackbone(
                pixel_shape,
                critic_hidden_dim,
                conv_channels,
                kernel_sizes,
                strides,
                final_pool_val,
            ).to(device)
            critic_backbone.load_state_dict(critic_backbone_state)
            critic_backbone.eval()

        critic_heads_state = ckpt['critic_heads']
        head_keys = [k for k in critic_heads_state.keys() if k.endswith('net.0.weight')]
        num_heads = len(head_keys)
        first_head = critic_heads_state['heads.0.net.0.weight']
        critic_head_hidden = first_head.shape[0]
        critic_heads = CriticEnsemble(critic_backbone.output_dim, act_dim, critic_head_hidden, num_heads).to(device)
        critic_heads.load_state_dict(critic_heads_state)
        critic_heads.eval()
        critic = CriticWrapper(critic_backbone, critic_heads)

        obs_norm = PixelNormalizer().to(device)
        obs_norm.eval()
        return actor, critic, obs_norm, args

    if obs_mode != 'state':
        raise NotImplementedError('Policy map supports only state observations for now')

    actor_backbone_state = ckpt['actor_backbone']
    first_weight = actor_backbone_state['net.0.weight']
    obs_dim = first_weight.shape[1]
    backbone_hidden = first_weight.shape[0]
    actor_backbone = MLPBackbone(obs_dim, backbone_hidden).to(device)
    actor_backbone.load_state_dict(actor_backbone_state)
    actor_backbone.eval()

    actor_head_state = ckpt['actor_head']
    act_dim = actor_head_state['fc_mu.weight'].shape[0]
    actor_head = GaussianPolicyHead(actor_backbone.output_dim, act_dim, args.get('actor_hidden_dim', backbone_hidden), args.get('init_scale', 0.01)).to(device)
    actor_head.load_state_dict(actor_head_state)
    actor_head.eval()
    actor = ActorWrapper(actor_backbone, actor_head)

    critic_backbone_state = ckpt.get('critic_backbone') or ckpt.get('shared_backbone')
    if critic_backbone_state is None:
        critic_backbone = actor_backbone
    else:
        first_w = critic_backbone_state['net.0.weight']
        critic_backbone = MLPBackbone(first_w.shape[1], first_w.shape[0]).to(device)
        critic_backbone.load_state_dict(critic_backbone_state)
        critic_backbone.eval()

    critic_heads_state = ckpt['critic_heads']
    head_keys = [k for k in critic_heads_state.keys() if k.endswith('net.0.weight')]
    num_heads = len(head_keys)
    head_hidden = critic_heads_state['heads.0.net.0.weight'].shape[0]
    critic_heads = CriticEnsemble(critic_backbone.output_dim, act_dim, head_hidden, num_heads).to(device)
    critic_heads.load_state_dict(critic_heads_state)
    critic_heads.eval()
    critic = CriticWrapper(critic_backbone, critic_heads)

    obs_state = ckpt.get('obs_normalizer_state')
    if obs_state:
        obs_norm = EmpiricalNormalization(shape=obs_dim, device=device)
        obs_norm.load_state_dict(obs_state)
        obs_norm.eval()
    else:
        obs_norm = IdentityNormalizer()
    return actor, critic, obs_norm, args


def infer_goal_and_bounds(train_args: dict, env_name: str | None, seed: int) -> tuple[
    np.ndarray,
    tuple[float, float, float, float],
    np.ndarray | None,
    float | None,
    tuple[float, float] | None,
    int | None,
]:
    env_id = env_name or train_args.get("env_name")
    if env_id is None:
        raise ValueError("Provide --env_name or ensure checkpoint stored env_name")

    env = gym.make(env_id, render_mode=None)
    env = FlexibleObsWrapper(
        env,
        include_goal=train_args.get("include_goal", True),
        include_distance=train_args.get("include_distance", False),
        include_direction=train_args.get("include_direction", False),
        include_velocity=train_args.get("include_velocity", False),
    )
    obs, info = env.reset(seed=seed)
    base_env = env.unwrapped
    env.close()

    goal = None
    if isinstance(info, dict):
        goal_candidate = info.get("goal")
        if goal_candidate is not None:
            goal_arr = np.asarray(goal_candidate)
            if goal_arr.ndim == 1 and goal_arr.size == 2:
                goal = goal_arr.astype(np.float32)
    if goal is None:
        obs_mode = train_args.get("obs_mode", "state")
        if obs_mode == "state" and isinstance(obs, np.ndarray) and obs.ndim == 1 and obs.size >= 4:
            goal = np.asarray(obs[2:4], dtype=np.float32)
    if goal is None:
        goal_attr = getattr(base_env, "cur_goal_xy", None)
        if goal_attr is not None:
            goal = np.asarray(goal_attr, dtype=np.float32)
    if goal is None:
        raise RuntimeError("Failed to infer goal position from environment reset")

    maze_map = getattr(base_env, "maze_map", None)
    maze_unit = getattr(base_env, "_maze_unit", None)
    offset_x = getattr(base_env, "_offset_x", None)
    offset_y = getattr(base_env, "_offset_y", None)

    dangerous_id = None
    if maze_map is not None and maze_unit is not None and offset_x is not None and offset_y is not None:
        # maze_map shape: (rows, cols). Row index increases downward (y direction).
        if hasattr(base_env, "_dangerous_tile_id"):
            dangerous_id = int(getattr(base_env, "_dangerous_tile_id"))

        h, w = maze_map.shape
        x_min = -float(offset_x)
        x_max = (w - 1) * float(maze_unit) - float(offset_x)
        y_min = -float(offset_y)
        y_max = (h - 1) * float(maze_unit) - float(offset_y)
        layout = np.array(maze_map, copy=True)
    else:
        # Fallback if maze parameters are not available; use goal as anchor with +/- 10 range
        x_min, x_max = goal[0] - 10.0, goal[0] + 10.0
        y_min, y_max = goal[1] - 10.0, goal[1] + 10.0
        layout = None

    offsets = None if offset_x is None or offset_y is None else (float(offset_x), float(offset_y))
    maze_unit_val = None if maze_unit is None else float(maze_unit)

    return goal, (x_min, x_max, y_min, y_max), layout, maze_unit_val, offsets, dangerous_id


def compose_obs(agent_xy: np.ndarray, goal_xy: np.ndarray, train_args: dict) -> np.ndarray:
    pieces = [agent_xy.astype(np.float32)]
    if train_args.get("include_goal", True):
        pieces.append(goal_xy.astype(np.float32))
    if train_args.get("include_distance", False):
        dist = np.linalg.norm(goal_xy - agent_xy)
        pieces.append(np.array([dist], dtype=np.float32))
    if train_args.get("include_direction", False):
        delta = goal_xy - agent_xy
        norm = np.linalg.norm(delta)
        if norm > 1e-8:
            pieces.append((delta / norm).astype(np.float32))
        else:
            pieces.append(np.zeros(2, dtype=np.float32))
    if train_args.get("include_velocity", False):
        pieces.append(np.zeros(2, dtype=np.float32))
    return np.concatenate(pieces)


def snap_goal_to_free(goal: np.ndarray,
                      maze_layout: np.ndarray | None,
                      maze_unit: float | None,
                      offsets: tuple[float, float] | None) -> np.ndarray:
    if maze_layout is None or maze_unit is None or offsets is None:
        return goal
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
    free_cells = np.argwhere(maze_layout == 0)
    if free_cells.size == 0:
        return goal
    off_x, off_y = float(offsets[0]), float(offsets[1])
    # Convert goal to nearest tile index
    unit = float(maze_unit)
    col = int(round(float((goal[0] + off_x) / unit)))
    row = int(round(float((goal[1] + off_y) / unit)))
    if (0 <= row < maze_layout.shape[0] and
            0 <= col < maze_layout.shape[1] and
            maze_layout[row, col] == 0):
        return goal

    xy_candidates = np.stack([
        free_cells[:, 1] * maze_unit - off_x,
        free_cells[:, 0] * maze_unit - off_y,
    ], axis=1)
    idx = int(np.argmin(np.linalg.norm(xy_candidates - goal, axis=1)))
    return xy_candidates[idx].astype(np.float32)


def evaluate_grid(actor: Actor, critic: Critic, obs_norm: EmpiricalNormalization,
                  train_args: dict, goal: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                  device: torch.device, free_mask: np.ndarray | None = None,
                  maze_unit: float | None = None, offsets: tuple[float, float] | None = None,
                  pixel_sampler: "PixelObservationSampler" | None = None) -> dict:
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    pts = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=-1)

    if free_mask is not None and maze_unit is not None and offsets is not None:
        free_mask = np.asarray(free_mask, dtype=bool)
        off_x, off_y = offsets
        # Convert world coordinates back to maze indices to mask non-traversable tiles.
        col = ((pts[:, 0] + off_x) / maze_unit)
        row = ((pts[:, 1] + off_y) / maze_unit)
        valid = (
            (row >= 0) & (row < free_mask.shape[0]) &
            (col >= 0) & (col < free_mask.shape[1])
        )
        row_idx = np.clip(np.floor(row).astype(int), 0, free_mask.shape[0] - 1)
        col_idx = np.clip(np.floor(col).astype(int), 0, free_mask.shape[1] - 1)
        traversable = np.zeros_like(valid, dtype=bool)
        traversable[valid] = free_mask[row_idx[valid], col_idx[valid]]
    else:
        traversable = np.ones(len(pts), dtype=bool)

    if pixel_sampler is not None:
        obs_tensor = pixel_sampler.sample(pts)
    else:
        obs_np = np.stack([compose_obs(p, goal, train_args) for p in pts], axis=0)
        obs_tensor = torch.from_numpy(obs_np).to(device)

    with torch.no_grad():
        norm_obs = obs_norm(obs_tensor)
        _, _, mean_actions = actor(norm_obs)
        q_outputs = critic(norm_obs, mean_actions)
        if isinstance(q_outputs, tuple):
            q_list = list(q_outputs)
        else:
            q_list = q_outputs
        q_stack = torch.stack(q_list, dim=0)
        value = torch.min(q_stack, dim=0).values.cpu().numpy().reshape(mesh_x.shape)
        if q_stack.shape[0] == 1:
            disagreement = np.zeros_like(value)
        else:
            q_max = torch.max(q_stack, dim=0).values
            q_min = torch.min(q_stack, dim=0).values
            disagreement = (q_max - q_min).cpu().numpy().reshape(mesh_x.shape)
        actions = mean_actions.cpu().numpy().reshape(*mesh_x.shape, -1)

    traversable_grid = traversable.reshape(mesh_x.shape)

    return {
        "grid_x": mesh_x,
        "grid_y": mesh_y,
        "value": value,
        "disagreement": disagreement,
        "actions": actions,
        "traversable": traversable_grid,
        "maze_unit": maze_unit,
        "offsets": offsets,
    }


def plot_maps(xs: np.ndarray, ys: np.ndarray, results: dict, output_dir: Path,
              base_name: str, quiver_stride: int,
              maze_layout: np.ndarray | None = None,
              maze_unit: float | None = None,
              offsets: tuple[float, float] | None = None,
              goal: np.ndarray | None = None,
              dangerous_id: int = DANGEROUS_TILE_ID) -> tuple[Path, Path]:
    value = results["value"]
    disagreement = results["disagreement"]
    actions = results["actions"]

    dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
    dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
    extent = [xs.min() - dx / 2, xs.max() + dx / 2, ys.min() - dy / 2, ys.max() + dy / 2]

    value_masked = np.ma.masked_invalid(value)
    # Histogram equalization for value map
    flat_vals = value.flatten()
    finite_mask = np.isfinite(flat_vals)
    equalized = np.full_like(flat_vals, np.nan, dtype=float)
    if np.any(finite_mask):
        hist, bin_edges = np.histogram(flat_vals[finite_mask], bins=512)
        if hist.sum() > 0:
            cdf = np.cumsum(hist).astype(float)
            cdf /= cdf[-1]
            equalized_values = np.interp(flat_vals[finite_mask], bin_edges[1:], cdf, left=0.0, right=1.0)
            equalized[finite_mask] = equalized_values
    equalized_map = equalized.reshape(value.shape)
    equalized_masked = np.ma.masked_invalid(equalized_map)

    mag = np.linalg.norm(actions[..., :2], axis=-1)
    mag_masked = np.ma.masked_invalid(mag)
    disagreement_masked = np.ma.masked_invalid(disagreement)

    legend_handles: list = []

    def overlay(ax: plt.Axes, add_legend: bool = False, render_walls: bool = True) -> None:
        nonlocal legend_handles
        if maze_layout is not None and maze_unit is not None and offsets is not None:
            off_x, off_y = offsets
            h, w = maze_layout.shape
            for i in range(h):
                for j in range(w):
                    tile = maze_layout[i, j]
                    x = j * maze_unit - off_x
                    y = i * maze_unit - off_y
                    if tile in WALL_TILE_IDS and render_walls:
                        rect = patches.Rectangle(
                            (x - maze_unit / 2, y - maze_unit / 2),
                            maze_unit,
                            maze_unit,
                            linewidth=0.6,
                            edgecolor="black",
                            facecolor="grey",
                            alpha=1.0,
                        )
                        ax.add_patch(rect)
                    elif tile == dangerous_id:
                        rect = patches.Rectangle(
                            (x - maze_unit / 2, y - maze_unit / 2),
                            maze_unit,
                            maze_unit,
                            linewidth=0.3,
                            edgecolor="red",
                            facecolor="red",
                            alpha=0.5,
                        )
                        ax.add_patch(rect)
        if add_legend and not any(isinstance(h, patches.Patch) and h.get_label() == "Wall" for h in legend_handles):
            legend_handles.extend([
                patches.Patch(facecolor="black", alpha=0.6, label="Wall"),
                patches.Patch(facecolor="red", alpha=0.35, label="Dangerous tile"),
            ])
        if goal is not None:
            ax.scatter(goal[0], goal[1], c="white", marker="*", s=100,
                       edgecolors="black", linewidths=0.6)
            if add_legend and not any(isinstance(h, Line2D) and h.get_label() == "Goal" for h in legend_handles):
                legend_handles.append(
                    Line2D([0], [0], marker="*", color="black", markerfacecolor="white",
                           markeredgewidth=0.6, markersize=10, linestyle="none", label="Goal")
                )


    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

    im0 = axes[0].imshow(value_masked, extent=extent, cmap="viridis", aspect="equal", origin="lower")
    axes[0].set_title("min(Q1, Q2) — higher → better")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.8)
    cbar0.set_label("Estimated return (normalized)")

    im_eq = axes[1].imshow(equalized_masked, extent=extent, cmap="viridis", aspect="equal", origin="lower", vmin=0.0, vmax=1.0)
    axes[1].set_title("min(Q1, Q2) — histogram equalized")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    cbar_eq = fig.colorbar(im_eq, ax=axes[1], shrink=0.8)
    cbar_eq.set_label("Equalized percentile")

    im1 = axes[2].imshow(mag_masked, extent=extent, cmap="plasma", aspect="equal", origin="lower")
    axes[2].set_title("Actor mean — color = |a|, arrows = direction")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    cbar1 = fig.colorbar(im1, ax=axes[2], shrink=0.8)
    cbar1.set_label("Mean action magnitude")

    stride = max(1, quiver_stride)
    sub_x = xs[::stride]
    sub_y = ys[::stride]
    sub_actions = actions[::stride, ::stride, :2]
    valid_arrows = ~np.isnan(sub_actions).any(axis=-1)
    if np.any(valid_arrows):
        rows, cols = np.where(valid_arrows)
        axes[2].quiver(
            sub_x[cols],
            sub_y[rows],
            sub_actions[rows, cols, 0],
            sub_actions[rows, cols, 1],
            color="white",
            pivot="mid",
            angles="xy",
        )
    axes[2].text(
        0.02,
        0.95,
        "White arrows → mean action direction",
        transform=axes[2].transAxes,
        color="white",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=4),
    )

    im2 = axes[3].imshow(disagreement_masked, extent=extent, cmap="magma", aspect="equal", origin="lower")
    axes[3].set_title("|Q1 - Q2| — critic disagreement")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")
    cbar2 = fig.colorbar(im2, ax=axes[3], shrink=0.8)
    cbar2.set_label("Absolute value difference")

    for idx, ax in enumerate(axes):
        overlay(ax, add_legend=False, render_walls=True)
        if legend_handles and idx == 3:
            ax.legend(handles=legend_handles, loc="lower right")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{base_name}.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    return png_path


def generate_policy_map(
    model_path: Path,
    output_dir: Path,
    tag: str | None = None,
    env_name: str | None = None,
    device: str = "cpu",
    grid_resolution: int = 64,
    quiver_stride: int = 2,
    seed: int = 0,
    goal_override: np.ndarray | None = None,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    cache_path: Path | None = None,
    write_meta: bool = True,
) -> tuple[Path, Path | None, dict]:
    device_t = torch.device(device)
    ckpt, policy_type = load_checkpoint(model_path, device_t)
    actor, critic, obs_norm, train_args = build_networks(ckpt, device_t, policy_type)
    is_pixels = train_args.get("obs_mode", "state") == "pixels"

    if goal_override is not None:
        goal_override = np.asarray(goal_override, dtype=np.float32)

    cached_goal = None
    cached_x_range = None
    cached_y_range = None
    cached_seed = seed
    if cache_path is not None and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            cache_data = json.load(f)
        cached_goal = np.asarray(cache_data.get("goal"), dtype=np.float32)
        cached_x_range = cache_data.get("x_range")
        cached_y_range = cache_data.get("y_range")
        cached_seed = cache_data.get("seed", seed)
    else:
        cache_data = None

    goal_auto, bounds, maze_layout, maze_unit, offsets, dangerous_id = infer_goal_and_bounds(
        train_args, env_name, cached_seed
    )

    if goal_override is not None:
        goal = goal_override
    elif cached_goal is not None:
        goal = cached_goal
    else:
        goal = goal_auto
    x_range = x_range or cached_x_range or [bounds[0], bounds[1]]
    y_range = y_range or cached_y_range or [bounds[2], bounds[3]]

    goal = snap_goal_to_free(goal.astype(np.float32), maze_layout, maze_unit, offsets)

    xs = np.linspace(x_range[0], x_range[1], grid_resolution, dtype=np.float32)
    ys = np.linspace(y_range[0], y_range[1], grid_resolution, dtype=np.float32)

    free_mask = None
    if maze_layout is not None:
        free_mask = maze_layout == 0

    pixel_sampler = None
    try:
        if is_pixels:
            pixel_shape = ckpt.get('pixel_shape') or train_args.get('pixel_shape')
            if pixel_shape is None:
                raise ValueError('Pixel checkpoint missing pixel_shape metadata')
            pixel_shape = tuple(int(x) for x in pixel_shape)
            env_id_pixels = env_name or train_args.get('env_name')
            if env_id_pixels is None:
                raise ValueError('Cannot infer environment id for pixel visualization')
            camera_name = train_args.get('pixel_camera')
            camera_mode = train_args.get('pixel_camera_mode', 'global')
            pixel_sampler = PixelObservationSampler(
                env_id_pixels,
                width=int(pixel_shape[2]),
                height=int(pixel_shape[1]),
                camera_name=camera_name,
                pixel_camera_mode=camera_mode,
                pixel_local_view_size=float(train_args.get('pixel_local_view_size', 12.0)),
                pixel_local_camera_height=train_args.get('pixel_local_camera_height'),
                pixel_first_person_distance=float(train_args.get('pixel_first_person_distance', 3.0)),
                pixel_first_person_height=float(train_args.get('pixel_first_person_height', 1.0)),
                pixel_first_person_lookahead=float(train_args.get('pixel_first_person_lookahead', 2.0)),
                pixel_first_person_pitch=float(train_args.get('pixel_first_person_pitch', -15.0)),
                seed=cached_seed,
                device=device_t,
            )
            pixel_sampler.prepare(goal)

        results = evaluate_grid(
            actor,
            critic,
            obs_norm,
            train_args,
            goal,
            xs,
            ys,
            device_t,
            free_mask=free_mask,
            maze_unit=maze_unit,
            offsets=offsets,
            pixel_sampler=pixel_sampler,
        )
    finally:
        if pixel_sampler is not None:
            pixel_sampler.close()

    base_name = f"vis_{model_path.stem}" + (f"_{tag}" if tag else "")
    png_path = plot_maps(
        xs,
        ys,
        results,
        output_dir,
        base_name,
        quiver_stride,
        maze_layout=maze_layout,
        maze_unit=maze_unit,
        offsets=offsets,
        goal=goal,
        dangerous_id=dangerous_id or DANGEROUS_TILE_ID,
    )

    meta = {
        "model_path": str(model_path),
        "goal": goal.tolist(),
        "x_range": [float(xs.min()), float(xs.max())],
        "y_range": [float(ys.min()), float(ys.max())],
        "grid_resolution": grid_resolution,
        "quiver_stride": quiver_stride,
        "value_min": float(np.nanmin(results["value"])),
        "value_max": float(np.nanmax(results["value"])),
        "disagreement_mean": float(np.nanmean(results["disagreement"])),
        "dangerous_tile_id": int(dangerous_id or DANGEROUS_TILE_ID),
    }

    meta_path = None
    if write_meta:
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / f"{base_name}.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    if cache_path is not None and cache_data is None:
        cache_payload = {
            "goal": goal.tolist(),
            "x_range": x_range,
            "y_range": y_range,
            "seed": seed,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache_payload, f, indent=2)

    return png_path, meta_path, meta


def main():
    args = parse_args()
    png_path, meta_path, _ = generate_policy_map(
        model_path=args.model_path,
        output_dir=args.output_dir,
        tag=args.tag,
        env_name=args.env_name,
        device=args.device,
        grid_resolution=args.grid_resolution,
        quiver_stride=args.quiver_stride,
        seed=args.seed,
        goal_override=np.asarray(args.goal, dtype=np.float32) if args.goal is not None else None,
        x_range=args.x_range,
        y_range=args.y_range,
        cache_path=args.goal_cache,
    )
    print(f"Saved visualization to {png_path}")
    if meta_path:
        print(f"Metadata stored in {meta_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FastSAC training for OGBench environments using VectorizedOGBenchEnv and teacher interventions.

Matches CLI style of train_rsl_rl_integrated.py where practical.
"""

import os
import sys
import argparse
import time
import math
import json
import copy
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

# Add FastSAC path
sys.path.append('fasttd3/fast_sac')

from fast_sac import Actor, Critic
from fast_sac_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    SimpleReplayBuffer,
    save_params,
)

# Register OGBench envs and wrappers
import ogbench
from ogbench.wrappers import FlexibleObsWrapper, DetailedRewardWrapper, InterventionWrapper
from fasttd3.fast_sac.environments.ogbench_env import OGBenchVecEnvAdapter

TOOLS_PATH = Path(__file__).resolve().parent / "tools"
if TOOLS_PATH.exists():
    sys.path.append(str(TOOLS_PATH))


class PixelNormalizer(nn.Module):
    def forward(self, x):
        return x / 255.0


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
    def __init__(self, input_shape, feature_dim: int):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, feature_dim),
            nn.ReLU(),
        )
        self.output_dim = feature_dim

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] not in (1, 3):
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


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

    def min_q(self, features, actions):
        qs = self.forward(features, actions)
        stacked = torch.stack(qs, dim=0)
        return torch.min(stacked, dim=0).values
try:
    from visualize_policy_map import generate_policy_map  # type: ignore
except Exception:  # pragma: no cover - optional dependency for viz
    generate_policy_map = None


def parse_args():
    p = argparse.ArgumentParser()
    # Env
    p.add_argument('--env_name', type=str, default='pointmaze-arena-danger-lethal-v0')
    p.add_argument('--num_envs', type=int, default=64)
    p.add_argument('--total_timesteps', type=int, default=1_000_000)
    p.add_argument('--device', type=str, default='auto')
    # Observations
    p.add_argument('--include_goal', action='store_true', default=True)
    p.add_argument('--include_distance', action='store_true', default=False)
    p.add_argument('--include_direction', action='store_true', default=False)
    p.add_argument('--include_velocity', action='store_true', default=False)
    # Rewards
    p.add_argument('--reward_type', type=str, default='sparse', choices=['sparse','dense','combined'])
    p.add_argument('--dense_reward_scale', type=float, default=0.01)
    p.add_argument('--step_penalty', type=float, default=0.0)
    p.add_argument('--reward_switch_after_steps', type=int, default=0)
    p.add_argument('--switch_env_name', type=str, default=None,
                   help='Optional OGBench env id to switch to after a curriculum step')
    p.add_argument('--switch_env_after_steps', type=int, default=0,
                   help='Global env steps after which to switch to switch_env_name (0 disables)')
    # Intervention / Teacher
    p.add_argument('--use_intervention', action='store_true', default=False)
    p.add_argument('--intervention_mode', type=str, default='agent', choices=['human','agent'])
    p.add_argument('--teacher_type', type=str, default='bfs', choices=['bfs'])
    p.add_argument('--tolerance_type', type=str, default='angle', choices=['angle','l2'])
    p.add_argument('--tolerance_value', type=float, default=30.0)
    p.add_argument('--hard_block_lethal', action='store_true', default=True)
    p.add_argument('--no_hard_block_lethal', dest='hard_block_lethal', action='store_false')
    p.add_argument('--intervention_enable_after_steps', type=int, default=0)
    # SAC core (trimmed reasonable defaults)
    p.add_argument('--actor_learning_rate', type=float, default=3e-4)
    p.add_argument('--critic_learning_rate', type=float, default=3e-4)
    p.add_argument('--batch_size', type=int, default=32768)
    p.add_argument('--buffer_size', type=int, default=1024*50)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--policy_frequency', type=int, default=2)
    p.add_argument('--num_updates', type=int, default=2)
    p.add_argument('--learning_starts', type=int, default=1000)
    p.add_argument('--max_grad_norm', type=float, default=0.0)
    p.add_argument('--init_scale', type=float, default=0.01)
    p.add_argument('--actor_hidden_dim', type=int, default=512)
    p.add_argument('--critic_hidden_dim', type=int, default=1024)

    p.add_argument('--arch_shared_trunk', action='store_true', default=False,
                   help='Share an observation trunk between actor and critic(s)')
    p.add_argument('--shared_hidden_dim', type=int, default=512,
                   help='Hidden size for shared trunk when enabled')
    p.add_argument('--num_critics', type=int, default=2,
                   help='Number of critic heads (2 or 3 supported)')
    p.add_argument('--obs_mode', type=str, default='state', choices=['state', 'pixels'],
                   help='Observation mode: vector state or pixel images')
    p.add_argument('--pixel_width', type=int, default=64,
                   help='Pixel observation width when obs_mode=pixels')
    p.add_argument('--pixel_height', type=int, default=64,
                   help='Pixel observation height when obs_mode=pixels')
    p.add_argument('--pixel_camera', type=str, default=None,
                   help='Optional MuJoCo camera name for pixel observations (defaults to env setting)')
    p.add_argument('--store_denied_actions', action='store_true', default=False,
                   help='Add denied student actions to replay buffer with penalty reward')
    p.add_argument('--denied_action_penalty', type=float, default=-1.0,
                   help='Reward assigned to denied student actions when stored')
    # Intervention reward shaping variants
    p.add_argument('--intervention_reward_mode', type=str, default='none',
                   choices=['none', 'penalty_student', 'bonus_teacher'],
                   help='How to shape reward/actions on intervention')
    p.add_argument('--intervention_reward_value', type=float, default=0.0,
                   help='Magnitude for intervention reward shaping (e.g., 0.1)')
    p.add_argument('--bonus_teacher_value', type=float, default=0.0,
                   help='Additional reward added when teacher action is applied (can combine with penalty modes)')
    # Logging
    p.add_argument('--use_wandb', action='store_true', default=False)
    p.add_argument('--project', type=str, default='ogbench-rsl-rl')
    p.add_argument('--exp_name', type=str, default=None)
    p.add_argument('--save_interval', type=int, default=200000,
                   help='Env-step interval for checkpoint saves (0 disables)')
    p.add_argument('--log_interval', type=int, default=200)
    p.add_argument('--post_switch_viz_multiplier', type=int, default=1,
                   help='Reduce checkpoint/viz interval by this factor after curriculum/env switch (>=1)')
    p.add_argument('--disagreement_hist_edges', type=str, default='',
                   help='Comma-separated positive edges for disagreement hist bins (teacher/non-teacher). Empty to disable histogram logging.')
    p.add_argument('--disagreement_thresholds', type=str, default='0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0,2.0,5.0',
                   help='Comma-separated thresholds to log fraction of interventions with disagreement >= threshold')
    # Misc
    p.add_argument('--compile', action='store_true', default=False)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16','fp16'])
    # Counterfactual buffer (student-denied actions) for fast adaptation
    p.add_argument('--cf_buffer_enable', action='store_true', default=False,
                   help='Enable counterfactual buffer for denied student actions')
    p.add_argument('--cf_capacity', type=int, default=100000,
                   help='Capacity of CF buffer (rows)')
    p.add_argument('--cf_sample_ratio', type=float, default=0.5,
                   help='Fraction of batch for CF critic loss (0..1)')
    p.add_argument('--cf_penalty', type=float, default=1.0,
                   help='Positive penalty magnitude; critic target becomes -abs(value) for denied actions')
    p.add_argument('--cf_q_weight', type=float, default=1.0,
                   help='Weight for CF critic penalty loss')
    # Preference buffer (pairwise ranking on intervened rows: teacher vs student)
    p.add_argument('--pref_buffer_enable', action='store_true', default=False,
                   help='Enable preference buffer storing pairs (s, a_teacher, a_student) and ranking loss')
    p.add_argument('--pref_capacity', type=int, default=100000,
                   help='Capacity of preference buffer (pairs)')
    p.add_argument('--pref_sample_ratio', type=float, default=0.5,
                   help='Fraction of update batch for preference pairs (0..1)')
    p.add_argument('--pref_rank_weight', type=float, default=1.0,
                   help='Weight for the pairwise ranking loss added to critic loss')
    p.add_argument('--pref_rank_margin', type=float, default=0.1,
                   help='Margin for ranking loss: softplus(margin - (Qpos - Qneg))')
    # Preference-TD buffer (balanced TD samples: teacher real transition; student synthetic terminal negative)
    p.add_argument('--pref_td_buffer_enable', action='store_true', default=False,
                   help='Enable preference-TD buffer that keeps balanced teacher/student TD transitions')
    p.add_argument('--pref_td_capacity', type=int, default=100000,
                   help='Capacity per-role (teacher/student) for preference-TD buffer')
    p.add_argument('--pref_td_sample_ratio', type=float, default=0.5,
                   help='Fraction of update batch to draw from preference-TD buffer (split 50/50 teacher/student)')
    p.add_argument('--pref_td_q_weight', type=float, default=1.0,
                   help='Weight for additional critic TD loss from preference-TD samples')
    p.add_argument('--pref_td_penalty_value', type=float, default=0.1,
                   help='Negative reward assigned to synthetic student terminal in preference-TD buffer')
    p.add_argument('--pref_td_teacher_bonus_value', type=float, default=0.0,
                   help='Optional extra reward added to teacher transitions in preference-TD buffer')
    # Replay buffer reset on curriculum switch
    p.add_argument('--reset_replay_on_switch', action='store_true', default=False,
                   help='Reset main replay buffer when reward_switch_after_steps is reached')
    p.add_argument('--reset_critic_on_switch', action='store_true', default=False,
                   help='Reload critic weights/optimiser to initial state at curriculum switch')
    # Policy visualization
    p.add_argument('--viz_on_checkpoint', action='store_true', default=False,
                   help='Render policy/critic maps whenever a checkpoint is saved')
    p.add_argument('--viz_grid_resolution', type=int, default=64,
                   help='Grid resolution for policy maps if enabled')
    p.add_argument('--viz_quiver_stride', type=int, default=1,
                   help='Stride for quiver arrows in policy maps')
    p.add_argument('--viz_device', type=str, default='cpu',
                   help='Device to use when generating policy maps')
    p.add_argument('--viz_seed', type=int, default=0,
                   help='Seed used to sample/lock the visualization goal location')
    return p.parse_args()


def make_wrappers(args):
    def _apply(env):
        if args.obs_mode == 'state':
            env = FlexibleObsWrapper(
                env,
                include_goal=args.include_goal,
                include_distance=args.include_distance,
                include_direction=args.include_direction,
                include_velocity=args.include_velocity,
            )
        env = DetailedRewardWrapper(
            env,
            reward_type=args.reward_type,
            dense_reward_scale=args.dense_reward_scale,
            step_penalty=args.step_penalty,
            switch_reward_to_sparse_after_steps_per_env=(args.reward_switch_after_steps // max(1, args.num_envs)),
        )
        if args.use_intervention and args.intervention_mode == 'agent':
            env = InterventionWrapper(
                env,
                mode='agent',
                teacher_type=args.teacher_type,
                tolerance_type=args.tolerance_type,
                tolerance_value=args.tolerance_value,
                hard_block_lethal=args.hard_block_lethal,
                enable_after_steps=args.intervention_enable_after_steps,
            )
        return env
    return [_apply]


def main():
    args = parse_args()
    device = torch.device('cuda' if (args.device=='auto' and torch.cuda.is_available()) or args.device=='cuda' else 'cpu')

    cf_penalty_target = -abs(float(args.cf_penalty))
    setattr(args, 'cf_penalty_target', cf_penalty_target)

    default_buffer_size = 1024 * 50
    if not args.exp_name:
        env_tag = args.env_name.replace('-v0', '').replace('-', '_')
        components = [env_tag]
        components.append(args.reward_type)
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        components.append(timestamp)
        args.exp_name = '_'.join(components)

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

    print(f"FastSAC OGBench on {args.env_name} device={device}")
    print(f"Log directory: {run_log_dir}")
    print(f"Model directory: {run_model_dir}")

    # Open progress log early so we can write init breadcrumbs
    log_file_path = run_log_dir / 'training.log'
    progress_file = open(log_file_path, 'a', encoding='utf-8')
    progress_file.write(f"# logging started {datetime.now().isoformat()}\n")
    progress_file.flush()

    def record_progress(message: str):
        progress_file.write(f"{datetime.now().isoformat()} {message}\n")
        progress_file.flush()

    config_path = run_log_dir / 'args.json'
    with open(config_path, 'w', encoding='utf-8') as cfg_file:
        json.dump(vars(args), cfg_file, indent=2)
    print(f"Saved run config: {config_path}")

    wrappers = make_wrappers(args)
    current_wrappers = wrappers
    current_env_name = args.env_name
    env_make_kwargs = {}
    if args.obs_mode == 'pixels':
        env_make_kwargs['render_mode'] = 'rgb_array'
        if args.pixel_width:
            env_make_kwargs['width'] = int(args.pixel_width)
        if args.pixel_height:
            env_make_kwargs['height'] = int(args.pixel_height)
        if args.pixel_camera:
            env_make_kwargs['camera_name'] = args.pixel_camera
    record_progress("[Init] constructing vector env adapter")
    envs = OGBenchVecEnvAdapter(
        env_name=current_env_name,
        num_envs=args.num_envs,
        device=device,
        wrappers=current_wrappers,
        clip_actions=1.0,
        **env_make_kwargs,
    )
    record_progress("[Init] env adapter constructed")
    print("[Init] Env adapter constructed", flush=True)

    n_obs = envs.num_obs
    n_act = envs.num_actions

    raw_obs_space = envs._env.envs[0].observation_space
    pixel_shape = None
    if args.obs_mode == 'pixels':
        if len(raw_obs_space.shape) != 3:
            raise RuntimeError('Pixel observation expected to have 3 dims')
        raw_shape = raw_obs_space.shape
        if raw_shape[0] in (1, 3, 4):
            pixel_shape = raw_shape
        elif raw_shape[-1] in (1, 3, 4):
            pixel_shape = (raw_shape[-1], raw_shape[0], raw_shape[1])
        else:
            raise RuntimeError(f'Unable to determine channel dimension for pixel observations: {raw_shape}')
        obs_normalizer = IdentityNormalizer().to(device)
        critic_obs_normalizer = IdentityNormalizer().to(device)
    else:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)

    reward_normalizer = RewardNormalizer(gamma=args.gamma, device=device, g_max=10.0)

    def build_backbone(mode, hidden_dim):
        if mode == 'pixels':
            c, h, w = pixel_shape
            return PixelBackbone((c, h, w), hidden_dim).to(device)
        return MLPBackbone(n_obs, hidden_dim).to(device)

    shared_backbone = None
    initial_shared_backbone_state = None
    initial_critic_backbone_state = None
    if args.arch_shared_trunk:
        shared_backbone = build_backbone(args.obs_mode, args.shared_hidden_dim)
        if args.obs_mode == 'pixels' and hasattr(shared_backbone, 'fc'):
            record_progress(f'[Init] shared trunk fc weight {shared_backbone.fc[0].weight.shape}')
        actor_backbone = shared_backbone
        actor_head = GaussianPolicyHead(shared_backbone.output_dim, n_act, args.actor_hidden_dim, args.init_scale).to(device)
        critic_backbone = shared_backbone
        critic_heads = CriticEnsemble(shared_backbone.output_dim, n_act, args.critic_hidden_dim, args.num_critics).to(device)
    else:
        actor_backbone = build_backbone(args.obs_mode, args.actor_hidden_dim)
        actor_head = GaussianPolicyHead(actor_backbone.output_dim, n_act, args.actor_hidden_dim, args.init_scale).to(device)
        critic_backbone = build_backbone(args.obs_mode, args.critic_hidden_dim)
        critic_heads = CriticEnsemble(critic_backbone.output_dim, n_act, args.critic_hidden_dim, args.num_critics).to(device)

    critic_feature_backbone = actor_backbone if args.arch_shared_trunk else critic_backbone
    # Target critic components
    critic_target_backbone = copy.deepcopy(critic_backbone)
    critic_target_heads = copy.deepcopy(critic_heads)

    # Optimizers
    if args.arch_shared_trunk:
        trunk_params = list(shared_backbone.parameters())
        actor_params = list(actor_head.parameters())
        critic_params = list(critic_heads.parameters())
        trunk_optimizer = optim.Adam(trunk_params, lr=args.critic_learning_rate)
        q_optimizer = optim.AdamW(critic_params, lr=args.critic_learning_rate, weight_decay=0.1)
        actor_optimizer = optim.AdamW(actor_params, lr=args.actor_learning_rate, weight_decay=0.1)
    else:
        trunk_optimizer = None
        actor_params = list(actor_backbone.parameters()) + list(actor_head.parameters())
        critic_params = list(critic_backbone.parameters()) + list(critic_heads.parameters())
        q_optimizer = optim.AdamW(critic_params, lr=args.critic_learning_rate, weight_decay=0.1)
        actor_optimizer = optim.AdamW(actor_params, lr=args.actor_learning_rate, weight_decay=0.1)

    if args.arch_shared_trunk:
        initial_shared_backbone_state = copy.deepcopy(shared_backbone.state_dict())
    else:
        initial_critic_backbone_state = copy.deepcopy(critic_backbone.state_dict())
    initial_critic_heads_state = copy.deepcopy(critic_heads.state_dict())
    initial_target_backbone_state = copy.deepcopy(critic_target_backbone.state_dict())
    initial_target_heads_state = copy.deepcopy(critic_target_heads.state_dict())


    def infer_pixel_shape(obs_input):
        data = obs_input
        if isinstance(data, tuple):
            data = data[0]
        if isinstance(data, dict):
            for key in ('policy', 'pixels', 'image', 'observation'):
                if key in data:
                    data = data[key]
                    break
            else:
                raise KeyError(f"Unknown observation keys {list(data.keys())}")
        if isinstance(data, torch.Tensor):
            sample = data[0]
        elif isinstance(data, np.ndarray):
            sample = data[0] if data.ndim == 4 else data
        else:
            sample = torch.as_tensor(data)[0]
        shape = tuple(sample.shape)
        if len(shape) == 4:
            shape = shape[1:]
        if shape[0] in (1, 3, 4):
            return shape
        if shape[-1] in (1, 3, 4):
            return (shape[-1], shape[0], shape[1])
        raise RuntimeError(f"Unable to determine channel dimension from {shape}")

    def prepare_obs(obs_input):
        tensor = obs_input
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        if isinstance(tensor, dict):
            for key in ('policy', 'pixels', 'image', 'observation'):
                if key in tensor:
                    tensor = tensor[key]
                    break
            else:
                raise KeyError(f"Unknown observation keys {list(tensor.keys())}")
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        tensor = tensor.to(device)
        if args.obs_mode == 'pixels':
            if tensor.ndim == 4:
                if tensor.shape[-1] in (1, 3, 4) and tensor.shape[1] not in (1, 3, 4):
                    tensor = tensor.permute(0, 3, 1, 2).contiguous()
            if tensor.dtype in (torch.uint8, torch.int8):
                tensor = tensor.float().div(255.0)
        return tensor.view(tensor.shape[0], -1)

    def reshape_obs(obs_flat: torch.Tensor):
        if args.obs_mode == 'pixels':
            return obs_flat.view(obs_flat.shape[0], *pixel_shape)
        return obs_flat

    def actor_forward(obs_flat: torch.Tensor):
        obs_in = reshape_obs(obs_flat)
        features = actor_backbone(obs_in)
        action, log_pi, mean = actor_head(features)
        return action, log_pi, mean, features

    def critic_forward(backbone_module, heads_module, obs_flat: torch.Tensor, actions: torch.Tensor):
        obs_in = reshape_obs(obs_flat)
        features = backbone_module(obs_in)
        q_values = heads_module(features, actions)
        return features, q_values

    target_entropy = -float(n_act)
    log_alpha = torch.ones(1, requires_grad=True, device=device)
    log_alpha.data.copy_(torch.tensor([np.log(0.001)], device=device))
    alpha_optimizer = optim.Adam([log_alpha], lr=args.critic_learning_rate)

    rb = SimpleReplayBuffer(
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
    )

    # Initialize a simple counterfactual buffer (on-device ring buffer) if enabled
    if args.cf_buffer_enable:
        cf_capacity = int(max(1, args.cf_capacity))
        cf_obs = torch.empty((cf_capacity, n_obs), dtype=torch.float32, device=device)
        cf_act = torch.empty((cf_capacity, n_act), dtype=torch.float32, device=device)
        cf_ptr = 0
        cf_size = 0

        def cf_append(s_batch: torch.Tensor, a_batch: torch.Tensor):
            nonlocal cf_ptr, cf_size
            if s_batch is None or a_batch is None:
                return
            b = int(s_batch.shape[0])
            if b <= 0:
                return
            # If incoming is larger than capacity, keep only the last cf_capacity rows
            if b > cf_capacity:
                s_batch = s_batch[-cf_capacity:]
                a_batch = a_batch[-cf_capacity:]
                b = cf_capacity
            end = cf_ptr + b
            if end <= cf_capacity:
                cf_obs[cf_ptr:end].copy_(s_batch)
                cf_act[cf_ptr:end].copy_(a_batch)
            else:
                first = cf_capacity - cf_ptr
                cf_obs[cf_ptr:].copy_(s_batch[:first])
                cf_act[cf_ptr:].copy_(a_batch[:first])
                remain = b - first
                cf_obs[:remain].copy_(s_batch[first:])
                cf_act[:remain].copy_(a_batch[first:])
            cf_ptr = (cf_ptr + b) % cf_capacity
            cf_size = min(cf_size + b, cf_capacity)

        def cf_sample(b: int):
            if cf_size <= 0:
                return None, None
            idx = torch.randint(low=0, high=cf_size, size=(int(b),), device=device)
            s = cf_obs[idx]
            a = cf_act[idx]
            # Normalize states consistently with main replay samples
            s = _normalize_obs(s)
            return s, a
    else:
        cf_obs = None
        cf_act = None
        cf_ptr = 0
        cf_size = 0

    # Initialize preference pair buffer if enabled: stores (s, a_teacher, a_student)
    if args.pref_buffer_enable:
        pref_capacity = int(max(1, args.pref_capacity))
        pref_s = torch.empty((pref_capacity, n_obs), dtype=torch.float32, device=device)
        pref_a_pos = torch.empty((pref_capacity, n_act), dtype=torch.float32, device=device)
        pref_a_neg = torch.empty((pref_capacity, n_act), dtype=torch.float32, device=device)
        pref_ptr = 0
        pref_size = 0

        def pref_append(s_batch: torch.Tensor, a_pos: torch.Tensor, a_neg: torch.Tensor):
            nonlocal pref_ptr, pref_size
            if s_batch is None or a_pos is None or a_neg is None:
                return
            b = int(s_batch.shape[0])
            if b <= 0:
                return
            if b > pref_capacity:
                s_batch = s_batch[-pref_capacity:]
                a_pos = a_pos[-pref_capacity:]
                a_neg = a_neg[-pref_capacity:]
                b = pref_capacity
            end = pref_ptr + b
            if end <= pref_capacity:
                pref_s[pref_ptr:end].copy_(s_batch)
                pref_a_pos[pref_ptr:end].copy_(a_pos)
                pref_a_neg[pref_ptr:end].copy_(a_neg)
            else:
                first = pref_capacity - pref_ptr
                pref_s[pref_ptr:].copy_(s_batch[:first])
                pref_a_pos[pref_ptr:].copy_(a_pos[:first])
                pref_a_neg[pref_ptr:].copy_(a_neg[:first])
                remain = b - first
                pref_s[:remain].copy_(s_batch[first:])
                pref_a_pos[:remain].copy_(a_pos[first:])
                pref_a_neg[:remain].copy_(a_neg[first:])
            pref_ptr = (pref_ptr + b) % pref_capacity
            pref_size = min(pref_size + b, pref_capacity)

        def pref_sample(b: int):
            if pref_size <= 0:
                return None, None, None
            idx = torch.randint(low=0, high=pref_size, size=(int(b),), device=device)
            s = pref_s[idx]
            a_pos = pref_a_pos[idx]
            a_neg = pref_a_neg[idx]
            s = _normalize_obs(s)
            return s, a_pos, a_neg
    else:
        pref_s = pref_a_pos = pref_a_neg = None
        pref_ptr = 0
        pref_size = 0

    # Initialize preference-TD buffer if enabled: balanced teacher/student TD samples
    if args.pref_td_buffer_enable:
        td_cap = int(max(1, args.pref_td_capacity))
        # Teacher ring buffers
        t_s = torch.empty((td_cap, n_obs), dtype=torch.float32, device=device)
        t_a = torch.empty((td_cap, n_act), dtype=torch.float32, device=device)
        t_r = torch.empty((td_cap, 1), dtype=torch.float32, device=device)
        t_next_s = torch.empty((td_cap, n_obs), dtype=torch.float32, device=device)
        t_done = torch.empty((td_cap, 1), dtype=torch.float32, device=device)
        t_ptr = 0
        t_size = 0
        # Student ring buffers (terminal negatives)
        s_s = torch.empty((td_cap, n_obs), dtype=torch.float32, device=device)
        s_a = torch.empty((td_cap, n_act), dtype=torch.float32, device=device)
        s_r = torch.empty((td_cap, 1), dtype=torch.float32, device=device)
        s_ptr = 0
        s_size = 0

        def pref_td_append_teacher(s_batch, a_batch, r_batch, next_s_batch, done_batch):
            nonlocal t_ptr, t_size
            if s_batch is None or a_batch is None:
                return
            b = int(s_batch.shape[0])
            if b <= 0:
                return
            if b > td_cap:
                s_batch = s_batch[-td_cap:]
                a_batch = a_batch[-td_cap:]
                r_batch = r_batch[-td_cap:]
                next_s_batch = next_s_batch[-td_cap:]
                done_batch = done_batch[-td_cap:]
                b = td_cap
            end = t_ptr + b
            if end <= td_cap:
                t_s[t_ptr:end].copy_(s_batch)
                t_a[t_ptr:end].copy_(a_batch)
                t_r[t_ptr:end].copy_(r_batch.view(-1, 1))
                t_next_s[t_ptr:end].copy_(next_s_batch)
                t_done[t_ptr:end].copy_(done_batch.view(-1, 1))
            else:
                first = td_cap - t_ptr
                t_s[t_ptr:].copy_(s_batch[:first])
                t_a[t_ptr:].copy_(a_batch[:first])
                t_r[t_ptr:].copy_(r_batch[:first].view(-1, 1))
                t_next_s[t_ptr:].copy_(next_s_batch[:first])
                t_done[t_ptr:].copy_(done_batch[:first].view(-1, 1))
                remain = b - first
                t_s[:remain].copy_(s_batch[first:])
                t_a[:remain].copy_(a_batch[first:])
                t_r[:remain].copy_(r_batch[first:].view(-1, 1))
                t_next_s[:remain].copy_(next_s_batch[first:])
                t_done[:remain].copy_(done_batch[first:].view(-1, 1))
            t_ptr = (t_ptr + b) % td_cap
            t_size = min(t_size + b, td_cap)

        def pref_td_append_student(s_batch, a_batch, r_batch):
            nonlocal s_ptr, s_size
            if s_batch is None or a_batch is None:
                return
            b = int(s_batch.shape[0])
            if b <= 0:
                return
            if b > td_cap:
                s_batch = s_batch[-td_cap:]
                a_batch = a_batch[-td_cap:]
                r_batch = r_batch[-td_cap:]
                b = td_cap
            end = s_ptr + b
            if end <= td_cap:
                s_s[s_ptr:end].copy_(s_batch)
                s_a[s_ptr:end].copy_(a_batch)
                s_r[s_ptr:end].copy_(r_batch.view(-1, 1))
            else:
                first = td_cap - s_ptr
                s_s[s_ptr:].copy_(s_batch[:first])
                s_a[s_ptr:].copy_(a_batch[:first])
                s_r[s_ptr:].copy_(r_batch[:first].view(-1, 1))
                remain = b - first
                s_s[:remain].copy_(s_batch[first:])
                s_a[:remain].copy_(a_batch[first:])
                s_r[:remain].copy_(r_batch[first:].view(-1, 1))
            s_ptr = (s_ptr + b) % td_cap
            s_size = min(s_size + b, td_cap)

        def pref_td_sample(b: int):
            b_teacher = max(1, int(b // 2))
            b_student = max(1, b - b_teacher)
            if t_size <= 0 or s_size <= 0:
                return None
            idx_t = torch.randint(low=0, high=t_size, size=(b_teacher,), device=device)
            idx_s = torch.randint(low=0, high=s_size, size=(b_student,), device=device)
            batch = {
                't_s': _normalize_obs(t_s[idx_t]),
                't_a': t_a[idx_t],
                't_r': t_r[idx_t],
                't_next_s': _normalize_obs(t_next_s[idx_t]),
                't_done': t_done[idx_t],
                's_s': _normalize_obs(s_s[idx_s]),
                's_a': s_a[idx_s],
                's_r': s_r[idx_s],
            }
            return batch
    else:
        t_s = t_a = t_r = t_next_s = t_done = None
        s_s = s_a = s_r = None
        t_ptr = t_size = s_ptr = s_size = 0

    amp_enabled = args.amp and (device.type=='cuda')
    amp_device_type = 'cuda' if device.type=='cuda' else 'cpu'
    amp_dtype = torch.bfloat16 if args.amp_dtype=='bf16' else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype==torch.float16)

    # Compile optional
    def _normalize_obs(x): return obs_normalizer(x)
    if args.compile:
        actor_backbone = torch.compile(actor_backbone)
        actor_head = torch.compile(actor_head)
        if not args.arch_shared_trunk:
            critic_backbone = torch.compile(critic_backbone)
        critic_heads = torch.compile(critic_heads)
        critic_target_backbone = torch.compile(critic_target_backbone)
        critic_target_heads = torch.compile(critic_target_heads)
        _normalize_obs = torch.compile(_normalize_obs)

    try:
        # Initialize wandb early so runs appear even if logging hasn't triggered yet
        if args.use_wandb:
            import wandb
            wandb_run = wandb.init(
                project=args.project,
                name=args.exp_name,
                id=args.exp_name,
                config=vars(args),
                reinit=True,
                resume="allow",
            )
        else:
            wandb_run = None
        obs_raw = envs.reset()
        if args.obs_mode == 'pixels':
            pixel_shape = infer_pixel_shape(obs_raw)
            record_progress(f'[Pixels] inferred pixel shape {pixel_shape}')
        obs = prepare_obs(obs_raw)
        n_obs = obs.shape[1]
        record_progress("[Init] envs.reset() returned; entering loop")
        print("[Init] Env reset complete; starting training loop", flush=True)
        total_env_steps = 0
        iteration_idx = 0
        start_time = time.time()
        # wandb_run may be set above
        # Episode buffers similar to RSL-RL
        cur_reward_sum = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
        cur_episode_length = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)
        rewbuffer = []
        lenbuffer = []
        last_denied_samples = 0
        next_log_step = args.log_interval if args.log_interval > 0 else None
        save_interval_current = args.save_interval if args.save_interval > 0 else None
        next_save_step = save_interval_current if save_interval_current else None

        run_prefix = current_env_name.replace('-', '_')

        # Learning warm-up threshold (also reused after any replay reset)
        next_learning_starts_at = int(args.learning_starts)

        def save_checkpoint(tag: str, step_value: int):
            save_path = run_model_dir / f"{run_prefix}_{tag}.pt"
            checkpoint = {
                'step': step_value,
                'actor_backbone': actor_backbone.state_dict(),
                'actor_head': actor_head.state_dict(),
                'critic_backbone': (None if args.arch_shared_trunk else critic_backbone.state_dict()),
                'shared_backbone': actor_backbone.state_dict() if args.arch_shared_trunk else None,
                'critic_heads': critic_heads.state_dict(),
                'critic_target_backbone': critic_target_backbone.state_dict(),
                'critic_target_heads': critic_target_heads.state_dict(),
                'obs_normalizer_state': (obs_normalizer.state_dict() if hasattr(obs_normalizer, 'state_dict') else None),
                'critic_obs_normalizer_state': (critic_obs_normalizer.state_dict() if hasattr(critic_obs_normalizer, 'state_dict') else None),
                'log_alpha': log_alpha.detach().cpu().item(),
                'pixel_shape': pixel_shape,
                'args': vars(args),
            }
            torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
            record_progress(f"[Checkpoint] saved {save_path}")
            return save_path

        def maybe_render_policy_map(tag: str, step_value: int, checkpoint_path: Path):
            if not args.viz_on_checkpoint or generate_policy_map is None:
                return
            try:
                png_path, meta_path, _ = generate_policy_map(
                    model_path=checkpoint_path,
                    output_dir=viz_output_dir,
                    tag=tag,
                    env_name=current_env_name,
                    device=args.viz_device,
                    grid_resolution=args.viz_grid_resolution,
                    quiver_stride=args.viz_quiver_stride,
                    seed=args.viz_seed,
                    cache_path=viz_cache_path,
                )
                record_progress(f"[Viz] generated {png_path}")
                if args.use_wandb and wandb_run is not None:
                    import wandb
                    wandb_run.log({
                        "viz/policy_map": wandb.Image(str(png_path), caption=tag),
                    }, step=step_value)
            except Exception as exc:  # pragma: no cover - best effort logging
                record_progress(f"[Viz] failed for {tag}: {exc}")

        print("[Init] Starting training loop", flush=True)

        # Track if we have reset the main replay buffer at the curriculum switch
        did_reset_replay = False
        did_switch_env = False
        env_switch_global_step = int(max(0, args.switch_env_after_steps))

        # Disagreement tracking
        if args.disagreement_hist_edges:
            try:
                parsed_edges = [float(edge.strip()) for edge in args.disagreement_hist_edges.split(',') if edge.strip()]
                parsed_edges = sorted([edge for edge in parsed_edges if edge > 0.0])
            except Exception:
                parsed_edges = []
            if parsed_edges:
                hist_edges_tensor = torch.tensor(parsed_edges, dtype=torch.float32, device=device)
                num_hist_bins = hist_edges_tensor.numel() + 1
                hist_labels = []
                for idx in range(num_hist_bins):
                    if idx == 0:
                        hist_labels.append(f"<= {parsed_edges[0]:.2f}")
                    elif idx == num_hist_bins - 1:
                        hist_labels.append(f">= {parsed_edges[-1]:.2f}")
                    else:
                        hist_labels.append(f"({parsed_edges[idx-1]:.2f}, {parsed_edges[idx]:.2f}]")
                teacher_hist_counts = torch.zeros(num_hist_bins, device=device)
                non_teacher_hist_counts = torch.zeros(num_hist_bins, device=device)
            else:
                hist_edges_tensor = None
                hist_labels = []
                teacher_hist_counts = None
                non_teacher_hist_counts = None
        else:
            hist_edges_tensor = None
            hist_labels = []
            teacher_hist_counts = None
            non_teacher_hist_counts = None

        teacher_disagreement_sum = 0.0
        non_teacher_disagreement_sum = 0.0
        teacher_disagreement_steps = 0.0
        non_teacher_disagreement_steps = 0.0
        corr_total_steps = 0.0
        corr_sum_mask = 0.0
        corr_sum_dis = 0.0
        corr_sum_mask_sq = 0.0
        corr_sum_dis_sq = 0.0
        corr_sum_mask_dis = 0.0

        # Threshold percentages (fraction of teacher interventions with disagreement >= threshold)
        if args.disagreement_thresholds:
            try:
                thresh_vals = [float(x.strip()) for x in args.disagreement_thresholds.split(',') if x.strip()]
                thresh_vals = sorted([t for t in thresh_vals if t > 0.0])
            except Exception:
                thresh_vals = []
        else:
            thresh_vals = []
        if thresh_vals:
            thresh_tensor = torch.tensor(thresh_vals, dtype=torch.float32, device=device)
            teacher_above_counts = torch.zeros(len(thresh_vals), dtype=torch.float32, device=device)
            teacher_steps_window = 0.0
        else:
            thresh_tensor = None
            teacher_above_counts = None
            teacher_steps_window = 0.0

        # Q and disagreement running averages for the current log window
        qmin_sum_all = 0.0
        qmin_steps_all = 0.0
        qmin_sum_teacher = 0.0
        qmin_steps_teacher = 0.0
        qmin_sum_non = 0.0
        qmin_steps_non = 0.0

        while total_env_steps < args.total_timesteps:
            norm_obs = _normalize_obs(obs)
            obs_actor_input = reshape_obs(norm_obs)
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                if args.obs_mode == 'pixels' and iteration_idx == 0 and total_env_steps == 0:
                    record_progress(f'[Pixels] actor input shape {obs_actor_input.shape}, pixel_shape {pixel_shape}')
                try:
                    features_actor = actor_backbone(obs_actor_input)
                except RuntimeError as exc:
                    record_progress(f'[Pixels] backbone failure: input shape {obs_actor_input.shape}, pixel_shape {pixel_shape}')
                    raise
                if args.obs_mode == 'pixels' and iteration_idx == 0 and total_env_steps == 0:
                    record_progress(f'[Pixels] backbone output shape {features_actor.shape}')
                pi_action, _, _ = actor_head(features_actor)
            next_obs_raw, rewards, dones, infos = envs.step(pi_action.float())
            actions = pi_action
            next_obs = prepare_obs(next_obs_raw)
            # Detach copies for any auxiliary buffers before we mutate below
            obs_detached_flat = obs.detach()
            next_obs_detached_flat = next_obs.detach()
            truncations = infos.get('time_outs', torch.zeros_like(dones, device=device))
            applied_actions = infos.get('applied_actions', actions)
            student_actions = infos.get('student_actions')
            teacher_mask = infos.get('teacher_intervened_mask')
            # Build mask if missing by comparing applied vs student actions
            if teacher_mask is None and student_actions is not None and applied_actions is not None:
                try:
                    teacher_mask = (torch.abs(applied_actions - student_actions).sum(dim=-1) > 1e-6)
                except Exception:
                    teacher_mask = None

            # Decide actions to store and reward shaping
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
                        # Store the STUDENT action for intervened rows and apply a negative adjustment
                        if student_actions is not None:
                            used_actions = used_actions.clone()
                            used_actions[denied_ids] = student_actions[denied_ids]
                        if val != 0.0:
                            rewards_eff = rewards_eff.clone()
                            rewards_eff[denied_ids] = rewards_eff[denied_ids] - abs(val)
                        # Mark these rows as terminal so bootstrapping stops
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

                    # Append to preference pair buffer (s, a_teacher, a_student)
                    if args.pref_buffer_enable and student_actions is not None and 'teacher_actions' in infos:
                        try:
                            a_teacher_all = infos['teacher_actions']
                            s_batch = obs_detached_flat[denied_ids]
                            pref_append(obs_detached_flat[denied_ids], a_teacher_all[denied_ids], student_actions[denied_ids])
                        except Exception:
                            pass

                    # Append to preference-TD buffer (balanced TD samples)
                    if args.pref_td_buffer_enable and student_actions is not None and 'teacher_actions' in infos:
                        try:
                            a_teacher_all = infos['teacher_actions']
                            s_now = obs_detached_flat[denied_ids]
                            s_next = next_obs_detached_flat[denied_ids]
                            r_teacher = rewards[denied_ids].clone().view(-1, 1)
                            if float(args.pref_td_teacher_bonus_value) != 0.0:
                                r_teacher = r_teacher + float(args.pref_td_teacher_bonus_value)
                            d_teacher = dones[denied_ids].clone().view(-1, 1).float()
                            pref_td_append_teacher(s_now, a_teacher_all[denied_ids], r_teacher.squeeze(1), s_next, d_teacher.squeeze(1))
                            # Student synthetic terminal negatives
                            r_student = -abs(float(args.pref_td_penalty_value)) * torch.ones((denied_ids.numel(), 1), device=device, dtype=torch.float32)
                            pref_td_append_student(s_now, student_actions[denied_ids], r_student.squeeze(1))
                        except Exception:
                            pass
            
            # Build transition
            obs_detached = obs_detached_flat
            next_obs_detached = next_obs_detached_flat
            transition = TensorDict(
                {
                    'observations': obs_detached,
                    'actions': used_actions.detach(),
                    'next': {
                        'observations': next_obs_detached,
                        'rewards': rewards_eff.detach(),
                        'truncations': truncations.long(),
                        'dones': dones_eff.long(),
                    },
                },
                batch_size=(envs.num_envs,),
                device=device,
            )
            rb.extend(transition)

            # Critic disagreement logging data (evaluate on executed action)
            if args.pref_buffer_enable or args.pref_td_buffer_enable or args.cf_buffer_enable or True:
                with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                    norm_obs_now = _normalize_obs(obs)
                    features_now = critic_feature_backbone(reshape_obs(norm_obs_now))
                    q_stack = torch.stack(critic_heads(features_now, used_actions), dim=0).squeeze(-1)
                    max_q = torch.max(q_stack, dim=0).values
                    min_q = torch.min(q_stack, dim=0).values
                    disagreement_step = max_q - min_q
                    qmin_step = min_q
                if teacher_mask is not None:
                    teacher_mask_float = teacher_mask.float()
                else:
                    teacher_mask_float = torch.zeros_like(disagreement_step, device=device)
                non_teacher_mask_float = 1.0 - teacher_mask_float

                teacher_disagreement_sum += float((disagreement_step * teacher_mask_float).sum().item())
                non_teacher_disagreement_sum += float((disagreement_step * non_teacher_mask_float).sum().item())
                teacher_disagreement_steps += float(teacher_mask_float.sum().item())
                non_teacher_disagreement_steps += float(non_teacher_mask_float.sum().item())

                corr_total_steps += float(disagreement_step.numel())
                corr_sum_mask += float(teacher_mask_float.sum().item())
                corr_sum_dis += float(disagreement_step.sum().item())
                corr_sum_mask_sq += float(teacher_mask_float.sum().item())
                corr_sum_dis_sq += float((disagreement_step ** 2).sum().item())
                corr_sum_mask_dis += float((teacher_mask_float * disagreement_step).sum().item())

                if hist_edges_tensor is not None:
                    bin_idx = torch.bucketize(disagreement_step, hist_edges_tensor)
                    teacher_hist_counts.scatter_add_(0, bin_idx, teacher_mask_float)
                    non_teacher_hist_counts.scatter_add_(0, bin_idx, non_teacher_mask_float)

                if thresh_tensor is not None:
                    comp = (disagreement_step.unsqueeze(1) >= thresh_tensor.unsqueeze(0)).float()
                    comp_teacher = comp * teacher_mask_float.unsqueeze(1)
                    teacher_above_counts += comp_teacher.sum(dim=0)
                    teacher_steps_window += float(teacher_mask_float.sum().item())

                qmin_sum_all += float(qmin_step.sum().item())
                qmin_steps_all += float(qmin_step.numel())
                qmin_sum_teacher += float((qmin_step * teacher_mask_float).sum().item())
                qmin_steps_teacher += float(teacher_mask_float.sum().item())
                qmin_sum_non += float((qmin_step * non_teacher_mask_float).sum().item())
                qmin_steps_non += float(non_teacher_mask_float.sum().item())
            # Append counterfactual rows to CF buffer (student-denied actions)
            if args.cf_buffer_enable and teacher_mask is not None and student_actions is not None and cf_obs is not None:
                denied_ids = torch.nonzero(teacher_mask, as_tuple=False).flatten()
                if denied_ids.numel() > 0:
                    # Store (s, a_student)
                    cf_s = obs_detached_flat[denied_ids]
                    cf_a = student_actions[denied_ids]
                    # Initialize CF tensors on first use
                    cf_append(cf_s, cf_a)
            
            # We already folded denied penalties into rewards_eff; just expose count via logging

            # Book-keeping for episode stats
            cur_reward_sum += rewards
            cur_episode_length += 1
            done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                rewbuffer += cur_reward_sum[done_ids].tolist()
                lenbuffer += cur_episode_length[done_ids].tolist()
                cur_reward_sum[done_ids] = 0
                cur_episode_length[done_ids] = 0

            # Learn
            iteration_idx += 1
            total_env_steps += envs.num_envs

            if (
                not did_switch_env
                and args.switch_env_name
                and args.switch_env_name != current_env_name
                and env_switch_global_step > 0
                and total_env_steps >= env_switch_global_step
            ):
                per_env_progress = total_env_steps // envs.num_envs
                record_progress(
                    f"[Env] Switching from {current_env_name} to {args.switch_env_name} at total_steps={total_env_steps}"
                )
                try:
                    obs = envs.switch_env(
                        args.switch_env_name,
                        wrappers=current_wrappers,
                        curriculum_steps=per_env_progress,
                    )
                except Exception as exc:
                    record_progress(f"[Env] switch failed: {exc}")
                    raise
                current_env_name = args.switch_env_name
                run_prefix = current_env_name.replace('-', '_')
                cur_reward_sum.zero_()
                cur_episode_length.zero_()
                did_switch_env = True
                if args.use_wandb and wandb_run is not None:
                    import wandb
                    wandb_run.log({
                        "env/switch_event": 1,
                        "env/current_env": current_env_name,
                    }, step=total_env_steps)
                # tighten checkpoint/viz interval after switch
                if save_interval_current is not None and args.post_switch_viz_multiplier > 1:
                    save_interval_current = max(1, args.save_interval // args.post_switch_viz_multiplier)
                    next_save_step = total_env_steps + save_interval_current

            # Optionally reset main replay buffer once at reward switch (curriculum)
            if (
                args.reset_replay_on_switch
                and not did_reset_replay
                and args.reward_switch_after_steps > 0
                and total_env_steps >= args.reward_switch_after_steps
            ):
                record_progress(f"[Replay] Resetting main replay buffer at step {total_env_steps}")
                rb = SimpleReplayBuffer(
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
                )
                did_reset_replay = True
                # Reapply warm-up: postpone learning by the same number of steps as initially
                next_learning_starts_at = total_env_steps + int(args.learning_starts)
                record_progress(f"[Replay] Post-reset warm-up: learning resumes at env_step >= {next_learning_starts_at}")

                if args.reset_critic_on_switch:
                    record_progress(f"[Critic] Resetting critic weights/optimizer at step {total_env_steps}")
                    if args.arch_shared_trunk:
                        actor_backbone.load_state_dict(initial_shared_backbone_state)
                        trunk_params = list(actor_backbone.parameters())
                        trunk_optimizer = optim.Adam(trunk_params, lr=args.critic_learning_rate)
                        critic_params = list(critic_heads.parameters())
                        q_optimizer = optim.AdamW(critic_params, lr=args.critic_learning_rate, weight_decay=0.1)
                    else:
                        critic_backbone.load_state_dict(initial_critic_backbone_state)
                        critic_params = list(critic_backbone.parameters()) + list(critic_heads.parameters())
                        q_optimizer = optim.AdamW(critic_params, lr=args.critic_learning_rate, weight_decay=0.1)
                    critic_heads.load_state_dict(initial_critic_heads_state)
                    critic_target_backbone.load_state_dict(initial_target_backbone_state)
                    critic_target_heads.load_state_dict(initial_target_heads_state)
                    critic_feature_backbone = actor_backbone if args.arch_shared_trunk else critic_backbone
                if save_interval_current is not None and args.post_switch_viz_multiplier > 1:
                    save_interval_current = max(1, args.save_interval // args.post_switch_viz_multiplier)
                    next_save_step = total_env_steps + save_interval_current

            if next_save_step is not None and total_env_steps >= next_save_step:
                tag_name = f"step{total_env_steps}"
                ckpt_path = save_checkpoint(tag_name, total_env_steps)
                maybe_render_policy_map(tag_name, total_env_steps, ckpt_path)
                next_save_step += save_interval_current

            # Only learn if we have enough data in replay (also after any reset)
            if total_env_steps >= next_learning_starts_at and getattr(rb, 'ptr', 0) > 0:
                base_batch = args.batch_size // max(1, args.num_envs)
                # Compute sub-batch allocations for auxiliary buffers
                b_pref = int(base_batch * args.pref_sample_ratio) if args.pref_buffer_enable else 0
                b_pref_td = int(base_batch * args.pref_td_sample_ratio) if args.pref_td_buffer_enable else 0
                main_batch = max(1, base_batch - b_pref - b_pref_td)

                for i in range(args.num_updates):
                    batch = rb.sample(main_batch)
                    obs_batch = batch['observations']
                    next_obs_batch = batch['next']['observations']
                    actions_batch = batch['actions']
                    rewards_batch = batch['next']['rewards'].unsqueeze(-1)
                    dones_batch = batch['next']['dones'].float().unsqueeze(-1)

                    obs_batch = _normalize_obs(obs_batch)
                    next_obs_batch = _normalize_obs(next_obs_batch)

                    if args.arch_shared_trunk and trunk_optimizer is not None:
                        trunk_optimizer.zero_grad(set_to_none=True)
                    q_optimizer.zero_grad(set_to_none=True)

                    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                        # Target critic evaluation
                        next_actions, next_log_pi, _, _ = actor_forward(next_obs_batch)
                        next_features_target = critic_target_backbone(reshape_obs(next_obs_batch))
                        target_q_list = critic_target_heads(next_features_target, next_actions)
                        min_next_q = torch.min(torch.stack(target_q_list, dim=0), dim=0).values
                        min_next_q = min_next_q - log_alpha.exp() * next_log_pi
                        target_q = rewards_batch + (1.0 - dones_batch) * (args.gamma * min_next_q)

                        current_features = (actor_backbone if args.arch_shared_trunk else critic_backbone)(reshape_obs(obs_batch))
                        current_q_list = critic_heads(current_features, actions_batch)
                        qf_loss = torch.tensor(0.0, device=device)
                        for q_pred in current_q_list:
                            qf_loss = qf_loss + F.mse_loss(q_pred, target_q)

                        # Counterfactual critic penalty
                        if args.cf_buffer_enable and args.cf_q_weight > 0.0 and args.cf_sample_ratio > 0.0 and 'cf_size' in locals() and cf_size > 0:
                            cf_b = max(1, int(base_batch * args.cf_sample_ratio))
                            s_cf, a_cf = cf_sample(cf_b)
                            if s_cf is not None:
                                cf_features = (actor_backbone if args.arch_shared_trunk else critic_backbone)(reshape_obs(s_cf))
                                cf_q_list = critic_heads(cf_features, a_cf)
                                for q_cf in cf_q_list:
                                    qf_loss = qf_loss + args.cf_q_weight * F.mse_loss(q_cf, torch.full_like(q_cf, cf_penalty_target))

                        # Preference ranking loss (teacher > student at s)
                        if args.pref_buffer_enable and args.pref_rank_weight > 0.0 and b_pref > 0 and 'pref_size' in locals() and pref_size > 0:
                            s_pair, a_pos, a_neg = pref_sample(b_pref)
                            if s_pair is not None:
                                pref_features = (actor_backbone if args.arch_shared_trunk else critic_backbone)(reshape_obs(s_pair))
                                q_pos = critic_heads(pref_features, a_pos)
                                q_neg = critic_heads(pref_features, a_neg)
                                qpos_min = torch.min(torch.stack(q_pos, dim=0), dim=0).values
                                qneg_min = torch.min(torch.stack(q_neg, dim=0), dim=0).values
                                margin = float(args.pref_rank_margin)
                                rank_loss = torch.nn.functional.softplus(margin - (qpos_min - qneg_min)).mean()
                                qf_loss = qf_loss + float(args.pref_rank_weight) * rank_loss

                        # Preference-TD balanced critic loss
                        if args.pref_td_buffer_enable and args.pref_td_q_weight > 0.0 and b_pref_td > 0 and 't_size' in locals() and t_size > 0 and s_size > 0:
                            td_batch = pref_td_sample(b_pref_td)
                            if td_batch is not None:
                                t_s = td_batch['t_s']
                                t_next_s = td_batch['t_next_s']
                                t_a = td_batch['t_a']
                                t_r = td_batch['t_r'].unsqueeze(-1)
                                t_done = td_batch['t_done'].unsqueeze(-1)
                                s_s = td_batch['s_s']
                                s_a = td_batch['s_a']
                                s_r = td_batch['s_r'].unsqueeze(-1)

                                next_actions_td, next_log_pi_td, _, _ = actor_forward(t_next_s)
                                next_features_td = critic_target_backbone(reshape_obs(t_next_s))
                                q_td_list = critic_target_heads(next_features_td, next_actions_td)
                                min_q_td = torch.min(torch.stack(q_td_list, dim=0), dim=0).values
                                min_q_td = min_q_td - log_alpha.exp() * next_log_pi_td
                                target_teacher = t_r + (1.0 - t_done) * (args.gamma * min_q_td)

                                teacher_features = (actor_backbone if args.arch_shared_trunk else critic_backbone)(reshape_obs(t_s))
                                teacher_q_list = critic_heads(teacher_features, t_a)
                                loss_teacher = sum(F.mse_loss(q_t, target_teacher) for q_t in teacher_q_list)

                                student_features = (actor_backbone if args.arch_shared_trunk else critic_backbone)(reshape_obs(s_s))
                                student_q_list = critic_heads(student_features, s_a)
                                loss_student = sum(F.mse_loss(q_s, s_r) for q_s in student_q_list)
                                qf_loss = qf_loss + float(args.pref_td_q_weight) * (loss_teacher + loss_student)

                    scaler.scale(qf_loss).backward()
                    scaler.unscale_(q_optimizer)
                    torch.nn.utils.clip_grad_norm_(critic_params, max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float('inf'))
                    scaler.step(q_optimizer)

                    # Actor update
                    actor_optimizer.zero_grad(set_to_none=True)
                    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                        pi_actions, log_pi, _, _ = actor_forward(obs_batch)
                        actor_critic_features = (actor_backbone if args.arch_shared_trunk else critic_backbone)(reshape_obs(obs_batch))
                        q_pi_list = critic_heads(actor_critic_features, pi_actions)
                        min_q_pi = torch.min(torch.stack(q_pi_list, dim=0), dim=0).values
                        actor_loss = (log_alpha.exp().detach() * log_pi - min_q_pi).mean()
                    scaler.scale(actor_loss).backward()
                    scaler.unscale_(actor_optimizer)
                    if args.arch_shared_trunk and trunk_optimizer is not None:
                        scaler.unscale_(trunk_optimizer)
                    torch.nn.utils.clip_grad_norm_(actor_params, max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float('inf'))
                    if args.arch_shared_trunk and trunk_optimizer is not None:
                        torch.nn.utils.clip_grad_norm_(trunk_params, max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float('inf'))
                    scaler.step(actor_optimizer)

                    if args.arch_shared_trunk and trunk_optimizer is not None:
                        scaler.step(trunk_optimizer)
                    scaler.update()

                    # Alpha update
                    alpha_optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        _, log_pi_curr, _, _ = actor_forward(obs_batch)
                    alpha_loss = -log_alpha.exp() * (log_pi_curr + target_entropy).detach().mean()
                    alpha_loss.backward()
                    alpha_optimizer.step()

                    # Soft update targets
                    source_backbone = actor_backbone if args.arch_shared_trunk else critic_backbone
                    for src_param, tgt_param in zip(source_backbone.parameters(), critic_target_backbone.parameters()):
                        tgt_param.data.copy_(args.tau * src_param.data + (1 - args.tau) * tgt_param.data)
                    for src_param, tgt_param in zip(critic_heads.parameters(), critic_target_heads.parameters()):
                        tgt_param.data.copy_(args.tau * src_param.data + (1 - args.tau) * tgt_param.data)
            should_log = False
            if next_log_step is not None and total_env_steps >= next_log_step:
                should_log = True
                next_log_step += args.log_interval
            if total_env_steps >= args.total_timesteps:
                should_log = True

            if should_log:
                collection_time = time.time() - start_time
                fps = int(total_env_steps / max(1e-6, collection_time))
                logs = {
                    'Perf/total_fps': fps,
                    'Perf/collection_time_sec': collection_time,
                    'Perf/env_steps': total_env_steps,
                    'Perf/iterations': iteration_idx,
                }
                if len(rewbuffer) > 0:
                    logs['Train/mean_reward'] = float(np.mean(rewbuffer[-100:]))
                    logs['Train/mean_episode_length'] = float(np.mean(lenbuffer[-100:]))
                if 'log' in infos and isinstance(infos['log'], dict):
                    for k, v in infos['log'].items():
                        try:
                            logs[k] = float(v.float().mean().item())
                        except Exception:
                            pass
                if args.store_denied_actions:
                    logs['/Teacher/denied_transition_samples'] = float(last_denied_samples)
                if teacher_disagreement_steps > 0:
                    logs['/Teacher/mean_disagreement_intervened'] = float(teacher_disagreement_sum / max(1e-8, teacher_disagreement_steps))
                if non_teacher_disagreement_steps > 0:
                    logs['/Teacher/mean_disagreement_no_intervention'] = float(non_teacher_disagreement_sum / max(1e-8, non_teacher_disagreement_steps))
                # Global disagreement average across all steps in the window
                total_dis_sum = teacher_disagreement_sum + non_teacher_disagreement_sum
                total_dis_steps = teacher_disagreement_steps + non_teacher_disagreement_steps
                if total_dis_steps > 0:
                    logs['/Critic/mean_disagreement_all'] = float(total_dis_sum / max(1e-8, total_dis_steps))
                if args.pref_buffer_enable:
                    logs['/Buffers/pref_pairs'] = float(pref_size)
                if args.pref_td_buffer_enable:
                    logs['/Buffers/pref_td_teacher'] = float(t_size)
                    logs['/Buffers/pref_td_student'] = float(s_size)
                if corr_total_steps > 0:
                    numer = corr_total_steps * corr_sum_mask_dis - corr_sum_mask * corr_sum_dis
                    denom_part_x = corr_total_steps * corr_sum_mask_sq - (corr_sum_mask ** 2)
                    denom_part_y = corr_total_steps * corr_sum_dis_sq - (corr_sum_dis ** 2)
                    if denom_part_x > 1e-8 and denom_part_y > 1e-8:
                        corr_value = numer / math.sqrt(denom_part_x * denom_part_y)
                        logs['/Teacher/corr(disagreement, intervention)'] = float(corr_value)
                if hist_edges_tensor is not None:
                    teacher_hist_cpu = teacher_hist_counts.detach().cpu()
                    non_teacher_hist_cpu = non_teacher_hist_counts.detach().cpu()
                    for idx, label in enumerate(hist_labels):
                        logs[f"/Teacher/disagreement_hist_teacher_{label}"] = float(teacher_hist_cpu[idx].item())
                        logs[f"/Teacher/disagreement_hist_non_teacher_{label}"] = float(non_teacher_hist_cpu[idx].item())
                    teacher_hist_counts.zero_()
                    non_teacher_hist_counts.zero_()
                # Threshold percentages since last log (teacher only)
                if thresh_tensor is not None and teacher_steps_window > 0:
                    pct = (teacher_above_counts / max(1.0, teacher_steps_window)).detach().cpu().numpy()
                    for i, thr in enumerate(thresh_vals):
                        logs[f"/Teacher/frac_interventions_dis_ge_{thr}"] = float(pct[i])
                    teacher_above_counts.zero_()
                    teacher_steps_window = 0.0

                # Mean Q(s, a_used) summaries (min over heads)
                if qmin_steps_all > 0:
                    logs['/Critic/mean_q_min_all'] = float(qmin_sum_all / max(1e-8, qmin_steps_all))
                if qmin_steps_teacher > 0:
                    logs['/Critic/mean_q_min_intervened'] = float(qmin_sum_teacher / max(1e-8, qmin_steps_teacher))
                if qmin_steps_non > 0:
                    logs['/Critic/mean_q_min_no_intervention'] = float(qmin_sum_non / max(1e-8, qmin_steps_non))
                # Reset Q-window accumulators
                qmin_sum_all = qmin_steps_all = qmin_sum_teacher = qmin_steps_teacher = qmin_sum_non = qmin_steps_non = 0.0

                log_line_parts = [
                    f"env_steps {total_env_steps}/{args.total_timesteps}",
                    f"iter {iteration_idx}",
                    f"fps {fps}",
                ]
                if 'Train/mean_reward' in logs:
                    log_line_parts.append(f"mean_reward {logs['Train/mean_reward']:.2f}")
                if '/Episode/goal_success_rate' in logs:
                    log_line_parts.append(f"success {logs['/Episode/goal_success_rate']:.2f}")
                if '/Teacher/teacher_fraction_steps' in logs:
                    log_line_parts.append(f"teacher_frac {logs['/Teacher/teacher_fraction_steps']:.2f}")
                if args.store_denied_actions and last_denied_samples > 0:
                    log_line_parts.append(f"denied {last_denied_samples}")
                console_line = "[FastSAC] " + " | ".join(log_line_parts)
                print(console_line, flush=True)
                record_progress(console_line)

                if args.use_wandb:
                    import wandb
                    if wandb_run is None:
                        wandb_run = wandb.init(
                            project=args.project,
                            name=args.exp_name,
                            id=args.exp_name,
                            config=vars(args),
                            reinit=True,
                            resume="allow",
                        )
                    wandb_run.log(logs, step=total_env_steps)

            obs = next_obs
        # Save final
        final_ckpt = save_checkpoint('final', total_env_steps)
        maybe_render_policy_map('final', total_env_steps, final_ckpt)
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass
        total_time = time.time() - start_time
        summary_line = (
            "✅ FastSAC training complete"
            f" env_steps={total_env_steps}"
            f" iterations={iteration_idx}"
            f" duration_sec={total_time:.1f}"
            f" models_dir={run_model_dir}"
        )
        print("=" * 80)
        print(summary_line)
        record_progress(summary_line)
    finally:
        try:
            progress_file.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()

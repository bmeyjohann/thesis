#!/usr/bin/env python3
"""
Interactive Evaluation Script for RSL-RL Trained Agents

Load a trained RSL-RL policy and run it interactively on OGBench environments
with visual rendering. Perfect for testing trained agents and debugging!

Usage:
    # Plain policy evaluation
    python eval_interactive.py --model_path models/pointmaze_medium_sparse.pt --env_name pointmaze-medium-v0

    # With human teleop intervention overlay
    python eval_interactive.py --model_path models/pointmaze_medium_sparse.pt --env_name pointmaze-arena-danger-lethal-v0 \
        --intervention_mode human

    # With BFS teacher interventions
    python eval_interactive.py --model_path models/pointmaze_medium_sparse.pt --env_name pointmaze-arena-danger-lethal-v0 \
        --intervention_mode agent --teacher_type bfs --tolerance_type angle --tolerance_value 30
    
Controls:
    - ESC: Exit
    - SPACE: Reset environment
    - R: Toggle manual reset mode
    - Q: Quit
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pygame
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import contextmanager

from ogbench_utils import (
    GaussianPolicyHead,
    MLPBackbone,
    PixelBackbone,
    PixelNormalizer,
    build_ogbench_wrapper,
    build_eval_parser,
    prepare_observation,
    reshape_observation,
)

# Fix WSL window positioning issues  
os.environ['SDL_VIDEO_CENTERED'] = '1'


def _canonical_pixel_shape(shape) -> tuple[int, int, int]:
    """Return pixel shape as (C, H, W)."""
    if shape is None:
        raise ValueError("pixel shape is None")
    if isinstance(shape, torch.Size):
        shape = tuple(int(s) for s in shape)
    elif isinstance(shape, (list, tuple)):
        shape = tuple(int(s) for s in shape)
    else:
        raise TypeError(f"Unsupported pixel shape type: {type(shape)}")
    if len(shape) != 3:
        raise ValueError(f"Pixel shape must have 3 dims, got {shape}")
    c_first = shape[0] in (1, 3, 4)
    c_last = shape[-1] in (1, 3, 4)
    if c_first and not c_last:
        return shape
    if c_last and not c_first:
        return (shape[-1], shape[0], shape[1])
    # Ambiguous but assume already canonical
    return shape


def _parse_int_tuple(value, *, allow_single: bool = False) -> tuple[int, ...] | None:
    """Parse tuples stored as tuple/list/str/int in checkpoints."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(',')]
        parsed = [int(item) for item in items if item]
        return tuple(parsed)
    if isinstance(value, int):
        if allow_single:
            return (int(value),)
        return (int(value),)
    return None


def _parse_optional_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def unwrap_maze_env(env):
    """Walk through nested gym wrappers to find the underlying MazeEnv."""
    visited = set()
    current = env
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "maze_map") and hasattr(current, "xy_to_ij"):
            return current
        current = getattr(current, "env", None)
    return None


class BirdsEyeRenderer:
    """Top-down pygame renderer for maze environments with optional observation panel."""

    def __init__(self, maze_env, initial_obs: np.ndarray | None):
        if not hasattr(maze_env, "maze_map"):
            raise ValueError("maze_env must expose maze_map")
        self.env = maze_env
        self.grid = np.array(maze_env.maze_map)
        self.rows, self.cols = self.grid.shape
        self.maze_unit = float(getattr(maze_env, "_maze_unit", 1.0))
        self.offset_x = float(getattr(maze_env, "_offset_x", 0.0))
        self.offset_y = float(getattr(maze_env, "_offset_y", 0.0))
        self.cell_size = self._suggest_cell_size(self.cols, self.rows)
        self.grid_width = self.cols * self.cell_size
        self.grid_height = self.rows * self.cell_size

        self.show_obs_panel = (
            initial_obs is not None
            and isinstance(initial_obs, np.ndarray)
            and initial_obs.ndim >= 2
        )
        if self.show_obs_panel:
            obs = initial_obs
            if obs.ndim == 1:
                side = int(np.sqrt(obs.size))
                obs = obs.reshape(side, side)
            if obs.ndim == 2:
                obs = np.stack([obs] * 3, axis=-1)
            if obs.shape[-1] == 1:
                obs = np.repeat(obs, 3, axis=-1)
            self.obs_shape = obs.shape
            scale = max(2, min(6, 480 // max(obs.shape[0], obs.shape[1])))
            self.obs_surface_size = (obs.shape[1] * scale, obs.shape[0] * scale)
        else:
            self.obs_shape = None
            self.obs_surface_size = (0, 0)

        total_width = self.grid_width + (self.obs_surface_size[0] + 16 if self.show_obs_panel else 0)
        total_height = max(self.grid_height, self.obs_surface_size[1])
        if total_width == 0 or total_height == 0:
            total_width = max(total_width, 320)
            total_height = max(total_height, 320)
        self.surface = pygame.display.set_mode((total_width, total_height))
        pygame.display.set_caption("Bird's Eye View")
        self.font = pygame.font.SysFont("Arial", 14)

    @staticmethod
    def _suggest_cell_size(cols: int, rows: int) -> int:
        if cols <= 0 or rows <= 0:
            return 32
        base = min(960 // max(cols, 1), 720 // max(rows, 1))
        return int(max(16, min(72, base)))

    def draw(self, pixel_obs: np.ndarray | None = None):
        if self.surface is None:
            return
        self.surface.fill((28, 28, 28))
        grid_surface = pygame.Surface((self.grid_width, self.grid_height))
        grid_surface.fill((235, 235, 235))

        for i in range(self.rows):
            for j in range(self.cols):
                y_pix = (self.rows - 1 - i) * self.cell_size
                rect = pygame.Rect(j * self.cell_size, y_pix, self.cell_size, self.cell_size)
                cell = int(self.grid[i, j])
                if cell == 1:
                    color = (60, 60, 60)
                elif cell == getattr(self.env, "_dangerous_tile_id", 2):
                    mode = getattr(self.env, "_dangerous_state_mode", "floor")
                    color = (190, 60, 40) if mode != "wall" else (200, 30, 30)
                else:
                    color = (225, 225, 225)
                pygame.draw.rect(grid_surface, color, rect)
                pygame.draw.rect(grid_surface, (180, 180, 180), rect, width=1)

        goal_xy = getattr(self.env, "cur_goal_xy", None)
        if goal_xy is not None:
            center = self._xy_to_screen(goal_xy)
            pygame.draw.circle(grid_surface, (50, 120, 255), center, max(6, self.cell_size // 3))

        if hasattr(self.env, "get_xy"):
            agent_xy = self.env.get_xy()
            center = self._xy_to_screen(agent_xy)
            pygame.draw.circle(grid_surface, (30, 30, 220), center, max(6, self.cell_size // 3))
            pygame.draw.circle(grid_surface, (255, 255, 255), center, max(6, self.cell_size // 3), width=2)

        self.surface.blit(grid_surface, (0, 0))

        if self.show_obs_panel:
            panel = pygame.Surface(self.obs_surface_size)
            panel.fill((20, 20, 20))
            label = self.font.render("Observation", True, (230, 230, 230))
            panel.blit(label, (4, 4))
            if pixel_obs is not None:
                obs_img = pixel_obs
                if isinstance(obs_img, torch.Tensor):
                    obs_img = obs_img.detach().cpu().numpy()
                if obs_img.ndim == 1:
                    side = int(np.sqrt(obs_img.size))
                    obs_img = obs_img.reshape(side, side)
                if obs_img.ndim == 2:
                    obs_img = np.stack([obs_img] * 3, axis=-1)
                if obs_img.shape[-1] == 1:
                    obs_img = np.repeat(obs_img, 3, axis=-1)
                obs_img = np.asarray(obs_img)
                if obs_img.dtype != np.uint8:
                    obs_min = float(np.min(obs_img))
                    obs_max = float(np.max(obs_img))
                    if obs_max > obs_min:
                        obs_img = (obs_img - obs_min) / (obs_max - obs_min)
                    obs_img = (obs_img * 255.0).clip(0, 255).astype(np.uint8)
                obs_img = np.ascontiguousarray(obs_img)
                surf = pygame.surfarray.make_surface(obs_img.swapaxes(0, 1))
                surf = pygame.transform.smoothscale(surf, self.obs_surface_size)
                panel.blit(surf, (0, 24))
            self.surface.blit(panel, (self.grid_width + 16, 0))

        pygame.display.flip()

    def _xy_to_screen(self, xy):
        x, y = xy
        col = (x + self.offset_x) / self.maze_unit
        row = (y + self.offset_y) / self.maze_unit
        center_x = (col + 0.5) * self.cell_size
        center_y = (self.rows - (row + 0.5)) * self.cell_size
        half_cell = 0.5 * self.cell_size
        center_x = float(np.clip(center_x, half_cell, self.grid_width - half_cell))
        center_y = float(np.clip(center_y, half_cell, self.grid_height - half_cell))
        return int(center_x), int(center_y)

@contextmanager
def mujoco_gl_context(backend: Optional[str]):
    """Temporarily set the MUJOCO_GL backend (restoring previous value afterwards)."""
    original = os.environ.get('MUJOCO_GL')
    try:
        if backend is None:
            if 'MUJOCO_GL' in os.environ:
                del os.environ['MUJOCO_GL']
        else:
            os.environ['MUJOCO_GL'] = backend
        yield
    finally:
        if original is None:
            if 'MUJOCO_GL' in os.environ:
                del os.environ['MUJOCO_GL']
        else:
            os.environ['MUJOCO_GL'] = original


def _mirror_env_worker(env_name: str, args_dict: dict, seed: int, action_queue, event_queue):
    """Run a human-render mirror env in a separate process."""
    try:
        args_ns = argparse.Namespace(**args_dict)
        with mujoco_gl_context('glfw'):
            env, render_mode = create_env(env_name, args_ns, render_override='human', mirror_mode=True)
        event_queue.put(('ready', render_mode))
        obs, info = env.reset(seed=seed)
        env.render()
        while True:
            cmd, payload = action_queue.get()
            if cmd == 'step':
                action = np.asarray(payload, dtype=np.float32)
                _, _, term, trunc, _ = env.step(action)
                env.render()
                if term or trunc:
                    obs, info = env.reset()
                    env.render()
                event_queue.put(('step_ok', None))
            elif cmd == 'reset':
                obs, info = env.reset()
                env.render()
                event_queue.put(('reset_ok', None))
            elif cmd == 'close':
                break
    except Exception as exc:
        event_queue.put(('error', repr(exc)))
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except Exception:
            pass
        event_queue.put(('closed', None))


class MirrorEnvProcess:
    def __init__(self, env_name: str, args: argparse.Namespace, seed: int):
        ctx = mp.get_context('spawn')
        self._action_queue = ctx.Queue()
        self._event_queue = ctx.Queue()
        args_dict = vars(args).copy()
        self._process = ctx.Process(
            target=_mirror_env_worker,
            args=(env_name, args_dict, seed, self._action_queue, self._event_queue),
            daemon=True,
        )
        self._process.start()
        status, payload = self._event_queue.get()
        if status == 'ready':
            self.render_mode = payload
            print(f"🪞 Mirror process ready (mode={payload})")
        elif status == 'error':
            raise RuntimeError(f"Mirror env failed to start: {payload}")
        else:
            raise RuntimeError(f"Unexpected mirror startup event: {status}")

    def alive(self) -> bool:
        return self._process.is_alive()

    def _drain(self):
        msgs = []
        while not self._event_queue.empty():
            status, payload = self._event_queue.get()
            msgs.append((status, payload))
        for status, payload in msgs:
            if status == 'error':
                raise RuntimeError(f"Mirror env error: {payload}")
        return msgs

    def step(self, action: np.ndarray):
        if not self.alive():
            return
        self._action_queue.put(('step', action.tolist()))
        status, payload = self._event_queue.get()
        if status == 'error':
            raise RuntimeError(f"Mirror env error during step: {payload}")

    def reset(self):
        if not self.alive():
            return
        self._action_queue.put(('reset', None))
        status, payload = self._event_queue.get()
        if status == 'error':
            raise RuntimeError(f"Mirror env error during reset: {payload}")

    def close(self):
        if self.alive():
            try:
                self._action_queue.put(('close', None))
            except Exception:
                pass
        if self._process.is_alive():
            self._process.join(timeout=1.0)


# Import ogbench to register environments
import ogbench

# Custom environments will be registered through OGBench

# Add RSL-RL to path
sys.path.append('fasttd3/fast_sac')

# Import FastSAC components lazily
try:
    from fast_sac import Actor  # type: ignore
    from fast_sac_utils import EmpiricalNormalization  # type: ignore
    FASTSAC_AVAILABLE = True
except ImportError:
    FASTSAC_AVAILABLE = False

# Import RSL-RL components
try:
    from rsl_rl.modules import ActorCritic  # ActorCritic is in modules, not algorithms!
    from tensordict import TensorDict
    print("✓ RSL-RL imports successful")
except ImportError as e:
    print(f"❌ RSL-RL import failed: {e}")
    print("Make sure RSL-RL is installed: pip install rsl_rl")
    sys.exit(1)

def get_args():
    """Parse command line arguments."""
    return build_eval_parser().parse_args()

class FastSACPolicy:
    """Unified wrapper for FastSAC policies (legacy and new architectures)."""

    def __init__(self, *,
                 obs_normalizer: nn.Module,
                 obs_mode: str,
                 pixel_shape=None,
                 actor_backbone: nn.Module | None = None,
                 actor_head: nn.Module | None = None,
                 legacy_actor: nn.Module | None = None):
        self.obs_normalizer = obs_normalizer
        self.obs_mode = obs_mode
        self.pixel_shape = tuple(pixel_shape) if pixel_shape is not None else None
        self.actor_backbone = actor_backbone
        self.actor_head = actor_head
        self.legacy_actor = legacy_actor
        module = legacy_actor if legacy_actor is not None else actor_head
        self.device = next(module.parameters()).device

    def eval(self):
        self.obs_normalizer.eval()
        if self.legacy_actor is not None:
            self.legacy_actor.eval()
        else:
            self.actor_backbone.eval()
            self.actor_head.eval()

    def _normalize(self, obs):
        try:
            return self.obs_normalizer(obs, center=True)
        except TypeError:
            return self.obs_normalizer(obs)

    def act(self, obs_dict, deterministic: bool = True):
        obs = obs_dict["policy"].to(self.device)
        if self.legacy_actor is not None:
            with torch.no_grad():
                norm_obs = self._normalize(obs)
                actions, _, means = self.legacy_actor(norm_obs)
            return means if deterministic else actions
        norm_obs = self._normalize(obs)
        obs_input = reshape_observation(norm_obs, obs_mode=self.obs_mode, pixel_shape=self.pixel_shape)
        with torch.no_grad():
            features = self.actor_backbone(obs_input)
            actions, _, means = self.actor_head(features)
        return means if deterministic else actions


def _resolve_policy_type(args_policy_type: str, checkpoint: Dict[str, Any]) -> str:
    if args_policy_type != 'auto':
        return args_policy_type
    keys = set(checkpoint.keys())
    if {'actor_backbone', 'actor_head'} <= keys:
        return 'fastsac_v2'
    if {'actor_state_dict', 'qnet_state_dict'} <= keys:
        return 'fastsac'
    return 'rsl-rl'


def load_trained_policy(model_path: str, env, device: torch.device, args):
    """Load a trained policy and return (policy, training_metadata)."""
    print(f"\n🔄 Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print("✓ Checkpoint loaded")

    policy_type = _resolve_policy_type(args.policy_type, checkpoint)
    print(f"📦 Detected policy format: {policy_type}")

    training_info: Dict[str, Any] = {}
    obs_space = env.observation_space
    act_dim = env.action_space.shape[0]
    obs_dim = int(np.prod(obs_space.shape))

    if policy_type == 'rsl-rl':
        dummy_obs = torch.zeros(1, obs_dim, device=device)
        dummy_obs_dict = TensorDict({"policy": dummy_obs}, batch_size=[1], device=device)
        config = checkpoint.get('policy_cfg') or checkpoint.get('model_config') or {
            'hidden_dims': [256, 256, 256],
            'activation': 'elu',
        }
        valid_keys = {'hidden_dims', 'activation', 'init_noise_std', 'actor_hidden_dims', 'critic_hidden_dims'}
        config = {k: v for k, v in config.items() if k in valid_keys}
        policy = ActorCritic(
            obs=dummy_obs_dict,
            obs_groups={"policy": ["policy"], "critic": ["policy"]},
            num_actions=act_dim,
            **config,
        ).to(device)
        state_dict = checkpoint.get('policy_state_dict') or checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        policy.load_state_dict(state_dict)
        policy.eval()
        print("✓ ActorCritic policy loaded")
    elif policy_type == 'fastsac':
        if not FASTSAC_AVAILABLE:
            raise ImportError("FastSAC components are unavailable; ensure 'fasttd3/fast_sac' is on PYTHONPATH or installed.")
        actor_hidden = checkpoint.get('args', {}).get('actor_hidden_dim', 512)
        init_scale = checkpoint.get('args', {}).get('init_scale', 0.01)
        actor = Actor(
            n_obs=obs_dim,
            n_act=act_dim,
            num_envs=1,
            init_scale=init_scale,
            hidden_dim=actor_hidden,
            device=device,
        ).to(device)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()
        obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
        if checkpoint.get('obs_normalizer_state'):
            obs_normalizer.load_state_dict(checkpoint['obs_normalizer_state'])
        obs_normalizer.eval()
        policy = FastSACPolicy(
            obs_normalizer=obs_normalizer,
            obs_mode='state',
            pixel_shape=None,
            legacy_actor=actor,
        )
        policy.eval()
    else:  # fastsac_v2
        if not FASTSAC_AVAILABLE:
            raise ImportError("FastSAC components are unavailable; ensure 'fasttd3/fast_sac' is on PYTHONPATH or installed.")
        train_args = checkpoint.get('args', {}) or {}
        obs_mode = train_args.get('obs_mode', getattr(args, 'obs_mode', 'state'))
        arch_shared = train_args.get('arch_shared_trunk', False)
        actor_hidden = train_args.get('actor_hidden_dim', 512)
        shared_hidden = train_args.get('shared_hidden_dim', actor_hidden)
        init_scale = train_args.get('init_scale', 0.01)
        feature_dim = shared_hidden if arch_shared else actor_hidden
        if obs_mode == 'pixels':
            pixel_shape_raw = checkpoint.get('pixel_shape') or obs_space.shape
            pixel_shape = _canonical_pixel_shape(pixel_shape_raw)
            conv_channels = (
                _parse_int_tuple(train_args.get('pixel_conv_channels_parsed'))
                or _parse_int_tuple(train_args.get('pixel_conv_channels'))
                or (32, 64, 64)
            )
            kernel_sizes = (
                _parse_int_tuple(train_args.get('pixel_kernel_sizes_parsed'))
                or _parse_int_tuple(train_args.get('pixel_kernel_sizes'))
                or (8, 4, 3)
            )
            strides = (
                _parse_int_tuple(train_args.get('pixel_strides_parsed'))
                or _parse_int_tuple(train_args.get('pixel_strides'))
                or (4, 2, 1)
            )
            final_pool = (
                _parse_optional_int(train_args.get('pixel_final_pool_parsed'))
                or _parse_optional_int(train_args.get('pixel_final_pool'))
            )
            backbone = PixelBackbone(
                pixel_shape,
                feature_dim,
                conv_channels=conv_channels,
                kernel_sizes=kernel_sizes,
                strides=strides,
                final_pool=final_pool,
            ).to(device)
            obs_normalizer = PixelNormalizer().to(device)
        else:
            pixel_shape = None
            backbone = MLPBackbone(obs_dim, feature_dim).to(device)
            obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
            if checkpoint.get('obs_normalizer_state'):
                obs_normalizer.load_state_dict(checkpoint['obs_normalizer_state'])
        backbone.load_state_dict(checkpoint['actor_backbone'])
        backbone.eval()
        actor_head = GaussianPolicyHead(backbone.output_dim, act_dim, actor_hidden, init_scale).to(device)
        actor_head.load_state_dict(checkpoint['actor_head'])
        actor_head.eval()
        policy = FastSACPolicy(
            obs_normalizer=obs_normalizer,
            obs_mode=obs_mode,
            pixel_shape=pixel_shape,
            actor_backbone=backbone,
            actor_head=actor_head,
        )
        policy.eval()
        training_info['obs_mode'] = obs_mode
        training_info['num_critics'] = train_args.get('num_critics', 2)
        training_info['arch_shared_trunk'] = arch_shared
        if pixel_shape is not None:
            training_info['pixel_shape'] = pixel_shape
            training_info['pixel_conv_channels'] = conv_channels
            training_info['pixel_kernel_sizes'] = kernel_sizes
            training_info['pixel_strides'] = strides
            training_info['pixel_final_pool'] = final_pool

    # Common training metadata
    args_obj = checkpoint.get('args')
    if isinstance(args_obj, dict):
        for key in ['env_name', 'reward_type', 'obs_mode', 'shared_hidden_dim', 'pixel_width', 'pixel_height', 'pixel_camera']:
            if key in args_obj:
                training_info[key] = args_obj[key]
    elif args_obj is not None:
        for key in ['env_name', 'reward_type', 'obs_mode', 'shared_hidden_dim', 'pixel_width', 'pixel_height', 'pixel_camera']:
            if hasattr(args_obj, key):
                training_info[key] = getattr(args_obj, key)
    for key in ['training_info', 'iteration', 'total_timesteps']:
        if key in checkpoint:
            training_info[key] = checkpoint[key]

    print("✓ Policy set to evaluation mode")
    return policy, training_info


def create_env(env_name: str, args, *, render_override: str | None = None, mirror_mode: bool = False):
    """Create the evaluation environment with appropriate wrappers."""
    print(f"🏗️  Creating environment: {env_name}")
    
    # Base environment creation parameters
    env_kwargs = {
        'max_episode_steps': args.max_episode_steps,
    }

    obs_mode = getattr(args, 'obs_mode', 'state')
    requested_render_mode = render_override if render_override is not None else args.render_mode
    render_mode = requested_render_mode

    if obs_mode == 'pixels':
        if render_mode != 'rgb_array' and not mirror_mode:
            print(f"ℹ️ Pixel-trained policy detected; overriding render_mode '{render_mode}' -> 'rgb_array' for correct observations.")
            render_mode = 'rgb_array'
    env_kwargs['render_mode'] = render_mode

    if render_mode == 'rgb_array':
        width = getattr(args, 'pixel_width', None) or getattr(args, 'width', None)
        height = getattr(args, 'pixel_height', None) or getattr(args, 'height', None)
        if width is not None:
            env_kwargs['width'] = int(width)
        if height is not None:
            env_kwargs['height'] = int(height)
        if getattr(args, 'pixel_camera', None):
            env_kwargs['camera_name'] = args.pixel_camera
    else:
        # Add width/height parameters for human-renderable OGBench environments
        if env_name.startswith(('pointmaze-', 'antmaze-', 'humanoidmaze-')) or render_mode == 'human':
            env_kwargs.update({
                'width': args.width,
                'height': args.height
            })
    debug_suffix = " (mirror)" if mirror_mode else ""
    print(f"   render_mode={render_mode}{debug_suffix}")
    print(f"   env_kwargs={env_kwargs}{debug_suffix}")
    
    try:
        env = gym.make(env_name, **env_kwargs)

        teleop = None
        if args.intervention_mode == 'human':
            from ogbench.teleop import ControlWindowTeleop

            teleop = ControlWindowTeleop(width=520, height=420, show_debug_info=True)

        wrapper = build_ogbench_wrapper(
            obs_mode=getattr(args, 'obs_mode', 'state'),
            include_goal=args.include_goal,
            include_distance=args.include_distance,
            include_direction=args.include_direction,
            include_velocity=args.include_velocity,
            reward_type=args.reward_type,
            dense_reward_scale=args.dense_reward_scale,
            step_penalty=args.step_penalty,
            reward_switch_after_steps=args.reward_switch_after_steps,
            intervention_mode=args.intervention_mode,
            teacher_type=args.teacher_type,
            tolerance_type=args.tolerance_type,
            tolerance_value=args.tolerance_value,
            hard_block_lethal=args.hard_block_lethal,
            intervention_enable_after_steps=args.intervention_enable_after_steps,
            teleop_interface=teleop,
        )
        env = wrapper(env)

        if getattr(args, 'obs_mode', 'state') == 'state':
            print('Applied FlexibleObsWrapper')
        print('Applied DetailedRewardWrapper (type={})'.format(args.reward_type))
        if args.intervention_mode == 'human':
            print('Applied InterventionWrapper (human teleop)')
        elif args.intervention_mode == 'agent':
            print('Applied InterventionWrapper (agent teacher: {})'.format(args.teacher_type))

        print('Environment created successfully')
        print('   Observation space:', env.observation_space)
        print('   Action space:', env.action_space)
        return env, render_mode
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        raise


def _format_policy_observation(obs, policy, device):
    """Match training-time preprocessing for policy inputs."""
    obs_mode = getattr(policy, 'obs_mode', 'state')
    pixel_shape = getattr(policy, 'pixel_shape', None)
    return prepare_observation(
        obs_input=obs,
        device=device,
        obs_mode=obs_mode,
        pixel_shape=pixel_shape,
        flatten=True,
    )

def run_interactive_evaluation(policy, env, args, device, mirror: MirrorEnvProcess | None):
    """Run interactive evaluation loop."""
    print(f"\n🎮 Starting Interactive Evaluation")
    print(f"Environment: {args.env_name}")
    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"\n🎮 Controls:")
    print(f"   ESC/Q: Exit")
    print(f"   SPACE: Reset environment")
    print(f"   R: Toggle auto-reset")
    print(f"   Click the window and use keys!")
    
    # Initialize pygame for event handling
    pygame.init()
    clock = pygame.time.Clock()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Reset environment
    obs, info = env.reset(seed=args.seed)
    maze_env = unwrap_maze_env(env)
    birds_eye = None
    if maze_env is not None:
        initial_obs = obs if isinstance(obs, np.ndarray) and obs.ndim >= 2 else None
        try:
            birds_eye = BirdsEyeRenderer(maze_env, initial_obs)
            birds_eye.draw(initial_obs)
        except Exception as exc:
            print(f"⚠️ Bird's eye renderer unavailable: {exc}")
            birds_eye = None
    if mirror is not None:
        try:
            mirror.reset()
        except Exception as exc:
            print(f"⚠️ Mirror process initialization failed: {exc}")
            mirror.close()
            mirror = None
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    total_reward = 0.0
    running = True
    auto_reset = True
    
    print(f"\n🚀 Starting evaluation...")
    
    episode_rewards = []
    episode_lengths = []
    step_idx = 0
    
    while running and (args.num_episodes == 0 or episode_count < args.num_episodes):
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    print(f"🔄 Manual reset triggered")
                    obs, info = env.reset()
                    if mirror is not None:
                        try:
                            mirror.reset()
                        except Exception as exc:
                            print(f"⚠️ Mirror reset failed: {exc}")
                            mirror.close()
                            mirror = None
                    if birds_eye is not None:
                        try:
                            birds_eye.draw(obs if birds_eye.show_obs_panel else None)
                        except Exception as exc:
                            print(f"⚠️ Bird's eye draw failed: {exc}")
                            birds_eye = None
                    episode_reward = 0.0
                    episode_length = 0
                elif event.key == pygame.K_r:
                    auto_reset = not auto_reset
                    print(f"🔄 Auto-reset: {'ON' if auto_reset else 'OFF'}")
        
        # Convert observation to tensor and add batch dimension
        obs_tensor = _format_policy_observation(obs, policy, device)
        obs_dict = TensorDict({
            "policy": obs_tensor,
        }, batch_size=[obs_tensor.shape[0]], device=device)
        
        # Get action from policy
        with torch.no_grad():
            actions = policy.act(obs_dict, deterministic=True)  # Use deterministic for evaluation
        
        # Convert action to numpy and remove batch dimension
        action = actions.cpu().numpy()[0]
        
        # Apply action processing (matching training settings)
        if args.action_scale != 1.0:
            action *= args.action_scale
        if args.clip_actions:
            action = np.clip(action, -1.0, 1.0)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if mirror is not None:
            try:
                mirror.step(action)
            except Exception as exc:
                print(f"⚠️ Mirror step failed: {exc}")
                mirror.close()
                mirror = None
        
        # Update episode tracking
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        if birds_eye is not None:
            try:
                birds_eye.draw(obs if birds_eye.show_obs_panel else None)
            except Exception as exc:
                print(f"⚠️ Bird's eye draw failed: {exc}")
                birds_eye = None
        
        # Print intervention events
        step_idx += 1
        if args.print_interventions and isinstance(info, dict) and info.get('teacher_intervened', False):
            reason = info.get('teacher_reason', 'unknown')
            # compute angle between actions if available
            angle_str = ''
            try:
                sa = np.array(info.get('student_action'), dtype=np.float32)
                ta = np.array(info.get('teacher_action'), dtype=np.float32)
                if sa is not None and ta is not None:
                    an = np.linalg.norm(sa)
                    bn = np.linalg.norm(ta)
                    if an > 1e-8 and bn > 1e-8:
                        cos = float(np.clip(np.dot(sa, ta) / (an * bn), -1.0, 1.0))
                        angle = float(np.degrees(np.arccos(cos)))
                        angle_str = f", angle={angle:.1f} deg"
            except Exception:
                pass
            print(f"🛟 Teacher intervention at step {episode_length}: reason={reason}{angle_str}")

        # Print step info (every 50 steps to avoid spam)
        if episode_length % 50 == 0:
            print(f"Step {episode_length:3d}: Reward {reward:6.3f}, Episode Reward: {episode_reward:8.3f}")
        
        # Handle episode completion
        if done:
            episode_count += 1
            total_reward += episode_reward
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            avg_reward = total_reward / episode_count
            
            print(f"\n📊 Episode {episode_count} Complete!")
            print(f"   Reward: {episode_reward:8.3f}")
            print(f"   Length: {episode_length:3d} steps")
            print(f"   Average Reward: {avg_reward:8.3f}")
            if len(episode_rewards) > 1:
                print(f"   Best Reward: {max(episode_rewards):8.3f}")
                print(f"   Worst Reward: {min(episode_rewards):8.3f}")
                print(f"   Reward Std: {np.std(episode_rewards):8.3f}")
            
            # Goal achievement check (sparse reward > 0 indicates goal reached)
            goal_reached = reward > 0 or episode_reward > 0.5  # Adjust threshold as needed
            if goal_reached:
                print(f"🎯 GOAL REACHED! 🎉")
            
            # If teacher metrics available, print summary
            if isinstance(info, dict) and 'teacher_num_interventions' in info:
                print("   Teacher summary:")
                print(f"     interventions: {int(info['teacher_num_interventions'])}")
                print(f"     steps: {int(info['teacher_intervention_steps'])} / {int(info.get('teacher_episode_steps', episode_length))}")
                print(f"     fraction: {float(info['teacher_fraction_steps']):.3f}")
                print(f"     avg_burst_len: {float(info['teacher_avg_burst_len']):.2f}")
                print(f"     safety: {int(info['teacher_num_safety_interventions'])}, divergence: {int(info['teacher_num_divergence_interventions'])}")

            # Reset for next episode
            if auto_reset:
                obs, info = env.reset()
                if mirror is not None:
                    try:
                        mirror.reset()
                    except Exception as exc:
                        print(f"⚠️ Mirror reset failed: {exc}")
                        mirror.close()
                        mirror = None
                if birds_eye is not None:
                    try:
                        birds_eye.draw(obs if birds_eye.show_obs_panel else None)
                    except Exception as exc:
                        print(f"⚠️ Bird's eye draw failed: {exc}")
                        birds_eye = None
            else:
                print(f"⏸️  Auto-reset disabled. Press SPACE to reset manually.")
                # Keep current state until manual reset

            episode_reward = 0.0
            episode_length = 0
        
        # Control frame rate
        clock.tick(args.fps)
    
    # Final statistics
    if episode_rewards:
        print(f"\n📊 Final Statistics ({len(episode_rewards)} episodes):")
        print(f"   Average Reward: {np.mean(episode_rewards):8.3f} ± {np.std(episode_rewards):6.3f}")
        print(f"   Best Reward: {np.max(episode_rewards):8.3f}")
        print(f"   Worst Reward: {np.min(episode_rewards):8.3f}")
        print(f"   Average Length: {np.mean(episode_lengths):6.1f} ± {np.std(episode_lengths):4.1f}")
        
        # Success rate (episodes with positive reward)
        successful_episodes = sum(1 for r in episode_rewards if r > 0.5)
        success_rate = successful_episodes / len(episode_rewards) * 100
        print(f"   Success Rate: {successful_episodes}/{len(episode_rewards)} ({success_rate:.1f}%)")
    
    print(f"\n✅ Evaluation completed!")

def main():
    args = get_args()
    device = select_device(args.device)

    print("🚀 Interactive Policy Evaluation")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    env: Optional[gym.Env] = None
    mirror: Optional[MirrorEnvProcess] = None
    try:
        update_args_from_checkpoint(args)
        env, mirror = setup_environment(args)
        policy, training_info = load_trained_policy(args.model_path, env, device, args)
        if training_info:
            print("Training info:")
            for key, value in training_info.items():
                print(f"   {key}: {value}")
        run_interactive_evaluation(policy, env, args, device, mirror=mirror)
    except Exception as exc:
        print(f"❌ Evaluation failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if mirror is not None:
            try:
                mirror.close()
            except Exception:
                pass
        pygame.quit()

    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        pygame.quit()
        sys.exit(0)

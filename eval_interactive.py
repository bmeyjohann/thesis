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
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import pygame
import multiprocessing as mp
from collections import deque
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
    maybe_set_goal_color,
)
from ogbench_utils.obs import (
    prepare_observation,
    reshape_observation,
    POLICY_OBS_KEYS,
)

# Fix WSL window positioning issues  
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Ensure DRQv2 package is importable
DRQV2_PATH = Path(__file__).resolve().parent / "drqv2"
if DRQV2_PATH.exists():
    sys.path.append(str(DRQV2_PATH))


def select_device(device_arg: str) -> torch.device:
    """Choose execution device, mirroring training entrypoint behavior."""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def _coerce_args_dict(raw: Any) -> dict[str, Any]:
    """Normalize checkpoint args to a plain dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return {k: v for k, v in vars(raw).items() if not k.startswith('_')}
    except TypeError:
        return {}


_CLI_FLAG_ALIASES: dict[str, tuple[str, ...]] = {
    'env_name': ('--env_name',),
    'include_goal': ('--include_goal', '--no_include_goal'),
    'include_distance': ('--include_distance',),
    'include_direction': ('--include_direction',),
    'include_velocity': ('--include_velocity',),
    'reward_type': ('--reward_type',),
    'dense_reward_scale': ('--dense_reward_scale',),
    'step_penalty': ('--step_penalty',),
    'reward_switch_after_steps': ('--reward_switch_after_steps',),
    'tolerance_type': ('--tolerance_type',),
    'tolerance_value': ('--tolerance_value',),
    'hard_block_lethal': ('--hard_block_lethal', '--no_hard_block_lethal'),
    'intervention_enable_after_steps': ('--intervention_enable_after_steps',),
    'pixel_width': ('--pixel_width',),
    'pixel_height': ('--pixel_height',),
    'pixel_camera': ('--pixel_camera',),
    'pixel_camera_mode': ('--pixel_camera_mode',),
    'pixel_local_view_size': ('--pixel_local_view_size',),
    'pixel_local_camera_height': ('--pixel_local_camera_height',),
    'pixel_first_person_distance': ('--pixel_first_person_distance',),
    'pixel_first_person_height': ('--pixel_first_person_height',),
    'pixel_first_person_lookahead': ('--pixel_first_person_lookahead',),
    'pixel_first_person_pitch': ('--pixel_first_person_pitch',),
    'decouple_view': ('--decouple_view',),
    'view_delta_scale': ('--view_delta_scale',),
    'goal_marker_color': ('--goal_marker_color',),
    'frame_stack': ('--frame_stack',),
    'use_local_actions': ('--use_local_actions', '--no_use_local_actions'),
    'se2_translation_scale': ('--se2_translation_scale',),
    'goal_relative_scale': ('--goal_relative_scale',),
    'teacher_type': ('--teacher_type',),
    'intervention_mode': ('--intervention_mode',),
    'intervention_safety_margin_frac': ('--intervention_safety_margin_frac',),
    'intervention_release_steps': ('--intervention_release_steps',),
}


def _to_pixels_array(arr: Any) -> np.ndarray:
    """Normalize arrays to uint8 HWC format."""
    array = np.asarray(arr)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[2] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 255.0)
        if array.max() <= 1.0:
            array = array * 255.0
        array = array.astype(np.uint8)
    return array


class RenderedPixelsWrapper(gym.Wrapper):
    """Rebuild observation['pixels'] from env.render(), mirroring training."""

    def __init__(self, env: gym.Env, *, width: Optional[int], height: Optional[int]):
        super().__init__(env)
        self._width = width
        self._height = height

    def _inject_pixels(self, obs: Any, pixels: np.ndarray):
        if isinstance(obs, dict):
            obs = dict(obs)
            obs['pixels'] = pixels
            return obs
        return {'pixels': pixels}

    def _grab_pixels(self, fallback: Any = None) -> np.ndarray:
        frame = None
        try:
            frame = self.env.render()
        except Exception:
            frame = None
        if frame is None:
            if isinstance(fallback, dict) and 'pixels' in fallback:
                frame = fallback['pixels']
            elif fallback is not None:
                frame = fallback
        if frame is None:
            width = self._width if self._width is not None else 84
            height = self._height if self._height is not None else 84
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        return _to_pixels_array(frame)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pixels = self._grab_pixels(obs)
        obs = self._inject_pixels(obs, pixels)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels = self._grab_pixels(obs)
        obs = self._inject_pixels(obs, pixels)
        return obs, reward, terminated, truncated, info


def _cli_flag_provided(flags: tuple[str, ...]) -> bool:
    argv = sys.argv[1:]
    for flag in flags:
        if flag in argv:
            return True
        prefix = flag + '='
        for arg in argv:
            if arg.startswith(prefix):
                return True
    return False


def _load_model_config_dict(model_path: Optional[str]) -> tuple[Optional[Path], dict[str, Any]]:
    if not model_path:
        return None, {}
    try:
        resolved = Path(model_path).resolve()
    except Exception:
        return None, {}
    candidates = []
    exp_dir = resolved.parent
    candidates.append(exp_dir / "args.json")
    # Mirror logs/<subdir>/<exp>/args.json if structure matches models/<subdir>/<exp>
    if len(exp_dir.parts) >= 2:
        maybe_subdir = exp_dir.parent.name
        log_candidate = Path("logs") / maybe_subdir / exp_dir.name / "args.json"
        candidates.append(log_candidate)
    for cand in candidates:
        if cand.is_file():
            try:
                with cand.open('r', encoding='utf-8') as fp:
                    return cand, json.load(fp)
            except Exception as exc:
                print(f"⚠️ Failed to read config {cand}: {exc}")
    return None, {}


def apply_model_config_defaults(args) -> Optional[dict[str, Any]]:
    config_path, config_dict = _load_model_config_dict(getattr(args, 'model_path', None))
    if not config_dict:
        return None
    applied_keys: list[str] = []
    for key, value in config_dict.items():
        if not hasattr(args, key):
            continue
        if key == 'model_path':
            continue
        flags = _CLI_FLAG_ALIASES.get(key)
        if flags and _cli_flag_provided(flags):
            continue
        setattr(args, key, value)
        applied_keys.append(key)
    return {
        'path': config_path,
        'applied_keys': sorted(applied_keys),
        'config': config_dict,
    }


def log_eval_configuration(args, *, config_info=None, checkpoint_args=None) -> None:
    print("\n⚙️ Evaluation configuration summary")
    if config_info and config_info.get('path'):
        applied = config_info.get('applied_keys') or []
        applied_str = ', '.join(applied) if applied else 'none'
        print(f"   Loaded model config: {config_info['path']} (applied: {applied_str})")
    if checkpoint_args:
        print("   Checkpoint metadata found; synced observation/camera settings.")
    fields = [
        ("Environment", 'env_name'),
        ("Observation Mode", 'obs_mode'),
        ("Include Goal", 'include_goal'),
        ("Include Distance", 'include_distance'),
        ("Include Direction", 'include_direction'),
        ("Include Velocity", 'include_velocity'),
        ("Frame Stack", 'frame_stack'),
        ("Use Local Actions", 'use_local_actions'),
        ("SE(2) Translation Scale", 'se2_translation_scale'),
        ("Goal Relative Scale", 'goal_relative_scale'),
        ("Reward Type", 'reward_type'),
        ("Dense Reward Scale", 'dense_reward_scale'),
        ("Step Penalty", 'step_penalty'),
        ("Teacher", 'teacher_type'),
        ("Intervention Mode", 'intervention_mode'),
        ("Tolerance Type", 'tolerance_type'),
        ("Tolerance Value", 'tolerance_value'),
    ]
    for label, attr in fields:
        if hasattr(args, attr):
            print(f"   {label}: {getattr(args, attr)}")
    if getattr(args, 'obs_mode', 'state') == 'pixels':
        pixel_fields = [
            ("Pixel Size", f"{getattr(args, 'pixel_width', 84)}x{getattr(args, 'pixel_height', 84)}"),
            ("Camera Mode", getattr(args, 'pixel_camera_mode', 'global')),
            ("Camera Name", getattr(args, 'pixel_camera', None)),
            ("Local View Size", getattr(args, 'pixel_local_view_size', None)),
            ("Local Camera Height", getattr(args, 'pixel_local_camera_height', None)),
            ("First-Person distance", getattr(args, 'pixel_first_person_distance', None)),
            ("First-Person height", getattr(args, 'pixel_first_person_height', None)),
            ("First-Person lookahead", getattr(args, 'pixel_first_person_lookahead', None)),
            ("First-Person pitch", getattr(args, 'pixel_first_person_pitch', None)),
        ]
        for label, value in pixel_fields:
            if value is not None:
                print(f"   {label}: {value}")
    print("")


def update_args_from_checkpoint(args, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Populate evaluation args with metadata saved during training."""
    ckpt_args = _coerce_args_dict(checkpoint.get('args'))
    if not ckpt_args:
        return ckpt_args

    obs_mode_ckpt = ckpt_args.get('obs_mode', 'state')
    if getattr(args, 'obs_mode', None) is None:
        args.obs_mode = obs_mode_ckpt
    elif args.obs_mode != obs_mode_ckpt:
        print(f"⚠️ CLI obs_mode={args.obs_mode} overrides checkpoint obs_mode={obs_mode_ckpt}")

    keys_to_sync = [
        'env_name',
        'include_goal',
        'include_distance',
        'include_direction',
        'include_velocity',
        'frame_stack',
        'use_local_actions',
        'decouple_view',
        'view_delta_scale',
        'se2_translation_scale',
        'goal_relative_scale',
        'reward_type',
        'dense_reward_scale',
        'step_penalty',
        'reward_switch_after_steps',
        'teacher_type',
        'tolerance_type',
        'tolerance_value',
        'hard_block_lethal',
        'intervention_enable_after_steps',
        'intervention_safety_margin_frac',
        'intervention_release_steps',
        'pixel_width',
        'pixel_height',
        'pixel_camera',
        'pixel_conv_channels',
        'pixel_kernel_sizes',
        'pixel_strides',
        'pixel_final_pool',
        'pixel_camera_mode',
        'pixel_local_view_size',
        'pixel_local_camera_height',
        'pixel_first_person_distance',
        'pixel_first_person_height',
        'pixel_first_person_lookahead',
        'pixel_first_person_pitch',
        'goal_marker_color',
    ]
    for key in keys_to_sync:
        if key not in ckpt_args or not hasattr(args, key):
            continue
        flags = _CLI_FLAG_ALIASES.get(key)
        if flags and _cli_flag_provided(flags):
            continue
        setattr(args, key, ckpt_args[key])
    # Intervention mode is opt-in for evaluation; only adopt checkpoint value if caller left it unspecified
    if hasattr(args, 'intervention_mode') and 'intervention_mode' in ckpt_args:
        current = getattr(args, 'intervention_mode', None)
        if current in (None, 'auto'):
            setattr(args, 'intervention_mode', ckpt_args['intervention_mode'])

    pixel_shape = checkpoint.get('pixel_shape')
    if pixel_shape is not None:
        args.pixel_shape_from_checkpoint = tuple(int(v) for v in pixel_shape)
    else:
        args.pixel_shape_from_checkpoint = None
    return ckpt_args


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


def clip_action_l2_tensor(actions: torch.Tensor, max_norm: float = 1.0) -> torch.Tensor:
    norms = actions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    mask = norms > max_norm
    if mask.any():
        scale = torch.ones_like(norms)
        scale[mask] = max_norm / norms[mask]
        actions = actions * scale
    return actions


def _extract_pixel_panel(obs: Any) -> np.ndarray | None:
    """Return an HxWx3 uint8 image when obs includes pixel data."""
    if obs is None:
        return None
    data = obs
    if isinstance(data, tuple) and data:
        data = data[0]
    if isinstance(data, dict):
        for key in ("pixels", "image", "policy", "observation"):
            if key in data:
                data = data[key]
                break
        else:
            return None
    arr = np.asarray(data)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
        return _extract_pixel_panel(arr)
    return None


def maybe_reset_policy_state(policy) -> None:
    reset_fn = getattr(policy, "reset", None)
    if callable(reset_fn):
        try:
            reset_fn()
        except TypeError:
            reset_fn(policy)


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

        pixel_panel = _extract_pixel_panel(initial_obs)
        self.show_obs_panel = pixel_panel is not None
        if self.show_obs_panel:
            obs = pixel_panel
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
        self.legend_lines = [
            "Controls:",
            "ESC/Q: Exit",
            "SPACE: Reset",
            "N: Next goal",
            "S/F: Slow/Fast",
            "R: Toggle auto-reset",
        ]

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

        legend_height = len(self.legend_lines) * 18 + 12
        legend_surface = pygame.Surface((self.grid_width, legend_height), pygame.SRCALPHA)
        legend_surface.fill((15, 15, 15, 200))
        for idx, line in enumerate(self.legend_lines):
            label = self.font.render(line, True, (230, 230, 230))
            legend_surface.blit(label, (8, 4 + idx * 18))
        self.surface.blit(legend_surface, (0, max(0, self.grid_height - legend_height)))

        if self.show_obs_panel:
            panel = pygame.Surface(self.obs_surface_size)
            panel.fill((20, 20, 20))
            label = self.font.render("Observation", True, (230, 230, 230))
            panel.blit(label, (4, 4))
            obs_img = _extract_pixel_panel(pixel_obs)
            if obs_img is not None:
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

try:
    from drqv2.drqv2 import DrQV2Agent  # type: ignore
    from drqv2.recurrent_agent import DrQV2RecurrentAgent  # type: ignore
    DRQV2_AVAILABLE = True
except ImportError:
    DRQV2_AVAILABLE = False

# Import RSL-RL components
try:
    from rsl_rl.modules import ActorCritic  # ActorCritic is in modules, not algorithms!
    from tensordict import TensorDict
    print("✓ RSL-RL imports successful")
except ImportError as e:
    print(f"❌ RSL-RL import failed: {e}")
    print("Make sure RSL-RL is installed: pip install rsl_rl")
    sys.exit(1)


def setup_environment(args: argparse.Namespace) -> tuple[gym.Env, Optional[MirrorEnvProcess]]:
    """Construct the evaluation environment (and optional mirror renderer)."""
    if args.headless:
        args.render_mode = 'rgb_array'
        args.mirror_human_render = False
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

    backend = args.policy_mujoco_gl
    env = None
    mirror = None
    try:
        mujoco_backend = None
        if backend == 'auto':
            wants_glfw = False
            if not args.headless:
                if args.obs_mode != 'pixels' and args.render_mode == 'human':
                    wants_glfw = True
                elif args.mirror_human_render:
                    wants_glfw = True
            mujoco_backend = 'glfw' if wants_glfw else 'egl'
        else:
            mujoco_backend = backend

        with mujoco_gl_context(mujoco_backend):
            env, render_mode = create_env(args.env_name, args)
        args.render_mode = render_mode

        if args.mirror_human_render and not args.headless and render_mode == 'rgb_array':
            try:
                mirror = MirrorEnvProcess(args.env_name, args, args.seed)
            except Exception as exc:
                print(f"⚠️ Mirror process unavailable: {exc}")
                mirror = None
        return env, mirror
    except Exception:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        raise

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

    def act(self, obs_dict, deterministic: bool = True, prev_actions: torch.Tensor | None = None):
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


class DrQPolicy:
    """Wrapper around DrQV2Agent/DrQV2RecurrentAgent for pixel observations."""

    def __init__(self, *, agent, pixel_shape: tuple[int, int, int], device: torch.device):
        self.agent = agent
        self.device = device
        self.pixel_shape = tuple(int(x) for x in pixel_shape)
        self.target_channels, self.target_height, self.target_width = self.pixel_shape
        base_channels = 3 if self.target_channels % 3 == 0 else self.target_channels
        stack = max(1, self.target_channels // base_channels)
        self.base_channels = base_channels
        self.frame_stack = stack
        self.obs_mode = 'pixels'
        self._frame_buffer: list[torch.Tensor] = []
        self._step = 0
        self.use_se2_warp = bool(getattr(agent, 'use_se2_warp', False))
        self.prev_action_dim = int(getattr(agent, 'prev_action_dim', 0))
        self.goal_history_dim = int(getattr(agent, 'goal_history_dim', 0))
        raw_hist = getattr(agent, 'action_history_len', 0) or 0
        self.action_history_len = max(0, int(raw_hist))
        self.action_dim = int(getattr(agent, 'action_dim', 0)) or self.prev_action_dim // max(1, self.action_history_len)
        self.agent_variant = getattr(agent, 'recurrent_type', 'standard') if hasattr(agent, 'recurrent_type') else 'standard'
        self.agent.train(False)

    def eval(self):
        self.agent.train(False)
        self.reset()

    def reset(self):
        self._frame_buffer = []
        self._step = 0
        reset_fn = getattr(self.agent, 'reset_memory', None)
        if callable(reset_fn):
            reset_fn()

    def prepare_obs(self, obs, device):
        data = obs
        if isinstance(data, tuple):
            data = data[0]
        if isinstance(data, dict):
            for key in POLICY_OBS_KEYS:
                if key in data:
                    data = data[key]
                    break
            else:
                data = next(iter(data.values()))
        tensor = torch.as_tensor(data, device=device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 4 and tensor.shape[1] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
        return tensor.float()

    def _reshape_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 2:
            obs = self._unflatten_frame(obs)
        if obs.ndim == 4:
            if obs.shape[1] not in (1, 3, 4, self.base_channels, self.target_channels) and obs.shape[-1] in (1, 3, 4):
                obs = obs.permute(0, 3, 1, 2).contiguous()
            elif obs.shape[1] not in (1, 3, 4, self.base_channels, self.target_channels):
                raise ValueError(f"Unexpected channels axis for DrQ obs: {tuple(obs.shape)}")
        else:
            raise ValueError(f"Unexpected observation tensor shape for DrQ policy: {tuple(obs.shape)}")

        if obs.shape[1] == self.target_channels:
            return obs

        if obs.shape[1] != self.base_channels:
            raise ValueError(f"Unexpected channel count {obs.shape[1]} for DrQ policy (expected {self.base_channels} per frame)")

        frame = self._resize_frame(obs)
        return self._stack_frames(frame)

    def _unflatten_frame(self, obs: torch.Tensor) -> torch.Tensor:
        batch, total = obs.shape
        base_channels = self.base_channels
        if total % base_channels != 0:
            base_channels = 3 if total % 3 == 0 else total
            self.base_channels = base_channels
            self.frame_stack = max(1, self.target_channels // self.base_channels)
        per_frame = total // base_channels
        side = int(math.sqrt(per_frame))
        if side * side * base_channels != total:
            raise ValueError(f"Cannot reshape flattened observation of size {total} into channels={base_channels}")
        frame = obs.view(batch, base_channels, side, side)
        return frame

    def _resize_frame(self, frame: torch.Tensor) -> torch.Tensor:
        _, _, h, w = frame.shape
        if h == self.target_height and w == self.target_width:
            return frame
        return F.interpolate(frame, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False)

    def _stack_frames(self, frame: torch.Tensor) -> torch.Tensor:
        if self.frame_stack <= 1:
            return frame
        cloned = frame.clone()
        if len(self._frame_buffer) < self.frame_stack:
            while len(self._frame_buffer) < self.frame_stack:
                self._frame_buffer.append(cloned)
        else:
            self._frame_buffer.pop(0)
            self._frame_buffer.append(cloned)
        return torch.cat(self._frame_buffer, dim=1)

    def act(
        self,
        obs_dict,
        deterministic: bool = True,
        prev_actions: torch.Tensor | None = None,
        goal_history: torch.Tensor | None = None,
    ):
        obs = obs_dict["policy"].to(self.device)
        obs = self._reshape_obs(obs)
        obs_np = obs.detach().cpu().numpy()[0]
        prev_np = None
        if prev_actions is not None:
            prev_np = prev_actions.detach().cpu().numpy()[0]
        goal_np = None
        if goal_history is not None:
            goal_np = goal_history.detach().cpu().numpy()[0]
        with torch.no_grad():
            action = self.agent.act(
                obs_np,
                step=self._step,
                eval_mode=True,
                prev_actions=prev_np,
                goal_history=goal_np,
            )
        self._step += 1
        return torch.as_tensor(action, device=self.device, dtype=torch.float32).view(1, -1)

    def register_pending_warp(self, warp_params: np.ndarray):
        fn = getattr(self.agent, 'register_pending_warp', None)
        if callable(fn):
            fn(warp_params)


def _infer_pixel_shape_from_space(space) -> tuple[int, int, int] | None:
    """Best-effort inference of (C, H, W) from a gym space."""
    if hasattr(space, 'spaces'):
        policy_space = space.spaces.get('policy')
        if policy_space is None and space.spaces:
            policy_space = next(iter(space.spaces.values()))
        if policy_space is not None:
            return _infer_pixel_shape_from_space(policy_space)
        return None
    shape = getattr(space, 'shape', None)
    if shape is None or len(shape) != 3:
        return None
    if shape[0] in (1, 3, 4):
        return (int(shape[0]), int(shape[1]), int(shape[2]))
    if shape[-1] in (1, 3, 4):
        return (int(shape[-1]), int(shape[0]), int(shape[1]))
    return None


class _ControllerPolicyBase:
    """Base class for simple controllers that mimic the policy API."""

    def __init__(self, *, action_space: gym.spaces.Box, obs_mode: str, pixel_shape, device: torch.device):
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("Controller policies require a continuous Box action space.")
        self.action_dim = int(np.prod(action_space.shape))
        self.device = device
        self.obs_mode = obs_mode
        self.pixel_shape = pixel_shape

    def eval(self):
        return self

    def reset(self):
        return None

    def act(
        self,
        obs_dict,
        deterministic: bool = True,
        prev_actions: torch.Tensor | None = None,
        goal_history: torch.Tensor | None = None,
    ):
        raise NotImplementedError


class RandomControllerPolicy(_ControllerPolicyBase):
    """Uniform random actions in [-1, 1]."""

    def act(
        self,
        obs_dict,
        deterministic: bool = True,
        prev_actions: torch.Tensor | None = None,
        goal_history: torch.Tensor | None = None,
    ):
        batch = obs_dict["policy"].shape[0]
        return torch.empty(batch, self.action_dim, device=self.device).uniform_(-1.0, 1.0)


class IdleControllerPolicy(_ControllerPolicyBase):
    """Always output zero actions (use with human teleop overrides)."""

    def act(
        self,
        obs_dict,
        deterministic: bool = True,
        prev_actions: torch.Tensor | None = None,
        goal_history: torch.Tensor | None = None,
    ):
        batch = obs_dict["policy"].shape[0]
        return torch.zeros(batch, self.action_dim, device=self.device)


class KeyboardControllerPolicy(_ControllerPolicyBase):
    """Keyboard-controlled policy (WASD for move, Q/E for view delta)."""

    def __init__(self, *, action_dim: int, device: torch.device, decouple_view: bool):
        self.action_dim = int(action_dim)
        self.device = device
        self.decouple_view = bool(decouple_view)
        self.teleop = InlineEvalTeleop(decouple_view=self.decouple_view)
        self.teleop.print_controls()

    def eval(self):
        return None

    def reset(self):
        self.teleop.reset()

    def act(
        self,
        obs_dict,
        deterministic: bool = True,
        prev_actions: torch.Tensor | None = None,
        goal_history: torch.Tensor | None = None,
    ):
        action = self.teleop.get_action()
        if action is None:
            action = np.zeros(self.action_dim, dtype=np.float32)
        else:
            action = np.asarray(action, dtype=np.float32)
            if action.shape[0] != self.action_dim:
                action = np.pad(action, (0, max(0, self.action_dim - action.shape[0])), mode='constant')
        return torch.as_tensor(action, device=self.device, dtype=torch.float32).view(1, -1)


class InlineEvalTeleop:
    """Keyboard teleop that shares the main pygame window (no extra display)."""

    def __init__(self, threshold: float = 0.05, hold_time: float = 0.25, *, decouple_view: bool = False):
        pygame.init()
        self.threshold = float(threshold)
        self.hold_time = float(hold_time)
        self.decouple_view = bool(decouple_view)
        self._last_action = np.zeros(3 if self.decouple_view else 2, dtype=np.float32)
        self._last_active_ts = 0.0

    def get_action(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        action = np.zeros(3 if self.decouple_view else 2, dtype=np.float32)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[1] = 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[1] = -1.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = -1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 1.0
        if self.decouple_view:
            if keys[pygame.K_q]:
                action[2] = -1.0
            if keys[pygame.K_e]:
                action[2] = 1.0

        magnitude = float(np.linalg.norm(action))
        now = time.perf_counter()
        if magnitude > self.threshold:
            self._last_action = action
            self._last_active_ts = now
            return action
        if now - self._last_active_ts < self.hold_time:
            return self._last_action
        return None

    def update_state(self, obs, info, step_count):
        return None

    def reset(self):
        self._last_action[:] = 0.0
        self._last_active_ts = 0.0

    def should_quit(self):
        return False

    def close(self):
        return None

    def print_controls(self):
        if self.decouple_view:
            print("🎮 Inline teleop active – WASD/arrow keys move, Q/E adjust view direction.")
        else:
            print("🎮 Inline teleop active – focus the render window and use WASD/arrow keys.")


def _resolve_policy_type(args_policy_type: str, checkpoint: Dict[str, Any]) -> str:
    if args_policy_type != 'auto':
        return args_policy_type
    keys = set(checkpoint.keys())
    if {'drq_encoder', 'drq_actor'} <= keys:
        return 'drqv2'
    if {'actor_backbone', 'actor_head'} <= keys:
        return 'fastsac_v2'
    if {'actor_state_dict', 'qnet_state_dict'} <= keys:
        return 'fastsac'
    return 'rsl-rl'


def load_trained_policy(
    model_path: str,
    env,
    device: torch.device,
    args,
    *,
    checkpoint: Optional[dict[str, Any]] = None,
):
    """Load a trained policy and return (policy, training_metadata)."""
    print(f"\n🔄 Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if checkpoint is None:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    print("✓ Checkpoint loaded")

    policy_type = _resolve_policy_type(args.policy_type, checkpoint)
    print(f"📦 Detected policy format: {policy_type}")

    training_info: Dict[str, Any] = {}
    train_args = _coerce_args_dict(checkpoint.get('args'))
    obs_space = env.observation_space
    act_dim = env.action_space.shape[0]
    if hasattr(obs_space, 'shape') and obs_space.shape is not None:
        obs_dim = int(np.prod(obs_space.shape))
    elif hasattr(obs_space, 'spaces'):
        policy_space = obs_space.spaces.get('policy')
        if policy_space is None:
            policy_space = next(iter(obs_space.spaces.values()))
        obs_dim = int(np.prod(policy_space.shape))
    else:
        raise RuntimeError(f"Unsupported observation space type: {type(obs_space)}")

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
        actor_hidden = train_args.get('actor_hidden_dim', 512)
        init_scale = train_args.get('init_scale', 0.01)
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
    elif policy_type == 'drqv2':
        if not DRQV2_AVAILABLE:
            raise ImportError("DRQ-v2 components unavailable; ensure drqv2 module is on PYTHONPATH.")
        pixel_shape_raw = checkpoint.get('pixel_shape') or getattr(args, 'pixel_shape_from_checkpoint', None) or obs_space.shape
        pixel_shape = _canonical_pixel_shape(pixel_shape_raw)
        action_shape = tuple(int(x) for x in env.action_space.shape)
        frame_stack = int(train_args.get('frame_stack', 1) or 1)
        action_history_len = max(1, frame_stack)
        agent_variant = train_args.get('agent_variant', 'standard')
        agent_kwargs = dict(
            obs_shape=pixel_shape,
            action_shape=action_shape,
            device=device,
            lr=float(train_args.get('learning_rate', 1e-4)),
            feature_dim=int(train_args.get('drq_feature_dim', 50)),
            hidden_dim=int(train_args.get('drq_hidden_dim', 1024)),
            critic_target_tau=float(train_args.get('critic_target_tau', 0.01)),
            num_expl_steps=int(train_args.get('num_expl_steps', 2000)),
            update_every_steps=int(train_args.get('update_every_steps', 1)),
            stddev_schedule=str(train_args.get('stddev_schedule', "linear(1.0,0.1,1e5)")),
            stddev_clip=float(train_args.get('stddev_clip', 0.3)),
            use_tb=False,
        )

        def _instantiate_agent(hist_len: int):
            kwargs = dict(agent_kwargs)
            kwargs['action_history_len'] = max(0, hist_len)
            if agent_variant == 'recurrent':
                return DrQV2RecurrentAgent(
                    recurrent_type=train_args.get('recurrent_type', 'convgru'),
                    recurrent_hidden_dim=int(train_args.get('recurrent_hidden_dim', 512)),
                    conv_hidden_channels=int(train_args.get('recurrent_conv_channels', 32)),
                    use_se2_warp=bool(train_args.get('recurrent_use_se2_warp', False)),
                    **kwargs,
                )
            return DrQV2Agent(**kwargs)

        history_candidate = max(1, action_history_len)
        tried_zero = False
        while True:
            agent = _instantiate_agent(history_candidate)
            encoder_module = getattr(agent, 'encoder', None)
            if encoder_module is None:
                encoder_module = getattr(agent, 'core', None)
            if encoder_module is None:
                raise AttributeError("DrQ agent is missing encoder/core module")
            encoder_module.load_state_dict(checkpoint['drq_encoder'])
            try:
                agent.actor.load_state_dict(checkpoint['drq_actor'])
                break
            except RuntimeError as exc:
                if history_candidate > 0 and not tried_zero:
                    print("⚠️ Actor input mismatch (prev-action history); retrying without prev-actions.")
                    history_candidate = 0
                    tried_zero = True
                    continue
                raise
        if 'drq_critic' in checkpoint:
            agent.critic.load_state_dict(checkpoint['drq_critic'])
        if 'drq_critic_target' in checkpoint:
            agent.critic_target.load_state_dict(checkpoint['drq_critic_target'])
        agent.train(False)
        policy = DrQPolicy(agent=agent, pixel_shape=pixel_shape, device=device)
        policy.eval()
    else:  # fastsac_v2
        if not FASTSAC_AVAILABLE:
            raise ImportError("FastSAC components are unavailable; ensure 'fasttd3/fast_sac' is on PYTHONPATH or installed.")
        obs_mode = train_args.get('obs_mode', getattr(args, 'obs_mode', 'state'))
        arch_shared = train_args.get('arch_shared_trunk', False)
        actor_hidden = train_args.get('actor_hidden_dim', 512)
        shared_hidden = train_args.get('shared_hidden_dim', actor_hidden)
        init_scale = train_args.get('init_scale', 0.01)
        feature_dim = shared_hidden if arch_shared else actor_hidden
        if obs_mode == 'pixels':
            pixel_shape_raw = (
                checkpoint.get('pixel_shape')
                or getattr(args, 'pixel_shape_from_checkpoint', None)
                or obs_space.shape
            )
            pixel_shape = _canonical_pixel_shape(pixel_shape_raw)
            backbone_state = checkpoint['actor_backbone']
            conv_channels: list[int] = []
            kernel_sizes: list[int] = []
            layer_idx = 0
            while True:
                weight_key = f'conv.{2 * layer_idx}.weight'
                if weight_key not in backbone_state:
                    break
                weight = backbone_state[weight_key]
                conv_channels.append(int(weight.shape[0]))
                kernel_sizes.append(int(weight.shape[2]))
                layer_idx += 1
            if not conv_channels:
                raise RuntimeError("Checkpoint missing convolutional layers for pixel backbone")
            stride_values = _parse_int_tuple(train_args.get('pixel_strides')) or (4, 2, 1)
            stride_list = list(stride_values)
            if not stride_list:
                stride_list = [1]
            if len(stride_list) < len(conv_channels):
                stride_list.extend([stride_list[-1]] * (len(conv_channels) - len(stride_list)))
            strides = tuple(stride_list[: len(conv_channels)])
            fc_in_features = backbone_state['fc.0.weight'].shape[1]
            pool_sources = [
                train_args.get('pixel_final_pool_parsed'),
                train_args.get('pixel_final_pool'),
                getattr(args, 'pixel_final_pool', None),
                getattr(args, 'pixel_final_pool_parsed', None),
            ]
            pool_candidates: list[int | None] = []
            for source in pool_sources:
                parsed = _parse_optional_int(source)
                if parsed is not None and parsed > 0:
                    pool_candidates.append(parsed)
            pool_candidates.extend([None, 2, 4, 8])
            chosen_backbone = None
            chosen_pool: int | None = None
            for pool_candidate in pool_candidates:
                pool_value = pool_candidate if pool_candidate and pool_candidate > 0 else None
                candidate = PixelBackbone(
                    pixel_shape,
                    feature_dim,
                    conv_channels=conv_channels,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    final_pool=pool_value,
                )
                flat_dim = candidate.fc[0].weight.shape[1]
                if flat_dim == fc_in_features:
                    chosen_backbone = candidate.to(device)
                    chosen_pool = pool_value
                    break
            if chosen_backbone is None:
                raise RuntimeError(
                    f"Unable to reconstruct pixel backbone (expected fc in-features {fc_in_features}); "
                    "pass --pixel_final_pool to disambiguate."
                )
            backbone = chosen_backbone
            final_pool = chosen_pool
            conv_channels = tuple(conv_channels)
            kernel_sizes = tuple(kernel_sizes)
            strides = tuple(strides)
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
        for key in [
            'env_name',
            'reward_type',
            'obs_mode',
            'shared_hidden_dim',
            'pixel_width',
            'pixel_height',
            'pixel_camera',
            'pixel_camera_mode',
            'pixel_local_view_size',
            'pixel_local_camera_height',
            'pixel_first_person_distance',
            'pixel_first_person_height',
            'pixel_first_person_lookahead',
            'pixel_first_person_pitch',
            'frame_stack',
            'use_local_actions',
            'se2_translation_scale',
            'agent_variant',
            'recurrent_type',
            'recurrent_use_se2_warp',
            'goal_marker_color',
        ]:
            if key in args_obj:
                training_info[key] = args_obj[key]
    elif args_obj is not None:
        for key in [
            'env_name',
            'reward_type',
            'obs_mode',
            'shared_hidden_dim',
            'pixel_width',
            'pixel_height',
            'pixel_camera',
            'pixel_camera_mode',
            'pixel_local_view_size',
            'pixel_local_camera_height',
            'pixel_first_person_distance',
            'pixel_first_person_height',
            'pixel_first_person_lookahead',
            'pixel_first_person_pitch',
            'frame_stack',
            'use_local_actions',
            'se2_translation_scale',
            'agent_variant',
            'recurrent_type',
            'recurrent_use_se2_warp',
            'goal_marker_color',
        ]:
            if hasattr(args_obj, key):
                training_info[key] = getattr(args_obj, key)
    for key in ['training_info', 'iteration', 'total_timesteps']:
        if key in checkpoint:
            training_info[key] = checkpoint[key]

    training_info['frame_stack'] = int(train_args.get('frame_stack', training_info.get('frame_stack', 1) or 1))
    training_info['use_local_actions'] = bool(train_args.get('use_local_actions', training_info.get('use_local_actions', False)))
    training_info['se2_translation_scale'] = float(train_args.get('se2_translation_scale', training_info.get('se2_translation_scale', 0.2)))
    training_info['agent_variant'] = train_args.get('agent_variant', training_info.get('agent_variant', 'standard'))
    if training_info['agent_variant'] == 'recurrent':
        training_info['recurrent_type'] = train_args.get('recurrent_type', training_info.get('recurrent_type', 'convgru'))
        training_info['recurrent_use_se2_warp'] = bool(train_args.get('recurrent_use_se2_warp', training_info.get('recurrent_use_se2_warp', False)))

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

    policy_width = getattr(args, 'pixel_width', None)
    policy_height = getattr(args, 'pixel_height', None)

    if obs_mode == 'pixels':
        render_mode = 'rgb_array'
        env_kwargs['render_mode'] = 'rgb_array'
        env_kwargs['width'] = int(policy_width or 84)
        env_kwargs['height'] = int(policy_height or 84)
    else:
        env_kwargs['render_mode'] = render_mode
        if render_mode == 'rgb_array':
            width = getattr(args, 'pixel_width', None) or getattr(args, 'width', None)
            height = getattr(args, 'pixel_height', None) or getattr(args, 'height', None)
            if width is not None:
                env_kwargs['width'] = int(width)
            if height is not None:
                env_kwargs['height'] = int(height)

    if render_mode == 'rgb_array':
        if getattr(args, 'pixel_camera', None):
            env_kwargs['camera_name'] = args.pixel_camera
        else:
            env_kwargs['pixel_camera_mode'] = getattr(args, 'pixel_camera_mode', 'global')
            env_kwargs['pixel_local_view_size'] = getattr(args, 'pixel_local_view_size', 12.0)
            env_kwargs['pixel_local_camera_height'] = getattr(args, 'pixel_local_camera_height', None)
            env_kwargs['pixel_first_person_distance'] = getattr(args, 'pixel_first_person_distance', 3.0)
            env_kwargs['pixel_first_person_height'] = getattr(args, 'pixel_first_person_height', 1.0)
            env_kwargs['pixel_first_person_lookahead'] = getattr(args, 'pixel_first_person_lookahead', 2.0)
            env_kwargs['pixel_first_person_pitch'] = getattr(args, 'pixel_first_person_pitch', -15.0)

    if (render_mode == 'human' or requested_render_mode == 'human') and obs_mode != 'pixels':
        env_kwargs['width'] = args.width
        env_kwargs['height'] = args.height
    debug_suffix = " (mirror)" if mirror_mode else ""
    print(f"   render_mode={render_mode}{debug_suffix}")
    print(f"   env_kwargs={env_kwargs}{debug_suffix}")
    
    try:
        env = gym.make(env_name, **env_kwargs)

        teleop = None
        if args.intervention_mode == 'human':
            if args.headless:
                from ogbench.teleop import ControlWindowTeleop

                teleop = ControlWindowTeleop(width=520, height=420, show_debug_info=True)
            else:
                teleop = InlineEvalTeleop(decouple_view=bool(getattr(args, 'decouple_view', False)))
                teleop.print_controls()

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
            intervention_safety_margin_frac=args.intervention_safety_margin_frac,
            intervention_release_steps=args.intervention_release_steps,
            teleop_interface=teleop,
        )
        env = wrapper(env)
        maybe_set_goal_color(env, getattr(args, 'goal_marker_color', 'auto'))

        if getattr(args, 'obs_mode', 'state') == 'state':
            print('Applied FlexibleObsWrapper')
        print('Applied DetailedRewardWrapper (type={})'.format(args.reward_type))
        if args.intervention_mode == 'human':
            print('Applied InterventionWrapper (human teleop)')
        elif args.intervention_mode == 'agent':
            print('Applied InterventionWrapper (agent teacher: {})'.format(args.teacher_type))
        if getattr(args, 'obs_mode', 'state') == 'pixels' and not mirror_mode:
            env = RenderedPixelsWrapper(
                env,
                width=env_kwargs.get('width'),
                height=env_kwargs.get('height'),
            )
            print('Applied RenderedPixelsWrapper for pixel observations')

        print('Environment created successfully')
        print('   Observation space:', env.observation_space)
        print('   Action space:', env.action_space)
        return env, render_mode
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        raise


def _format_policy_observation(obs, policy, device):
    """Match training-time preprocessing for policy inputs."""
    if isinstance(policy, DrQPolicy):
        return policy.prepare_obs(obs, device)
    obs_mode = getattr(policy, 'obs_mode', 'state')
    pixel_shape = getattr(policy, 'pixel_shape', None)
    return prepare_observation(
        obs_input=obs,
        device=device,
        obs_mode=obs_mode,
        pixel_shape=pixel_shape,
        flatten=True,
    )


def _init_action_history(action_dim: int, stack: int) -> deque:
    length = max(0, int(stack))
    if length <= 0:
        return deque()
    history = deque(maxlen=length)
    zero = np.zeros(action_dim, dtype=np.float32)
    for _ in range(length):
        history.append(zero.copy())
    return history


def _flatten_action_history(history: deque) -> np.ndarray:
    if not history:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(list(history), axis=0).astype(np.float32, copy=False)


class ActionFrameTransformer:
    """Convert between local (agent-centric) and global action frames."""

    def __init__(self, use_local: bool, translation_scale: float = 0.2):
        self.use_local = bool(use_local)
        self.translation_scale = float(translation_scale)
        self.heading = np.array([1.0, 0.0], dtype=np.float32)
        self.last_action = np.array([1.0, 0.0], dtype=np.float32)
        self._last_xy: np.ndarray | None = None

    def reset(self):
        self.heading[:] = np.array([1.0, 0.0], dtype=np.float32)
        self.last_action[:] = np.array([1.0, 0.0], dtype=np.float32)
        self._last_xy = None

    def to_global(self, action: np.ndarray) -> np.ndarray:
        if not self.use_local:
            return np.asarray(action, dtype=np.float32)
        local = np.asarray(action, dtype=np.float32)
        cos_h, sin_h = self.heading
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
        return (rot @ local.reshape(-1, 1)).reshape(local.shape)

    def to_local(self, action: np.ndarray) -> np.ndarray:
        if not self.use_local:
            return np.asarray(action, dtype=np.float32)
        glob = np.asarray(action, dtype=np.float32)
        cos_h, sin_h = self.heading
        rot = np.array([[cos_h, sin_h], [-sin_h, cos_h]], dtype=np.float32)
        return (rot @ glob.reshape(-1, 1)).reshape(glob.shape)

    def update_heading(self, info: dict | None):
        if not isinstance(info, dict):
            return
        delta = None
        prev_qpos = info.get('prev_qpos')
        qpos = info.get('qpos')
        if prev_qpos is not None and qpos is not None:
            delta = np.asarray(qpos[:2], dtype=np.float32) - np.asarray(prev_qpos[:2], dtype=np.float32)
        elif 'xy' in info:
            xy = np.asarray(info['xy'], dtype=np.float32)
            if self._last_xy is not None:
                delta = xy - self._last_xy
            self._last_xy = xy.copy()
        if delta is not None and delta.shape[0] >= 2:
            norm = np.linalg.norm(delta[:2])
            if norm > 1e-6:
                self.heading[:] = delta[:2] / norm

    def compute_warp_from_action(self, action_local: np.ndarray) -> np.ndarray:
        vec = np.asarray(action_local, dtype=np.float32).reshape(-1)
        if vec.shape[0] > 2:
            vec = vec[:2]
        warp = np.zeros(3, dtype=np.float32)
        warp[:2] = vec[:2] * self.translation_scale
        prev = self.last_action
        norm_prev = np.linalg.norm(prev)
        norm_new = np.linalg.norm(vec)
        heading_prev = np.arctan2(prev[1], prev[0]) if norm_prev > 1e-6 else 0.0
        heading_new = np.arctan2(vec[1], vec[0]) if norm_new > 1e-6 else heading_prev
        warp[2] = float(heading_new - heading_prev)
        if norm_new > 1e-6:
            self.last_action[:] = vec
        return warp


def clip_action_l2_np(action: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    vec = np.asarray(action, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > max_norm and norm > 0:
        vec = vec / norm
    return vec


def split_action_components(action: np.ndarray, decouple_view: bool) -> tuple[np.ndarray, float]:
    vec = np.asarray(action, dtype=np.float32).reshape(-1)
    if not decouple_view:
        return vec, 0.0
    if vec.shape[0] < 3:
        vec = np.pad(vec, (0, 3 - vec.shape[0]), mode='constant')
    move = vec[:2]
    view_delta = float(np.clip(vec[2], -1.0, 1.0))
    return move, view_delta


def _resolve_action_history_len(args, policy) -> int:
    if hasattr(policy, 'action_history_len'):
        try:
            return max(0, int(getattr(policy, 'action_history_len')))
        except Exception:
            pass
    frame_stack = getattr(args, 'frame_stack', None)
    if frame_stack is not None:
        try:
            return max(0, int(frame_stack))
        except Exception:
            return 1
    return 1


def _init_goal_history(goal_dim: int, stack: int, fill: Optional[np.ndarray] = None) -> deque:
    history = deque(maxlen=stack)
    if fill is None:
        base = np.zeros(goal_dim, dtype=np.float32)
    else:
        base = np.asarray(fill, dtype=np.float32).reshape(goal_dim)
    for _ in range(stack):
        history.append(base.copy())
    return history


def _flatten_goal_history(history: deque) -> np.ndarray:
    if not history:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(list(history), axis=0).astype(np.float32, copy=False)


def _compute_goal_relative_local(env, transformer: ActionFrameTransformer, scale: float) -> np.ndarray:
    base = getattr(env, "unwrapped", env)
    if hasattr(base, "get_xy") and hasattr(base, "cur_goal_xy"):
        try:
            agent_xy = np.asarray(base.get_xy(), dtype=np.float32)
            goal_xy = np.asarray(base.cur_goal_xy, dtype=np.float32)
            rel_global = goal_xy - agent_xy
        except Exception:
            rel_global = np.zeros(2, dtype=np.float32)
    else:
        rel_global = np.zeros(2, dtype=np.float32)
    rel_local = transformer.to_local(rel_global.astype(np.float32, copy=False))
    if scale and scale > 0:
        rel_local = rel_local / float(scale)
    return np.clip(rel_local, -1.0, 1.0).astype(np.float32, copy=False)


def _set_view_dir_from_delta(env, move_vec: np.ndarray, view_delta: float, view_delta_scale: float) -> None:
    base = getattr(env, "unwrapped", env)
    base_dir = None
    move = np.asarray(move_vec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(move)
    if norm > 1e-6:
        base_dir = move[:2] / norm
    elif hasattr(base, "_last_move_dir"):
        try:
            base_dir = np.asarray(getattr(base, "_last_move_dir"), dtype=np.float32)[:2]
        except Exception:
            base_dir = None
    if base_dir is None:
        base_dir = np.array([1.0, 0.0], dtype=np.float32)
    angle = float(np.clip(view_delta, -1.0, 1.0)) * float(view_delta_scale)
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    view_dir = (rot @ base_dir.reshape(-1, 1)).reshape(2)
    if hasattr(base, "set_view_dir"):
        base.set_view_dir(view_dir)


def run_headless_evaluation(policy, env, args, device):
    """Minimal evaluation loop without pygame for automated tests."""
    print("\n🧪 Running headless evaluation")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    target_episodes = args.num_episodes if args.num_episodes and args.num_episodes > 0 else 1
    max_steps = args.max_episode_steps
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    action_dim = int(getattr(policy, 'action_dim', 0) or np.prod(env.action_space.shape))
    history_len = _resolve_action_history_len(args, policy)
    prev_action_dim = int(getattr(policy, 'prev_action_dim', 0))
    goal_history_dim = int(getattr(policy, 'goal_history_dim', 0))
    goal_history_len = goal_history_dim // 2 if goal_history_dim > 0 else 0
    if prev_action_dim <= 0:
        history_len = 0
    use_local_actions = bool(getattr(args, 'use_local_actions', False))
    translation_scale = float(getattr(args, 'se2_translation_scale', 0.2))
    decouple_view = bool(getattr(args, 'decouple_view', False))
    view_delta_scale = float(getattr(args, 'view_delta_scale', np.pi))
    policy_has_warp = bool(getattr(policy, 'use_se2_warp', False))
    action_history = _init_action_history(action_dim, history_len)
    transformer = ActionFrameTransformer(use_local_actions, translation_scale)
    goal_history = None
    if goal_history_len > 0:
        goal_history = _init_goal_history(2, goal_history_len)

    for episode_idx in range(target_episodes):
        obs, info = env.reset(seed=args.seed + episode_idx)
        maybe_reset_policy_state(policy)
        action_history = _init_action_history(action_dim, history_len)
        transformer.reset()
        transformer.update_heading(info if isinstance(info, dict) else None)
        if goal_history_len > 0:
            initial_goal = _compute_goal_relative_local(env, transformer, getattr(args, 'goal_relative_scale', 10.0))
            goal_history = _init_goal_history(2, goal_history_len, fill=initial_goal)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < max_steps:
            obs_tensor = _format_policy_observation(obs, policy, device)
            obs_dict = TensorDict({"policy": obs_tensor}, batch_size=[obs_tensor.shape[0]], device=device)
            with torch.no_grad():
                if history_len > 0 and action_history:
                    prev_stack = _flatten_action_history(action_history)
                    prev_tensor = torch.as_tensor(prev_stack, device=device).view(1, -1)
                else:
                    prev_tensor = None
                if goal_history_len > 0 and goal_history:
                    goal_stack = _flatten_goal_history(goal_history)
                    goal_tensor = torch.as_tensor(goal_stack, device=device).view(1, -1)
                else:
                    goal_tensor = None
                try:
                    actions = policy.act(
                        obs_dict,
                        deterministic=True,
                        prev_actions=prev_tensor,
                        goal_history=goal_tensor,
                    )
                except TypeError:
                    actions = policy.act(obs_dict, deterministic=True, prev_actions=prev_tensor)
            action_local_raw = actions.cpu().numpy()[0]
            move_local, view_delta = split_action_components(action_local_raw, decouple_view)
            move_local = clip_action_l2_np(move_local)
            if decouple_view:
                action_local = np.concatenate([move_local, np.array([view_delta], dtype=np.float32)], axis=0)
            else:
                action_local = move_local
            if policy_has_warp and hasattr(policy, 'register_pending_warp'):
                warp = transformer.compute_warp_from_action(move_local)
                policy.register_pending_warp(warp)
            move_global = transformer.to_global(move_local)
            move_global = clip_action_l2_np(move_global)
            if decouple_view:
                _set_view_dir_from_delta(env, move_global, view_delta, view_delta_scale)
                action = np.concatenate([move_global, np.array([view_delta], dtype=np.float32)], axis=0)
            else:
                action = move_global
            if args.action_scale != 1.0:
                action *= args.action_scale
            if args.clip_actions:
                action = np.clip(action, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(action)
            if history_len > 0 and action_history is not None:
                action_history.append(action_local.copy())
            if goal_history_len > 0 and goal_history is not None:
                goal_rel = _compute_goal_relative_local(env, transformer, getattr(args, 'goal_relative_scale', 10.0))
                goal_history.append(goal_rel.copy())
            transformer.update_heading(info if isinstance(info, dict) else None)
            ep_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)

        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        print(f"   Episode {episode_idx + 1}: reward={ep_reward:.3f} len={steps}")

    if episode_rewards:
        avg_reward = float(np.mean(episode_rewards))
        avg_length = float(np.mean(episode_lengths))
        print(
            f"\n📊 Headless summary over {len(episode_rewards)} episode(s): "
            f"avg_reward={avg_reward:.3f} avg_len={avg_length:.1f}"
        )
    print("✅ Headless evaluation completed")


def _evaluate_goal_reached(info: Any, *, distance_epsilon: float) -> tuple[bool, Optional[float]]:
    """Determine whether goal was reached using wrapper metadata."""
    def _iter_infos(entry: Any):
        if isinstance(entry, dict):
            yield entry
            nested = entry.get('final_info')
            if isinstance(nested, dict):
                yield nested

    goal_flag = False
    distance = None
    for entry in _iter_infos(info):
        if not goal_flag and 'goal_reached' in entry:
            goal_flag = bool(entry['goal_reached'])
        if distance is None and 'distance_to_goal' in entry:
            try:
                distance = float(entry['distance_to_goal'])
            except (TypeError, ValueError):
                distance = None
    if not goal_flag and distance is not None and distance <= distance_epsilon:
        goal_flag = True
    return goal_flag, distance


def run_interactive_evaluation(policy, env, args, device, mirror: MirrorEnvProcess | None):
    """Run interactive evaluation loop."""
    print(f"\n🎮 Starting Interactive Evaluation")
    print(f"Environment: {args.env_name}")
    model_label = args.model_path if args.model_path else "None"
    print(f"Model: {model_label}")
    print(f"Controller: {getattr(args, 'controller', 'policy')}")
    print(f"Device: {device}")
    print(f"\n🎮 Controls:")
    print(f"   ESC/Q: Exit")
    print(f"   SPACE: Reset environment")
    print(f"   N: Skip to next goal")
    print(f"   S: Slow down  |  F: Speed up")
    print(f"   R: Toggle auto-reset")
    print(f"   Click the window and use keys!")

    # Initialize pygame for event handling
    pygame.init()
    clock = pygame.time.Clock()
    current_fps = float(max(1, args.fps))
    FPS_MIN = 1.0
    FPS_MAX = 240.0
    FPS_SCALE = 1.5
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    action_dim = int(getattr(policy, 'action_dim', 0) or np.prod(env.action_space.shape))
    history_len = _resolve_action_history_len(args, policy)
    prev_action_dim = int(getattr(policy, 'prev_action_dim', 0))
    goal_history_dim = int(getattr(policy, 'goal_history_dim', 0))
    goal_history_len = goal_history_dim // 2 if goal_history_dim > 0 else 0
    if prev_action_dim <= 0:
        history_len = 0
    use_local_actions = bool(getattr(args, 'use_local_actions', False))
    translation_scale = float(getattr(args, 'se2_translation_scale', 0.2))
    decouple_view = bool(getattr(args, 'decouple_view', False))
    view_delta_scale = float(getattr(args, 'view_delta_scale', np.pi))
    policy_has_warp = bool(getattr(policy, 'use_se2_warp', False))
    action_history = _init_action_history(action_dim, history_len)
    transformer = ActionFrameTransformer(use_local_actions, translation_scale)
    goal_history = None
    if goal_history_len > 0:
        goal_history = _init_goal_history(2, goal_history_len)

    # Reset environment
    obs, info = env.reset(seed=args.seed)
    maybe_reset_policy_state(policy)
    action_history = _init_action_history(action_dim, history_len)
    transformer.reset()
    transformer.update_heading(info if isinstance(info, dict) else None)
    if goal_history_len > 0:
        initial_goal = _compute_goal_relative_local(env, transformer, getattr(args, 'goal_relative_scale', 10.0))
        goal_history = _init_goal_history(2, goal_history_len, fill=initial_goal)
    maze_env = unwrap_maze_env(env)
    birds_eye = None
    if maze_env is not None:
        initial_obs = _extract_pixel_panel(obs)
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
    
    def reset_environment(reason: Optional[str] = None):
        nonlocal obs, info, episode_reward, episode_length, mirror, birds_eye, action_history, transformer, goal_history
        if reason:
            print(reason)
        obs, info = env.reset()
        maybe_reset_policy_state(policy)
        action_history = _init_action_history(action_dim, history_len)
        transformer.reset()
        transformer.update_heading(info if isinstance(info, dict) else None)
        if goal_history_len > 0:
            initial_goal = _compute_goal_relative_local(env, transformer, getattr(args, 'goal_relative_scale', 10.0))
            goal_history = _init_goal_history(2, goal_history_len, fill=initial_goal)
        if mirror is not None:
            try:
                mirror.reset()
            except Exception as exc:
                print(f"⚠️ Mirror reset failed: {exc}")
                mirror.close()
                mirror = None
        if birds_eye is not None:
            try:
                birds_eye.draw(obs)
            except Exception as exc:
                print(f"⚠️ Bird's eye draw failed: {exc}")
                birds_eye = None
        episode_reward = 0.0
        episode_length = 0

    while running and (args.num_episodes == 0 or episode_count < args.num_episodes):
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    reset_environment("🔄 Manual reset triggered")
                elif event.key == pygame.K_n:
                    reset_environment("⏭️  Skipping to next goal")
                elif event.key == pygame.K_r:
                    auto_reset = not auto_reset
                    print(f"🔄 Auto-reset: {'ON' if auto_reset else 'OFF'}")
                elif event.key == pygame.K_s:
                    current_fps = max(FPS_MIN, current_fps / FPS_SCALE)
                    print(f"🐢 Slowdown: target FPS {current_fps:.1f}")
                elif event.key == pygame.K_f:
                    current_fps = min(FPS_MAX, current_fps * FPS_SCALE)
                    print(f"⚡ Speedup: target FPS {current_fps:.1f}")
        
        # Convert observation to tensor and add batch dimension
        obs_tensor = _format_policy_observation(obs, policy, device)
        obs_dict = TensorDict({
            "policy": obs_tensor,
        }, batch_size=[obs_tensor.shape[0]], device=device)
        
        # Get action from policy
        with torch.no_grad():
            if history_len > 0 and action_history:
                prev_stack = _flatten_action_history(action_history)
                prev_tensor = torch.as_tensor(prev_stack, device=device).view(1, -1)
            else:
                prev_tensor = None
            if goal_history_len > 0 and goal_history:
                goal_stack = _flatten_goal_history(goal_history)
                goal_tensor = torch.as_tensor(goal_stack, device=device).view(1, -1)
            else:
                goal_tensor = None
            try:
                actions = policy.act(
                    obs_dict,
                    deterministic=True,
                    prev_actions=prev_tensor,
                    goal_history=goal_tensor,
                )
            except TypeError:
                actions = policy.act(obs_dict, deterministic=True, prev_actions=prev_tensor)
        action_local_raw = actions.cpu().numpy()[0]
        move_local, view_delta = split_action_components(action_local_raw, decouple_view)
        move_local = clip_action_l2_np(move_local)
        if decouple_view:
            action_local = np.concatenate([move_local, np.array([view_delta], dtype=np.float32)], axis=0)
        else:
            action_local = move_local
        warp_params = None
        if policy_has_warp and hasattr(policy, 'register_pending_warp'):
            warp_params = transformer.compute_warp_from_action(move_local)
            policy.register_pending_warp(warp_params)
        move_global = transformer.to_global(move_local)
        move_global = clip_action_l2_np(move_global)
        if decouple_view:
            _set_view_dir_from_delta(env, move_global, view_delta, view_delta_scale)
            action_global = move_global
        else:
            action_global = move_global
        
        # Apply action processing (matching training settings)
        if args.action_scale != 1.0:
            action_global *= args.action_scale
        if args.clip_actions:
            action_global = np.clip(action_global, -1.0, 1.0)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action_global)
        done = terminated or truncated
        if mirror is not None:
            try:
                mirror.step(action_global)
            except Exception as exc:
                print(f"⚠️ Mirror step failed: {exc}")
                mirror.close()
                mirror = None
        if history_len > 0 and action_history is not None:
            action_history.append(action_local.copy())
        if goal_history_len > 0 and goal_history is not None:
            goal_rel = _compute_goal_relative_local(env, transformer, getattr(args, 'goal_relative_scale', 10.0))
            goal_history.append(goal_rel.copy())
        transformer.update_heading(info if isinstance(info, dict) else None)
        
        # Update episode tracking
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        if birds_eye is not None:
            try:
                birds_eye.draw(obs)
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
            
            goal_reached, distance_to_goal = _evaluate_goal_reached(info, distance_epsilon=args.success_distance_epsilon)
            if goal_reached:
                if distance_to_goal is not None:
                    print(f"🎯 GOAL REACHED! 🎉 (distance={distance_to_goal:.3f})")
                else:
                    print("🎯 GOAL REACHED! 🎉")
            elif distance_to_goal is not None:
                print(f"   Final distance to goal: {distance_to_goal:.3f}")
            
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
                reset_environment(None)
            else:
                print(f"⏸️  Auto-reset disabled. Press SPACE to reset manually.")
                # Keep current state until manual reset

            episode_reward = 0.0
            episode_length = 0
        
        # Control frame rate
        clock.tick(current_fps)
    
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
    if getattr(args, 'decouple_view', False):
        args.use_local_actions = True
    device = select_device(args.device)

    print("🚀 Interactive Policy Evaluation")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    if args.controller == 'human' and args.intervention_mode != 'human':
        print("ℹ️ Enabling human teleop intervention wrapper for manual control.")
        args.intervention_mode = 'human'

    config_info = apply_model_config_defaults(args) if args.model_path else None

    env: Optional[gym.Env] = None
    mirror: Optional[MirrorEnvProcess] = None
    policy = None
    training_info: dict[str, Any] = {}
    checkpoint = None
    ckpt_args_dict = None
    try:
        if args.controller == 'policy':
            if not args.model_path:
                raise ValueError("A --model_path must be provided when controller='policy'.")
            checkpoint = torch.load(args.model_path, map_location='cpu')
            ckpt_args_dict = update_args_from_checkpoint(args, checkpoint)
        else:
            if args.model_path:
                print("⚠️ controller!='policy'; ignoring provided --model_path.")
        log_eval_configuration(args, config_info=config_info, checkpoint_args=ckpt_args_dict)
        env, mirror = setup_environment(args)
        if args.controller == 'policy':
            policy, training_info = load_trained_policy(
                args.model_path,
                env,
                device,
                args,
                checkpoint=checkpoint,
            )
        else:
            obs_mode = getattr(args, 'obs_mode', 'state') or 'state'
            pixel_shape_hint = None
            if obs_mode == 'pixels':
                try:
                    pixel_shape_hint = _infer_pixel_shape_from_space(env.observation_space)
                except Exception:
                    pixel_shape_hint = None
            if args.controller == 'random':
                policy = RandomControllerPolicy(
                    action_space=env.action_space,
                    obs_mode=obs_mode,
                    pixel_shape=pixel_shape_hint,
                    device=device,
                )
            elif args.controller == 'human':
                policy = IdleControllerPolicy(
                    action_space=env.action_space,
                    obs_mode=obs_mode,
                    pixel_shape=pixel_shape_hint,
                    device=device,
                )
            elif args.controller == 'keyboard':
                action_dim = 3 if getattr(args, 'decouple_view', False) else int(np.prod(env.action_space.shape))
                policy = KeyboardControllerPolicy(
                    action_dim=action_dim,
                    device=device,
                    decouple_view=bool(getattr(args, 'decouple_view', False)),
                )
            else:
                raise ValueError(f"Unknown controller '{args.controller}'")
            training_info = {'controller': args.controller}
        if hasattr(policy, 'eval'):
            policy.eval()
        if training_info:
            print("Training info:")
            for key, value in training_info.items():
                print(f"   {key}: {value}")
        if args.headless:
            run_headless_evaluation(policy, env, args, device)
        else:
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
        if not args.headless and pygame.get_init():
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

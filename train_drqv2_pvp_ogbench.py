#!/usr/bin/env python3
"""
DrQ-v2 + Proxy Value Propagation (PVP) training entrypoint for OGBench.

This script implements the PVP workflow described in the paper: reward-free
TD learning plus proxy value loss on human-intervened actions, with balanced
sampling between novice and human buffers. A toggle allows the reward-based
ablation by storing environment rewards in the replay buffer.
"""

from __future__ import annotations

import os
import sys
import json
import time
import warnings
from collections import deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import dm_env
from dm_env import specs
import gymnasium as gym
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Ensure EGL is preferred for MuJoCo-powered envs.
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

# Default WANDB behaviour mirrors FastSAC: offline by default, quiet console.
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")

PROJECT_ROOT = Path(__file__).resolve().parent
TOOLS_PATH = PROJECT_ROOT / "tools"
if TOOLS_PATH.exists():
    sys.path.append(str(TOOLS_PATH))
DRQV2_PATH = PROJECT_ROOT / "drqv2"
if DRQV2_PATH.exists() and str(DRQV2_PATH) not in sys.path:
    sys.path.append(str(DRQV2_PATH))

try:
    from visualize_policy_map import generate_policy_map  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    generate_policy_map = None

from drqv2.drqv2 import DrQV2Agent  # noqa: E402
from drqv2 import utils  # noqa: E402
from drqv2.replay_buffer import (  # noqa: E402
    ReplayBufferStorage,
    InMemoryReplayStorage,
    make_replay_loader,
    make_replay_loader_in_memory,
)
from drqv2.pref_buffer import PreferencePairStorage, PreferencePairDataset  # noqa: E402

from ogbench_utils import (  # noqa: E402
    TeacherMetricsAccumulator,
    build_train_parser,
    maybe_set_goal_color,
)
from ogbench_utils.env_wrappers_manip import build_ogbench_manip_wrapper  # noqa: E402
from ogbench_utils.env_wrappers_maze import build_ogbench_maze_wrapper  # noqa: E402


# ---------------------------------------------------------------------------
# CLI / configuration helpers
# ---------------------------------------------------------------------------


def build_arg_parser():
    """Extend the shared OGBench parser with DrQ-v2 knobs."""
    parser = build_train_parser()
    parser.set_defaults(
        obs_mode="pixels",
        pixel_width=84,
        pixel_height=84,
        include_goal=True,
        reward_type="dense",
        dense_reward_scale=1.0,
        batch_size=512,
        save_interval=50_000,
        viz_first_step=0,
        viz_grid_resolution=32,
        viz_quiver_stride=2,
        pref_loss_type="pvp",
        pref_buffer_enable=True,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--frame_stack", type=int, default=3)
    parser.add_argument("--save_snapshot", action="store_true", default=False)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--save_train_video", action="store_true", default=False)
    parser.add_argument("--video_render_size", type=int, default=256)

    # Agent hyperparameters
    parser.add_argument("--drq_feature_dim", type=int, default=50)
    parser.add_argument("--drq_hidden_dim", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--critic_target_tau", type=float, default=0.01)
    parser.add_argument("--stddev_schedule", type=str, default="linear(1.0,0.1,1e5)")
    parser.add_argument("--stddev_clip", type=float, default=0.3)
    parser.add_argument("--num_expl_steps", type=int, default=2000)
    parser.add_argument("--update_every_steps", type=int, default=1)
    parser.add_argument("--se2_translation_scale", type=float, default=0.2,
                        help="Scale factor applied to action x/y when computing SE(2) translation (in latent pixels)")
    parser.add_argument("--use_local_actions", action="store_true", default=False,
                        help="Interpret policy actions in the agent first-person frame when using first-person camera")
    parser.add_argument("--context_mlp", action="store_true", default=False,
                        help="Encode action/goal history with an MLP before concatenation")
    parser.add_argument("--context_hidden_dim", type=int, default=128,
                        help="Hidden dimension for the action/goal context MLP")
    parser.add_argument("--context_dim", type=int, default=64,
                        help="Output dimension for the action/goal context MLP")

    # Replay / optimisation knobs
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--nstep", type=int, default=3)
    parser.add_argument("--replay_buffer_size", type=int, default=1_000_000)
    parser.add_argument("--replay_buffer_num_workers", type=int, default=4)
    parser.add_argument("--replay_fetch_every", type=int, default=1000,
                        help="How often to scan disk replay buffers for new episodes")

    # Evaluation / logging
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--eval_every_frames", type=int, default=50_000)
    parser.add_argument("--smoke_test_steps", type=int, default=0,
                        help="Override total timesteps for very short smoke tests (0 disables override)")
    parser.add_argument("--pvp_use_reward_in_replay", action="store_true", default=False,
                        help="Store environment rewards in the replay buffer (PVP reward ablation)")
    parser.add_argument("--pref_loss_type", type=str, default="pvp", choices=["pvp"],
                        help="Proxy value loss type (fixed for PVP runs)")
    parser.add_argument("--buffer_root", type=str, default=None,
                        help="Optional root directory for replay/pref buffers (e.g., tmpfs/ramdisk).")
    parser.add_argument("--replay_mode", type=str, default="disk", choices=["disk", "ram"],
                        help="Replay buffer backend: disk (default) or in-memory")
    parser.add_argument("--pref_chunk_size", type=int, default=64,
                        help="Number of preference events per on-disk shard")
    parser.add_argument("--pref_fetch_every", type=int, default=512,
                        help="How often (in samples) to attempt loading new preference shards")
    return parser

def clip_action_l2(actions: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    if not isinstance(actions, np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        norm = np.linalg.norm(actions)
        if norm > max_norm and norm > 0:
            actions = actions / norm
        return actions
    norms = np.linalg.norm(actions, axis=-1, keepdims=True)
    mask = norms > max_norm
    result = actions.copy()
    result[mask & (norms > 0)] = result[mask & (norms > 0)] / norms[mask & (norms > 0)]
    return result


def is_manip_env(env_name: str) -> bool:
    lowered = env_name.lower()
    return any(tag in lowered for tag in ("cube", "scene", "puzzle"))


def split_action_components(action: np.ndarray, decouple_view: bool) -> tuple[np.ndarray, float]:
    vec = np.asarray(action, dtype=np.float32)
    if not decouple_view:
        return vec, 0.0
    if vec.shape[0] < 3:
        padded = np.pad(vec, (0, 3 - vec.shape[0]), mode='constant')
    else:
        padded = vec
    move = padded[:2]
    view_delta = float(np.clip(padded[2], -1.0, 1.0))
    return move, view_delta


def init_action_history(action_dim: int, stack: int) -> deque:
    history = deque(maxlen=stack)
    zero = np.zeros(action_dim, dtype=np.float32)
    for _ in range(stack):
        history.append(zero.copy())
    return history


def flatten_action_history(history: deque) -> np.ndarray:
    if not history:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(list(history), axis=0).astype(np.float32, copy=False)


def init_goal_history(goal_dim: int, stack: int, fill: Optional[np.ndarray] = None) -> deque:
    history = deque(maxlen=stack)
    if fill is None:
        base = np.zeros(goal_dim, dtype=np.float32)
    else:
        base = np.asarray(fill, dtype=np.float32).reshape(goal_dim)
    for _ in range(stack):
        history.append(base.copy())
    return history


def flatten_goal_history(history: deque) -> np.ndarray:
    if not history:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(list(history), axis=0).astype(np.float32, copy=False)


class ActionFrameTransformer:
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

    def update_heading(self, info: dict):
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

def parse_args():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.nstep != 1:
        print(
            "[Warn] nstep>1 is not supported in current buffer setups. "
            "Forcing --nstep 1 to avoid invalid multi-step samples."
        )
        args.nstep = 1
    if args.smoke_test_steps and args.smoke_test_steps > 0:
        args.total_timesteps = args.smoke_test_steps
        args.eval_every_frames = max(1, args.smoke_test_steps // 2)
        args.save_interval = max(1, args.smoke_test_steps)
        args.num_eval_episodes = 1
    if getattr(args, "decouple_view", False):
        args.use_local_actions = True
    if not getattr(args, "use_intervention", False):
        args.intervention_mode = "none"
    return args


def ensure_experiment_name(args) -> None:
    if args.exp_name:
        return
    env_tag = args.env_name.replace("-v0", "").replace("-", "_")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    args.exp_name = f"drqv2_{env_tag}_{stamp}"


# ---------------------------------------------------------------------------
# Environment adapter
# ---------------------------------------------------------------------------


class OGBenchPixelsEnv(dm_env.Environment):
    """Minimal dm_env-compatible wrapper over a single Gym OGBench environment."""

    def __init__(
        self,
        *,
        env_name: str,
        wrappers,
        width: int,
        height: int,
        camera_name: Optional[str],
        pixel_camera_mode: str,
        pixel_local_view_size: float,
        pixel_local_camera_height: Optional[float],
        pixel_first_person_distance: float,
        pixel_first_person_height: float,
        pixel_first_person_lookahead: float,
        pixel_first_person_pitch: float,
        decouple_view: bool,
        view_delta_scale: float,
        goal_marker_color: str,
        seed: int,
    ):
        env_kwargs = dict(
            render_mode="rgb_array",
            width=width,
            height=height,
            pixel_camera_mode=pixel_camera_mode,
            pixel_local_view_size=pixel_local_view_size,
            pixel_local_camera_height=pixel_local_camera_height,
            pixel_first_person_distance=pixel_first_person_distance,
            pixel_first_person_height=pixel_first_person_height,
            pixel_first_person_lookahead=pixel_first_person_lookahead,
            pixel_first_person_pitch=pixel_first_person_pitch,
            disable_env_checker=True,
        )
        if camera_name is not None:
            env_kwargs["camera_name"] = camera_name

        try:
            base_env = gym.make(env_name, **env_kwargs)
        except TypeError as exc:
            # Fallback for environments that don't yet support the new kwargs.
            drop_keys = [
                'pixel_camera_mode',
                'pixel_local_view_size',
                'pixel_local_camera_height',
                'pixel_first_person_distance',
                'pixel_first_person_height',
                'pixel_first_person_lookahead',
                'pixel_first_person_pitch',
                'disable_env_checker',
                'camera_name',
            ]
            if any(key in str(exc) for key in drop_keys):
                for key in drop_keys:
                    env_kwargs.pop(key, None)
                warnings.warn(
                    f"{env_name} does not accept pixel camera kwargs; falling back to default camera behaviour."
                )
                base_env = gym.make(env_name, **env_kwargs)
            else:
                raise
        for wrap in wrappers:
            base_env = wrap(base_env)
        maybe_set_goal_color(base_env, goal_marker_color)
        self._env = base_env
        self._seed = seed
        self._last_info: Dict[str, Any] = {}
        self._next_seed: Optional[int] = seed + 1
        self._decouple_view = bool(decouple_view)
        self._view_delta_scale = float(view_delta_scale)

        action_space = base_env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise RuntimeError("OGBench DrQ wrapper expects a continuous Box action space.")
        if self._decouple_view:
            low = np.concatenate([action_space.low, np.array([-1.0], dtype=np.float32)], axis=0)
            high = np.concatenate([action_space.high, np.array([1.0], dtype=np.float32)], axis=0)
            shape = low.shape
        else:
            low = action_space.low
            high = action_space.high
            shape = action_space.shape
        self._action_spec = specs.BoundedArray(
            shape=shape,
            dtype=np.float32,
            minimum=low,
            maximum=high,
            name="action",
        )

        # Probe observation shape for spec construction.
        obs_sample, info = self._env.reset(seed=self._seed)
        self._last_info = info or {}
        obs_array = self._render_pixels(obs_sample)
        self._obs_key = "pixels"
        self._obs_shape = obs_array.shape
        self._obs_spec = {
            self._obs_key: specs.BoundedArray(
                shape=self._obs_shape,
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name=self._obs_key,
            )
        }

    @staticmethod
    def _to_pixels(obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            for key in ("pixels", "image", "policy", "observation"):
                if key in obs:
                    obs = obs[key]
                    break
        arr = np.asarray(obs)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 255.0)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = arr.astype(np.uint8)
        return arr

    def _render_pixels(self, raw_obs: Any) -> np.ndarray:
        try:
            frame = self._env.render()
        except Exception:
            frame = None
        if frame is None:
            frame = raw_obs
        if isinstance(frame, dict):
            for key in ("pixels", "image", "policy", "observation"):
                if key in frame:
                    frame = frame[key]
                    break
        return self._to_pixels(frame)

    def goal_relative_global(self) -> Optional[np.ndarray]:
        base = getattr(self._env, "unwrapped", self._env)
        if hasattr(base, "get_xy") and hasattr(base, "cur_goal_xy"):
            try:
                agent_xy = np.asarray(base.get_xy(), dtype=np.float32)
                goal_xy = np.asarray(base.cur_goal_xy, dtype=np.float32)
                return goal_xy - agent_xy
            except Exception:
                return None
        return None

    def reset(self) -> dm_env.TimeStep:
        if self._next_seed is not None:
            obs, info = self._env.reset(seed=self._next_seed)
            self._next_seed = None
        else:
            obs, info = self._env.reset()
        self._last_info = info or {}
        pixels = self._render_pixels(obs)
        return dm_env.restart({self._obs_key: pixels})

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        move_action = action
        if self._decouple_view:
            move_action = np.asarray(action, dtype=np.float32)[:2]
            view_delta = float(np.asarray(action, dtype=np.float32)[2])
            view_dir = self._compute_view_dir(move_action, view_delta)
            if hasattr(self._env, "unwrapped") and hasattr(self._env.unwrapped, "set_view_dir"):
                self._env.unwrapped.set_view_dir(view_dir)
        obs, reward, terminated, truncated, info = self._env.step(move_action)
        info = dict(info or {})
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        self._last_info = info
        pixels = self._render_pixels(obs)
        done = bool(terminated or truncated)
        step_type = dm_env.StepType.LAST if done else dm_env.StepType.MID
        discount = np.array(0.0 if done else 1.0, dtype=np.float32)
        reward_arr = np.array(reward, dtype=np.float32)
        return dm_env.TimeStep(step_type=step_type, reward=reward_arr, discount=discount, observation={self._obs_key: pixels})

    def last_info(self) -> Dict[str, Any]:
        return self._last_info

    def _compute_view_dir(self, move_action: np.ndarray, view_delta: float) -> np.ndarray | None:
        base = getattr(self._env, "unwrapped", self._env)
        base_dir = None
        move_vec = np.asarray(move_action, dtype=np.float32)
        norm = np.linalg.norm(move_vec)
        if norm > 1e-6:
            base_dir = move_vec / norm
        elif hasattr(base, "_last_move_dir"):
            try:
                base_dir = np.asarray(getattr(base, "_last_move_dir"), dtype=np.float32)[:2]
            except Exception:
                base_dir = None
        if base_dir is None:
            base_dir = np.array([1.0, 0.0], dtype=np.float32)
        angle = float(np.clip(view_delta, -1.0, 1.0)) * self._view_delta_scale
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        return (rot @ base_dir.reshape(-1, 1)).reshape(2)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Logging + checkpoints
# ---------------------------------------------------------------------------


@dataclass
class RunPaths:
    log_dir: Path
    model_dir: Path
    policy_map_dir: Path
    policy_cache_path: Path
    record_progress: Any
    log_args_path: Path
    model_args_path: Path


def prepare_run_dirs(args) -> RunPaths:
    logs_root = Path("logs") / "drqv2"
    models_root = Path("models") / "drqv2"
    logs_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    run_log_dir = logs_root / args.exp_name
    run_model_dir = models_root / args.exp_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = run_log_dir / "policy_maps"
    cache_path = run_log_dir / "policy_map_goal.json"
    log_file = run_log_dir / "training.log"
    progress_fp = open(log_file, "a", encoding="utf-8")
    progress_fp.write(f"# logging started {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    progress_fp.flush()

    args_payload = vars(args)

    def _record(msg: str) -> None:
        stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
        progress_fp.write(f"{stamp} {msg}\n")
        progress_fp.flush()

    log_args_path = run_log_dir / "args.json"
    with open(log_args_path, "w", encoding="utf-8") as fp:
        json.dump(args_payload, fp, indent=2)

    model_args_path = run_model_dir / "args.json"
    with open(model_args_path, "w", encoding="utf-8") as fp:
        json.dump(args_payload, fp, indent=2)

    return RunPaths(
        log_dir=run_log_dir,
        model_dir=run_model_dir,
        policy_map_dir=viz_dir,
        policy_cache_path=cache_path,
        record_progress=_record,
        log_args_path=log_args_path,
        model_args_path=model_args_path,
    )


class DrQCheckpointManager:
    def __init__(self, *, args, run_paths: RunPaths, logger: DrQLogger | None):
        self.args = args
        self.paths = run_paths
        self.logger = logger

    def save(self, *, tag: str, step_value: int, agent: DrQV2Agent, pixel_shape: Tuple[int, int, int], action_shape: Tuple[int, ...]) -> Path:
        save_path = self.paths.model_dir / f"{self.args.exp_name}_{tag}.pt"
        encoder_module = getattr(agent, 'encoder', None)
        if encoder_module is None:
            encoder_module = getattr(agent, 'core', None)
        if encoder_module is None:
            raise AttributeError("Agent does not expose encoder/core for checkpointing")
        payload = {
            "algo": "drqv2",
            "step": step_value,
            "drq_encoder": encoder_module.state_dict(),
            "drq_actor": agent.actor.state_dict(),
            "drq_critic": agent.critic.state_dict(),
            "drq_critic_target": agent.critic_target.state_dict(),
            "pixel_shape": pixel_shape,
            "action_shape": action_shape,
            "args": vars(self.args),
        }
        torch.save(payload, save_path)
        self.paths.record_progress(f"[Checkpoint] saved {save_path}")
        return save_path

    def maybe_render_policy_map(self, *, checkpoint_path: Path, step_value: int, force: bool = False) -> None:
        if not self.args.viz_on_checkpoint or generate_policy_map is None:
            return
        if (self.args.viz_first_step and step_value < self.args.viz_first_step) and not force:
            return
        try:
            png_path, _, _ = generate_policy_map(
                model_path=checkpoint_path,
                output_dir=self.paths.policy_map_dir,
                tag=f"{step_value}",
                env_name=self.args.env_name,
                device=self.args.viz_device,
                grid_resolution=self.args.viz_grid_resolution,
                quiver_stride=self.args.viz_quiver_stride,
                seed=self.args.viz_seed,
                cache_path=self.paths.policy_cache_path,
            )
            self.paths.record_progress(f"[Viz] generated policy map for step {step_value}")
            if self.logger is not None:
                self.logger.log_policy_map(image_path=png_path, step=step_value)
        except Exception as exc:  # pragma: no cover - viz is best-effort
            self.paths.record_progress(f"[Viz] failed at step {step_value}: {exc}")


class DrQLogger:
    def __init__(self, *, args, record_progress, teacher_metrics: TeacherMetricsAccumulator):
        self.args = args
        self.record_progress = record_progress
        self.teacher_metrics = teacher_metrics
        self.next_log_step = args.log_interval if args.log_interval > 0 else None
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.wandb_run = None
        self._env_metric_accum: Dict[str, Dict[str, float]] = {}
        self._env_metric_bounds: Dict[str, Dict[str, float]] = {}

    def maybe_log(
        self,
        *,
        total_env_steps: int,
        total_timesteps: int,
        rewbuffer: list,
        lenbuffer: list,
        update_metrics: Dict[str, float],
        update_count: int,
        perf_metrics: Optional[Dict[str, float]] = None,
        log_alpha: float = 0.0,
        last_denied_samples: int = 0,
        pref_buffer_size: int = -1,
        goal_successes: int = 0,
        goal_distance_sum: float = 0.0,
        goal_distance_count: int = 0,
    ) -> bool:
        if self.next_log_step is None or total_env_steps < self.next_log_step:
            return False

        now = time.perf_counter()
        elapsed = now - self.last_log_time
        total_elapsed = now - self.start_time
        fps = (total_env_steps - (self.next_log_step - self.args.log_interval)) / max(elapsed, 1e-6) if self.args.log_interval else total_env_steps / max(total_elapsed, 1e-6)

        logs: Dict[str, float] = {
            "Perf/env_steps": float(total_env_steps),
            "Perf/total_time_sec": float(total_elapsed),
            "Perf/fps": float(fps),
        }
        if rewbuffer:
            logs["Train/episode_reward_mean"] = float(np.mean(rewbuffer[-100:]))
        if lenbuffer:
            logs["Train/episode_length_mean"] = float(np.mean(lenbuffer[-100:]))
        if update_count > 0:
            for key, value in update_metrics.items():
                logs[f"Train/{key}"] = float(value / max(1, update_count))
        else:
            logs["Train/alpha"] = float(np.exp(log_alpha))

        if last_denied_samples > 0:
            logs["/Teacher/denied_transition_samples"] = float(last_denied_samples)
        if pref_buffer_size >= 0:
            logs["/Buffers/pref_pairs"] = float(pref_buffer_size)
        if goal_successes > 0:
            logs["/Env/goal_successes"] = float(goal_successes)
        if goal_distance_count > 0:
            logs["/Env/final_goal_distance"] = float(goal_distance_sum / max(1, goal_distance_count))

        if perf_metrics:
            logs.update({k: float(v) for k, v in perf_metrics.items()})

        teacher_snapshot = self.teacher_metrics.snapshot()
        if teacher_snapshot.qmin_all is not None:
            logs["/Critic/mean_q_min_all"] = teacher_snapshot.qmin_all
        if teacher_snapshot.qmin_teacher is not None:
            logs["/Critic/mean_q_min_intervened"] = teacher_snapshot.qmin_teacher
        if teacher_snapshot.qmin_non is not None:
            logs["/Critic/mean_q_min_no_intervention"] = teacher_snapshot.qmin_non

        self._flush_env_metrics(logs)
        for key in list(logs.keys()):
            lower_key = key.lower()
            if "disagreement" in lower_key or "frac_interventions_dis_ge_" in lower_key:
                logs.pop(key, None)

        msg_parts = [
            f"env_steps {total_env_steps}/{total_timesteps}",
            f"fps {int(fps)}",
        ]
        if "Train/episode_reward_mean" in logs:
            msg_parts.append(f"rew {logs['Train/episode_reward_mean']:.2f}")
        if "Train/episode_length_mean" in logs:
            msg_parts.append(f"len {logs['Train/episode_length_mean']:.1f}")
        if "/Wrapper/teacher_fraction_steps" in logs:
            msg_parts.append(f"teacher_frac {logs['/Wrapper/teacher_fraction_steps']:.2f}")
        self.record_progress("[DrQ] " + " | ".join(msg_parts))

        self._log_to_wandb(logs, step=total_env_steps)

        self.teacher_metrics.reset_after_log()
        self.last_log_time = now
        if self.next_log_step is not None:
            self.next_log_step += self.args.log_interval
        return True

    def log_eval(self, *, total_env_steps: int, metrics: Dict[str, float]) -> None:
        payload = {f"Eval/{key}": float(value) for key, value in metrics.items()}
        msg = ", ".join(f"{k}={v:.3f}" for k, v in payload.items())
        self.record_progress(f"[Eval] steps={total_env_steps} {msg}")
        self._log_to_wandb(payload, step=total_env_steps)

    def log_policy_map(self, *, image_path: Path, step: int) -> None:
        if not self.args.use_wandb:
            return
        run = self._ensure_wandb_run()
        if run is None:
            return
        import wandb  # type: ignore

        run.log(
            {
                "viz/policy_map": wandb.Image(str(image_path), caption=f"policy_map_step_{step}"),
            },
            step=step,
        )

    def accumulate_env_metrics(self, infos) -> None:
        def _accumulate_metric(metric_key: str, value: Any) -> None:
            try:
                tensor = value
                if hasattr(tensor, "float"):
                    tensor = tensor.float()
                metric_value = float(torch.as_tensor(tensor).mean().item())
            except Exception:
                return
            slot = self._env_metric_accum.setdefault(metric_key, {"sum": 0.0, "count": 0})
            slot["sum"] += metric_value
            slot["count"] += 1
            bounds = self._env_metric_bounds.setdefault(metric_key, {"min": metric_value, "max": metric_value})
            bounds["min"] = min(bounds["min"], metric_value)
            bounds["max"] = max(bounds["max"], metric_value)

        log_dict = None
        if isinstance(infos, dict):
            candidate = infos.get("log")
            if isinstance(candidate, dict):
                log_dict = candidate
            else:
                log_dict = None
        if isinstance(log_dict, dict) and log_dict:
            for key, value in log_dict.items():
                _accumulate_metric(key, value)

        if isinstance(infos, dict):
            for key, value in infos.items():
                if not isinstance(key, str):
                    continue
                if key.startswith("teacher_") or key in {"terminated", "truncated"}:
                    _accumulate_metric(f"/Wrapper/{key}", value)
            if "proprio/gripper_opening" in infos:
                opening = infos["proprio/gripper_opening"]
                _accumulate_metric("/Wrapper/gripper_opening", opening)
            if "proprio/gripper_contact" in infos:
                contact = infos["proprio/gripper_contact"]
                _accumulate_metric("/Wrapper/gripper_contact", contact)
                try:
                    closed = (np.asarray(contact, dtype=np.float32) > 0.5).astype(np.float32)
                    _accumulate_metric("/Wrapper/gripper_closed", closed)
                except Exception:
                    pass
            if "proprio/effector_yaw" in infos:
                _accumulate_metric("/Wrapper/gripper_yaw_rad", infos["proprio/effector_yaw"])

    def _flush_env_metrics(self, logs: Dict[str, float]) -> None:
        if not self._env_metric_accum:
            return
        for key, data in self._env_metric_accum.items():
            if data["count"] > 0:
                logs[key] = data["sum"] / data["count"]
                bounds = self._env_metric_bounds.get(key)
                if bounds is not None:
                    logs[f"{key}_min"] = bounds["min"]
                    logs[f"{key}_max"] = bounds["max"]
        self._env_metric_accum.clear()
        self._env_metric_bounds.clear()

    def _ensure_wandb_run(self):
        if not self.args.use_wandb:
            return None
        if self.wandb_run is not None:
            return self.wandb_run
        import wandb  # type: ignore

        self.wandb_run = wandb.init(
            project=self.args.project,
            name=self.args.exp_name,
            id=self.args.exp_name,
            config=vars(self.args),
            reinit=True,
            resume="allow",
        )
        return self.wandb_run

    def _log_to_wandb(self, payload: Dict[str, float], *, step: int) -> None:
        run = self._ensure_wandb_run()
        if run is not None:
            run.log(payload, step=step)

    def finish(self):
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Teacher metrics helper
# ---------------------------------------------------------------------------


def build_teacher_metrics(args, device: torch.device) -> TeacherMetricsAccumulator:
    hist_edges_tensor = None
    hist_labels: list[str] = []
    if args.disagreement_hist_edges:
        try:
            parsed_edges = [float(edge.strip()) for edge in args.disagreement_hist_edges.split(",") if edge.strip()]
            parsed_edges = sorted(edge for edge in parsed_edges if edge > 0.0)
        except Exception:
            parsed_edges = []
        if parsed_edges:
            hist_edges_tensor = torch.tensor(parsed_edges, dtype=torch.float32, device=device)
            num_bins = hist_edges_tensor.numel() + 1
            for idx in range(num_bins):
                if idx == 0:
                    hist_labels.append(f"<= {parsed_edges[0]:.2f}")
                elif idx == num_bins - 1:
                    hist_labels.append(f">= {parsed_edges[-1]:.2f}")
                else:
                    hist_labels.append(f"({parsed_edges[idx-1]:.2f}, {parsed_edges[idx]:.2f}]")

    if args.disagreement_thresholds:
        try:
            thresh_vals = [float(x.strip()) for x in args.disagreement_thresholds.split(",") if x.strip()]
            thresh_vals = sorted(t for t in thresh_vals if t > 0.0)
        except Exception:
            thresh_vals = []
    else:
        thresh_vals = []
    thresh_tensor = torch.tensor(thresh_vals, dtype=torch.float32, device=device) if thresh_vals else None
    return TeacherMetricsAccumulator(
        device=device,
        hist_edges_tensor=hist_edges_tensor,
        hist_labels=hist_labels,
        thresh_tensor=thresh_tensor,
        thresh_values=thresh_vals,
    )


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def ensure_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(obs, device=device)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.float()


def concat_replay_batches(batch_a: tuple, batch_b: tuple) -> tuple:
    if len(batch_a) != len(batch_b):
        raise ValueError("Replay batch mismatch when balancing novice/human samples.")
    combined = []
    for item_a, item_b in zip(batch_a, batch_b):
        if torch.is_tensor(item_a) or torch.is_tensor(item_b):
            combined.append(torch.cat([item_a, item_b], dim=0))
        else:
            combined.append(np.concatenate([item_a, item_b], axis=0))
    return tuple(combined)


def collect_replay_stats(loaders) -> Dict[str, float]:
    totals = defaultdict(float)
    counts = defaultdict(int)
    for loader in loaders:
        if loader is None or not hasattr(loader, "dataset"):
            continue
        dataset = loader.dataset
        if hasattr(dataset, "get_stats"):
            stats, stat_counts = dataset.get_stats(reset=True)
            for key, value in stats.items():
                totals[key] += float(value)
            for key, value in stat_counts.items():
                counts[key] += int(value)
    metrics = {}
    if totals:
        metrics["Perf/replay_scan_ms"] = totals.get("replay_scan_s", 0.0) * 1000.0
        metrics["Perf/replay_load_ms"] = totals.get("replay_load_s", 0.0) * 1000.0
        metrics["Perf/replay_sample_ms"] = totals.get("replay_sample_s", 0.0) * 1000.0
        metrics["Perf/replay_scan_calls"] = counts.get("replay_scan_s", 0)
        metrics["Perf/replay_load_calls"] = counts.get("replay_load_s", 0)
        metrics["Perf/replay_sample_calls"] = counts.get("replay_sample_s", 0)
    return metrics


def build_env(args, wrappers, seed: int) -> dm_env.Environment:
    env = OGBenchPixelsEnv(
        env_name=args.env_name,
        wrappers=wrappers,
        width=int(args.pixel_width),
        height=int(args.pixel_height),
        camera_name=args.pixel_camera,
        pixel_camera_mode=args.pixel_camera_mode,
        pixel_local_view_size=float(args.pixel_local_view_size),
        pixel_local_camera_height=args.pixel_local_camera_height,
        pixel_first_person_distance=float(args.pixel_first_person_distance),
        pixel_first_person_height=float(args.pixel_first_person_height),
        pixel_first_person_lookahead=float(args.pixel_first_person_lookahead),
        pixel_first_person_pitch=float(args.pixel_first_person_pitch),
        decouple_view=bool(getattr(args, "decouple_view", False)),
        view_delta_scale=float(getattr(args, "view_delta_scale", np.pi)),
        goal_marker_color=getattr(args, "goal_marker_color", "auto"),
        seed=seed,
    )
    return env


def wrap_env_for_drq(env: dm_env.Environment, args) -> dm_env.Environment:
    from drqv2.dmc import ActionRepeatWrapper, FrameStackWrapper, ActionDTypeWrapper, ExtendedTimeStepWrapper

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, args.action_repeat)
    env = FrameStackWrapper(env, args.frame_stack, pixels_key="pixels")
    env = ExtendedTimeStepWrapper(env)
    return env


def compute_goal_relative_local(
    env: dm_env.Environment,
    action_transformer: ActionFrameTransformer,
    scale: float,
    fallback_dim: int = 2,
) -> np.ndarray:
    rel_global = getattr(env, "goal_relative_global", lambda: None)()
    if rel_global is None:
        return np.zeros(fallback_dim, dtype=np.float32)
    rel_local = action_transformer.to_local(np.asarray(rel_global, dtype=np.float32))
    if scale and scale > 0:
        rel_local = rel_local / float(scale)
    return np.clip(rel_local, -1.0, 1.0).astype(np.float32, copy=False)


def run_eval_metrics(env,
                     agent: DrQV2Agent,
                     num_episodes: int,
                     device: torch.device,
                     action_dim: int,
                     history_len: int,
                     goal_history_len: int,
                     use_local_actions: bool,
                     translation_scale: float,
                     goal_scale: float,
                     manip_env: bool) -> Dict[str, float]:
    total_reward = 0.0
    total_length = 0
    success_count = 0
    lethal_count = 0
    timeout_count = 0
    distance_sum = 0.0
    distance_count = 0
    success_length_sum = 0
    action_transformer = ActionFrameTransformer(use_local_actions, translation_scale)
    for _ in range(num_episodes):
        action_history = init_action_history(action_dim, history_len)
        prev_stack = flatten_action_history(action_history)
        goal_history = None
        if goal_history_len > 0:
            initial_goal = compute_goal_relative_local(
                env,
                action_transformer,
                scale=goal_scale,
                fallback_dim=2,
            )
            goal_history = init_goal_history(2, goal_history_len, fill=initial_goal)
        goal_stack = flatten_goal_history(goal_history) if goal_history is not None else None
        time_step = env.reset()
        action_transformer.reset()
        done = False
        episode_reward = 0.0
        episode_len = 0
        final_info = {}
        while not time_step.last():
            obs = time_step.observation
            with torch.no_grad(), utils.eval_mode(agent):
                action_local = agent.act(obs,
                                         step=0,
                                         eval_mode=True,
                                         prev_actions=prev_stack,
                                         goal_history=goal_stack)
            if manip_env:
                action_global = np.clip(np.asarray(action_local, dtype=np.float32), -1.0, 1.0)
                action_local_full = action_global
            else:
                move_local, view_delta = split_action_components(
                    action_local, bool(getattr(env, "action_spec", lambda: None)() and env.action_spec().shape[0] == 3)
                )
                move_local = clip_action_l2(move_local)
                action_local_full = (
                    np.concatenate([move_local, np.array([view_delta], dtype=np.float32)], axis=0)
                    if getattr(env, "action_spec", lambda: None)() and env.action_spec().shape[0] == 3
                    else move_local
                )
                move_global = action_transformer.to_global(move_local)
                move_global = clip_action_l2(move_global)
                if action_local_full.shape[0] == 3:
                    action_global = np.concatenate([move_global, np.array([view_delta], dtype=np.float32)], axis=0)
                else:
                    action_global = move_global
            time_step = env.step(action_global)
            action_history.append(np.asarray(action_local_full, dtype=np.float32).copy())
            prev_stack = flatten_action_history(action_history)
            action_transformer.update_heading(getattr(env, "last_info", lambda: {})() or {})
            if goal_history is not None:
                goal_rel_local = compute_goal_relative_local(
                    env,
                    action_transformer,
                    scale=goal_scale,
                    fallback_dim=2,
                )
                goal_history.append(goal_rel_local.copy())
                goal_stack = flatten_goal_history(goal_history)
            episode_reward += float(time_step.reward)
            episode_len += 1
            if time_step.last():
                final_info = getattr(env, "last_info", lambda: {})() or {}
        total_reward += episode_reward
        total_length += episode_len
        if final_info:
            if bool(final_info.get("goal_reached", False)) or bool(final_info.get("success", False)):
                success_count += 1
                success_length_sum += episode_len
            if bool(final_info.get("killed", 0.0)):
                lethal_count += 1
            if bool(final_info.get("time_out", False)) or bool(final_info.get("truncated", False)):
                timeout_count += 1
            if "distance_to_goal" in final_info:
                distance_sum += float(final_info["distance_to_goal"])
                distance_count += 1

    denom = max(1, num_episodes)
    metrics = {
        "avg_return": total_reward / denom,
        "avg_length": total_length / denom,
        "success_rate": success_count / denom,
        "lethal_rate": lethal_count / denom,
        "timeout_rate": timeout_count / denom,
    }
    if distance_count > 0:
        metrics["avg_final_distance"] = distance_sum / max(1, distance_count)
    if success_count > 0:
        metrics["avg_success_length"] = success_length_sum / max(1, success_count)
    return metrics


def train():
    args = parse_args()
    ensure_experiment_name(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    cudnn.benchmark = True
    utils.set_seed_everywhere(args.seed)
    build_wrapper = (
        build_ogbench_manip_wrapper if is_manip_env(args.env_name) else build_ogbench_maze_wrapper
    )
    train_wrappers = [build_wrapper(
        env_name=args.env_name,
        obs_mode=args.obs_mode,
        include_goal=args.include_goal,
        include_distance=args.include_distance,
        include_direction=args.include_direction,
        include_velocity=args.include_velocity,
        reward_type=args.reward_type,
        dense_reward_scale=args.dense_reward_scale,
        step_penalty=args.step_penalty,
        reward_switch_after_steps=args.reward_switch_after_steps // max(1, args.num_envs),
        intervention_mode=args.intervention_mode if args.use_intervention else "none",
        teacher_type=args.teacher_type,
        tolerance_type=args.tolerance_type,
        tolerance_value=args.tolerance_value,
        hard_block_lethal=args.hard_block_lethal,
        intervention_enable_after_steps=args.intervention_enable_after_steps,
        intervention_safety_margin_frac=args.intervention_safety_margin_frac,
        intervention_release_steps=args.intervention_release_steps,
        teacher_target_mode=args.teacher_target_mode,
        cube_success_tolerance=args.cube_success_tolerance,
    )]
    eval_wrappers = [build_wrapper(
        env_name=args.env_name,
        obs_mode=args.obs_mode,
        include_goal=args.include_goal,
        include_distance=args.include_distance,
        include_direction=args.include_direction,
        include_velocity=args.include_velocity,
        reward_type=args.reward_type,
        dense_reward_scale=args.dense_reward_scale,
        step_penalty=args.step_penalty,
        reward_switch_after_steps=args.reward_switch_after_steps // max(1, args.num_envs),
        intervention_mode="none",
        teacher_type=args.teacher_type,
        tolerance_type=args.tolerance_type,
        tolerance_value=args.tolerance_value,
        hard_block_lethal=args.hard_block_lethal,
        intervention_enable_after_steps=args.intervention_enable_after_steps,
        intervention_safety_margin_frac=args.intervention_safety_margin_frac,
        intervention_release_steps=args.intervention_release_steps,
        teacher_target_mode=args.teacher_target_mode,
        cube_success_tolerance=args.cube_success_tolerance,
    )]

    base_env = build_env(args, train_wrappers, seed=args.seed)
    eval_env_base = build_env(args, eval_wrappers, seed=args.seed + 1)
    train_env = wrap_env_for_drq(base_env, args)
    eval_env = wrap_env_for_drq(eval_env_base, args)

    obs_spec = train_env.observation_spec()
    action_spec = train_env.action_spec()
    obs_shape = obs_spec.shape
    action_shape = action_spec.shape
    if len(obs_shape) != 3:
        raise ValueError(f"DrQ-v2 trainer expects 3D pixel observations, got shape {obs_shape}")
    pixel_shape_tuple = tuple(int(x) for x in obs_shape)
    action_history_len = max(1, int(args.frame_stack))
    action_dim = int(np.prod(action_shape))
    prev_action_dim = action_dim * action_history_len
    goal_history_len = action_history_len if getattr(args, "goal_relative_history", False) else 0
    goal_vector_dim = 2
    goal_history_dim = goal_vector_dim * goal_history_len
    manip_env = is_manip_env(args.env_name)
    agent = DrQV2Agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        lr=args.learning_rate,
        feature_dim=args.drq_feature_dim,
        hidden_dim=args.drq_hidden_dim,
        critic_target_tau=args.critic_target_tau,
        num_expl_steps=args.num_expl_steps,
        update_every_steps=args.update_every_steps,
        stddev_schedule=args.stddev_schedule,
        stddev_clip=args.stddev_clip,
        use_tb=False,
        action_history_len=action_history_len,
        goal_history_dim=goal_history_dim,
        use_context_mlp=args.context_mlp,
        context_hidden_dim=args.context_hidden_dim,
        context_dim=args.context_dim,
    )

    run_paths = prepare_run_dirs(args)
    teacher_metrics = build_teacher_metrics(args, device)
    logger = DrQLogger(args=args, record_progress=run_paths.record_progress, teacher_metrics=teacher_metrics)
    checkpoint_mgr = DrQCheckpointManager(args=args, run_paths=run_paths, logger=logger)
    if args.use_wandb:
        try:
            logger._ensure_wandb_run()
            logger._log_to_wandb({"Run/startup": 1.0}, step=0)
        except Exception as exc:  # pragma: no cover - wandb is best-effort
            run_paths.record_progress(f"[WandB] init failed: {exc}")

    data_specs_list = [train_env.observation_spec()]
    if prev_action_dim > 0:
        data_specs_list.append(
            specs.Array((prev_action_dim,), np.float32, "prev_actions")
        )
    if goal_history_dim > 0:
        data_specs_list.append(
            specs.Array((goal_history_dim,), np.float32, "goal_history")
        )
    data_specs_list.extend([
        train_env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
    ])
    data_specs = tuple(data_specs_list)
    buffer_root = Path(args.buffer_root).expanduser() if args.buffer_root else run_paths.log_dir
    buffer_root.mkdir(parents=True, exist_ok=True)
    batch_h = max(1, args.batch_size // 2)
    batch_n = max(1, args.batch_size - batch_h)
    if args.replay_mode == "ram":
        replay_storage_novice = InMemoryReplayStorage(data_specs, max_size=args.replay_buffer_size)
        replay_storage_human = InMemoryReplayStorage(data_specs, max_size=args.replay_buffer_size)
        novice_loader_full = make_replay_loader_in_memory(
            replay_storage_novice,
            batch_size=args.batch_size,
            nstep=args.nstep,
            discount=args.discount,
        )
        novice_loader_half = make_replay_loader_in_memory(
            replay_storage_novice,
            batch_size=batch_n,
            nstep=args.nstep,
            discount=args.discount,
        )
        human_loader_half = make_replay_loader_in_memory(
            replay_storage_human,
            batch_size=batch_h,
            nstep=args.nstep,
            discount=args.discount,
        )
    else:
        replay_dir_novice = buffer_root / "buffer_novice"
        replay_dir_human = buffer_root / "buffer_human"
        replay_storage_novice = ReplayBufferStorage(data_specs, replay_dir_novice)
        replay_storage_human = ReplayBufferStorage(data_specs, replay_dir_human)
        novice_loader_full = make_replay_loader(
            replay_dir_novice,
            max_size=args.replay_buffer_size,
            batch_size=args.batch_size,
            num_workers=args.replay_buffer_num_workers,
            save_snapshot=args.save_snapshot,
            nstep=args.nstep,
            discount=args.discount,
            fetch_every=args.replay_fetch_every,
        )
        novice_loader_half = make_replay_loader(
            replay_dir_novice,
            max_size=args.replay_buffer_size,
            batch_size=batch_n,
            num_workers=args.replay_buffer_num_workers,
            save_snapshot=args.save_snapshot,
            nstep=args.nstep,
            discount=args.discount,
            fetch_every=args.replay_fetch_every,
        )
        human_loader_half = make_replay_loader(
            replay_dir_human,
            max_size=args.replay_buffer_size,
            batch_size=batch_h,
            num_workers=args.replay_buffer_num_workers,
            save_snapshot=args.save_snapshot,
            nstep=args.nstep,
            discount=args.discount,
            fetch_every=args.replay_fetch_every,
        )
    novice_iter_full = None
    novice_iter_half = None
    human_iter_half = None

    pref_storage = None
    pref_dataset = None
    if args.pref_buffer_enable:
        pref_dir = buffer_root / "pref_pairs"
        pref_storage = PreferencePairStorage(
            obs_shape=obs_shape,
            action_shape=action_shape,
            prev_action_shape=(prev_action_dim,),
            goal_history_shape=(goal_history_dim,) if goal_history_dim > 0 else None,
            storage_dir=pref_dir,
            chunk_size=args.pref_chunk_size,
            hidden_state_shape=None,
            warp_param_dim=0,
        )
        pref_dataset = PreferencePairDataset(
            storage_dir=pref_dir,
            max_size=args.pref_capacity,
            fetch_every=args.pref_fetch_every,
            prev_action_shape=(prev_action_dim,),
            goal_history_shape=(goal_history_dim,) if goal_history_dim > 0 else None,
            hidden_state_shape=None,
            warp_param_dim=0,
        )

    action_history = init_action_history(action_dim, action_history_len)
    action_transformer = ActionFrameTransformer(
        use_local=args.use_local_actions,
        translation_scale=args.se2_translation_scale,
    )
    action_transformer.reset()
    time_step = train_env.reset()
    goal_history = None
    if goal_history_len > 0:
        initial_goal = compute_goal_relative_local(
            train_env,
            action_transformer,
            scale=getattr(args, "goal_relative_scale", 10.0),
            fallback_dim=goal_vector_dim,
        )
        goal_history = init_goal_history(goal_vector_dim, goal_history_len, fill=initial_goal)
    time_step = time_step._replace(
        prev_actions=flatten_action_history(action_history),
        goal_history=flatten_goal_history(goal_history) if goal_history is not None else None,
    )
    replay_storage_novice.add(time_step)

    initial_ckpt = checkpoint_mgr.save(
        tag="step0",
        step_value=0,
        agent=agent,
        pixel_shape=pixel_shape_tuple,
        action_shape=action_shape,
    )
    checkpoint_mgr.maybe_render_policy_map(checkpoint_path=initial_ckpt, step_value=0, force=True)

    total_env_steps = 0
    episode_reward = 0.0
    episode_length = 0
    rewbuffer: list = []
    lenbuffer: list = []
    metrics_accum: Dict[str, float] = {}
    metrics_updates = 0
    perf_accum: Dict[str, float] = {
        "loop_time": 0.0,
        "env_step": 0.0,
        "update": 0.0,
        "sample": 0.0,
    }
    perf_steps = 0
    goal_success_counter = 0
    goal_final_distance_sum = 0.0
    goal_final_distance_count = 0

    log_alpha = 0.0

    while total_env_steps < args.total_timesteps:
        loop_start = time.perf_counter()
        last_denied_samples = 0
        if time_step.last():
            rewbuffer.append(episode_reward)
            lenbuffer.append(episode_length)
            action_history = init_action_history(action_dim, action_history_len)
            action_transformer.reset()
            time_step = train_env.reset()
            if goal_history_len > 0:
                initial_goal = compute_goal_relative_local(
                    train_env,
                    action_transformer,
                    scale=getattr(args, "goal_relative_scale", 10.0),
                    fallback_dim=goal_vector_dim,
                )
                goal_history = init_goal_history(goal_vector_dim, goal_history_len, fill=initial_goal)
            time_step = time_step._replace(
                prev_actions=flatten_action_history(action_history),
                goal_history=flatten_goal_history(goal_history) if goal_history is not None else None,
            )
            replay_storage_novice.add(time_step)
            episode_reward = 0.0
            episode_length = 0

        obs = time_step.observation
        prev_action_stack = time_step.prev_actions
        goal_history_stack = time_step.goal_history if goal_history_len > 0 else None
        with torch.no_grad(), utils.eval_mode(agent):
            policy_action_local = agent.act(
                obs,
                step=total_env_steps,
                eval_mode=False,
                prev_actions=prev_action_stack,
                goal_history=goal_history_stack,
            )
        if manip_env:
            policy_action_global = np.clip(np.asarray(policy_action_local, dtype=np.float32), -1.0, 1.0)
            policy_action_local_full = policy_action_global.copy()
            move_global = policy_action_global
            view_delta = 0.0
        else:
            move_local, view_delta = split_action_components(
                policy_action_local, bool(getattr(args, "decouple_view", False))
            )
            move_local = clip_action_l2(move_local)
            policy_action_local_full = (
                np.concatenate([move_local, np.array([view_delta], dtype=np.float32)], axis=0)
                if getattr(args, "decouple_view", False)
                else move_local
            )
            move_global = action_transformer.to_global(move_local)
            move_global = clip_action_l2(move_global)
            if getattr(args, "decouple_view", False):
                policy_action_global = np.concatenate(
                    [move_global, np.array([view_delta], dtype=np.float32)], axis=0
                )
            else:
                policy_action_global = move_global
        env_step_start = time.perf_counter()
        next_time_step = train_env.step(policy_action_global)
        env_step_end = time.perf_counter()
        info = getattr(train_env, "last_info", lambda: {})() or {}
        logger.accumulate_env_metrics(info)
        if float(info.get("success", 0.0)) > 0.0:
            goal_success_counter += 1

        teacher_intervened = bool(info.get("teacher_intervened"))
        teacher_action_global = info.get("teacher_action")
        teacher_action_local = None
        if teacher_action_global is not None:
            if manip_env:
                teacher_action_local = np.asarray(teacher_action_global, dtype=np.float32)
            else:
                teacher_action_local = action_transformer.to_local(
                    np.asarray(teacher_action_global, dtype=np.float32)
                ).astype(np.float32, copy=False)
        student_action_global = info.get("student_action", move_global)
        student_action_local = policy_action_local_full.copy()
        if teacher_intervened and teacher_action_global is not None:
            applied_action_global = np.asarray(teacher_action_global, dtype=np.float32)
        else:
            applied_action_global = np.asarray(student_action_global, dtype=np.float32)
        if manip_env:
            applied_action_local = applied_action_global.astype(np.float32, copy=False)
        else:
            applied_move_local = action_transformer.to_local(applied_action_global).astype(np.float32, copy=False)
            if getattr(args, "decouple_view", False):
                applied_action_local = np.concatenate(
                    [applied_move_local, np.array([view_delta], dtype=np.float32)], axis=0
                )
                if teacher_action_local is not None:
                    teacher_action_local = np.concatenate(
                        [teacher_action_local, np.array([view_delta], dtype=np.float32)], axis=0
                    )
            else:
                applied_action_local = applied_move_local
        action_for_logging = applied_action_local.copy()
        action_history.append(applied_action_local.copy())
        next_prev_stack = flatten_action_history(action_history)
        action_transformer.update_heading(info)
        if goal_history is not None:
            goal_rel_local = compute_goal_relative_local(
                train_env,
                action_transformer,
                scale=getattr(args, "goal_relative_scale", 10.0),
                fallback_dim=goal_vector_dim,
            )
            goal_history.append(goal_rel_local.copy())
        next_time_step = next_time_step._replace(
            action=applied_action_local,
            prev_actions=next_prev_stack,
            goal_history=flatten_goal_history(goal_history) if goal_history is not None else None,
        )
        reward_arr = np.asarray(next_time_step.reward, dtype=np.float32)
        if reward_arr.shape == ():
            reward_arr = reward_arr.reshape(1)
        stored_reward = reward_arr if args.pvp_use_reward_in_replay else np.zeros_like(reward_arr)
        next_time_step = next_time_step._replace(reward=stored_reward)
        if next_time_step.last():
            distance_val = info.get("distance_to_goal")
            if distance_val is not None:
                goal_final_distance_sum += float(distance_val)
                goal_final_distance_count += 1
        target_storage = replay_storage_human if teacher_intervened else replay_storage_novice
        target_storage.add(next_time_step)

        if teacher_intervened:
            last_denied_samples = 1
        if (
            pref_storage is not None
            and teacher_intervened
            and teacher_action_local is not None
            and student_action_local is not None
        ):
            try:
                pref_storage.add(
                    np.asarray(obs).copy(),
                    np.asarray(prev_action_stack, dtype=np.float32).copy(),
                    np.asarray(goal_history_stack, dtype=np.float32).copy() if goal_history_stack is not None else None,
                    teacher_action_local.copy(),
                    student_action_local.copy(),
                    hidden_state=None,
                    warp_params=None,
                )
            except Exception as exc:
                run_paths.record_progress(f"[PrefBuffer] append failed: {exc}")

        episode_reward += float(np.asarray(reward_arr).reshape(-1)[0])
        episode_length += 1
        total_env_steps += 1
        time_step = next_time_step

        if total_env_steps >= args.num_expl_steps:
            if replay_storage_novice.num_episodes() == 0:
                continue
            use_balanced = replay_storage_human.num_episodes() > 0
            pref_batch_np = None
            if (
                pref_dataset is not None
                and args.pref_rank_weight > 0.0
                and args.pref_sample_ratio > 0.0
            ):
                pref_sample_start = time.perf_counter()
                pref_batch_size = max(1, int(args.batch_size * args.pref_sample_ratio))
                pref_batch_np = pref_dataset.sample(pref_batch_size)
                perf_accum["sample"] += time.perf_counter() - pref_sample_start
            if use_balanced:
                if novice_iter_half is None:
                    novice_iter_half = iter(novice_loader_half)
                if human_iter_half is None:
                    human_iter_half = iter(human_loader_half)
                sample_start = time.perf_counter()
                try:
                    batch_n = next(novice_iter_half)
                except StopIteration:
                    novice_iter_half = iter(novice_loader_half)
                    batch_n = next(novice_iter_half)
                try:
                    batch_h = next(human_iter_half)
                except StopIteration:
                    human_iter_half = iter(human_loader_half)
                    batch_h = next(human_iter_half)
                combined_batch = concat_replay_batches(batch_n, batch_h)
                replay_iter = iter([combined_batch])
                perf_accum["sample"] += time.perf_counter() - sample_start
            else:
                if novice_iter_full is None:
                    novice_iter_full = iter(novice_loader_full)
                sample_start = time.perf_counter()
                try:
                    batch_n = next(novice_iter_full)
                except StopIteration:
                    novice_iter_full = iter(novice_loader_full)
                    batch_n = next(novice_iter_full)
                replay_iter = iter([batch_n])
                perf_accum["sample"] += time.perf_counter() - sample_start

            update_start = time.perf_counter()
            metrics = agent.update(
                replay_iter,
                total_env_steps,
                pref_batch=pref_batch_np,
                pref_weight=args.pref_rank_weight,
                pref_margin=args.pref_rank_margin,
                pref_loss_type=args.pref_loss_type,
            )
            perf_accum["update"] += time.perf_counter() - update_start
            if metrics:
                for key, value in metrics.items():
                    metrics_accum[key] = metrics_accum.get(key, 0.0) + float(value)
                metrics_updates += 1

        with torch.no_grad(), utils.eval_mode(agent):
            obs_tensor = ensure_tensor(obs, device)
            act_tensor = torch.as_tensor(action_for_logging, device=device).view(1, -1)
            repr_obs = agent.encoder(obs_tensor)
            if agent.context_input_dim > 0:
                prev_tensor = torch.as_tensor(prev_action_stack, device=device).view(1, -1)
                goal_tensor = None
                if goal_history_stack is not None:
                    goal_tensor = torch.as_tensor(goal_history_stack, device=device).view(1, -1)
                context_tensor = agent._build_context(prev_tensor, goal_tensor, obs_tensor.shape[0])
            else:
                context_tensor = None
            q1, q2 = agent.critic(repr_obs, context_tensor, act_tensor)
            disagreement = torch.abs(q1 - q2).view(-1)
            qmin = torch.min(q1, q2).view(-1)
        teacher_flag = torch.tensor(
            [1.0 if info.get("teacher_intervened") else 0.0],
            device=device,
            dtype=torch.float32,
        )
        teacher_metrics.update(disagreement, teacher_flag, 1.0 - teacher_flag, qmin)

        loop_end = time.perf_counter()
        perf_accum["loop_time"] += loop_end - loop_start
        perf_accum["env_step"] += env_step_end - env_step_start
        perf_steps += 1

        if args.save_interval and total_env_steps % args.save_interval == 0:
            ckpt = checkpoint_mgr.save(
                tag=f"step{total_env_steps}",
                step_value=total_env_steps,
                agent=agent,
                pixel_shape=pixel_shape_tuple,
                action_shape=action_shape,
            )
            checkpoint_mgr.maybe_render_policy_map(checkpoint_path=ckpt, step_value=total_env_steps)

        pref_buffer_size = pref_storage.num_pairs() if pref_storage is not None else -1
        perf_metrics = None
        if perf_steps > 0 and perf_accum["loop_time"] > 0:
            loop_avg = perf_accum["loop_time"] / perf_steps
            env_avg = perf_accum["env_step"] / perf_steps
            update_avg = perf_accum["update"] / max(1, perf_steps)
            sample_avg = perf_accum["sample"] / max(1, perf_steps)
            perf_metrics = {
                "Perf/step_ms": loop_avg * 1000.0,
                "Perf/env_step_ms": env_avg * 1000.0,
                "Perf/update_ms": update_avg * 1000.0,
                "Perf/sample_ms": sample_avg * 1000.0,
                "Perf/env_step_pct": (perf_accum["env_step"] / perf_accum["loop_time"]) * 100.0,
                "Perf/update_pct": (perf_accum["update"] / perf_accum["loop_time"]) * 100.0,
                "Perf/sample_pct": (perf_accum["sample"] / perf_accum["loop_time"]) * 100.0,
            }
            replay_stats = collect_replay_stats([novice_loader_full, novice_loader_half, human_loader_half])
            if replay_stats:
                perf_metrics.update(replay_stats)
            logged = logger.maybe_log(
                total_env_steps=total_env_steps,
                total_timesteps=args.total_timesteps,
                rewbuffer=rewbuffer,
                lenbuffer=lenbuffer,
                update_metrics=metrics_accum,
                update_count=metrics_updates,
                perf_metrics=perf_metrics,
                log_alpha=log_alpha,
                last_denied_samples=last_denied_samples,
                pref_buffer_size=pref_buffer_size,
                goal_successes=goal_success_counter,
                goal_distance_sum=goal_final_distance_sum,
                goal_distance_count=goal_final_distance_count,
            )
            if logged:
                if metrics_updates > 0:
                    metrics_accum = {}
                    metrics_updates = 0
                perf_accum = {"loop_time": 0.0, "env_step": 0.0, "update": 0.0, "sample": 0.0}
                perf_steps = 0
                goal_success_counter = 0
                goal_final_distance_sum = 0.0
                goal_final_distance_count = 0

            if args.eval_every_frames and total_env_steps % args.eval_every_frames == 0:
                eval_metrics = run_eval_metrics(
                    eval_env,
                    agent,
                    args.num_eval_episodes,
                    device,
                    action_dim,
                    action_history_len,
                    goal_history_len,
                    use_local_actions=args.use_local_actions,
                    translation_scale=args.se2_translation_scale,
                    goal_scale=getattr(args, "goal_relative_scale", 10.0),
                    manip_env=manip_env,
                )
                logger.log_eval(total_env_steps=total_env_steps, metrics=eval_metrics)

    final_ckpt = checkpoint_mgr.save(
        tag="final",
        step_value=total_env_steps,
        agent=agent,
        pixel_shape=pixel_shape_tuple,
        action_shape=action_shape,
    )
    checkpoint_mgr.maybe_render_policy_map(checkpoint_path=final_ckpt, step_value=total_env_steps)
    if pref_storage is not None:
        pref_storage.close()
    logger.finish()


if __name__ == "__main__":
    train()

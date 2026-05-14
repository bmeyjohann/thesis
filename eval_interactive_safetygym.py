#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import time
import sys
from pathlib import Path
from typing import Any, Dict
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from safetygym_utils.controllers import build_human_controller
from safetygym_utils.gamepad import (
    DEFAULT_SAFETY_GAMEPAD_CACHE_PATH,
    DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
    DEFAULT_SAFETY_GAMEPAD_PORT,
)
from safetygym_utils.env import clip_action_to_space, extract_goal_distance, extract_step_limit, make_safety_env, resolve_control_scheme, scale_action_np
from safetygym_utils.io import load_args_json, maybe_find_args_json_from_model
from safetygym_utils.metrics import EpisodeWindow, augment_rollout_summary, classify_outcome
from safetygym_utils.policy_viz import (
    _extract_bounds,
    _extract_overlay_specs,
    plot_episode_contact_sheet,
    plot_eval_episode_trajectory,
)
from safetygym_utils.rendering import build_external_viewer, resolve_env_render_mode, wants_external_viewer
from safetygym_utils.sac import SafetyActor
from safetygym_utils.wrappers import HumanInterventionWrapper, RewardModeWrapper, TerminateOnGoalWrapper

_FAST_SAC_PATH = Path(__file__).resolve().parent / "fasttd3" / "fast_sac"
if _FAST_SAC_PATH.exists():
    _fast_sac_path_str = str(_FAST_SAC_PATH)
    if _fast_sac_path_str not in sys.path:
        sys.path.insert(0, _fast_sac_path_str)

try:
    from fast_sac_utils import EmpiricalNormalization  # type: ignore
except Exception:  # pragma: no cover - optional dependency for minimal checkpoints only
    EmpiricalNormalization = None


class EvalTelemetryPanel:
    def __init__(self, *, width: int = 720, height: int = 420, draw_hz: float = 20.0):
        self.width = int(width)
        self.height = int(height)
        self.draw_hz = float(draw_hz)
        self._pygame = None
        self._screen = None
        self._font = None
        self._small_font = None
        self._clock = None
        self._last_draw_ts = 0.0
        self._history_len = 180
        self._reward_hist: deque[float] = deque(maxlen=self._history_len)
        self._dense_hist: deque[float] = deque(maxlen=self._history_len)
        self._sparse_hist: deque[float] = deque(maxlen=self._history_len)
        self._dist_hist: deque[float] = deque(maxlen=self._history_len)
        self._current: Dict[str, float] = {}
        self._action = np.zeros((2,), dtype=np.float32)
        self._control_help: list[str] = []

    @staticmethod
    def _has_graphical_display() -> bool:
        if os.name == "nt":
            return True
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            return True
        return False

    def _ensure(self) -> bool:
        if self._pygame is not None:
            return True
        if not self._has_graphical_display():
            return False
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        pygame.init()
        pygame.display.set_caption("SafetyGym Reward Telemetry")
        self._screen = pygame.display.set_mode((self.width, self.height))
        self._font = pygame.font.Font(None, 24)
        self._small_font = pygame.font.Font(None, 20)
        self._clock = pygame.time.Clock()
        self._pygame = pygame
        return True

    @staticmethod
    def _plot_rect(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
        return (int(x), int(y), int(w), int(h))

    @staticmethod
    def _draw_series(pygame, screen, rect, values, color, *, symmetric: bool = False) -> None:
        x, y, w, h = rect
        pygame.draw.rect(screen, (55, 55, 70), rect, width=1)
        vals = [float(v) for v in values]
        if len(vals) < 2:
            return
        if symmetric:
            vmax = max(1e-6, max(abs(v) for v in vals))
            vmin = -vmax
        else:
            finite = [v for v in vals if np.isfinite(v)]
            if not finite:
                return
            vmin = min(finite)
            vmax = max(finite)
            if abs(vmax - vmin) < 1e-6:
                vmax = vmin + 1e-6
        pts = []
        for idx, value in enumerate(vals):
            frac_x = idx / max(1, len(vals) - 1)
            frac_y = (float(value) - vmin) / max(1e-6, vmax - vmin)
            px = x + int(frac_x * (w - 1))
            py = y + h - 1 - int(frac_y * (h - 1))
            pts.append((px, py))
        if symmetric:
            zero_y = y + h - 1 - int((0.0 - vmin) / max(1e-6, vmax - vmin) * (h - 1))
            pygame.draw.line(screen, (70, 70, 90), (x, zero_y), (x + w, zero_y), width=1)
        pygame.draw.lines(screen, color, False, pts, width=2)

    def update(
        self,
        *,
        info: Dict[str, Any],
        reward: float,
        action: np.ndarray | None = None,
        control_help: list[str] | None = None,
    ) -> None:
        self._current = {
            "reward": float(reward),
            "dense": float(info.get("reward_dense_component", 0.0)),
            "sparse": float(info.get("reward_sparse_component", 0.0)),
            "distance": float(info.get("goal_distance", float("nan"))),
            "distance_prev": float(info.get("goal_distance_prev", float("nan"))),
            "goal_met": 1.0 if bool(info.get("goal_met", False)) else 0.0,
        }
        if action is not None:
            self._action = np.asarray(action, dtype=np.float32).reshape(-1)
        if control_help is not None:
            self._control_help = list(control_help)
        self._reward_hist.append(self._current["reward"])
        self._dense_hist.append(self._current["dense"])
        self._sparse_hist.append(self._current["sparse"])
        self._dist_hist.append(self._current["distance"])

    def draw(self) -> None:
        if not self._ensure():
            return
        now = time.perf_counter()
        if self.draw_hz > 0.0 and (now - self._last_draw_ts) < (1.0 / self.draw_hz):
            return
        self._last_draw_ts = now

        pygame = self._pygame
        screen = self._screen
        font = self._font
        small_font = self._small_font
        clock = self._clock
        assert pygame is not None and screen is not None and font is not None and small_font is not None and clock is not None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

        screen.fill((20, 20, 24))
        lines = [
            "Reward telemetry",
            (
                f"reward={self._current.get('reward', 0.0):+.4f} "
                f"dense={self._current.get('dense', 0.0):+.4f} "
                f"sparse={self._current.get('sparse', 0.0):+.1f}"
            ),
            (
                f"dist={self._current.get('distance', float('nan')):+.4f} "
                f"prev={self._current.get('distance_prev', float('nan')):+.4f} "
                f"goal_met={int(self._current.get('goal_met', 0.0) > 0.5)}"
            ),
            f"action={np.array2string(np.asarray(self._action), precision=2)}",
        ]
        y = 16
        for idx, text in enumerate(lines):
            surf = (font if idx == 0 else small_font).render(text, True, (230, 230, 230))
            screen.blit(surf, (16, y))
            y += 28

        help_y = 16
        for text in self._control_help[:5]:
            surf = small_font.render(text, True, (180, 180, 180))
            screen.blit(surf, (410, help_y))
            help_y += 22

        left = 16
        top = 110
        plot_w = self.width - 32
        plot_h = 90
        self._draw_series(
            pygame,
            screen,
            self._plot_rect(left, top, plot_w, plot_h),
            self._reward_hist,
            (94, 201, 255),
            symmetric=True,
        )
        screen.blit(small_font.render("Shaped reward", True, (200, 200, 200)), (left, top - 20))
        self._draw_series(
            pygame,
            screen,
            self._plot_rect(left, top + 120, plot_w, plot_h),
            self._dense_hist,
            (255, 196, 87),
            symmetric=True,
        )
        screen.blit(small_font.render("Dense component", True, (200, 200, 200)), (left, top + 100))
        self._draw_series(
            pygame,
            screen,
            self._plot_rect(left, top + 240, plot_w, plot_h),
            self._dist_hist,
            (120, 232, 146),
            symmetric=False,
        )
        screen.blit(small_font.render("Goal distance", True, (200, 200, 200)), (left, top + 220))
        pygame.display.flip()
        clock.tick(60)

    def close(self) -> None:
        pygame = self._pygame
        if pygame is not None:
            pygame.display.quit()
            pygame.quit()
        self._pygame = None
        self._screen = None
        self._font = None
        self._small_font = None
        self._clock = None


class EvalEpisodeControlPanel:
    def __init__(self, *, width: int = 420, height: int = 220, title: str = "SafetyGym Eval Controls"):
        self.width = int(width)
        self.height = int(height)
        self.title = str(title)
        self._initialized = False
        self._warned_no_display = False
        self._closed = False
        self._root = None
        self._label = None
        self._requests = {
            "prev": False,
            "next": False,
            "reset": False,
            "quit": False,
        }
        self._fps_delta = 0.0
        self._current_episode = 1
        self._num_episodes = 1
        self._fps = 30.0

    @staticmethod
    def _has_graphical_display() -> bool:
        if os.name == "nt":
            return True
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            return True
        return False

    def _on_key(self, event) -> None:
        key = str(getattr(event, "keysym", "")).strip().lower()
        if key in {"escape", "q"}:
            self._requests["quit"] = True
        elif key == "left":
            self._requests["prev"] = True
        elif key == "right":
            self._requests["next"] = True
        elif key in {"r", "backspace"}:
            self._requests["reset"] = True
        elif key == "up":
            self._fps_delta += 2.0
        elif key == "down":
            self._fps_delta -= 2.0
        self._render_text()

    def _on_close(self) -> None:
        self._closed = True
        root = self._root
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        self._root = None
        self._label = None

    def _ensure(self) -> bool:
        if self._initialized:
            return self._root is not None
        if not self._has_graphical_display():
            if not self._warned_no_display:
                print("eval controls disabled: no graphical display detected", flush=True)
                self._warned_no_display = True
            self._initialized = True
            return False
        try:
            import tkinter as tk
        except Exception as exc:
            print(f"eval controls disabled: tkinter unavailable ({exc})", flush=True)
            self._initialized = True
            return False

        root = tk.Tk()
        root.title(self.title)
        root.geometry(f"{self.width}x{self.height}")
        root.resizable(False, False)
        root.configure(bg="#141418")
        root.bind("<KeyPress>", self._on_key)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        label = tk.Label(
            root,
            text="",
            justify="left",
            anchor="nw",
            bg="#141418",
            fg="#e0e0e0",
            font=("TkDefaultFont", 11),
        )
        label.pack(fill="both", expand=True, padx=12, pady=12)
        self._root = root
        self._label = label
        try:
            root.focus_force()
        except Exception:
            pass
        self._initialized = True
        self._render_text()
        return True

    def _render_text(self) -> None:
        label = self._label
        if label is None:
            return
        lines = [
            "Eval episode controls",
            "",
            f"Episode: {int(self._current_episode)}/{int(max(1, self._num_episodes))}",
            f"FPS: {float(self._fps):.1f}",
            "",
            "Right: next episode",
            "Left: previous episode",
            "R / Backspace: reset current episode",
            "Up / Down: FPS +/- 2",
            "Esc / Q: quit",
        ]
        try:
            label.config(text="\n".join(lines))
        except Exception:
            pass

    def set_status(self, *, episode_idx: int, num_episodes: int, fps: float) -> None:
        self._current_episode = int(max(1, episode_idx))
        self._num_episodes = int(max(1, num_episodes))
        self._fps = float(fps)
        self._render_text()

    def poll(self) -> tuple[bool, bool, bool, bool, float]:
        if not self._ensure():
            return False, False, False, False, 0.0
        root = self._root
        if root is None or self._closed:
            return False, False, False, False, 0.0
        try:
            root.update_idletasks()
            root.update()
        except Exception:
            self._closed = True
            self._root = None
            self._label = None
            return False, False, True, False, 0.0
        prev_requested = bool(self._requests["prev"])
        next_requested = bool(self._requests["next"])
        reset_requested = bool(self._requests["reset"])
        quit_requested = bool(self._requests["quit"])
        fps_delta = float(self._fps_delta)
        self._requests = {k: False for k in self._requests}
        self._fps_delta = 0.0
        return prev_requested, next_requested, quit_requested, reset_requested, fps_delta

    def close(self) -> None:
        self._on_close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive evaluation for Safety-Gymnasium FastSAC (.pt) and SB3 PPO (.zip) checkpoints")
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--controller", type=str, default="policy", choices=["policy", "random", "human", "keyboard", "gamepad", "scripted"])
    p.add_argument("--policy_format", type=str, default="auto", choices=["auto", "fastsac", "ppo"])
    p.add_argument("--intervention_mode", type=str, default="none", choices=["none", "human"])
    p.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array", "none", "pygame", "topdown"])
    p.add_argument("--viewer_fps", type=float, default=20.0)
    p.add_argument("--viewer_scale", type=float, default=1.0)
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=2.0)
    p.add_argument("--car_force_scale", type=float, default=2.0)
    p.add_argument("--car_action_mode", type=str, default="raw_wheels", choices=["raw_wheels", "throttle_turn", "cardinal"])
    p.add_argument("--point_action_mode", type=str, default="native", choices=["native", "world_velocity"])
    p.add_argument("--point_turn_gain", type=float, default=2.5)
    p.add_argument("--point_alignment_power", type=float, default=1.0)
    p.add_argument("--point_allow_backward", action="store_true", default=False)
    p.add_argument("--obs_mask_mode", type=str, default="none", choices=["none", "goal_only_lidar"])
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--fps", type=int, default=30)

    p.add_argument(
        "--reward_mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "dense_plus_sparse", "potential_diff", "dual", "native", "none"],
    )
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--success_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--cost_penalty", type=float, default=0.0)
    p.add_argument("--cost_penalty_warmup_steps", type=int, default=0)
    p.add_argument("--cost_penalty_ramp_steps", type=int, default=0)
    p.add_argument("--clearance_penalty_scale", type=float, default=0.0)
    p.add_argument("--clearance_margin", type=float, default=0.0)
    p.add_argument("--clearance_penalty_power", type=float, default=1.0)
    p.add_argument("--clearance_penalty_mode", type=str, default="hinge_power", choices=["hinge_power", "softplus"])
    p.add_argument("--clearance_penalty_temperature", type=float, default=0.08)
    p.add_argument("--clearance_penalty_warmup_steps", type=int, default=0)
    p.add_argument("--clearance_penalty_ramp_steps", type=int, default=0)
    p.add_argument("--forward_reward_scale", type=float, default=0.0)
    p.add_argument("--backward_penalty_scale", type=float, default=0.0)
    p.add_argument("--heading_reward_scale", type=float, default=0.0)
    p.add_argument("--heading_positive_only", action="store_true", default=True)
    p.add_argument("--no_heading_positive_only", dest="heading_positive_only", action="store_false")
    p.add_argument("--terminate_on_goal", action="store_true", default=False)
    p.add_argument("--no_terminate_on_goal", dest="terminate_on_goal", action="store_false")

    p.add_argument("--intervention_threshold", type=float, default=0.1)
    p.add_argument("--intervention_hold_seconds", type=float, default=0.25)
    p.add_argument("--human_action_scale", type=float, default=1.0)
    p.add_argument("--human_input_device", type=str, default="keyboard", choices=["keyboard", "gamepad", "scripted"])
    p.add_argument("--controller_fps_limit", type=int, default=0)
    p.add_argument("--controller_overlay_hz", type=float, default=20.0)
    p.add_argument("--gamepad_config_path", type=str, default=str(DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH))
    p.add_argument("--gamepad_mode", type=str, default="local", choices=["local", "connect"])
    p.add_argument("--gamepad_host", type=str, default="")
    p.add_argument("--gamepad_port", type=int, default=0)
    p.add_argument("--gamepad_cache_path", type=str, default=str(DEFAULT_SAFETY_GAMEPAD_CACHE_PATH))
    p.add_argument("--gamepad_reconnect_seconds", type=float, default=2.0)
    p.add_argument("--gamepad_use_saved_config", action="store_true", default=True)
    p.add_argument("--no_gamepad_use_saved_config", dest="gamepad_use_saved_config", action="store_false")
    p.add_argument("--gamepad_device_index", type=int, default=0)

    p.add_argument("--actor_hidden_dim", type=int, default=256)
    p.add_argument("--use_layer_norm", action="store_true", default=False)
    p.add_argument("--layer_norm_eps", type=float, default=1e-5)
    p.add_argument("--init_scale", type=float, default=0.01)
    p.add_argument("--scale_actor_to_env_bounds", action="store_true", default=False)
    p.add_argument("--no_scale_actor_to_env_bounds", dest="scale_actor_to_env_bounds", action="store_false")
    p.add_argument("--load_checkpoint_args", action="store_true", default=True)
    p.add_argument("--no_load_checkpoint_args", dest="load_checkpoint_args", action="store_false")
    p.add_argument("--show_telemetry_overlay", action="store_true", default=False)
    p.add_argument("--telemetry_overlay_hz", type=float, default=20.0)
    p.add_argument("--show_episode_controls", action="store_true", default=True)
    p.add_argument("--no_show_episode_controls", dest="show_episode_controls", action="store_false")
    p.add_argument("--save_episode_plots", action="store_true", default=False)
    p.add_argument("--episode_plot_dir", type=str, default="")
    p.add_argument("--episode_plot_max_episodes", type=int, default=9)
    return p


def _apply_ckpt_defaults(args: argparse.Namespace) -> None:
    if not args.model_path or not args.load_checkpoint_args:
        return
    args_path = maybe_find_args_json_from_model(Path(args.model_path))
    if args_path is None or not args_path.exists():
        return
    cfg = load_args_json(args_path)
    cli_args = set(sys.argv[1:])

    def _set_if_default(name: str, default_val):
        if getattr(args, name) == default_val and name in cfg:
            setattr(args, name, cfg[name])

    def _bool_flag_explicit(*flags: str) -> bool:
        return any(flag in cli_args for flag in flags)

    _set_if_default("env_name", "SafetyCarGoal2-v0")
    _set_if_default("reward_mode", "sparse")
    _set_if_default("dense_reward_scale", 1.0)
    _set_if_default("success_reward_scale", 1.0)
    _set_if_default("step_penalty", 0.0)
    _set_if_default("cost_penalty", 0.0)
    _set_if_default("cost_penalty_warmup_steps", 0)
    _set_if_default("cost_penalty_ramp_steps", 0)
    _set_if_default("clearance_penalty_scale", 0.0)
    _set_if_default("clearance_margin", 0.0)
    _set_if_default("clearance_penalty_power", 1.0)
    _set_if_default("clearance_penalty_mode", "hinge_power")
    _set_if_default("clearance_penalty_temperature", 0.08)
    _set_if_default("clearance_penalty_warmup_steps", 0)
    _set_if_default("clearance_penalty_ramp_steps", 0)
    _set_if_default("forward_reward_scale", 0.0)
    _set_if_default("backward_penalty_scale", 0.0)
    _set_if_default("heading_reward_scale", 0.0)
    if not _bool_flag_explicit("--heading_positive_only", "--no_heading_positive_only"):
        _set_if_default("heading_positive_only", True)
    if not _bool_flag_explicit("--terminate_on_goal", "--no_terminate_on_goal"):
        _set_if_default("terminate_on_goal", False)
    _set_if_default("surface_mode", "default")
    _set_if_default("car_wheel_command_limit", 2.0)
    _set_if_default("car_force_scale", 2.0)
    _set_if_default("car_action_mode", "raw_wheels")
    _set_if_default("point_action_mode", "native")
    _set_if_default("point_turn_gain", 2.5)
    _set_if_default("point_alignment_power", 1.0)
    _set_if_default("point_allow_backward", False)
    _set_if_default("obs_mask_mode", "none")
    _set_if_default("actor_hidden_dim", 256)
    _set_if_default("use_layer_norm", False)
    _set_if_default("layer_norm_eps", 1e-5)
    _set_if_default("init_scale", 0.01)
    if not _bool_flag_explicit("--scale_actor_to_env_bounds", "--no_scale_actor_to_env_bounds"):
        _set_if_default("scale_actor_to_env_bounds", False)


def _resolve_policy_format(model_path: str, requested: str) -> str:
    requested = str(requested or "auto").lower()
    if requested in {"fastsac", "ppo"}:
        return requested
    suffix = Path(str(model_path)).suffix.lower()
    if suffix == ".zip":
        return "ppo"
    return "fastsac"


def _load_ppo_vecnormalize(model_path: Path, args: argparse.Namespace):
    vecnorm_path = model_path.parent / "vecnormalize.pkl"
    stem = model_path.stem
    if stem.startswith("ppo_step_") and stem.endswith("_steps"):
        step_text = stem[len("ppo_step_") :]
        candidate = model_path.parent / f"ppo_step_vecnormalize_{step_text}.pkl"
        if candidate.exists():
            vecnorm_path = candidate
    if not vecnorm_path.exists():
        return None

    # VecNormalize.load requires a VecEnv, but we only need the saved obs_rms for prediction.
    tmp_args = argparse.Namespace(**vars(args))
    tmp_args.render_mode = "none"
    tmp_env = DummyVecEnv([lambda: _build_env(tmp_args, controller=None)])
    vecnorm = VecNormalize.load(str(vecnorm_path), tmp_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


def _build_env(args: argparse.Namespace, controller):
    env = make_safety_env(
        args.env_name,
        render_mode=resolve_env_render_mode(args.render_mode),
        max_episode_steps=args.max_episode_steps,
        surface_mode=args.surface_mode,
        car_wheel_command_limit=args.car_wheel_command_limit,
        car_force_scale=args.car_force_scale,
        car_action_mode=args.car_action_mode,
        point_action_mode=str(getattr(args, "point_action_mode", "native")),
        point_turn_gain=float(getattr(args, "point_turn_gain", 2.5)),
        point_alignment_power=float(getattr(args, "point_alignment_power", 1.0)),
        point_allow_backward=bool(getattr(args, "point_allow_backward", False)),
        obs_mask_mode=str(getattr(args, "obs_mask_mode", "none")),
        seed=args.seed,
    )
    env = RewardModeWrapper(
        env,
        reward_mode=args.reward_mode,
        dense_reward_scale=args.dense_reward_scale,
        success_reward_scale=float(getattr(args, "success_reward_scale", 1.0)),
        step_penalty=args.step_penalty,
        cost_penalty=args.cost_penalty,
        cost_penalty_warmup_steps=int(getattr(args, "cost_penalty_warmup_steps", 0)),
        cost_penalty_ramp_steps=int(getattr(args, "cost_penalty_ramp_steps", 0)),
        clearance_penalty_scale=float(getattr(args, "clearance_penalty_scale", 0.0)),
        clearance_margin=float(getattr(args, "clearance_margin", 0.0)),
        clearance_penalty_power=float(getattr(args, "clearance_penalty_power", 1.0)),
        clearance_penalty_mode=str(getattr(args, "clearance_penalty_mode", "hinge_power")),
        clearance_penalty_temperature=float(getattr(args, "clearance_penalty_temperature", 0.08)),
        clearance_penalty_warmup_steps=int(getattr(args, "clearance_penalty_warmup_steps", 0)),
        clearance_penalty_ramp_steps=int(getattr(args, "clearance_penalty_ramp_steps", 0)),
        forward_reward_scale=float(getattr(args, "forward_reward_scale", 0.0)),
        backward_penalty_scale=float(getattr(args, "backward_penalty_scale", 0.0)),
        heading_reward_scale=float(getattr(args, "heading_reward_scale", 0.0)),
        heading_positive_only=bool(getattr(args, "heading_positive_only", True)),
    )
    if bool(getattr(args, "terminate_on_goal", False)):
        env = TerminateOnGoalWrapper(env)
    if args.intervention_mode == "human":
        if controller is None:
            raise ValueError("controller is required for intervention_mode=human")
        env = HumanInterventionWrapper(
            env,
            controller=controller,
            threshold=args.intervention_threshold,
            hold_seconds=args.intervention_hold_seconds,
        )
    return env


def _episode_metrics(
    info: Dict[str, Any],
    ep_return: float,
    ep_cost: float,
    ep_reward_raw_env: float,
    ep_reward_dense: float,
    ep_reward_sparse: float,
    ep_reward_step_penalty: float,
    ep_reward_cost_penalty: float,
    ep_len: int,
    max_steps: int,
    terminated: bool,
    truncated: bool,
    final_distance: float,
    goal_met_any: bool,
    goal_met_count: int,
    first_goal_hit_step: int | None,
    first_goal_reward_sum: float,
    first_goal_dense_reward_sum: float,
) -> Dict[str, float]:
    goal_met = bool(goal_met_any)
    outcome = classify_outcome(goal_met=goal_met, episode_steps=ep_len, max_episode_steps=max_steps)
    first_hit = int(first_goal_hit_step) if first_goal_hit_step is not None else int(max_steps)
    return {
        "episode_return": float(ep_return),
        "episode_cost_sum": float(ep_cost),
        "episode_cost_rate": float(ep_cost / max(1, ep_len)),
        "episode_length": float(ep_len),
        "reward_shaped_sum": float(ep_return),
        "reward_raw_env_sum": float(ep_reward_raw_env),
        "reward_dense_sum": float(ep_reward_dense),
        "reward_sparse_sum": float(ep_reward_sparse),
        "reward_step_penalty_sum": float(ep_reward_step_penalty),
        "reward_cost_penalty_sum": float(ep_reward_cost_penalty),
        "intervention_steps": float(info.get("teacher_intervention_steps", 0.0)),
        "intervention_fraction": float(info.get("teacher_fraction_steps", 0.0)),
        "intervention_num_bursts": float(info.get("teacher_num_bursts", 0.0)),
        "intervention_avg_burst_len": float(info.get("teacher_avg_burst_len", 0.0)),
        "goal_met": 1.0 if goal_met else 0.0,
        "goal_met_count": float(max(0, int(goal_met_count))),
        "first_goal_success": 1.0 if first_goal_hit_step is not None else 0.0,
        "first_goal_hit_step": float(first_hit),
        "first_goal_hit_step_success_only": float(first_hit if first_goal_hit_step is not None else 0.0),
        "first_goal_within_100": 1.0 if first_goal_hit_step is not None and first_hit <= 100 else 0.0,
        "first_goal_within_200": 1.0 if first_goal_hit_step is not None and first_hit <= 200 else 0.0,
        "first_goal_reward_sum": float(first_goal_reward_sum),
        "first_goal_dense_reward_sum": float(first_goal_dense_reward_sum),
        "final_distance_to_goal": float(final_distance),
        "outcome_success": 1.0 if outcome == "success" else 0.0,
        "outcome_timeout": 1.0 if outcome == "timeout" else 0.0,
        "outcome_kill": 1.0 if outcome == "kill" else 0.0,
        "outcome_other_failure": 1.0 if outcome not in {"success", "timeout", "kill"} else 0.0,
        "terminated": 1.0 if terminated else 0.0,
        "truncated": 1.0 if truncated else 0.0,
    }


def main() -> int:
    args = build_parser().parse_args()
    _apply_ckpt_defaults(args)

    if args.controller in {"human", "keyboard", "gamepad", "scripted"} and args.intervention_mode == "human":
        # Avoid double-human override (controller action + intervention wrapper action).
        args.intervention_mode = "none"
    if args.controller == "keyboard":
        args.human_input_device = "keyboard"
    elif args.controller == "gamepad":
        args.human_input_device = "gamepad"
    elif args.controller == "scripted":
        args.human_input_device = "scripted"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    controller = None
    control_help: list[str] = []
    if args.controller in {"human", "keyboard", "gamepad", "scripted"} or args.intervention_mode == "human":
        # Temporary env for action-dim detection.
        tmp_env = make_safety_env(
            args.env_name,
            render_mode="none",
            max_episode_steps=args.max_episode_steps,
            surface_mode=args.surface_mode,
            car_wheel_command_limit=args.car_wheel_command_limit,
            car_force_scale=args.car_force_scale,
            car_action_mode=args.car_action_mode,
            point_action_mode=str(getattr(args, "point_action_mode", "native")),
            point_turn_gain=float(getattr(args, "point_turn_gain", 2.5)),
            point_alignment_power=float(getattr(args, "point_alignment_power", 1.0)),
            point_allow_backward=bool(getattr(args, "point_allow_backward", False)),
            obs_mask_mode=str(getattr(args, "obs_mask_mode", "none")),
            seed=args.seed,
        )
        act_dim = int(np.prod(tmp_env.action_space.shape))
        obs_dim_tmp = int(np.prod(tmp_env.observation_space.shape))
        action_low_tmp = np.asarray(tmp_env.action_space.low, dtype=np.float32)
        action_high_tmp = np.asarray(tmp_env.action_space.high, dtype=np.float32)
        tmp_env.close()
        controller = build_human_controller(
            input_device=str(getattr(args, "human_input_device", "keyboard")),
            action_dim=act_dim,
            obs_dim=obs_dim_tmp,
            env_name=args.env_name,
            action_scale=args.human_action_scale,
            wheel_command_limit=float(args.car_wheel_command_limit),
            overlay_fps_limit=int(args.controller_fps_limit),
            overlay_draw_hz=float(args.controller_overlay_hz),
            gamepad_mode=str(getattr(args, "gamepad_mode", "local")),
            gamepad_host=str(getattr(args, "gamepad_host", "")),
            gamepad_port=int(getattr(args, "gamepad_port", 0) or DEFAULT_SAFETY_GAMEPAD_PORT),
            gamepad_cache_path=getattr(args, "gamepad_cache_path", DEFAULT_SAFETY_GAMEPAD_CACHE_PATH),
            gamepad_reconnect_seconds=float(getattr(args, "gamepad_reconnect_seconds", 2.0)),
            gamepad_config_path=args.gamepad_config_path,
            gamepad_use_saved_config=bool(getattr(args, "gamepad_use_saved_config", True)),
            gamepad_device_index=int(getattr(args, "gamepad_device_index", 0)),
            action_low=action_low_tmp,
            action_high=action_high_tmp,
            show_overlay=not bool(args.show_telemetry_overlay),
            prefer_separate_keyboard_window=wants_external_viewer(getattr(args, "render_mode", "human")),
            control_scheme_override=resolve_control_scheme(
                str(args.env_name),
                car_action_mode=str(getattr(args, "car_action_mode", "raw_wheels")),
                point_action_mode=str(getattr(args, "point_action_mode", "native")),
            ),
        )
        if str(getattr(args, "human_input_device", "keyboard")).lower() == "keyboard":
            control_help = [
                "Keyboard controls",
                "W/S: forward/backward",
                "A/D: turn left/right",
            ]

    env = _build_env(args, controller)
    viewer = build_external_viewer(
        render_mode=args.render_mode,
        title=f"SafetyGym Eval {args.env_name}",
        draw_hz=float(args.viewer_fps),
        scale=float(args.viewer_scale),
    ) if wants_external_viewer(args.render_mode) else None
    max_steps = extract_step_limit(env)
    telemetry_panel = EvalTelemetryPanel(draw_hz=float(args.telemetry_overlay_hz)) if bool(args.show_telemetry_overlay) else None
    control_panel = (
        EvalEpisodeControlPanel()
        if bool(getattr(args, "show_episode_controls", True)) and str(args.render_mode).lower() != "none"
        else None
    )

    obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if viewer is not None:
        viewer.draw_env(env)
    task = env.unwrapped.task
    overlay_specs = _extract_overlay_specs(task)
    bounds = _extract_bounds(task, x_range=None, y_range=None)
    obs_dim = int(obs.shape[0])
    act_dim = int(np.prod(env.action_space.shape))

    actor = None
    obs_preprocess = None
    ppo_model = None
    ppo_vecnorm = None
    policy_format = "fastsac"
    if args.controller == "policy":
        if not args.model_path:
            raise ValueError("--model_path is required for --controller policy")
        policy_format = _resolve_policy_format(args.model_path, getattr(args, "policy_format", "auto"))
        if policy_format == "ppo":
            model_path = Path(args.model_path).expanduser().resolve()
            ppo_model = PPO.load(str(model_path), device=device)
            ppo_vecnorm = _load_ppo_vecnormalize(model_path, args)
            print(f"loaded SB3 PPO policy: {model_path}", flush=True)
            if ppo_vecnorm is not None:
                print("loaded PPO VecNormalize stats", flush=True)
        else:
            actor = SafetyActor(
                n_obs=obs_dim,
                n_act=act_dim,
                num_envs=1,
                init_scale=args.init_scale,
                hidden_dim=args.actor_hidden_dim,
                use_layer_norm=bool(getattr(args, "use_layer_norm", False)),
                layer_norm_eps=float(getattr(args, "layer_norm_eps", 1e-5)),
                device=device,
            )
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
            actor.load_state_dict(checkpoint["actor_state_dict"])
            actor.eval()
            obs_norm_state = checkpoint.get("obs_normalizer_state_dict", None)
            if obs_norm_state and EmpiricalNormalization is not None:
                obs_preprocess = EmpiricalNormalization(shape=obs_dim, device=device)
                obs_preprocess.load_state_dict(obs_norm_state, strict=False)
                obs_preprocess.eval()
            else:
                obs_preprocess = torch.nn.Identity()

    win = EpisodeWindow(size=max(10, args.num_episodes))

    ep_ret = 0.0
    ep_cost = 0.0
    ep_reward_raw_env = 0.0
    ep_reward_dense = 0.0
    ep_reward_sparse = 0.0
    ep_reward_step_penalty = 0.0
    ep_reward_cost_penalty = 0.0
    ep_len = 0
    ep_goal_met_any = False
    ep_goal_met_count = 0
    ep_first_goal_hit_step: int | None = None
    ep_first_goal_reward_sum = 0.0
    ep_first_goal_dense_reward_sum = 0.0
    ep_path = [np.asarray(task.agent.pos[:2], dtype=np.float64).copy()]
    ep_goal_positions = [np.asarray(task.goal.pos[:2], dtype=np.float64).copy()]
    ep_goal_hit_points: list[np.ndarray] = []
    episodes = 0
    episode_idx = 0
    eval_fps = float(args.fps)
    saved_episode_plot_paths: list[Path] = []

    while episodes < args.num_episodes:
        if control_panel is not None:
            control_panel.set_status(episode_idx=episode_idx + 1, num_episodes=args.num_episodes, fps=eval_fps)
            prev_requested, next_requested, quit_requested, reset_requested, fps_delta = control_panel.poll()
            if abs(float(fps_delta)) > 0.0:
                eval_fps = max(0.0, float(eval_fps + fps_delta))
            if quit_requested:
                break
            if prev_requested:
                episode_idx = max(0, int(episode_idx - 1))
                obs, _ = env.reset(seed=args.seed + episode_idx)
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                if viewer is not None:
                    viewer.draw_env(env)
                ep_ret = 0.0
                ep_cost = 0.0
                ep_reward_raw_env = 0.0
                ep_reward_dense = 0.0
                ep_reward_sparse = 0.0
                ep_reward_step_penalty = 0.0
                ep_reward_cost_penalty = 0.0
                ep_len = 0
                ep_goal_met_any = False
                ep_goal_met_count = 0
                ep_first_goal_hit_step = None
                ep_first_goal_reward_sum = 0.0
                ep_first_goal_dense_reward_sum = 0.0
                ep_path = [np.asarray(task.agent.pos[:2], dtype=np.float64).copy()]
                ep_goal_positions = [np.asarray(task.goal.pos[:2], dtype=np.float64).copy()]
                ep_goal_hit_points = []
                continue
            if next_requested:
                episode_idx = min(max(0, int(args.num_episodes) - 1), int(episode_idx + 1))
                obs, _ = env.reset(seed=args.seed + episode_idx)
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                if viewer is not None:
                    viewer.draw_env(env)
                ep_ret = 0.0
                ep_cost = 0.0
                ep_reward_raw_env = 0.0
                ep_reward_dense = 0.0
                ep_reward_sparse = 0.0
                ep_reward_step_penalty = 0.0
                ep_reward_cost_penalty = 0.0
                ep_len = 0
                ep_goal_met_any = False
                ep_goal_met_count = 0
                ep_first_goal_hit_step = None
                ep_first_goal_reward_sum = 0.0
                ep_first_goal_dense_reward_sum = 0.0
                ep_path = [np.asarray(task.agent.pos[:2], dtype=np.float64).copy()]
                ep_goal_positions = [np.asarray(task.goal.pos[:2], dtype=np.float64).copy()]
                ep_goal_hit_points = []
                continue
            if reset_requested:
                obs, _ = env.reset(seed=args.seed + episode_idx)
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                if viewer is not None:
                    viewer.draw_env(env)
                ep_ret = 0.0
                ep_cost = 0.0
                ep_reward_raw_env = 0.0
                ep_reward_dense = 0.0
                ep_reward_sparse = 0.0
                ep_reward_step_penalty = 0.0
                ep_reward_cost_penalty = 0.0
                ep_len = 0
                ep_goal_met_any = False
                ep_goal_met_count = 0
                ep_first_goal_hit_step = None
                ep_first_goal_reward_sum = 0.0
                ep_first_goal_dense_reward_sum = 0.0
                ep_path = [np.asarray(task.agent.pos[:2], dtype=np.float64).copy()]
                ep_goal_positions = [np.asarray(task.goal.pos[:2], dtype=np.float64).copy()]
                ep_goal_hit_points = []
                continue

        if args.controller == "policy":
            if ppo_model is not None:
                ppo_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
                if ppo_vecnorm is not None:
                    ppo_obs = ppo_vecnorm.normalize_obs(ppo_obs)
                action, _ = ppo_model.predict(ppo_obs, deterministic=True)
                action = np.asarray(action, dtype=np.float32).reshape(-1)
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
                    if obs_preprocess is not None:
                        obs_t = obs_preprocess(obs_t)
                    _, _, mean = actor(obs_t)
                    action = mean[0].detach().cpu().numpy().astype(np.float32)
                if bool(getattr(args, "scale_actor_to_env_bounds", False)):
                    action = scale_action_np(action, env.action_space)
        elif args.controller == "random":
            action = env.action_space.sample().astype(np.float32)
        else:
            if controller is None:
                raise RuntimeError("Controller requested but not initialized")
            try:
                action = controller.get_action(obs=obs, env=env)
            except TypeError:
                try:
                    action = controller.get_action(obs=obs)
                except TypeError:
                    action = controller.get_action()
            if action is None:
                raise RuntimeError("Controller returned no action")
            action = np.asarray(action, dtype=np.float32)

        action = clip_action_to_space(action, env.action_space)
        next_obs, reward, cost, terminated, truncated, info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        if viewer is not None:
            viewer.draw_env(env)
        if telemetry_panel is not None:
            telemetry_panel.update(
                info=dict(info),
                reward=float(reward),
                action=action,
                control_help=control_help,
            )
            telemetry_panel.draw()

        ep_ret += float(reward)
        ep_cost += float(cost)
        ep_reward_raw_env += float(info.get("reward_raw_env", 0.0))
        ep_reward_dense += float(info.get("reward_dense_component", 0.0))
        ep_reward_sparse += float(info.get("reward_sparse_component", 0.0))
        ep_reward_step_penalty += float(info.get("reward_step_penalty_component", 0.0))
        ep_reward_cost_penalty += float(info.get("reward_cost_penalty_component", 0.0))
        ep_len += 1
        ep_path.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
        if bool(info.get("goal_met", False)):
            ep_goal_met_any = True
            ep_goal_met_count += 1
            ep_goal_hit_points.append(np.asarray(task.agent.pos[:2], dtype=np.float64).copy())
            if ep_first_goal_hit_step is None:
                ep_first_goal_hit_step = int(ep_len)
                ep_first_goal_reward_sum = float(ep_ret)
                ep_first_goal_dense_reward_sum = float(ep_reward_dense)
        current_goal_xy = np.asarray(task.goal.pos[:2], dtype=np.float64).copy()
        if np.linalg.norm(current_goal_xy - np.asarray(ep_goal_positions[-1], dtype=np.float64)) > 1e-6:
            ep_goal_positions.append(current_goal_xy)

        if terminated or truncated:
            final_dist = extract_goal_distance(env)
            ep = _episode_metrics(
                dict(info),
                ep_return=ep_ret,
                ep_cost=ep_cost,
                ep_reward_raw_env=ep_reward_raw_env,
                ep_reward_dense=ep_reward_dense,
                ep_reward_sparse=ep_reward_sparse,
                ep_reward_step_penalty=ep_reward_step_penalty,
                ep_reward_cost_penalty=ep_reward_cost_penalty,
                ep_len=ep_len,
                max_steps=max_steps,
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_distance=final_dist,
                goal_met_any=ep_goal_met_any,
                goal_met_count=ep_goal_met_count,
                first_goal_hit_step=ep_first_goal_hit_step,
                first_goal_reward_sum=(
                    ep_first_goal_reward_sum if ep_first_goal_hit_step is not None else float(ep_ret)
                ),
                first_goal_dense_reward_sum=(
                    ep_first_goal_dense_reward_sum if ep_first_goal_hit_step is not None else float(ep_reward_dense)
                ),
            )
            win.add(ep)
            if bool(getattr(args, "save_episode_plots", False)):
                plot_dir = (
                    Path(str(args.episode_plot_dir)).expanduser()
                    if str(getattr(args, "episode_plot_dir", "")).strip()
                    else Path("logs") / "safetygym_eval_plots" / Path(str(args.model_path or "policy")).stem
                )
                if len(saved_episode_plot_paths) < int(max(0, getattr(args, "episode_plot_max_episodes", 9))):
                    plot_path = plot_eval_episode_trajectory(
                        output_path=plot_dir / f"episode_{episodes + 1:03d}.png",
                        task=task,
                        overlay_specs=overlay_specs,
                        bounds=bounds,
                        path=np.asarray(ep_path, dtype=np.float64),
                        goal_positions=np.asarray(ep_goal_positions, dtype=np.float64),
                        goal_hit_points=(
                            np.asarray(ep_goal_hit_points, dtype=np.float64)
                            if ep_goal_hit_points
                            else np.zeros((0, 2), dtype=np.float64)
                        ),
                        episode_idx=episodes + 1,
                        total_episodes=int(args.num_episodes),
                        episode_reward=float(ep_ret),
                        goals_reached=int(ep_goal_met_count),
                        final_distance=float(final_dist),
                    )
                    saved_episode_plot_paths.append(Path(plot_path))
            outcome = "success" if ep["outcome_success"] > 0.5 else ("timeout" if ep["outcome_timeout"] > 0.5 else "kill")
            print(
                f"episode={episodes+1} outcome={outcome} return={ep_ret:.3f} cost={ep_cost:.3f} "
                f"dense={ep_reward_dense:.3f} sparse={ep_reward_sparse:.3f} "
                f"len={ep_len} final_dist={final_dist:.3f} interventions={int(ep['intervention_steps'])}",
                flush=True,
            )
            episodes += 1
            episode_idx = min(int(episodes), max(0, int(args.num_episodes) - 1))
            obs, _ = env.reset(seed=args.seed + episodes)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            if viewer is not None:
                viewer.draw_env(env)
            ep_ret = 0.0
            ep_cost = 0.0
            ep_reward_raw_env = 0.0
            ep_reward_dense = 0.0
            ep_reward_sparse = 0.0
            ep_reward_step_penalty = 0.0
            ep_reward_cost_penalty = 0.0
            ep_len = 0
            ep_goal_met_any = False
            ep_goal_met_count = 0
            ep_first_goal_hit_step = None
            ep_first_goal_reward_sum = 0.0
            ep_first_goal_dense_reward_sum = 0.0
            ep_path = [np.asarray(task.agent.pos[:2], dtype=np.float64).copy()]
            ep_goal_positions = [np.asarray(task.goal.pos[:2], dtype=np.float64).copy()]
            ep_goal_hit_points = []
        else:
            obs = next_obs

        if eval_fps > 0:
            time.sleep(1.0 / float(eval_fps))

    summary = augment_rollout_summary(win.summary("eval"), "eval")
    if "eval/mean_reward" not in summary and "eval/episode_return_mean" in summary:
        summary["eval/mean_reward"] = float(summary["eval/episode_return_mean"])
    if bool(getattr(args, "save_episode_plots", False)) and saved_episode_plot_paths:
        plot_dir = (
            Path(str(args.episode_plot_dir)).expanduser()
            if str(getattr(args, "episode_plot_dir", "")).strip()
            else Path("logs") / "safetygym_eval_plots" / Path(str(args.model_path or "policy")).stem
        )
        sheet_path = plot_episode_contact_sheet(
            image_paths=saved_episode_plot_paths,
            output_path=plot_dir / "episode_contact_sheet.png",
            title=f"SafetyGym eval episodes: {Path(str(args.model_path or 'policy')).stem}",
            max_cols=3,
        )
        if sheet_path is not None:
            print(f"saved episode contact sheet to {sheet_path}", flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)

    env.close()
    if ppo_vecnorm is not None:
        ppo_vecnorm.close()
    if controller is not None:
        controller.close()
    if telemetry_panel is not None:
        telemetry_panel.close()
    if viewer is not None:
        viewer.close()
    if control_panel is not None:
        control_panel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

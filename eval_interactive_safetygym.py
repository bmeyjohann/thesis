#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path
from typing import Any, Dict
from collections import deque

import numpy as np
import torch

from safetygym_utils.controllers import build_human_controller
from safetygym_utils.gamepad import (
    DEFAULT_SAFETY_GAMEPAD_CACHE_PATH,
    DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
    DEFAULT_SAFETY_GAMEPAD_PORT,
)
from safetygym_utils.env import clip_action_to_space, extract_goal_distance, extract_step_limit, make_safety_env
from safetygym_utils.io import load_args_json, maybe_find_args_json_from_model
from safetygym_utils.metrics import EpisodeWindow, classify_outcome
from safetygym_utils.rendering import build_external_viewer, resolve_env_render_mode, wants_external_viewer
from safetygym_utils.sac import SafetyActor
from safetygym_utils.wrappers import HumanInterventionWrapper, RewardModeWrapper


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive evaluation for Safety-Gymnasium FastSAC checkpoints")
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--controller", type=str, default="policy", choices=["policy", "random", "human", "keyboard", "gamepad"])
    p.add_argument("--intervention_mode", type=str, default="none", choices=["none", "human"])
    p.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array", "none", "pygame", "topdown"])
    p.add_argument("--viewer_fps", type=float, default=20.0)
    p.add_argument("--viewer_scale", type=float, default=1.0)
    p.add_argument("--surface_mode", type=str, default="default", choices=["default", "grippy"])
    p.add_argument("--car_wheel_command_limit", type=float, default=2.0)
    p.add_argument("--car_force_scale", type=float, default=2.0)
    p.add_argument("--max_episode_steps", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--fps", type=int, default=30)

    p.add_argument(
        "--reward_mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "dense_plus_sparse", "dual", "none"],
    )
    p.add_argument("--dense_reward_scale", type=float, default=1.0)
    p.add_argument("--step_penalty", type=float, default=0.0)

    p.add_argument("--intervention_threshold", type=float, default=0.1)
    p.add_argument("--intervention_hold_seconds", type=float, default=0.25)
    p.add_argument("--human_action_scale", type=float, default=1.0)
    p.add_argument("--human_input_device", type=str, default="keyboard", choices=["keyboard", "gamepad"])
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
    p.add_argument("--load_checkpoint_args", action="store_true", default=True)
    p.add_argument("--no_load_checkpoint_args", dest="load_checkpoint_args", action="store_false")
    p.add_argument("--show_telemetry_overlay", action="store_true", default=False)
    p.add_argument("--telemetry_overlay_hz", type=float, default=20.0)
    return p


def _apply_ckpt_defaults(args: argparse.Namespace) -> None:
    if not args.model_path or not args.load_checkpoint_args:
        return
    args_path = maybe_find_args_json_from_model(Path(args.model_path))
    if args_path is None or not args_path.exists():
        return
    cfg = load_args_json(args_path)

    def _set_if_default(name: str, default_val):
        if getattr(args, name) == default_val and name in cfg:
            setattr(args, name, cfg[name])

    _set_if_default("env_name", "SafetyCarGoal2-v0")
    _set_if_default("reward_mode", "sparse")
    _set_if_default("dense_reward_scale", 1.0)
    _set_if_default("step_penalty", 0.0)
    _set_if_default("surface_mode", "default")
    _set_if_default("car_wheel_command_limit", 2.0)
    _set_if_default("car_force_scale", 2.0)
    _set_if_default("actor_hidden_dim", 256)
    _set_if_default("use_layer_norm", False)
    _set_if_default("layer_norm_eps", 1e-5)
    _set_if_default("init_scale", 0.01)


def _build_env(args: argparse.Namespace, controller):
    env = make_safety_env(
        args.env_name,
        render_mode=resolve_env_render_mode(args.render_mode),
        max_episode_steps=args.max_episode_steps,
        surface_mode=args.surface_mode,
        car_wheel_command_limit=args.car_wheel_command_limit,
        car_force_scale=args.car_force_scale,
        seed=args.seed,
    )
    env = RewardModeWrapper(
        env,
        reward_mode=args.reward_mode,
        dense_reward_scale=args.dense_reward_scale,
        step_penalty=args.step_penalty,
    )
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


def _episode_metrics(info: Dict[str, Any], ep_return: float, ep_cost: float, ep_len: int, max_steps: int, terminated: bool, truncated: bool, final_distance: float) -> Dict[str, float]:
    goal_met = bool(info.get("goal_met", False))
    outcome = classify_outcome(goal_met=goal_met, episode_steps=ep_len, max_episode_steps=max_steps)
    return {
        "episode_return": float(ep_return),
        "episode_cost_sum": float(ep_cost),
        "episode_cost_rate": float(ep_cost / max(1, ep_len)),
        "episode_length": float(ep_len),
        "intervention_steps": float(info.get("teacher_intervention_steps", 0.0)),
        "intervention_fraction": float(info.get("teacher_fraction_steps", 0.0)),
        "intervention_num_bursts": float(info.get("teacher_num_bursts", 0.0)),
        "intervention_avg_burst_len": float(info.get("teacher_avg_burst_len", 0.0)),
        "goal_met": 1.0 if goal_met else 0.0,
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

    if args.controller in {"human", "keyboard", "gamepad"} and args.intervention_mode == "human":
        # Avoid double-human override (controller action + intervention wrapper action).
        args.intervention_mode = "none"
    if args.controller == "keyboard":
        args.human_input_device = "keyboard"
    elif args.controller == "gamepad":
        args.human_input_device = "gamepad"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    controller = None
    control_help: list[str] = []
    if args.controller in {"human", "keyboard", "gamepad"} or args.intervention_mode == "human":
        # Temporary env for action-dim detection.
        tmp_env = make_safety_env(
            args.env_name,
            render_mode="none",
            max_episode_steps=args.max_episode_steps,
            surface_mode=args.surface_mode,
            car_wheel_command_limit=args.car_wheel_command_limit,
            car_force_scale=args.car_force_scale,
            seed=args.seed,
        )
        act_dim = int(np.prod(tmp_env.action_space.shape))
        tmp_env.close()
        controller = build_human_controller(
            input_device=str(getattr(args, "human_input_device", "keyboard")),
            action_dim=act_dim,
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
            show_overlay=not bool(args.show_telemetry_overlay),
            prefer_separate_keyboard_window=wants_external_viewer(getattr(args, "render_mode", "human")),
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

    obs, _ = env.reset(seed=args.seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if viewer is not None:
        viewer.draw_env(env)
    obs_dim = int(obs.shape[0])
    act_dim = int(np.prod(env.action_space.shape))

    actor = None
    if args.controller == "policy":
        if not args.model_path:
            raise ValueError("--model_path is required for --controller policy")
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

    win = EpisodeWindow(size=max(10, args.num_episodes))

    ep_ret = 0.0
    ep_cost = 0.0
    ep_len = 0
    episodes = 0

    while episodes < args.num_episodes:
        if args.controller == "policy":
            with torch.no_grad():
                obs_t = torch.as_tensor(obs[None, :], device=device, dtype=torch.float32)
                _, _, mean = actor(obs_t)
                action = mean[0].detach().cpu().numpy().astype(np.float32)
        elif args.controller == "random":
            action = env.action_space.sample().astype(np.float32)
        else:
            if controller is None:
                raise RuntimeError("Controller requested but not initialized")
            action = controller.get_action().astype(np.float32)

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
        ep_len += 1

        if terminated or truncated:
            final_dist = extract_goal_distance(env)
            ep = _episode_metrics(
                dict(info),
                ep_return=ep_ret,
                ep_cost=ep_cost,
                ep_len=ep_len,
                max_steps=max_steps,
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_distance=final_dist,
            )
            win.add(ep)
            outcome = "success" if ep["outcome_success"] > 0.5 else ("timeout" if ep["outcome_timeout"] > 0.5 else "kill")
            print(
                f"episode={episodes+1} outcome={outcome} return={ep_ret:.3f} cost={ep_cost:.3f} "
                f"len={ep_len} final_dist={final_dist:.3f} interventions={int(ep['intervention_steps'])}",
                flush=True,
            )
            episodes += 1
            obs, _ = env.reset(seed=args.seed + episodes)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            if viewer is not None:
                viewer.draw_env(env)
            ep_ret = 0.0
            ep_cost = 0.0
            ep_len = 0
        else:
            obs = next_obs

        if args.fps > 0:
            time.sleep(1.0 / float(args.fps))

    summary = win.summary("eval")
    print(json.dumps(summary, sort_keys=True), flush=True)

    env.close()
    if controller is not None:
        controller.close()
    if telemetry_panel is not None:
        telemetry_panel.close()
    if viewer is not None:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

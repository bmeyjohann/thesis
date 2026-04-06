from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .env import extract_min_constrained_clearance
from .gamepad import (
    DEFAULT_SAFETY_GAMEPAD_CACHE_PATH,
    DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
    DEFAULT_SAFETY_GAMEPAD_PORT,
    GamepadMappingConfig,
    GamepadStateClient,
    PygameGamepadController,
    RemoteGamepadController,
    apply_gamepad_mapping_profile,
    infer_control_scheme,
    resolve_cached_gamepad_endpoint,
)


@dataclass
class KeyboardConfig:
    action_scale: float = 1.0
    window_width: int = 420
    window_height: int = 220
    window_title: str = "SafetyGym Controls"
    control_scheme: str = "planar_velocity"
    overlay_fps_limit: int = 0
    overlay_draw_hz: float = 20.0
    wheel_command_limit: float = 2.0
    show_overlay: bool = True

class PygameKeyboardController:
    """Focused keyboard control window for continuous actions."""

    def __init__(self, action_dim: int, config: KeyboardConfig | None = None):
        self.action_dim = int(action_dim)
        self.config = config or KeyboardConfig()
        self._initialized = False
        self._warned_no_display = False
        self._pygame = None
        self._screen = None
        self._font = None
        self._clock = None
        self._last_overlay_draw_ts = 0.0

    @staticmethod
    def _mix_differential_wheels(throttle: float, turn: float) -> tuple[float, float]:
        left_wheel = float(throttle - turn)
        right_wheel = float(throttle + turn)
        return left_wheel, right_wheel

    @staticmethod
    def _has_graphical_display() -> bool:
        if os.name == "nt":
            return True
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            return True
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            return False
        return False

    def _init_pygame(self) -> None:
        if self._initialized:
            return
        if not self._has_graphical_display() and not self._warned_no_display:
            print(
                "Warning: no graphical display detected; keyboard intervention will not work in this session.",
                file=sys.stderr,
                flush=True,
            )
            self._warned_no_display = True
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        pygame.init()
        pygame.display.set_caption(self.config.window_title)
        self._screen = pygame.display.set_mode((self.config.window_width, self.config.window_height))
        self._font = pygame.font.Font(None, 20)
        self._clock = pygame.time.Clock()
        self._pygame = pygame
        self._initialized = True

    def _read_keys(self) -> np.ndarray:
        self._init_pygame()
        pygame = self._pygame
        assert pygame is not None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

        keys = pygame.key.get_pressed()
        a = np.zeros((self.action_dim,), dtype=np.float32)
        s = float(self.config.action_scale)

        # Differential-drive style controls:
        # action[0] -> forward/backward velocity
        # action[1] -> turning (left/right)
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]

        if self.config.control_scheme == "differential_wheels" and self.action_dim >= 2:
            # Wheel-space mixing:
            # W/S -> both wheels +/- ; A/D -> opposite wheel directions.
            throttle = (1.0 if up else 0.0) - (1.0 if down else 0.0)
            # Sign convention matches observed car dynamics in SafetyCar:
            # A should steer left, D should steer right.
            turn = (1.0 if right else 0.0) - (1.0 if left else 0.0)
            a[0], a[1] = self._mix_differential_wheels(throttle=throttle, turn=turn)
        else:
            if self.action_dim >= 1:
                a[0] = (1.0 if up else 0.0) - (1.0 if down else 0.0)
            if self.action_dim >= 2:
                a[1] = (1.0 if left else 0.0) - (1.0 if right else 0.0)

        # Additional channels for larger action spaces.
        if self.action_dim >= 3:
            a[2] = (1.0 if keys[pygame.K_q] else 0.0) - (1.0 if keys[pygame.K_e] else 0.0)
        if self.action_dim >= 4:
            a[3] = (1.0 if keys[pygame.K_r] else 0.0) - (1.0 if keys[pygame.K_f] else 0.0)
        if self.action_dim >= 5:
            a[4] = (1.0 if keys[pygame.K_t] else 0.0) - (1.0 if keys[pygame.K_g] else 0.0)

        a *= s
        if self.config.control_scheme == "differential_wheels" and self.action_dim >= 2:
            lim = float(max(0.0, self.config.wheel_command_limit))
            a[:2] = np.clip(a[:2], -lim, lim)
            if self.action_dim > 2:
                a[2:] = np.clip(a[2:], -1.0, 1.0)
            return a.astype(np.float32, copy=False)
        return np.clip(a, -1.0, 1.0).astype(np.float32, copy=False)

    def _draw_overlay(self, action: np.ndarray) -> None:
        if not self._initialized or not bool(self.config.show_overlay):
            return
        draw_hz = float(self.config.overlay_draw_hz)
        now = time.perf_counter()
        if draw_hz > 0.0 and (now - self._last_overlay_draw_ts) < (1.0 / draw_hz):
            return
        self._last_overlay_draw_ts = now

        pygame = self._pygame
        screen = self._screen
        font = self._font
        clock = self._clock
        assert pygame is not None and screen is not None and font is not None and clock is not None

        screen.fill((20, 20, 24))
        if self.config.control_scheme == "differential_wheels":
            lines = [
                "Focus this window for intervention controls.",
                "Car wheel mixing (left_wheel, right_wheel):",
                "W/S: both wheels forward/backward",
                "A/D: opposite wheel directions for turning",
                f"Wheel command cap: +/-{float(self.config.wheel_command_limit):.1f}",
                f"Action: {np.array2string(np.asarray(action), precision=2)}",
            ]
        else:
            lines = [
                "Focus this window for intervention controls.",
                "Forward/Back: W/S or Up/Down",
                "Turn Left/Right: A/D or Left/Right",
                "Extra dims: Q/E, R/F, T/G",
                f"Action: {np.array2string(np.asarray(action), precision=2)}",
            ]
        y = 20
        for text in lines:
            surf = font.render(text, True, (220, 220, 220))
            screen.blit(surf, (16, y))
            y += 28
        pygame.display.flip()
        if int(self.config.overlay_fps_limit) > 0:
            clock.tick(int(self.config.overlay_fps_limit))

    def get_action(self) -> np.ndarray:
        action = self._read_keys()
        self._draw_overlay(action)
        return action

    def close(self) -> None:
        if not self._initialized:
            return
        pygame = self._pygame
        if pygame is not None:
            pygame.display.quit()
            pygame.quit()
        self._initialized = False
        self._pygame = None
        self._screen = None
        self._font = None
        self._clock = None


class TkKeyboardController:
    """Independent keyboard control window that does not use pygame.display."""

    def __init__(self, action_dim: int, config: KeyboardConfig | None = None):
        self.action_dim = int(action_dim)
        self.config = config or KeyboardConfig()
        self._keys_down: set[str] = set()
        self._warned_no_display = False
        self._initialized = False
        self._closed = False
        self._last_overlay_draw_ts = 0.0
        self._tk = None
        self._root = None
        self._label: Optional[object] = None

    @staticmethod
    def _mix_differential_wheels(throttle: float, turn: float) -> tuple[float, float]:
        left_wheel = float(throttle - turn)
        right_wheel = float(throttle + turn)
        return left_wheel, right_wheel

    @staticmethod
    def _has_graphical_display() -> bool:
        if os.name == "nt":
            return True
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            return True
        return False

    def _normalized_key(self, key: str) -> str:
        key_l = str(key).strip().lower()
        aliases = {
            "left": "left",
            "right": "right",
            "up": "up",
            "down": "down",
            "a": "a",
            "d": "d",
            "w": "w",
            "s": "s",
            "q": "q",
            "e": "e",
            "r": "r",
            "f": "f",
            "t": "t",
            "g": "g",
        }
        return aliases.get(key_l, key_l)

    def _on_press(self, event) -> None:
        key = self._normalized_key(getattr(event, "keysym", ""))
        if key:
            self._keys_down.add(key)

    def _on_release(self, event) -> None:
        key = self._normalized_key(getattr(event, "keysym", ""))
        if key:
            self._keys_down.discard(key)

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

    def _init_tk(self) -> None:
        if self._initialized:
            return
        if not self._has_graphical_display() and not self._warned_no_display:
            print(
                "Warning: no graphical display detected; keyboard intervention will not work in this session.",
                file=sys.stderr,
                flush=True,
            )
            self._warned_no_display = True
        try:
            import tkinter as tk
        except Exception:
            # Fall back to the existing pygame overlay when Tk is unavailable.
            fallback = PygameKeyboardController(self.action_dim, self.config)
            self.get_action = fallback.get_action  # type: ignore[method-assign]
            self.close = fallback.close  # type: ignore[method-assign]
            self._initialized = True
            return

        root = tk.Tk()
        root.title(self.config.window_title)
        root.geometry(f"{int(self.config.window_width)}x{int(self.config.window_height)}")
        root.resizable(False, False)
        root.configure(bg="#141418")
        root.bind("<KeyPress>", self._on_press)
        root.bind("<KeyRelease>", self._on_release)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        label = tk.Label(
            root,
            text="Focus this window for intervention controls.",
            justify="left",
            anchor="nw",
            bg="#141418",
            fg="#e0e0e0",
            font=("TkDefaultFont", 11),
        )
        label.pack(fill="both", expand=True, padx=12, pady=12)
        try:
            root.focus_force()
        except Exception:
            pass
        self._tk = tk
        self._root = root
        self._label = label
        self._initialized = True

    def _pump_window(self) -> None:
        self._init_tk()
        root = self._root
        if root is None or self._closed:
            return
        try:
            root.update_idletasks()
            root.update()
        except Exception:
            self._closed = True
            self._root = None
            self._label = None

    def _read_keys(self) -> np.ndarray:
        self._pump_window()
        a = np.zeros((self.action_dim,), dtype=np.float32)
        s = float(self.config.action_scale)
        keys = self._keys_down

        left = ("left" in keys) or ("a" in keys)
        right = ("right" in keys) or ("d" in keys)
        up = ("up" in keys) or ("w" in keys)
        down = ("down" in keys) or ("s" in keys)

        if self.config.control_scheme == "differential_wheels" and self.action_dim >= 2:
            throttle = (1.0 if up else 0.0) - (1.0 if down else 0.0)
            turn = (1.0 if right else 0.0) - (1.0 if left else 0.0)
            a[0], a[1] = self._mix_differential_wheels(throttle=throttle, turn=turn)
        else:
            if self.action_dim >= 1:
                a[0] = (1.0 if up else 0.0) - (1.0 if down else 0.0)
            if self.action_dim >= 2:
                a[1] = (1.0 if left else 0.0) - (1.0 if right else 0.0)

        if self.action_dim >= 3:
            a[2] = (1.0 if "q" in keys else 0.0) - (1.0 if "e" in keys else 0.0)
        if self.action_dim >= 4:
            a[3] = (1.0 if "r" in keys else 0.0) - (1.0 if "f" in keys else 0.0)
        if self.action_dim >= 5:
            a[4] = (1.0 if "t" in keys else 0.0) - (1.0 if "g" in keys else 0.0)

        a *= s
        if self.config.control_scheme == "differential_wheels" and self.action_dim >= 2:
            lim = float(max(0.0, self.config.wheel_command_limit))
            a[:2] = np.clip(a[:2], -lim, lim)
            if self.action_dim > 2:
                a[2:] = np.clip(a[2:], -1.0, 1.0)
            return a.astype(np.float32, copy=False)
        return np.clip(a, -1.0, 1.0).astype(np.float32, copy=False)

    def _draw_overlay(self, action: np.ndarray) -> None:
        if not bool(self.config.show_overlay):
            return
        draw_hz = float(self.config.overlay_draw_hz)
        now = time.perf_counter()
        if draw_hz > 0.0 and (now - self._last_overlay_draw_ts) < (1.0 / draw_hz):
            return
        self._last_overlay_draw_ts = now
        label = self._label
        if label is None:
            return
        if self.config.control_scheme == "differential_wheels":
            lines = [
                "Focus this window for keyboard intervention.",
                "",
                "W/S: both wheels forward/backward",
                "A/D: turn left/right",
                f"Wheel command cap: +/-{float(self.config.wheel_command_limit):.1f}",
                "",
                f"Action: {np.array2string(np.asarray(action), precision=2)}",
            ]
        else:
            lines = [
                "Focus this window for keyboard intervention.",
                "",
                "W/S or Up/Down: forward/backward",
                "A/D or Left/Right: turn left/right",
                "Extra dims: Q/E, R/F, T/G",
                "",
                f"Action: {np.array2string(np.asarray(action), precision=2)}",
            ]
        try:
            label.config(text="\n".join(lines))
        except Exception:
            pass

    def get_action(self) -> np.ndarray:
        action = self._read_keys()
        self._draw_overlay(action)
        return action

    def close(self) -> None:
        root = self._root
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        self._root = None
        self._label = None
        self._closed = True


class ExpertPolicyController:
    """Checkpoint-backed policy controller for expert-as-human intervention tests."""

    always_active = True

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: str = "cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Expert checkpoint not found: {self.checkpoint_path}")
        self.obs_dim = int(obs_dim)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.device = device
        self._load_model()

    @staticmethod
    def _resolve_args_path(checkpoint_path: Path) -> Path | None:
        for candidate in (checkpoint_path.parent / "args.json", checkpoint_path.parent.parent / "args.json"):
            if candidate.is_file():
                return candidate
        return None

    def _load_model(self) -> None:
        import json
        import torch
        import sys

        repo_root = Path(__file__).resolve().parent.parent
        fast_sac_path = repo_root / "fasttd3" / "fast_sac"
        fast_sac_path_str = str(fast_sac_path)
        if fast_sac_path.exists() and fast_sac_path_str not in sys.path:
            sys.path.insert(0, fast_sac_path_str)

        from fast_sac import Actor
        from fast_sac_utils import EmpiricalNormalization

        from .sac import SafetyActor

        args_path = self._resolve_args_path(self.checkpoint_path)
        ckpt_args = {}
        if args_path is not None:
            try:
                ckpt_args = json.loads(args_path.read_text())
            except Exception:
                ckpt_args = {}

        module_impl = str(ckpt_args.get("module_impl", "fastsac")).strip().lower()
        actor_hidden_dim = int(ckpt_args.get("actor_hidden_dim", 256))
        init_scale = float(ckpt_args.get("init_scale", 0.01))
        use_layer_norm = bool(ckpt_args.get("use_layer_norm", False))
        layer_norm_eps = float(ckpt_args.get("layer_norm_eps", 1e-5))
        self.scale_to_env_bounds = bool(ckpt_args.get("scale_actor_to_env_bounds", False))

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        action_dim = int(self.action_low.shape[0])
        device = torch.device(self.device)
        if module_impl == "custom":
            actor = SafetyActor(
                n_obs=self.obs_dim,
                n_act=action_dim,
                num_envs=1,
                init_scale=init_scale,
                hidden_dim=actor_hidden_dim,
                use_layer_norm=use_layer_norm,
                layer_norm_eps=layer_norm_eps,
                device=device,
            )
        else:
            actor = Actor(
                n_obs=self.obs_dim,
                n_act=action_dim,
                num_envs=1,
                init_scale=init_scale,
                hidden_dim=actor_hidden_dim,
                device=device,
            )
        actor.load_state_dict(checkpoint["actor_state_dict"])
        actor.eval()
        self.actor = actor
        self._torch = torch

        state = checkpoint.get("obs_normalizer_state_dict") or {}
        if state:
            normalizer = EmpiricalNormalization(shape=self.obs_dim, device=device)
            normalizer.load_state_dict(state, strict=False)
            self.obs_normalizer = normalizer
        else:
            self.obs_normalizer = torch.nn.Identity()

    def get_action(self, obs: np.ndarray | None = None, env=None) -> np.ndarray | None:
        if obs is None:
            return None
        torch = self._torch
        device = next(self.actor.parameters()).device
        with torch.no_grad():
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=device)
            obs_t = self.obs_normalizer(obs_t)
            _, _, mean_t = self.actor(obs_t)
            action = mean_t[0].detach().cpu().numpy().astype(np.float32)
        if self.scale_to_env_bounds:
            center = 0.5 * (self.action_high + self.action_low)
            half = 0.5 * (self.action_high - self.action_low)
            action = center + action * half
        return np.clip(action, self.action_low, self.action_high).astype(np.float32, copy=False)

    def close(self) -> None:
        return None


class SwitchingExpertPolicyController:
    """Route between a goal-reaching expert and a safe expert using live clearance."""

    always_active = True

    def __init__(
        self,
        *,
        goal_checkpoint_path: str | Path,
        safe_checkpoint_path: str | Path,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        clearance_threshold: float,
        device: str = "cpu",
    ):
        self.clearance_threshold = float(clearance_threshold)
        self.goal_controller = ExpertPolicyController(
            checkpoint_path=goal_checkpoint_path,
            obs_dim=obs_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
        )
        self.safe_controller = ExpertPolicyController(
            checkpoint_path=safe_checkpoint_path,
            obs_dim=obs_dim,
            action_low=action_low,
            action_high=action_high,
            device=device,
        )

    def get_action(self, obs: np.ndarray | None = None, env=None) -> np.ndarray | None:
        if obs is None:
            return None
        clearance = extract_min_constrained_clearance(env) if env is not None else None
        if clearance is not None and clearance <= self.clearance_threshold:
            return self.safe_controller.get_action(obs=obs)
        return self.goal_controller.get_action(obs=obs)

    def close(self) -> None:
        self.goal_controller.close()
        self.safe_controller.close()


def build_human_controller(
    *,
    input_device: str,
    action_dim: int,
    obs_dim: int | None = None,
    env_name: str,
    action_scale: float,
    wheel_command_limit: float,
    overlay_fps_limit: int,
    overlay_draw_hz: float,
    gamepad_mode: str = "local",
    gamepad_host: str = "",
    gamepad_port: int = 0,
    gamepad_cache_path: Path | str = DEFAULT_SAFETY_GAMEPAD_CACHE_PATH,
    gamepad_reconnect_seconds: float = 2.0,
    gamepad_config_path: Path | str = DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
    gamepad_use_saved_config: bool = True,
    gamepad_device_index: int = 0,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    expert_checkpoint_path: str = "",
    expert_safe_checkpoint_path: str = "",
    expert_switch_clearance_threshold: float = 0.08,
    expert_device: str = "cpu",
    show_overlay: bool = True,
    prefer_separate_keyboard_window: bool = False,
    control_scheme_override: str | None = None,
):
    input_device = str(input_device).lower()
    control_scheme = str(control_scheme_override).strip() if control_scheme_override else infer_control_scheme(env_name)
    if input_device == "expert":
        if obs_dim is None or action_low is None or action_high is None:
            raise ValueError("Expert controller requires obs_dim plus action_low/action_high.")
        checkpoint_path = str(expert_checkpoint_path).strip()
        if not checkpoint_path:
            raise ValueError("input_device=expert requires expert_checkpoint_path.")
        return ExpertPolicyController(
            checkpoint_path=checkpoint_path,
            obs_dim=int(obs_dim),
            action_low=np.asarray(action_low, dtype=np.float32),
            action_high=np.asarray(action_high, dtype=np.float32),
            device=str(expert_device),
        )
    if input_device == "expert_switch":
        if obs_dim is None or action_low is None or action_high is None:
            raise ValueError("Expert switch controller requires obs_dim plus action_low/action_high.")
        goal_checkpoint_path = str(expert_checkpoint_path).strip()
        safe_checkpoint_path = str(expert_safe_checkpoint_path).strip()
        if not goal_checkpoint_path:
            raise ValueError("input_device=expert_switch requires expert_checkpoint_path for the goal expert.")
        if not safe_checkpoint_path:
            raise ValueError("input_device=expert_switch requires expert_safe_checkpoint_path for the safe expert.")
        return SwitchingExpertPolicyController(
            goal_checkpoint_path=goal_checkpoint_path,
            safe_checkpoint_path=safe_checkpoint_path,
            obs_dim=int(obs_dim),
            action_low=np.asarray(action_low, dtype=np.float32),
            action_high=np.asarray(action_high, dtype=np.float32),
            clearance_threshold=float(expert_switch_clearance_threshold),
            device=str(expert_device),
        )
    if input_device == "gamepad":
        config = GamepadMappingConfig(
            device_index=int(gamepad_device_index),
            action_scale=float(action_scale),
            wheel_command_limit=float(wheel_command_limit),
        )
        if bool(gamepad_use_saved_config):
            config, _loaded = apply_gamepad_mapping_profile(config, Path(gamepad_config_path).expanduser())
        mode = str(gamepad_mode).lower()
        if mode == "connect":
            host, port, from_cache = resolve_cached_gamepad_endpoint(
                host=str(gamepad_host),
                port=int(gamepad_port),
                cache_path=gamepad_cache_path,
            )
            print(
                f"gamepad endpoint resolved to {host}:{port} (from_cache={1 if from_cache else 0})",
                flush=True,
            )
            client = GamepadStateClient(
                host=host,
                port=port,
                reconnect_seconds=float(gamepad_reconnect_seconds),
                cache_path=gamepad_cache_path,
            )
            return RemoteGamepadController(
                action_dim=action_dim,
                control_scheme=control_scheme,
                client=client,
                config=config,
                config_path=gamepad_config_path,
                hot_reload=True,
            )
        return PygameGamepadController(
            action_dim=action_dim,
            control_scheme=control_scheme,
            config=config,
            config_path=gamepad_config_path,
            hot_reload=True,
        )
    keyboard_cls = TkKeyboardController if bool(prefer_separate_keyboard_window) else PygameKeyboardController
    return keyboard_cls(
        action_dim=action_dim,
        config=KeyboardConfig(
            action_scale=action_scale,
            control_scheme=control_scheme,
            overlay_fps_limit=int(overlay_fps_limit),
            overlay_draw_hz=float(overlay_draw_hz),
            wheel_command_limit=float(wheel_command_limit),
            show_overlay=bool(show_overlay),
        ),
    )

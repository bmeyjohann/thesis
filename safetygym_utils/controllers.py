from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .env import (
    build_privileged_geometry_observation,
    extract_agent_forward_xy,
    extract_agent_xy,
    extract_goal_xy,
    extract_min_constrained_clearance,
    _task_constrained_object_specs,
    unwrap_env,
)
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
from .imitation_teacher import load_imitation_checkpoint, normalize_observations, scale_unit_action
from .flow_imitation_teacher import flow_action_to_env, load_flow_imitation_checkpoint


@dataclass
class KeyboardConfig:
    action_scale: float = 1.0
    window_width: int = 420
    window_height: int = 260
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
        self._paused = False
        self._intervening = False
        self._intervention_probability: float | None = None
        self._cost_active = False
        self._cost_label = ""
        self._bc_eval_requested = False

    def is_paused(self) -> bool:
        return bool(self._paused)

    def pop_bc_eval_request(self) -> bool:
        requested = bool(self._bc_eval_requested)
        self._bc_eval_requested = False
        return requested

    def set_intervention_status(self, *, active: bool, probability: float | None = None) -> None:
        self._intervening = bool(active)
        self._intervention_probability = None if probability is None else float(probability)

    def set_cost_status(self, *, active: bool, label: str = "") -> None:
        self._cost_active = bool(active)
        self._cost_label = str(label or "")

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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self._paused = True
                elif event.key in {pygame.K_c, pygame.K_SPACE}:
                    self._paused = False
                elif event.key == pygame.K_b:
                    self._bc_eval_requested = True
                    self._paused = True

        keys = pygame.key.get_pressed()
        a = np.zeros((self.action_dim,), dtype=np.float32)
        if self._paused:
            return a
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

        if self._paused:
            bg = (80, 64, 16)
        elif self._intervening:
            bg = (96, 18, 18)
        elif self._cost_active:
            bg = (64, 20, 12)
        else:
            bg = (12, 12, 14)
        screen.fill(bg)
        status = "PAUSED" if self._paused else ("INTERVENING" if self._intervening else "policy active")
        prob_text = (
            f" | p(intervene)={self._intervention_probability:.2f}"
            if self._intervention_probability is not None
            else ""
        )
        # Filled dot = active on the latest step; outline-only = inactive.
        dot_y = 18
        pygame.draw.circle(screen, (255, 255, 255), (22, dot_y), 8, width=2)
        if self._intervening:
            pygame.draw.circle(screen, (230, 40, 40), (22, dot_y), 6)
        pygame.draw.circle(screen, (255, 255, 255), (150, dot_y), 8, width=2)
        if self._cost_active:
            pygame.draw.circle(screen, (255, 150, 20), (150, dot_y), 6)
        screen.blit(font.render("intervention", True, (230, 230, 230)), (34, dot_y - 9))
        cost_text = f"cost {self._cost_label}".strip()
        screen.blit(font.render(cost_text, True, (230, 230, 230)), (162, dot_y - 9))
        if self.config.control_scheme == "differential_wheels":
            lines = [
                f"Status: {status}{prob_text}",
                "Focus this window for intervention controls.",
                "P: pause | C/Space: continue | B: train/eval BC teacher",
                "Car wheel mixing (left_wheel, right_wheel):",
                "W/S: both wheels forward/backward",
                "A/D: opposite wheel directions for turning",
                f"Wheel command cap: +/-{float(self.config.wheel_command_limit):.1f}",
                f"Action: {np.array2string(np.asarray(action), precision=2)}",
            ]
        else:
            lines = [
                f"Status: {status}{prob_text}",
                "Focus this window for intervention controls.",
                "P: pause | C/Space: continue | B: train/eval BC teacher",
                "Forward/Back: W/S or Up/Down",
                "Turn Left/Right: A/D or Left/Right",
                "Extra dims: Q/E, R/F, T/G",
                f"Action: {np.array2string(np.asarray(action), precision=2)}",
            ]
        y = 44
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
        self._status_canvas: Optional[object] = None
        self._paused = False
        self._intervening = False
        self._intervention_probability: float | None = None
        self._cost_active = False
        self._cost_label = ""
        self._bc_eval_requested = False

    def is_paused(self) -> bool:
        return bool(self._paused)

    def pop_bc_eval_request(self) -> bool:
        requested = bool(self._bc_eval_requested)
        self._bc_eval_requested = False
        return requested

    def set_intervention_status(self, *, active: bool, probability: float | None = None) -> None:
        self._intervening = bool(active)
        self._intervention_probability = None if probability is None else float(probability)

    def set_cost_status(self, *, active: bool, label: str = "") -> None:
        self._cost_active = bool(active)
        self._cost_label = str(label or "")

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
            "p": "p",
            "b": "b",
            "c": "c",
            "space": "space",
        }
        return aliases.get(key_l, key_l)

    def _on_press(self, event) -> None:
        key = self._normalized_key(getattr(event, "keysym", ""))
        if key == "p":
            self._paused = True
            return
        if key == "b":
            self._bc_eval_requested = True
            self._paused = True
            return
        if key in {"c", "space"}:
            self._paused = False
            return
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
        canvas = tk.Canvas(root, height=38, bg="#141418", highlightthickness=0)
        canvas.pack(fill="x", padx=12, pady=(10, 0))
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
        self._status_canvas = canvas
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
            self._status_canvas = None

    def _read_keys(self) -> np.ndarray:
        self._pump_window()
        a = np.zeros((self.action_dim,), dtype=np.float32)
        if self._paused:
            return a
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
        status = "PAUSED" if self._paused else ("INTERVENING" if self._intervening else "policy active")
        prob_text = (
            f" | p(intervene)={self._intervention_probability:.2f}"
            if self._intervention_probability is not None
            else ""
        )
        if self.config.control_scheme == "differential_wheels":
            lines = [
                f"Status: {status}{prob_text}",
                "Focus this window for keyboard intervention.",
                "P: pause | C/Space: continue | B: train/eval BC teacher",
                "",
                "W/S: both wheels forward/backward",
                "A/D: turn left/right",
                f"Wheel command cap: +/-{float(self.config.wheel_command_limit):.1f}",
                "",
                f"Action: {np.array2string(np.asarray(action), precision=2)}",
            ]
        else:
            lines = [
                f"Status: {status}{prob_text}",
                "Focus this window for keyboard intervention.",
                "P: pause | C/Space: continue | B: train/eval BC teacher",
                "",
                "W/S or Up/Down: forward/backward",
                "A/D or Left/Right: turn left/right",
                "Extra dims: Q/E, R/F, T/G",
                "",
                f"Action: {np.array2string(np.asarray(action), precision=2)}",
            ]
        try:
            bg = "#504010" if self._paused else ("#601212" if self._intervening else "#0c0c0e")
            label.config(bg=bg)
            root = self._root
            if root is not None:
                root.configure(bg=bg)
            canvas = self._status_canvas
            if canvas is not None:
                canvas.configure(bg=bg)
                canvas.delete("all")

                def draw_dot(x: int, label_text: str, active: bool, fill: str) -> None:
                    canvas.create_oval(x, 9, x + 18, 27, outline="#f5f5f5", width=2)
                    if active:
                        canvas.create_oval(x + 4, 13, x + 14, 23, outline=fill, fill=fill)
                    canvas.create_text(x + 28, 18, text=label_text, fill="#e8e8e8", anchor="w", font=("TkDefaultFont", 10))

                draw_dot(8, "intervention", self._intervening, "#e53935")
                cost_label = "cost" if not self._cost_label else f"cost {self._cost_label}"
                draw_dot(170, cost_label, self._cost_active, "#ff9800")
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
        self._status_canvas = None
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
        temporal_encoder = str(ckpt_args.get("temporal_encoder", "none"))
        obs_frame_stack = int(ckpt_args.get("obs_frame_stack", 1) or 1)
        intervention_aux_head = bool(ckpt_args.get("intervention_aux_head", False))
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
                temporal_encoder=temporal_encoder,
                obs_frame_stack=obs_frame_stack,
                intervention_aux_head=intervention_aux_head,
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


class SafeRLPolicyController:
    """Checkpoint-backed CPO/PCPO teacher from the local safe_rl package."""

    always_active = True

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        config_path: str | Path,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: str = "cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Safe-RL checkpoint not found: {self.checkpoint_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Safe-RL config not found: {self.config_path}")
        self.obs_dim = int(obs_dim)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.device = device
        self._load_model()

    def _load_model(self) -> None:
        import torch
        import yaml

        repo_root = Path(__file__).resolve().parent.parent
        safe_rl_root = repo_root / "safe_rl"
        safe_rl_root_str = str(safe_rl_root)
        if safe_rl_root.exists() and safe_rl_root_str not in sys.path:
            sys.path.insert(0, safe_rl_root_str)

        from safe_rl.modules import ActorCritic

        with self.config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        policy_cfg = dict(cfg["policy"])
        policy_cfg.pop("class_name", None)
        cost_limits = cfg.get("algorithm", {}).get("cost_limits") or [1.0]
        policy_cfg["num_costs"] = len(cost_limits)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        action_dim = int(self.action_low.shape[0])
        device = torch.device(self.device)
        policy = ActorCritic(self.obs_dim, self.obs_dim, action_dim, **policy_cfg).to(device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()
        self.policy = policy
        self._torch = torch

    def get_action(self, obs: np.ndarray | None = None, env=None) -> np.ndarray | None:
        if obs is None:
            return None
        torch = self._torch
        device = next(self.policy.parameters()).device
        with torch.inference_mode():
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=device)
            action = self.policy.act_inference(obs_t)[0].detach().cpu().numpy().astype(np.float32)
        return np.clip(action, self.action_low, self.action_high).astype(np.float32, copy=False)

    def evaluate_value(self, obs: np.ndarray | None = None, env=None) -> float | None:
        if obs is None:
            return None
        torch = self._torch
        device = next(self.policy.parameters()).device
        with torch.inference_mode():
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=device)
            value = self.policy.evaluate(obs_t)
        return float(value.reshape(-1)[0].detach().cpu().item())

    def evaluate_cost_value(self, obs: np.ndarray | None = None, env=None) -> float | None:
        if obs is None or not hasattr(self.policy, "evaluate_cost"):
            return None
        torch = self._torch
        device = next(self.policy.parameters()).device
        with torch.inference_mode():
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=device)
            try:
                value = self.policy.evaluate_cost(obs_t)
            except RuntimeError:
                return None
        return float(value.reshape(-1)[0].detach().cpu().item())

    def close(self) -> None:
        return None


class HeadingBCPolicyController:
    """Heading-level DAgger policy used as an always-active teacher."""

    always_active = True

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: str = "cpu",
    ):
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Heading-BC checkpoint not found: {checkpoint}")
        from .heading_policy import load_heading_policy

        self.policy = load_heading_policy(checkpoint, device=device)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.device = device

    def get_action(self, obs: np.ndarray | None = None, env=None, student_action=None) -> np.ndarray | None:
        if obs is None or env is None:
            return None
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        expected = int(getattr(getattr(self.policy, "config", None), "obs_dim", obs_arr.shape[0]))
        if obs_arr.shape[0] != expected:
            rich_obs = build_privileged_geometry_observation(env, obs_arr, rich=True)
            if rich_obs.shape[0] == expected:
                obs_arr = rich_obs
        action = self.policy.act(obs_arr, env=env, device=self.device)
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


class LearnedInterventionPolicyController:
    """BC/HG-DAGGER teacher with an intervention-probability head."""

    always_active = True

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        action_low: np.ndarray,
        action_high: np.ndarray,
        intervention_threshold: float = 0.5,
        device: str = "cpu",
    ):
        import torch

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.model, self.state = load_imitation_checkpoint(self.checkpoint_path, device=torch.device(device))
        self.device = torch.device(device)
        self.intervention_threshold = float(intervention_threshold)
        self.action_low_np = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high_np = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self._torch = torch
        meta = self.state.get("metadata", {})
        self.context_len = int(max(1, meta.get("context_len", 1)))
        self.env_obs_dim = int(meta.get("env_obs_dim", meta.get("obs_dim", 0)) or 0)
        self.segment_dim = int(meta.get("segment_dim", 0) or 0)
        self._history: deque[np.ndarray] = deque(maxlen=self.context_len)
        self.trained_action_low = self.state["action_low"].detach().cpu().numpy().reshape(-1)
        self.trained_action_high = self.state["action_high"].detach().cpu().numpy().reshape(-1)
        if self.trained_action_low.shape != self.action_low_np.shape:
            raise ValueError(
                f"Learned teacher action dim {self.trained_action_low.shape[0]} does not match env action dim {self.action_low_np.shape[0]}."
            )
        if meta.get("act_dim") is not None and int(meta["act_dim"]) != int(self.action_low_np.shape[0]):
            raise ValueError(
                f"Learned teacher checkpoint act_dim={meta['act_dim']} does not match env action dim={self.action_low_np.shape[0]}."
            )

    def reset(self) -> None:
        self._history.clear()
        self.last_intervention_probability = 0.0

    def _features(self, obs: np.ndarray, student_action: np.ndarray | None) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.env_obs_dim > 0 and obs_arr.shape[0] != self.env_obs_dim:
            raise ValueError(
                f"Learned teacher checkpoint env_obs_dim={self.env_obs_dim} does not match obs dim={obs_arr.shape[0]}."
            )
        if student_action is None:
            student_arr = np.zeros_like(self.action_low_np, dtype=np.float32)
        else:
            student_arr = np.asarray(student_action, dtype=np.float32).reshape(-1)
            if student_arr.shape[0] != self.action_low_np.shape[0]:
                raise ValueError(
                    f"student_action dim={student_arr.shape[0]} does not match action dim={self.action_low_np.shape[0]}."
                )
        prev_intervened = 1.0 if float(getattr(self, "last_intervention_probability", 0.0)) >= self.intervention_threshold else 0.0
        segment = np.concatenate([obs_arr, student_arr, np.asarray([prev_intervened], dtype=np.float32)], axis=0)
        self._history.append(segment.astype(np.float32, copy=False))
        segment_dim = int(self.segment_dim or segment.shape[0])
        features = np.zeros((self.context_len, segment_dim), dtype=np.float32)
        for idx, hist_segment in enumerate(list(self._history)[-self.context_len :]):
            features[self.context_len - len(self._history) + idx, : hist_segment.shape[0]] = hist_segment
        return features.reshape(1, -1)

    def get_action(self, obs: np.ndarray | None = None, env=None, student_action: np.ndarray | None = None) -> np.ndarray | None:
        if obs is None:
            return None
        torch = self._torch
        with torch.inference_mode():
            obs_t = torch.as_tensor(self._features(obs, student_action), device=self.device)
            obs_t = normalize_observations(obs_t, self.state["obs_mean"], self.state["obs_std"])
            action_unit, logit = self.model(obs_t)
            prob = torch.sigmoid(logit)[0].detach().cpu().item()
            self.last_intervention_probability = float(prob)
            if float(prob) < self.intervention_threshold:
                return None
            action_t = scale_unit_action(action_unit, self.state["action_low"], self.state["action_high"])
            action = action_t[0].detach().cpu().numpy().astype(np.float32)
        return np.clip(action, self.action_low_np, self.action_high_np).astype(np.float32, copy=False)

    def close(self) -> None:
        return None


class ScriptedLidarTeacherController:
    """Rule-based SafetyCar teacher from goal and obstacle lidar baskets."""

    always_active = True

    def __init__(
        self,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        align_tolerance_bins: int = 1,
        min_goal_signal: float = 1e-6,
    ):
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.align_tolerance_bins = int(max(0, align_tolerance_bins))
        self.min_goal_signal = float(max(0.0, min_goal_signal))
        self._flat_slices: dict[str, slice] | None = None
        self._lidar_keys: tuple[str, ...] = ()
        self._front_bin: int | None = None
        self._num_bins: int | None = None

    @staticmethod
    def _wheel_limit(action_low: np.ndarray, action_high: np.ndarray) -> float:
        low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if low.size < 2 or high.size < 2:
            return 1.0
        return float(max(np.max(np.abs(low[:2])), np.max(np.abs(high[:2])), 1.0))

    @staticmethod
    def _iter_wrappers(env):
        cur = env
        seen: set[int] = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            yield cur
            cur = getattr(cur, "env", None)

    def _adapt_raw_wheels_to_env(self, env, wheel_action: np.ndarray) -> np.ndarray:
        action = np.asarray(wheel_action, dtype=np.float32).reshape(-1)
        for wrapper in self._iter_wrappers(env):
            if hasattr(wrapper, "reverse_action") and getattr(wrapper, "action_mode", "raw_wheels") in {"throttle_turn", "cardinal"}:
                action = np.asarray(wrapper.reverse_action(action), dtype=np.float32).reshape(-1)
                break
        low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        if action.shape[0] < low.shape[0]:
            action = np.pad(action, (0, low.shape[0] - action.shape[0]), mode="constant")
        elif action.shape[0] > low.shape[0]:
            action = action[: low.shape[0]]
        return np.clip(action, low, high).astype(np.float32, copy=False)

    def _maybe_world_velocity_action(self, *, signed_offset: int, env) -> np.ndarray | None:
        for wrapper in self._iter_wrappers(env):
            if getattr(wrapper, "action_mode", "") == "world_velocity":
                agent_xy = extract_agent_xy(env)
                goal_xy = extract_goal_xy(env)
                if agent_xy is not None and goal_xy is not None:
                    desired = np.asarray(goal_xy - agent_xy, dtype=np.float32).reshape(-1)
                    norm = float(np.linalg.norm(desired))
                    if norm > 1e-6:
                        desired = desired / norm
                        low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
                        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
                        return np.clip(desired[: low.shape[0]], low, high).astype(np.float32, copy=False)
                forward = extract_agent_forward_xy(env)
                if forward is None:
                    return None
                num_bins = int(self._num_bins or 16)
                angle = float(signed_offset) * (2.0 * np.pi / max(1, num_bins))
                c = float(np.cos(angle))
                s = float(np.sin(angle))
                fx = float(forward[0])
                fy = float(forward[1])
                desired = np.asarray([c * fx - s * fy, s * fx + c * fy], dtype=np.float32)
                norm = float(np.linalg.norm(desired))
                if norm > 1e-6:
                    desired = desired / norm
                low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
                high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
                return np.clip(desired[: low.shape[0]], low, high).astype(np.float32, copy=False)
        return None

    def _build_obs_slices(self, env) -> None:
        if self._flat_slices is not None:
            return
        base = unwrap_env(env)
        task = getattr(base, "task", None)
        obs_info = getattr(task, "obs_info", None)
        obs_space_dict = getattr(obs_info, "obs_space_dict", None)
        spaces = getattr(obs_space_dict, "spaces", None)
        if not isinstance(spaces, dict):
            raise TypeError("Scripted lidar teacher requires Dict-backed Safety-Gym observations.")
        offset = 0
        self._flat_slices = {}
        lidar_keys: list[str] = []
        for key, space in spaces.items():
            shape = tuple(int(v) for v in getattr(space, "shape", ()))
            width = int(np.prod(shape)) if shape else 1
            self._flat_slices[str(key)] = slice(offset, offset + width)
            offset += width
            if str(key).endswith("_lidar"):
                lidar_keys.append(str(key))
        self._lidar_keys = tuple(lidar_keys)
        goal_slice = self._flat_slices.get("goal_lidar")
        if goal_slice is None:
            raise KeyError("goal_lidar not found in Safety-Gym observation slices.")
        self._num_bins = int(goal_slice.stop - goal_slice.start)

    def _obs_components(self, obs: np.ndarray, env) -> tuple[np.ndarray, np.ndarray]:
        self._build_obs_slices(env)
        assert self._flat_slices is not None
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        goal_slice = self._flat_slices["goal_lidar"]
        goal = obs_arr[goal_slice].astype(np.float32, copy=True)
        merged = np.zeros_like(goal)
        for key in self._lidar_keys:
            if key == "goal_lidar":
                continue
            sl = self._flat_slices.get(key)
            if sl is None:
                continue
            merged = np.maximum(merged, obs_arr[sl].astype(np.float32, copy=False))
        return goal, merged

    def _ensure_front_bin(self, env) -> int:
        if self._front_bin is not None:
            return self._front_bin
        base = unwrap_env(env)
        task = getattr(base, "task", None)
        if task is None or not hasattr(task, "_obs_lidar"):
            raise RuntimeError("Scripted lidar teacher requires direct access to the Safety-Gym task.")
        forward = extract_agent_forward_xy(env)
        if forward is None:
            raise RuntimeError("Could not extract agent forward direction for scripted lidar teacher.")
        agent_xy = np.asarray(task.agent.pos[:2], dtype=np.float64)
        probe_xy = agent_xy + np.asarray(forward, dtype=np.float64) * 1.0
        probe_lidar = np.asarray(task._obs_lidar(np.asarray([probe_xy], dtype=np.float64), task.goal.group), dtype=np.float32)
        self._front_bin = int(np.argmax(probe_lidar))
        self._num_bins = int(probe_lidar.shape[0])
        return self._front_bin

    def _signed_bin_offset(self, target_bin: int, env) -> int:
        front_bin = self._ensure_front_bin(env)
        num_bins = int(self._num_bins or 16)
        half = num_bins // 2
        return int(((int(target_bin) - front_bin + half) % num_bins) - half)

    def _select_target_bin(self, goal_lidar: np.ndarray, obstacle_lidar: np.ndarray, env) -> tuple[int, dict[str, float]]:
        goal = np.asarray(goal_lidar, dtype=np.float32).reshape(-1)
        obstacles = np.asarray(obstacle_lidar, dtype=np.float32).reshape(-1)
        goal_idx = int(np.argmax(goal))
        goal_peak = float(goal[goal_idx])
        num_bins = int(goal.shape[0])
        goal_threshold = max(goal_peak, self.min_goal_signal)
        clear_mask = obstacles < goal_threshold

        def circ_dist(idx: int) -> int:
            diff = abs(int(idx) - goal_idx)
            return min(diff, num_bins - diff)

        if bool(clear_mask[goal_idx]):
            target_idx = goal_idx
        else:
            candidates = [idx for idx in range(num_bins) if bool(clear_mask[idx])]
            if candidates:
                target_idx = min(candidates, key=lambda idx: (circ_dist(idx), -float(goal[idx]), float(obstacles[idx])))
            else:
                target_idx = min(range(num_bins), key=lambda idx: (float(obstacles[idx]), circ_dist(idx), -float(goal[idx])))
        diag = {
            "goal_idx": float(goal_idx),
            "goal_peak": float(goal_peak),
            "goal_obstacle": float(obstacles[goal_idx]),
            "target_idx": float(target_idx),
            "target_obstacle": float(obstacles[target_idx]),
            "target_goal_signal": float(goal[target_idx]),
        }
        return int(target_idx), diag

    def _wheel_action(self, *, signed_offset: int, wheel_limit: float) -> np.ndarray:
        lim = float(max(1e-6, wheel_limit))
        if abs(int(signed_offset)) <= self.align_tolerance_bins:
            return np.asarray([lim, lim], dtype=np.float32)
        if int(signed_offset) > 0:
            return np.asarray([lim, -lim], dtype=np.float32)
        return np.asarray([-lim, lim], dtype=np.float32)

    def get_action(self, obs: np.ndarray | None = None, env=None) -> np.ndarray | None:
        if obs is None or env is None:
            return None
        goal_lidar, obstacle_lidar = self._obs_components(obs, env)
        target_idx, _diag = self._select_target_bin(goal_lidar, obstacle_lidar, env)
        signed_offset = self._signed_bin_offset(target_idx, env)
        world_velocity_action = self._maybe_world_velocity_action(signed_offset=signed_offset, env=env)
        if world_velocity_action is not None:
            return world_velocity_action
        wheel_action = self._wheel_action(
            signed_offset=signed_offset,
            wheel_limit=self._wheel_limit(self.action_low, self.action_high),
        )
        return self._adapt_raw_wheels_to_env(env, wheel_action)

    def close(self) -> None:
        return None


class ScriptedGeometricTeacherController:
    """Rule-based SafetyCar teacher using true task geometry for demo generation."""

    always_active = True

    def __init__(
        self,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        heading_tolerance: float = 0.20,
        lookahead: float = 1.0,
        safety_margin: float = 0.18,
        grid_resolution: float = 0.08,
        emergency_clearance: float = 0.08,
        action_shield_steps: int = 1,
    ):
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.heading_tolerance = float(max(0.01, heading_tolerance))
        self.lookahead = float(max(0.1, lookahead))
        self.safety_margin = float(max(0.0, safety_margin))
        self.grid_resolution = float(max(0.03, grid_resolution))
        self.emergency_clearance = float(max(0.0, emergency_clearance))
        self.action_shield_steps = int(max(0, action_shield_steps))
        self._pos_history: deque[np.ndarray] = deque(maxlen=40)
        self._recovery_steps = 0
        self._recovery_turn_sign = 1.0

    @staticmethod
    def _iter_wrappers(env):
        cur = env
        seen: set[int] = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            yield cur
            cur = getattr(cur, "env", None)

    @staticmethod
    def _signed_angle(src: np.ndarray, dst: np.ndarray) -> float:
        src = np.asarray(src, dtype=np.float64).reshape(2)
        dst = np.asarray(dst, dtype=np.float64).reshape(2)
        src = src / max(1e-9, float(np.linalg.norm(src)))
        dst = dst / max(1e-9, float(np.linalg.norm(dst)))
        cross = float(src[0] * dst[1] - src[1] * dst[0])
        dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
        return float(np.arctan2(cross, dot))

    def _obstacles(self, env) -> list[tuple[np.ndarray, float]]:
        base = unwrap_env(env)
        task = getattr(base, "task", None)
        world_cfg = getattr(getattr(task, "world_info", None), "world_config_dict", None)
        obstacles: list[tuple[np.ndarray, float]] = []
        if task is None or not isinstance(world_cfg, dict):
            return obstacles
        for section in ("geoms", "free_geoms"):
            items = world_cfg.get(section, {})
            if not isinstance(items, dict):
                continue
            for name, cfg in items.items():
                label = str(name).lower()
                if not any(key in label for key in ("hazard", "vase", "pillar", "gremlin", "wall")):
                    continue
                try:
                    pos = np.asarray(task.data.body(str(name)).xpos[:2], dtype=np.float64).reshape(2)
                except Exception:
                    continue
                geom_list = cfg.get("geoms", []) if isinstance(cfg, dict) else []
                geom = geom_list[0] if geom_list else {}
                size = np.asarray(geom.get("size", [0.1]), dtype=np.float64).reshape(-1)
                if str(geom.get("type", "")).lower() == "box" and size.size >= 2:
                    radius = float(np.linalg.norm(size[:2]))
                else:
                    radius = float(size[0] if size.size else 0.1)
                obstacles.append((pos, radius + self.safety_margin))
        return obstacles

    def _score_direction(self, *, agent_xy: np.ndarray, goal_xy: np.ndarray, direction: np.ndarray, obstacles: list[tuple[np.ndarray, float]]) -> float:
        goal_vec = goal_xy - agent_xy
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist < 1e-6:
            return 1e6
        goal_dir = goal_vec / goal_dist
        direction = direction / max(1e-9, float(np.linalg.norm(direction)))
        score = 2.0 * float(np.dot(direction, goal_dir))
        score += 0.4 * max(0.0, goal_dist - float(np.linalg.norm(goal_xy - (agent_xy + 0.35 * direction))))
        for pos, radius in obstacles:
            rel = pos - agent_xy
            along = float(np.dot(rel, direction))
            if along <= 0.0 or along > self.lookahead:
                continue
            lateral = float(np.linalg.norm(rel - along * direction))
            clearance = lateral - radius
            if clearance < 0.0:
                score -= 8.0 + 8.0 * min(1.0, -clearance / max(1e-6, radius))
            elif clearance < 0.18:
                score -= 3.0 * (0.18 - clearance) / 0.18
        return float(score)

    def _bounds(self, env) -> tuple[float, float, float, float]:
        base = unwrap_env(env)
        task = getattr(base, "task", None)
        extents = getattr(getattr(task, "placements_conf", None), "extents", None)
        arr = None if extents is None else np.asarray(extents, dtype=np.float64).reshape(-1)
        if arr is not None and arr.size == 4 and np.isfinite(arr).all():
            x_min, y_min, x_max, y_max = map(float, arr.tolist())
        else:
            x_min, y_min, x_max, y_max = -1.5, -1.5, 1.5, 1.5
        pad = 0.05
        return x_min + pad, x_max - pad, y_min + pad, y_max - pad

    def _plan_direction(
        self,
        *,
        env,
        agent_xy: np.ndarray,
        goal_xy: np.ndarray,
        obstacles: list[tuple[np.ndarray, float]],
    ) -> np.ndarray | None:
        x_min, x_max, y_min, y_max = self._bounds(env)
        res = self.grid_resolution
        nx = int(max(8, np.ceil((x_max - x_min) / res))) + 1
        ny = int(max(8, np.ceil((y_max - y_min) / res))) + 1

        def to_idx(pos: np.ndarray) -> tuple[int, int]:
            ix = int(np.clip(round((float(pos[0]) - x_min) / res), 0, nx - 1))
            iy = int(np.clip(round((float(pos[1]) - y_min) / res), 0, ny - 1))
            return ix, iy

        def to_xy(idx: tuple[int, int]) -> np.ndarray:
            return np.asarray([x_min + idx[0] * res, y_min + idx[1] * res], dtype=np.float64)

        blocked = np.zeros((nx, ny), dtype=bool)
        for ix in range(nx):
            x = x_min + ix * res
            for iy in range(ny):
                y = y_min + iy * res
                p = np.asarray([x, y], dtype=np.float64)
                for pos, radius in obstacles:
                    if float(np.linalg.norm(p - pos)) <= float(radius):
                        blocked[ix, iy] = True
                        break

        start = to_idx(agent_xy)
        goal = to_idx(goal_xy)
        blocked[start] = False
        blocked[goal] = False

        import heapq

        def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
            return float(np.hypot(a[0] - b[0], a[1] - b[1]))

        neighbors = [
            (-1, -1, np.sqrt(2.0)),
            (-1, 0, 1.0),
            (-1, 1, np.sqrt(2.0)),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (1, -1, np.sqrt(2.0)),
            (1, 0, 1.0),
            (1, 1, np.sqrt(2.0)),
        ]
        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], float] = {start: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for dx, dy, step_cost in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if nxt[0] < 0 or nxt[0] >= nx or nxt[1] < 0 or nxt[1] >= ny or blocked[nxt]:
                    continue
                new_cost = cost_so_far[current] + float(step_cost)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + heuristic(nxt, goal)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current
        if goal not in came_from:
            return None

        path_idx: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = goal
        while cur is not None:
            path_idx.append(cur)
            cur = came_from.get(cur)
        path_idx.reverse()
        if len(path_idx) < 2:
            goal_vec = goal_xy - agent_xy
            return goal_vec / max(1e-9, float(np.linalg.norm(goal_vec)))

        target = to_xy(path_idx[-1])
        waypoint_distance = max(0.14, min(0.22, 3.0 * float(res)))
        for idx in path_idx[1:]:
            candidate = to_xy(idx)
            if float(np.linalg.norm(candidate - agent_xy)) >= waypoint_distance:
                target = candidate
                break
        direction = target - agent_xy
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return None
        return direction / norm

    def _desired_direction(self, env) -> np.ndarray | None:
        agent_xy = extract_agent_xy(env)
        goal_xy = extract_goal_xy(env)
        forward = extract_agent_forward_xy(env)
        if agent_xy is None or goal_xy is None or forward is None:
            return None
        agent_xy = np.asarray(agent_xy, dtype=np.float64).reshape(2)
        goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        forward = np.asarray(forward, dtype=np.float64).reshape(2)
        forward = forward / max(1e-9, float(np.linalg.norm(forward)))
        obstacles = self._obstacles(env)
        # Emergency recovery is handled in get_action() from rendered footprint
        # clearance. Using inflated planning circles here was too broad and led
        # to safe but stuck/timeouting behavior.
        use_legacy_inflated_emergency = False
        if use_legacy_inflated_emergency and self.emergency_clearance > 0.0 and obstacles:
            nearest_clearance = float("inf")
            nearest_away = None
            for pos, radius in obstacles:
                rel = agent_xy - pos
                dist = float(np.linalg.norm(rel))
                clearance = dist - float(radius)
                if clearance < nearest_clearance:
                    nearest_clearance = clearance
                    if dist > 1e-9:
                        nearest_away = rel / dist
            if nearest_away is not None and nearest_clearance < self.emergency_clearance:
                goal_vec = goal_xy - agent_xy
                goal_norm = float(np.linalg.norm(goal_vec))
                goal_dir = goal_vec / max(goal_norm, 1e-9)
                # Bias mostly away from the obstacle, with a small goal component
                # to avoid endless pure retreat when the path is only slightly tight.
                escape = 0.85 * nearest_away + 0.15 * goal_dir
                escape_norm = float(np.linalg.norm(escape))
                if escape_norm > 1e-9:
                    return escape / escape_norm
        planned = self._plan_direction(env=env, agent_xy=agent_xy, goal_xy=goal_xy, obstacles=obstacles)
        if planned is not None:
            return planned
        angles = np.linspace(-np.pi, np.pi, 33, endpoint=False)
        best_dir = None
        best_score = -float("inf")
        for angle in angles:
            c = float(np.cos(angle))
            s = float(np.sin(angle))
            direction = np.asarray([c * forward[0] - s * forward[1], s * forward[0] + c * forward[1]], dtype=np.float64)
            score = self._score_direction(agent_xy=agent_xy, goal_xy=goal_xy, direction=direction, obstacles=obstacles)
            if score > best_score:
                best_score = score
                best_dir = direction
        if best_dir is None:
            goal_vec = goal_xy - agent_xy
            norm = float(np.linalg.norm(goal_vec))
            return goal_vec / max(1e-9, norm)
        return best_dir / max(1e-9, float(np.linalg.norm(best_dir)))

    def _adapt_raw_wheels_to_env(self, env, wheel_action: np.ndarray) -> np.ndarray:
        action = np.asarray(wheel_action, dtype=np.float32).reshape(-1)
        for wrapper in self._iter_wrappers(env):
            if hasattr(wrapper, "reverse_action") and getattr(wrapper, "action_mode", "raw_wheels") in {"throttle_turn", "cardinal"}:
                action = np.asarray(wrapper.reverse_action(action), dtype=np.float32).reshape(-1)
                break
        low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        if action.shape[0] < low.shape[0]:
            action = np.pad(action, (0, low.shape[0] - action.shape[0]), mode="constant")
        elif action.shape[0] > low.shape[0]:
            action = action[: low.shape[0]]
        return np.clip(action, low, high).astype(np.float32, copy=False)

    def _visual_clearance(self, env) -> float:
        for wrapper in self._iter_wrappers(env):
            if hasattr(wrapper, "_visual_min_clearance"):
                try:
                    clearance = wrapper._visual_min_clearance()
                except TypeError:
                    continue
                if clearance is not None and np.isfinite(clearance):
                    return float(clearance)
        return float("nan")

    def _visual_clearance_and_escape(self, env) -> tuple[float, np.ndarray | None]:
        base = unwrap_env(env)
        task = getattr(base, "task", None)
        model = getattr(task, "model", None)
        data = getattr(task, "data", None)
        if task is None or model is None or data is None:
            return float("nan"), None
        try:
            from .wrappers import FootprintCostWrapper
        except Exception:
            return float("nan"), None

        constrained_bodies = {str(spec.get("body_name", "")) for spec in _task_constrained_object_specs(task)}
        constrained_bodies.discard("")
        agent_shapes = []
        obstacle_shapes = []
        for geom_id in range(int(getattr(model, "ngeom", 0))):
            body_id = int(model.geom_bodyid[geom_id])
            body_name = FootprintCostWrapper._body_name(model, body_id)
            geom_name = FootprintCostWrapper._geom_name(model, geom_id)
            shape = FootprintCostWrapper._geom_to_shape(model, data, geom_id)
            if shape is None:
                continue
            if body_name in {"agent", "left", "right", "rear"} or geom_name in {
                "agent",
                "back_bumper",
                "back_connector",
                "front_bumper",
                "front_connector",
            }:
                agent_shapes.append(shape)
            elif body_name in constrained_bodies:
                obstacle_shapes.append(shape)
        if not agent_shapes or not obstacle_shapes:
            return float("nan"), None

        best_clearance = float("inf")
        best_away = None
        for agent_shape in agent_shapes:
            for obstacle_shape in obstacle_shapes:
                clearance = FootprintCostWrapper._shape_clearance(agent_shape, obstacle_shape)
                if clearance >= best_clearance:
                    continue
                away = np.asarray(agent_shape.center, dtype=np.float64) - np.asarray(
                    obstacle_shape.center,
                    dtype=np.float64,
                )
                norm = float(np.linalg.norm(away))
                if norm <= 1e-9:
                    continue
                best_clearance = float(clearance)
                best_away = away / norm
        return (float(best_clearance) if np.isfinite(best_clearance) else float("nan")), best_away

    def _candidate_raw_wheels(self, desired: np.ndarray, lim: float) -> list[np.ndarray]:
        desired = np.asarray(desired, dtype=np.float32).reshape(-1)
        candidates = [
            desired,
            np.asarray([0.35 * lim, 0.35 * lim], dtype=np.float32),
            np.asarray([lim, -lim], dtype=np.float32),
            np.asarray([-lim, lim], dtype=np.float32),
            np.asarray([0.3 * lim, -0.8 * lim], dtype=np.float32),
            np.asarray([-0.8 * lim, 0.3 * lim], dtype=np.float32),
            np.asarray([0.6 * lim, -0.2 * lim], dtype=np.float32),
            np.asarray([-0.2 * lim, 0.6 * lim], dtype=np.float32),
            np.asarray([-0.35 * lim, -0.35 * lim], dtype=np.float32),
            np.asarray([0.0, 0.0], dtype=np.float32),
        ]
        unique: list[np.ndarray] = []
        for cand in candidates:
            arr = np.clip(np.asarray(cand, dtype=np.float32).reshape(-1), self.action_low[:2], self.action_high[:2])
            if not any(np.allclose(arr, prev, atol=1e-6) for prev in unique):
                unique.append(arr)
        return unique

    def _shield_raw_wheels(self, env, desired_wheel: np.ndarray) -> np.ndarray:
        if self.action_shield_steps <= 0:
            return np.asarray(desired_wheel, dtype=np.float32)
        base = unwrap_env(env)
        task = getattr(base, "task", None)
        model = getattr(task, "model", None)
        data = getattr(task, "data", None)
        agent = getattr(task, "agent", None)
        if task is None or model is None or data is None or agent is None:
            return np.asarray(desired_wheel, dtype=np.float32)
        if int(getattr(model, "nu", 0)) != 2:
            return np.asarray(desired_wheel, dtype=np.float32)

        try:
            import mujoco
        except Exception:
            return np.asarray(desired_wheel, dtype=np.float32)

        qpos = np.asarray(data.qpos).copy()
        qvel = np.asarray(data.qvel).copy()
        ctrl = np.asarray(data.ctrl).copy()
        time_before = float(data.time)
        act = np.asarray(data.act).copy() if getattr(data, "act", None) is not None and np.asarray(data.act).size else None
        mocap_pos = np.asarray(data.mocap_pos).copy() if getattr(data, "mocap_pos", None) is not None and np.asarray(data.mocap_pos).size else None
        mocap_quat = np.asarray(data.mocap_quat).copy() if getattr(data, "mocap_quat", None) is not None and np.asarray(data.mocap_quat).size else None

        start_xy = extract_agent_xy(env)
        goal_xy = extract_goal_xy(env)
        start_dist = float(np.linalg.norm(np.asarray(goal_xy) - np.asarray(start_xy))) if start_xy is not None and goal_xy is not None else float("nan")
        start_clearance = self._visual_clearance(env)
        clearance_priority = (
            np.isfinite(start_clearance)
            and float(start_clearance) < max(0.25, float(self.emergency_clearance))
        )
        lim = ScriptedLidarTeacherController._wheel_limit(self.action_low, self.action_high)
        candidates = self._candidate_raw_wheels(np.asarray(desired_wheel, dtype=np.float32), lim)
        frames = 1
        sim_conf = getattr(task, "sim_conf", None)
        try:
            frames = int(sim_conf.frameskip_binom_n)
        except Exception:
            frames = 1
        segment_frames = int(max(1, frames))
        shield_segments = int(max(1, self.action_shield_steps))

        best_action = np.asarray(desired_wheel, dtype=np.float32)
        best_score = -float("inf")
        if clearance_priority and shield_segments > 1:
            sequences = [(first, second) for first in candidates for second in candidates]
        else:
            sequences = [(cand,) for cand in candidates]
        for seq in sequences:
            try:
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                data.ctrl[:] = ctrl
                data.time = time_before
                if act is not None:
                    data.act[:] = act
                if mocap_pos is not None:
                    data.mocap_pos[:] = mocap_pos
                if mocap_quat is not None:
                    data.mocap_quat[:] = mocap_quat
                mujoco.mj_forward(model, data)
                min_clearance = self._visual_clearance(env)
                for cand in seq:
                    agent.apply_action(cand)
                    for _ in range(segment_frames):
                        mujoco.mj_step(model, data)
                        step_clearance = self._visual_clearance(env)
                        if np.isfinite(step_clearance) and (
                            not np.isfinite(min_clearance) or float(step_clearance) < float(min_clearance)
                        ):
                            min_clearance = float(step_clearance)
                clearance = float(min_clearance)
                after_xy = extract_agent_xy(env)
                after_dist = (
                    float(np.linalg.norm(np.asarray(goal_xy) - np.asarray(after_xy)))
                    if after_xy is not None and goal_xy is not None
                    else float("nan")
                )
                progress = (start_dist - after_dist) if np.isfinite(start_dist) and np.isfinite(after_dist) else 0.0
                # This is a veto-style shield, not a clearance maximizer:
                # reject predicted visual overlap, otherwise choose the action
                # that still makes task progress. Over-valuing clearance caused
                # safe but useless orbiting/timeouts near obstacles.
                clearance_f = float(clearance if np.isfinite(clearance) else -1.0)
                first_action = np.asarray(seq[0], dtype=np.float32)
                forward_bias = float(first_action[0] + first_action[1]) / max(1e-6, 2.0 * lim)
                turn_mag = float(abs(first_action[0] - first_action[1])) / max(1e-6, 2.0 * lim)
                if clearance_priority:
                    # Near rendered contact, prefer any action that increases
                    # clearance even if it sacrifices goal progress.
                    score = 300.0 * float(np.clip(clearance_f, -0.10, 0.40))
                    score += 8.0 * float(progress)
                    score += 0.1 * forward_bias
                    score -= 0.05 * turn_mag
                    if np.isfinite(clearance) and clearance <= 0.0:
                        score -= 2000.0
                else:
                    score = 80.0 * float(progress)
                    score += 10.0 * float(np.clip(clearance_f, -0.05, 0.20))
                    score += 0.4 * forward_bias
                    score -= 0.1 * turn_mag
                    if np.isfinite(clearance) and clearance <= 0.03:
                        score -= 1000.0
                if score > best_score:
                    best_score = score
                    best_action = np.asarray(first_action, dtype=np.float32)
            except Exception:
                continue
            finally:
                try:
                    data.qpos[:] = qpos
                    data.qvel[:] = qvel
                    data.ctrl[:] = ctrl
                    data.time = time_before
                    if act is not None:
                        data.act[:] = act
                    if mocap_pos is not None:
                        data.mocap_pos[:] = mocap_pos
                    if mocap_quat is not None:
                        data.mocap_quat[:] = mocap_quat
                    mujoco.mj_forward(model, data)
                except Exception:
                    pass
        return np.asarray(best_action, dtype=np.float32)

    def get_action(self, obs: np.ndarray | None = None, env=None) -> np.ndarray | None:
        if env is None:
            return None
        forward = extract_agent_forward_xy(env)
        visual_clearance, visual_escape = self._visual_clearance_and_escape(env)
        desired = self._desired_direction(env)
        if forward is None or desired is None:
            return None
        agent_xy = extract_agent_xy(env)
        goal_xy = extract_goal_xy(env)
        if agent_xy is not None:
            cur_xy = np.asarray(agent_xy, dtype=np.float64).reshape(2)
            if self._pos_history and float(np.linalg.norm(cur_xy - self._pos_history[-1])) > 0.75:
                self._pos_history.clear()
                self._recovery_steps = 0
            self._pos_history.append(cur_xy.copy())
            if self._recovery_steps <= 0 and len(self._pos_history) >= self._pos_history.maxlen:
                displacement = float(np.linalg.norm(cur_xy - self._pos_history[0]))
                if displacement < 0.08:
                    self._recovery_steps = 35
                    if visual_escape is not None and forward is not None:
                        escape_angle = self._signed_angle(
                            np.asarray(forward, dtype=np.float64),
                            np.asarray(visual_escape, dtype=np.float64),
                        )
                        self._recovery_turn_sign = 1.0 if escape_angle >= 0.0 else -1.0
                    else:
                        self._recovery_turn_sign *= -1.0
        if (
            self.emergency_clearance > 0.0
            and visual_escape is not None
            and goal_xy is not None
            and agent_xy is not None
            and np.isfinite(visual_clearance)
            and float(visual_clearance) < max(0.35, 1.4 * float(self.safety_margin))
            and float(visual_clearance) >= self.emergency_clearance
        ):
            # Before rendered contact is imminent, only override the planned
            # direction if it still points into the obstacle. A previous
            # always-wall-following blend was safe but caused near-permanent
            # intervention/timeouts in cluttered layouts.
            away = np.asarray(visual_escape, dtype=np.float64).reshape(2)
            away = away / max(1e-9, float(np.linalg.norm(away)))
            goal_vec = np.asarray(goal_xy, dtype=np.float64).reshape(2) - np.asarray(agent_xy, dtype=np.float64).reshape(2)
            goal_dir = goal_vec / max(1e-9, float(np.linalg.norm(goal_vec)))
            desired_unit = np.asarray(desired, dtype=np.float64).reshape(2)
            desired_unit = desired_unit / max(1e-9, float(np.linalg.norm(desired_unit)))
            if float(np.dot(desired_unit, away)) < 0.10:
                tangent_l = np.asarray([-away[1], away[0]], dtype=np.float64)
                tangent_r = np.asarray([away[1], -away[0]], dtype=np.float64)
                tangent = tangent_l if float(np.dot(tangent_l, goal_dir)) >= float(np.dot(tangent_r, goal_dir)) else tangent_r
                blended = 0.25 * away + 0.70 * tangent + 0.45 * goal_dir
                blended_norm = float(np.linalg.norm(blended))
                if blended_norm > 1e-9:
                    desired = blended / blended_norm

        angle = self._signed_angle(np.asarray(forward, dtype=np.float64), np.asarray(desired, dtype=np.float64))
        lim = ScriptedLidarTeacherController._wheel_limit(self.action_low, self.action_high)
        if self._recovery_steps > 0 and not (
            self.emergency_clearance > 0.0
            and visual_escape is not None
            and np.isfinite(visual_clearance)
            and float(visual_clearance) < self.emergency_clearance
        ):
            self._recovery_steps -= 1
            turn = float(self._recovery_turn_sign)
            throttle = -0.45
            wheel = np.asarray([throttle - 0.65 * turn, throttle + 0.65 * turn], dtype=np.float32) * lim
        elif (
            self.emergency_clearance > 0.0
            and visual_escape is not None
            and np.isfinite(visual_clearance)
            and float(visual_clearance) < self.emergency_clearance
        ):
            escape_angle = self._signed_angle(np.asarray(forward, dtype=np.float64), np.asarray(visual_escape, dtype=np.float64))
            alignment = float(np.dot(
                np.asarray(forward, dtype=np.float64) / max(1e-9, float(np.linalg.norm(forward))),
                np.asarray(visual_escape, dtype=np.float64) / max(1e-9, float(np.linalg.norm(visual_escape))),
            ))
            if abs(escape_angle) > self.heading_tolerance:
                # Close to rendered contact, side-scrapes are more dangerous
                # than timeouts. Rotate first, then translate away.
                wheel = (
                    np.asarray([lim, -lim], dtype=np.float32)
                    if escape_angle > 0.0
                    else np.asarray([-lim, lim], dtype=np.float32)
                )
            else:
                turn = float(np.clip(-escape_angle / max(self.heading_tolerance, 1e-6), -0.4, 0.4))
                throttle = 0.4 if alignment >= 0.0 else -0.35
                wheel = np.asarray([throttle - turn, throttle + turn], dtype=np.float32) * lim
        elif abs(angle) > self.heading_tolerance:
            # Positive angle means the target is to the left; SafetyCar left-turn
            # wheel convention is left wheel forward, right wheel backward.
            wheel = np.asarray([lim, -lim], dtype=np.float32) if angle > 0.0 else np.asarray([-lim, lim], dtype=np.float32)
        else:
            turn = float(np.clip(-angle / self.heading_tolerance, -1.0, 1.0))
            throttle = 0.5
            wheel = np.asarray([throttle - turn, throttle + turn], dtype=np.float32) * lim
        wheel = self._shield_raw_wheels(env, wheel)
        return self._adapt_raw_wheels_to_env(env, wheel)

    def close(self) -> None:
        return None


class LegacyScriptedGeometricTeacherController(ScriptedGeometricTeacherController):
    """May-2026 geometric teacher before visual clearance and action shielding.

    The historical native-center-cost reference was collected before the
    visual-footprint recovery logic was added.  Keep this isolated so old
    benchmarks can be reproduced without changing the newer controller.
    """

    def __init__(
        self,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        heading_tolerance: float = 0.20,
        lookahead: float = 1.0,
        safety_margin: float = 0.18,
        grid_resolution: float = 0.08,
        emergency_clearance: float = 0.08,
        action_shield_steps: int = 0,
    ):
        # Do not call the newer initializer: its history/recovery state is
        # coupled to visual-footprint clearance that did not exist in May.
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.heading_tolerance = float(max(0.01, heading_tolerance))
        self.lookahead = float(max(0.1, lookahead))
        self.safety_margin = float(max(0.0, safety_margin))
        self.grid_resolution = float(max(0.03, grid_resolution))
        self.emergency_clearance = float(max(0.0, emergency_clearance))
        self.action_shield_steps = 0

    def get_action(self, obs: np.ndarray | None = None, env=None) -> np.ndarray | None:
        if env is None:
            return None
        forward = extract_agent_forward_xy(env)
        desired = self._desired_direction(env)
        if forward is None or desired is None:
            return None
        angle = self._signed_angle(np.asarray(forward, dtype=np.float64), np.asarray(desired, dtype=np.float64))
        lim = ScriptedLidarTeacherController._wheel_limit(self.action_low, self.action_high)
        if abs(angle) > self.heading_tolerance:
            wheel = np.asarray([lim, -lim], dtype=np.float32) if angle > 0.0 else np.asarray([-lim, lim], dtype=np.float32)
        else:
            turn = float(np.clip(-angle / self.heading_tolerance, -1.0, 1.0))
            wheel = np.asarray([0.5 - turn, 0.5 + turn], dtype=np.float32) * lim
        return self._adapt_raw_wheels_to_env(env, wheel)


class ScriptedVisualMpcTeacherController(ScriptedGeometricTeacherController):
    """Short-horizon visual-footprint MPC teacher for SafetyCar diagnostics.

    This teacher is intentionally privileged: it probes candidate raw wheel
    actions in the live MuJoCo state, restores the state, and selects the action
    that trades off goal progress, heading alignment, and rendered-footprint
    clearance. It is meant as a safety-supervisor diagnostic, not as a deployable
    policy.
    """

    def __init__(self, *args, horizon_steps: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon_steps = int(max(1, horizon_steps))

    @staticmethod
    def _copy_optional_array(value):
        arr = getattr(value, "copy", None)
        return arr() if callable(arr) else None

    def _restore_state(self, *, model, data, qpos, qvel, ctrl, time_before, act, mocap_pos, mocap_quat) -> None:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        data.time = time_before
        if act is not None:
            data.act[:] = act
        if mocap_pos is not None:
            data.mocap_pos[:] = mocap_pos
        if mocap_quat is not None:
            data.mocap_quat[:] = mocap_quat
        try:
            import mujoco

            mujoco.mj_forward(model, data)
        except Exception:
            pass

    def _candidate_actions(self, env, desired_wheel: np.ndarray) -> list[np.ndarray]:
        lim = ScriptedLidarTeacherController._wheel_limit(self.action_low, self.action_high)
        base = self._candidate_raw_wheels(np.asarray(desired_wheel, dtype=np.float32), lim)
        extra = [
            np.asarray([0.15 * lim, 0.15 * lim], dtype=np.float32),
            np.asarray([0.45 * lim, 0.05 * lim], dtype=np.float32),
            np.asarray([0.05 * lim, 0.45 * lim], dtype=np.float32),
            np.asarray([-0.20 * lim, -0.20 * lim], dtype=np.float32),
        ]
        unique: list[np.ndarray] = []
        for cand in [*base, *extra]:
            arr = np.clip(np.asarray(cand, dtype=np.float32).reshape(-1), self.action_low[:2], self.action_high[:2])
            if not any(np.allclose(arr, prev, atol=1e-6) for prev in unique):
                unique.append(arr)
        return unique

    def _score_candidate_rollout(
        self,
        *,
        env,
        action: np.ndarray,
        start_dist: float,
        goal_xy: np.ndarray | None,
        start_clearance: float,
        frames_per_step: int,
        model,
        data,
        agent,
    ) -> float:
        min_clearance = self._visual_clearance(env)
        horizon = int(max(1, self.horizon_steps))
        try:
            import mujoco

            for _ in range(horizon):
                agent.apply_action(action)
                for _ in range(int(max(1, frames_per_step))):
                    mujoco.mj_step(model, data)
                    clearance = self._visual_clearance(env)
                    if np.isfinite(clearance) and (
                        not np.isfinite(min_clearance) or float(clearance) < float(min_clearance)
                    ):
                        min_clearance = float(clearance)
        except Exception:
            return -float("inf")

        after_xy = extract_agent_xy(env)
        after_forward = extract_agent_forward_xy(env)
        after_dist = (
            float(np.linalg.norm(np.asarray(goal_xy, dtype=np.float64) - np.asarray(after_xy, dtype=np.float64)))
            if goal_xy is not None and after_xy is not None
            else float("nan")
        )
        progress = (start_dist - after_dist) if np.isfinite(start_dist) and np.isfinite(after_dist) else 0.0
        clearance_f = float(min_clearance) if np.isfinite(min_clearance) else -1.0
        score = 45.0 * float(progress)
        if np.isfinite(after_dist):
            score -= 1.5 * float(after_dist)
        if goal_xy is not None and after_xy is not None and after_forward is not None:
            goal_vec = np.asarray(goal_xy, dtype=np.float64) - np.asarray(after_xy, dtype=np.float64)
            goal_norm = float(np.linalg.norm(goal_vec))
            fwd = np.asarray(after_forward, dtype=np.float64).reshape(2)
            fwd = fwd / max(1e-9, float(np.linalg.norm(fwd)))
            if goal_norm > 1e-9:
                score += 1.2 * float(np.dot(fwd, goal_vec / goal_norm))

        # Hardly tolerate predicted rendered-footprint contact. The small
        # positive buffer handles numerical/rendering mismatch between probe and
        # executed step.
        if clearance_f <= 0.02:
            score -= 10000.0 + 1000.0 * float(0.02 - clearance_f)
        else:
            target_clearance = max(0.08, float(self.emergency_clearance))
            if clearance_f < target_clearance:
                score -= 180.0 * float(target_clearance - clearance_f)
            if np.isfinite(start_clearance) and clearance_f > float(start_clearance):
                score += 4.0 * float(min(0.2, clearance_f - float(start_clearance)))
            score += 0.3 * float(min(0.25, clearance_f))

        forward_bias = float(action[0] + action[1]) / max(
            1e-6,
            2.0 * ScriptedLidarTeacherController._wheel_limit(self.action_low, self.action_high),
        )
        score += 0.05 * forward_bias
        return float(score)

    def get_action(self, obs: np.ndarray | None = None, env=None) -> np.ndarray | None:
        if env is None:
            return None
        desired_direction = self._desired_direction(env)
        forward = extract_agent_forward_xy(env)
        if desired_direction is None or forward is None:
            return super().get_action(obs=obs, env=env)

        angle = self._signed_angle(np.asarray(forward, dtype=np.float64), np.asarray(desired_direction, dtype=np.float64))
        lim = ScriptedLidarTeacherController._wheel_limit(self.action_low, self.action_high)
        if abs(angle) > self.heading_tolerance:
            desired_wheel = np.asarray([lim, -lim], dtype=np.float32) if angle > 0.0 else np.asarray([-lim, lim], dtype=np.float32)
        else:
            turn = float(np.clip(-angle / self.heading_tolerance, -1.0, 1.0))
            throttle = 0.5
            desired_wheel = np.asarray([throttle - turn, throttle + turn], dtype=np.float32) * lim

        base = unwrap_env(env)
        task = getattr(base, "task", None)
        model = getattr(task, "model", None)
        data = getattr(task, "data", None)
        agent = getattr(task, "agent", None)
        if task is None or model is None or data is None or agent is None or int(getattr(model, "nu", 0)) != 2:
            return self._adapt_raw_wheels_to_env(env, desired_wheel)
        try:
            import mujoco  # noqa: F401
        except Exception:
            return self._adapt_raw_wheels_to_env(env, desired_wheel)

        qpos = np.asarray(data.qpos).copy()
        qvel = np.asarray(data.qvel).copy()
        ctrl = np.asarray(data.ctrl).copy()
        time_before = float(data.time)
        act = np.asarray(data.act).copy() if getattr(data, "act", None) is not None and np.asarray(data.act).size else None
        mocap_pos = np.asarray(data.mocap_pos).copy() if getattr(data, "mocap_pos", None) is not None and np.asarray(data.mocap_pos).size else None
        mocap_quat = np.asarray(data.mocap_quat).copy() if getattr(data, "mocap_quat", None) is not None and np.asarray(data.mocap_quat).size else None

        start_xy = extract_agent_xy(env)
        goal_xy = extract_goal_xy(env)
        start_dist = (
            float(np.linalg.norm(np.asarray(goal_xy, dtype=np.float64) - np.asarray(start_xy, dtype=np.float64)))
            if goal_xy is not None and start_xy is not None
            else float("nan")
        )
        start_clearance = self._visual_clearance(env)
        try:
            sim_conf = getattr(task, "sim_conf", None)
            frames = int(max(1, getattr(sim_conf, "frameskip_binom_n", 1)))
        except Exception:
            frames = 1

        best_action = np.asarray(desired_wheel, dtype=np.float32)
        best_score = -float("inf")
        for cand in self._candidate_actions(env, desired_wheel):
            try:
                self._restore_state(
                    model=model,
                    data=data,
                    qpos=qpos,
                    qvel=qvel,
                    ctrl=ctrl,
                    time_before=time_before,
                    act=act,
                    mocap_pos=mocap_pos,
                    mocap_quat=mocap_quat,
                )
                score = self._score_candidate_rollout(
                    env=env,
                    action=np.asarray(cand, dtype=np.float32),
                    start_dist=start_dist,
                    goal_xy=np.asarray(goal_xy, dtype=np.float64) if goal_xy is not None else None,
                    start_clearance=start_clearance,
                    frames_per_step=frames,
                    model=model,
                    data=data,
                    agent=agent,
                )
                if score > best_score:
                    best_score = float(score)
                    best_action = np.asarray(cand, dtype=np.float32)
            finally:
                try:
                    self._restore_state(
                        model=model,
                        data=data,
                        qpos=qpos,
                        qvel=qvel,
                        ctrl=ctrl,
                        time_before=time_before,
                        act=act,
                        mocap_pos=mocap_pos,
                        mocap_quat=mocap_quat,
                    )
                except Exception:
                    pass
        return self._adapt_raw_wheels_to_env(env, best_action)


class ScriptedGoalGeometricController:
    """Naive no-safety controller: turn to the goal center, then drive forward."""

    always_active = True

    def __init__(
        self,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        heading_tolerance: float = 0.08,
        forward_throttle: float = 0.9,
        turn_throttle: float = 1.0,
    ):
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.heading_tolerance = float(max(0.005, heading_tolerance))
        self.forward_throttle = float(max(0.0, forward_throttle))
        self.turn_throttle = float(max(0.0, turn_throttle))

    @staticmethod
    def _iter_wrappers(env):
        cur = env
        seen: set[int] = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            yield cur
            cur = getattr(cur, "env", None)

    @staticmethod
    def _signed_angle(src: np.ndarray, dst: np.ndarray) -> float:
        src = np.asarray(src, dtype=np.float64).reshape(2)
        dst = np.asarray(dst, dtype=np.float64).reshape(2)
        src = src / max(1e-9, float(np.linalg.norm(src)))
        dst = dst / max(1e-9, float(np.linalg.norm(dst)))
        cross = float(src[0] * dst[1] - src[1] * dst[0])
        dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
        return float(np.arctan2(cross, dot))

    def _adapt_raw_wheels_to_env(self, env, wheel_action: np.ndarray) -> np.ndarray:
        action = np.asarray(wheel_action, dtype=np.float32).reshape(-1)
        for wrapper in self._iter_wrappers(env):
            if hasattr(wrapper, "reverse_action") and getattr(wrapper, "action_mode", "raw_wheels") in {"throttle_turn", "cardinal"}:
                action = np.asarray(wrapper.reverse_action(action), dtype=np.float32).reshape(-1)
                break
        low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        if action.shape[0] < low.shape[0]:
            action = np.pad(action, (0, low.shape[0] - action.shape[0]), mode="constant")
        elif action.shape[0] > low.shape[0]:
            action = action[: low.shape[0]]
        return np.clip(action, low, high).astype(np.float32, copy=False)

    def get_action(self, obs: np.ndarray | None = None, env=None) -> np.ndarray | None:
        if env is None:
            return None
        agent_xy = extract_agent_xy(env)
        goal_xy = extract_goal_xy(env)
        forward = extract_agent_forward_xy(env)
        if agent_xy is None or goal_xy is None or forward is None:
            return None
        agent_xy = np.asarray(agent_xy, dtype=np.float64).reshape(2)
        goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        forward = np.asarray(forward, dtype=np.float64).reshape(2)
        desired = goal_xy - agent_xy
        if float(np.linalg.norm(desired)) < 1e-9:
            wheel = np.zeros((2,), dtype=np.float32)
            return self._adapt_raw_wheels_to_env(env, wheel)

        angle = self._signed_angle(forward, desired)
        lim = ScriptedLidarTeacherController._wheel_limit(self.action_low, self.action_high)
        if abs(angle) > self.heading_tolerance:
            turn = self.turn_throttle * lim
            wheel = np.asarray([turn, -turn], dtype=np.float32) if angle > 0.0 else np.asarray([-turn, turn], dtype=np.float32)
        else:
            throttle = self.forward_throttle * lim
            # Small proportional correction prevents dithering after alignment.
            correction = float(np.clip(-angle / self.heading_tolerance, -0.35, 0.35)) * throttle
            wheel = np.asarray([throttle - correction, throttle + correction], dtype=np.float32)
        return self._adapt_raw_wheels_to_env(env, wheel)

    def close(self) -> None:
        return None


class LearnedGateScriptedGeometricTeacherController:
    """Use a learned intervention head as the gate and scripted geometry for movement."""

    always_active = True

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        action_low: np.ndarray,
        action_high: np.ndarray,
        intervention_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.gate = LearnedInterventionPolicyController(
            checkpoint_path=checkpoint_path,
            action_low=action_low,
            action_high=action_high,
            intervention_threshold=intervention_threshold,
            device=device,
        )
        self.movement = ScriptedGeometricTeacherController(
            action_low=action_low,
            action_high=action_high,
        )
        self.last_intervention_probability = 0.0

    def reset(self) -> None:
        self.gate.reset()
        self.last_intervention_probability = 0.0

    def get_action(self, obs: np.ndarray | None = None, env=None, student_action: np.ndarray | None = None) -> np.ndarray | None:
        gate_action = self.gate.get_action(obs=obs, env=env, student_action=student_action)
        self.last_intervention_probability = float(getattr(self.gate, "last_intervention_probability", 0.0))
        if gate_action is None:
            return None
        return self.movement.get_action(obs=obs, env=env)

    def close(self) -> None:
        self.gate.close()
        self.movement.close()


class FlowInterventionPolicyController:
    """Flow-matching action teacher with a learned intervention head."""

    always_active = True

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        action_low: np.ndarray,
        action_high: np.ndarray,
        intervention_threshold: float = 0.5,
        device: str = "cpu",
        sample_steps: int = 12,
        num_samples: int = 8,
        sample_noise_scale: float = 1.0,
        sample_selector: str = "first",
    ):
        import torch

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.model, self.state = load_flow_imitation_checkpoint(self.checkpoint_path, device=torch.device(device))
        self.device = torch.device(device)
        self.intervention_threshold = float(intervention_threshold)
        self.action_low_np = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high_np = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self._torch = torch
        meta = self.state.get("metadata", {})
        self.sample_steps = int(max(1, meta.get("eval_sample_steps", sample_steps)))
        self.num_samples = int(max(1, meta.get("eval_num_samples", num_samples)))
        self.sample_noise_scale = float(meta.get("eval_sample_noise_scale", sample_noise_scale))
        self.sample_selector = str(meta.get("eval_sample_selector", sample_selector))
        self.context_len = int(max(1, meta.get("context_len", 1)))
        self.env_obs_dim = int(meta.get("env_obs_dim", meta.get("obs_dim", 0)) or 0)
        self.segment_dim = int(meta.get("segment_dim", 0) or 0)
        self._history: deque[np.ndarray] = deque(maxlen=self.context_len)
        if int(meta.get("act_dim", self.action_low_np.shape[0])) != int(self.action_low_np.shape[0]):
            raise ValueError(
                f"Flow teacher checkpoint act_dim={meta.get('act_dim')} does not match env action dim={self.action_low_np.shape[0]}."
            )

    def reset(self) -> None:
        self._history.clear()
        self.last_intervention_probability = 0.0

    def _features(self, obs: np.ndarray, student_action: np.ndarray | None) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.env_obs_dim > 0 and obs_arr.shape[0] != self.env_obs_dim:
            raise ValueError(
                f"Flow teacher checkpoint env_obs_dim={self.env_obs_dim} does not match obs dim={obs_arr.shape[0]}."
            )
        if student_action is None:
            student_arr = np.zeros_like(self.action_low_np, dtype=np.float32)
        else:
            student_arr = np.asarray(student_action, dtype=np.float32).reshape(-1)
            if student_arr.shape[0] != self.action_low_np.shape[0]:
                raise ValueError(
                    f"student_action dim={student_arr.shape[0]} does not match action dim={self.action_low_np.shape[0]}."
                )
        prev_intervened = 1.0 if float(getattr(self, "last_intervention_probability", 0.0)) >= self.intervention_threshold else 0.0
        segment = np.concatenate([obs_arr, student_arr, np.asarray([prev_intervened], dtype=np.float32)], axis=0)
        self._history.append(segment.astype(np.float32, copy=False))
        segment_dim = int(self.segment_dim or segment.shape[0])
        features = np.zeros((self.context_len, segment_dim), dtype=np.float32)
        hist = list(self._history)[-self.context_len :]
        start = self.context_len - len(hist)
        for idx, hist_segment in enumerate(hist):
            features[start + idx, : hist_segment.shape[0]] = hist_segment
        return features.reshape(1, -1)

    def get_action(self, obs: np.ndarray | None = None, env=None, student_action: np.ndarray | None = None) -> np.ndarray | None:
        if obs is None:
            return None
        torch = self._torch
        with torch.inference_mode():
            features = torch.as_tensor(self._features(obs, student_action), device=self.device, dtype=torch.float32)
            obs_norm = normalize_observations(features, self.state["obs_mean"], self.state["obs_std"])
            logit = self.model.gate_logit(obs_norm)
            prob = torch.sigmoid(logit)[0].detach().cpu().item()
            self.last_intervention_probability = float(prob)
            if float(prob) < self.intervention_threshold:
                return None
            action_t = flow_action_to_env(
                model=self.model,
                state=self.state,
                obs_features=features,
                sample_steps=self.sample_steps,
                num_samples=self.num_samples,
                noise_scale=self.sample_noise_scale,
                selector=self.sample_selector,
            )
            action = action_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
        return np.clip(action, self.action_low_np, self.action_high_np).astype(np.float32, copy=False)

    def close(self) -> None:
        return None


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
    expert_config_path: str = "",
    expert_safe_checkpoint_path: str = "",
    expert_switch_clearance_threshold: float = 0.08,
    learned_intervention_threshold: float = 0.5,
    expert_device: str = "cpu",
    show_overlay: bool = True,
    prefer_separate_keyboard_window: bool = False,
    control_scheme_override: str | None = None,
    scripted_geo_heading_tolerance: float = 0.20,
    scripted_geo_lookahead: float = 1.0,
    scripted_geo_safety_margin: float = 0.18,
    scripted_geo_grid_resolution: float = 0.08,
    scripted_geo_emergency_clearance: float = 0.08,
    scripted_geo_action_shield_steps: int = 1,
):
    input_device = str(input_device).lower()
    control_scheme = str(control_scheme_override).strip() if control_scheme_override else infer_control_scheme(env_name)
    if input_device == "scripted_goal_geom":
        if action_low is None or action_high is None:
            raise ValueError("Goal-geometric scripted controller requires action_low/action_high.")
        return ScriptedGoalGeometricController(
            action_low=np.asarray(action_low, dtype=np.float32),
            action_high=np.asarray(action_high, dtype=np.float32),
        )
    if input_device in {"scripted_geo", "scripted_geo_legacy", "scripted_visual_mpc"}:
        if action_low is None or action_high is None:
            raise ValueError("Geometric scripted controller requires action_low/action_high.")
        controller_cls = {
            "scripted_geo": ScriptedGeometricTeacherController,
            "scripted_geo_legacy": LegacyScriptedGeometricTeacherController,
            "scripted_visual_mpc": ScriptedVisualMpcTeacherController,
        }[input_device]
        return controller_cls(
            action_low=np.asarray(action_low, dtype=np.float32),
            action_high=np.asarray(action_high, dtype=np.float32),
            heading_tolerance=float(scripted_geo_heading_tolerance),
            lookahead=float(scripted_geo_lookahead),
            safety_margin=float(scripted_geo_safety_margin),
            grid_resolution=float(scripted_geo_grid_resolution),
            emergency_clearance=float(scripted_geo_emergency_clearance),
            action_shield_steps=int(scripted_geo_action_shield_steps),
        )
    if input_device == "scripted":
        if action_low is None or action_high is None:
            raise ValueError("Scripted controller requires action_low/action_high.")
        return ScriptedLidarTeacherController(
            action_low=np.asarray(action_low, dtype=np.float32),
            action_high=np.asarray(action_high, dtype=np.float32),
        )
    if input_device == "heading_bc":
        if action_low is None or action_high is None:
            raise ValueError("Heading-BC controller requires action_low/action_high.")
        checkpoint_path = str(expert_checkpoint_path).strip()
        if not checkpoint_path:
            raise ValueError("input_device=heading_bc requires expert_checkpoint_path.")
        return HeadingBCPolicyController(
            checkpoint_path=checkpoint_path,
            action_low=np.asarray(action_low, dtype=np.float32),
            action_high=np.asarray(action_high, dtype=np.float32),
            device=str(expert_device),
        )
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
    if input_device in {"learned", "learned_intervention", "imitation"}:
        if action_low is None or action_high is None:
            raise ValueError("Learned intervention controller requires action_low/action_high.")
        checkpoint_path = str(expert_checkpoint_path).strip()
        if not checkpoint_path:
            raise ValueError("input_device=learned requires expert_checkpoint_path.")
        return LearnedInterventionPolicyController(
            checkpoint_path=checkpoint_path,
            action_low=np.asarray(action_low, dtype=np.float32),
            action_high=np.asarray(action_high, dtype=np.float32),
            intervention_threshold=float(learned_intervention_threshold),
            device=str(expert_device),
        )
    if input_device in {"flow_imitation", "flow_bc", "flow_teacher"}:
        if action_low is None or action_high is None:
            raise ValueError("Flow intervention controller requires action_low/action_high.")
        checkpoint_path = str(expert_checkpoint_path).strip()
        if not checkpoint_path:
            raise ValueError("input_device=flow_imitation requires expert_checkpoint_path.")
        return FlowInterventionPolicyController(
            checkpoint_path=checkpoint_path,
            action_low=np.asarray(action_low, dtype=np.float32),
            action_high=np.asarray(action_high, dtype=np.float32),
            intervention_threshold=float(learned_intervention_threshold),
            device=str(expert_device),
        )
    if input_device in {"bc_gate_scripted_geo", "imitation_gate_scripted_geo", "learned_gate_scripted_geo"}:
        if action_low is None or action_high is None:
            raise ValueError("Learned-gate scripted-geo controller requires action_low/action_high.")
        checkpoint_path = str(expert_checkpoint_path).strip()
        if not checkpoint_path:
            raise ValueError("input_device=bc_gate_scripted_geo requires expert_checkpoint_path.")
        return LearnedGateScriptedGeometricTeacherController(
            checkpoint_path=checkpoint_path,
            action_low=np.asarray(action_low, dtype=np.float32),
            action_high=np.asarray(action_high, dtype=np.float32),
            intervention_threshold=float(learned_intervention_threshold),
            device=str(expert_device),
        )
    if input_device == "safe_rl":
        if obs_dim is None or action_low is None or action_high is None:
            raise ValueError("Safe-RL controller requires obs_dim plus action_low/action_high.")
        checkpoint_path = str(expert_checkpoint_path).strip()
        config_path = str(expert_config_path).strip()
        if not checkpoint_path:
            raise ValueError("input_device=safe_rl requires expert_checkpoint_path.")
        if not config_path:
            raise ValueError("input_device=safe_rl requires expert_config_path.")
        return SafeRLPolicyController(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
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

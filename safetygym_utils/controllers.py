from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np


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


def infer_control_scheme(env_name: str) -> str:
    name = str(env_name).lower()
    if "car" in name:
        return "differential_wheels"
    return "planar_velocity"


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
        if not self._initialized:
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

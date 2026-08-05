"""High-level Unitree navigation gamepad controls.

The low-level G1 locomotion policy remains autonomous.  This adapter only
maps a human gamepad to the three high-level navigation commands consumed by
the navigation policy: body-forward velocity, body-lateral velocity, and yaw
rate.  It can read a local SDL/Pygame device or the existing Windows-to-WSL
raw gamepad transport.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from safetygym_utils.gamepad import (
    GamepadMappingConfig,
    GamepadStateClient,
    PygameGamepadController,
    axis_value_from_state,
    button_pressed_from_state,
)


DEFAULT_UNITREE_GAMEPAD_STATE_DIR = Path.home() / ".config" / "thesis" / "unitree_nav_gamepad"
DEFAULT_UNITREE_GAMEPAD_CONFIG_PATH = DEFAULT_UNITREE_GAMEPAD_STATE_DIR / "mapping_profile.json"
DEFAULT_UNITREE_GAMEPAD_CACHE_PATH = DEFAULT_UNITREE_GAMEPAD_STATE_DIR / "last_endpoint.json"
DEFAULT_UNITREE_GAMEPAD_PORT = 8794


@dataclass
class UnitreeGamepadConfig:
    """Controls for the Unitree high-level navigation command."""

    device_index: int = 0
    deadzone: float = 0.12
    intervention_mode: str = "stick"
    intervention_threshold: float = 0.05
    # Retained only for the optional legacy button mode.
    require_gate_button: bool = True
    gate_button: str = "rb"
    forward_axis: str = "left_y"
    lateral_axis: str = "left_x"
    yaw_axis: str = "right_x"
    invert_forward: bool = True
    # Xbox left-stick horizontal is mirrored relative to the Unitree body-y
    # convention used by the navigation environment.
    invert_lateral: bool = True
    invert_yaw: bool = True
    forward_scale: float = 0.85
    lateral_scale: float = 0.65
    yaw_scale: float = 0.85


@dataclass
class UnitreeGamepadSample:
    action: np.ndarray
    intervening: bool
    gate_held: bool
    command_norm: float
    connected: bool
    stale: bool
    state_age_s: float
    state: dict[str, Any]
    transport: dict[str, Any]


def _apply_deadzone(value: float, deadzone: float) -> float:
    deadzone = float(np.clip(deadzone, 0.0, 0.95))
    value = float(value)
    if abs(value) <= deadzone:
        return 0.0
    return float(np.clip(np.sign(value) * (abs(value) - deadzone) / max(1e-6, 1.0 - deadzone), -1.0, 1.0))


def _config_from_payload(payload: dict[str, Any]) -> UnitreeGamepadConfig:
    config = UnitreeGamepadConfig()
    for key, default in asdict(config).items():
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(default, bool):
            value = bool(value)
        elif isinstance(default, int):
            value = int(value)
        elif isinstance(default, float):
            value = float(value)
        else:
            value = str(value)
        setattr(config, key, value)
    return config


def load_unitree_gamepad_config(path: str | Path = DEFAULT_UNITREE_GAMEPAD_CONFIG_PATH) -> tuple[UnitreeGamepadConfig, bool]:
    source = Path(path).expanduser()
    if not source.exists():
        return UnitreeGamepadConfig(), False
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return UnitreeGamepadConfig(), False
    return _config_from_payload(payload if isinstance(payload, dict) else {}), True


def save_unitree_gamepad_config(
    config: UnitreeGamepadConfig,
    path: str | Path = DEFAULT_UNITREE_GAMEPAD_CONFIG_PATH,
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


class UnitreeGamepadController:
    """Read raw controller state and map it to high-level Unitree commands."""

    def __init__(
        self,
        *,
        mode: str,
        host: str,
        port: int,
        config_path: str | Path = DEFAULT_UNITREE_GAMEPAD_CONFIG_PATH,
        device_index: int = 0,
        reconnect_seconds: float = 2.0,
        stale_timeout_s: float = 0.25,
        intervention_mode: str | None = None,
        intervention_threshold: float | None = None,
        invert_lateral: bool | None = None,
    ) -> None:
        self.mode = str(mode).lower()
        if self.mode not in {"local", "connect"}:
            raise ValueError(f"Unsupported Unitree gamepad mode: {mode!r}")
        self.config_path = Path(config_path).expanduser()
        self.config, loaded = load_unitree_gamepad_config(self.config_path)
        if not loaded:
            self.config.device_index = int(device_index)
            save_unitree_gamepad_config(self.config, self.config_path)
        self._intervention_mode_override = (
            str(intervention_mode).lower() if intervention_mode is not None else None
        )
        self._intervention_threshold_override = (
            float(intervention_threshold) if intervention_threshold is not None else None
        )
        self._invert_lateral_override = bool(invert_lateral) if invert_lateral is not None else None
        self._apply_runtime_overrides()
        self.stale_timeout_s = float(max(0.02, stale_timeout_s))
        self._config_mtime_ns: int | None = None
        self._local: PygameGamepadController | None = None
        self._client: GamepadStateClient | None = None
        if self.mode == "local":
            # PygameGamepadController supplies raw SDL state. Its own action
            # mapping is deliberately ignored in favor of this 3-axis mapper.
            self._local = PygameGamepadController(
                action_dim=3,
                control_scheme="planar_velocity",
                config=GamepadMappingConfig(device_index=int(self.config.device_index)),
                hot_reload=False,
            )
        else:
            self._client = GamepadStateClient(
                host=str(host),
                port=int(port),
                reconnect_seconds=float(reconnect_seconds),
                cache_path=DEFAULT_UNITREE_GAMEPAD_CACHE_PATH,
            )
            self._client.start()

    def _apply_runtime_overrides(self) -> None:
        """Keep CLI takeover semantics authoritative over a reloaded profile."""
        if self._intervention_mode_override is not None:
            self.config.intervention_mode = self._intervention_mode_override
        if self._intervention_threshold_override is not None:
            self.config.intervention_threshold = self._intervention_threshold_override
        if self._invert_lateral_override is not None:
            self.config.invert_lateral = self._invert_lateral_override

    def _maybe_reload_config(self) -> None:
        try:
            mtime_ns = int(self.config_path.stat().st_mtime_ns)
        except FileNotFoundError:
            return
        if mtime_ns == self._config_mtime_ns:
            return
        config, loaded = load_unitree_gamepad_config(self.config_path)
        if loaded:
            self.config = config
            self._apply_runtime_overrides()
            self._config_mtime_ns = mtime_ns

    def _read_state(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._local is not None:
            state = self._local.read_state()
            return state, {"connected": bool(state.get("connected", False)), "target": "local", "error": ""}
        assert self._client is not None
        state = dict(self._client.latest_state)
        connected = bool(self._client.connected)
        state["connected"] = bool(state.get("connected", False) and connected)
        return state, {
            "connected": connected,
            "target": f"{self._client.host}:{self._client.port}",
            "peer": self._client.last_peer,
            "error": self._client.last_error,
        }

    def sample(self) -> UnitreeGamepadSample:
        self._maybe_reload_config()
        state, transport = self._read_state()
        connected = bool(state.get("connected", False))
        timestamp_key = "_transport_received_at" if self.mode == "connect" else "timestamp"
        timestamp = float(state.get(timestamp_key, 0.0) or 0.0)
        now = time.time()
        state_age_s = (
            float("inf")
            if not np.isfinite(timestamp) or timestamp <= 0.0
            else max(0.0, now - timestamp)
        )
        stale = not np.isfinite(timestamp) or state_age_s > self.stale_timeout_s
        def axis(name: str, invert: bool, scale: float) -> float:
            value = _apply_deadzone(axis_value_from_state(state, name), self.config.deadzone)
            if invert:
                value = -value
            return float(np.clip(float(scale) * value, -1.0, 1.0))

        action = np.array(
            [
                axis(self.config.forward_axis, self.config.invert_forward, self.config.forward_scale),
                axis(self.config.lateral_axis, self.config.invert_lateral, self.config.lateral_scale),
                axis(self.config.yaw_axis, self.config.invert_yaw, self.config.yaw_scale),
            ],
            dtype=np.float32,
        )
        command_norm = float(np.linalg.norm(action))
        if str(self.config.intervention_mode).lower() == "stick":
            gate_held = bool(
                connected
                and not stale
                and command_norm >= max(0.0, float(self.config.intervention_threshold))
            )
        else:
            gate_held = connected and not stale and (
                not bool(self.config.require_gate_button)
                or button_pressed_from_state(state, self.config.gate_button)
            )
        if not gate_held:
            action.fill(0.0)
        command_norm = float(np.linalg.norm(action))
        return UnitreeGamepadSample(
            action=action,
            intervening=bool(gate_held),
            gate_held=bool(gate_held),
            command_norm=command_norm,
            connected=connected,
            stale=bool(stale),
            state_age_s=state_age_s,
            state=state,
            transport=transport,
        )

    def close(self) -> None:
        if self._local is not None:
            self._local.close()
        if self._client is not None:
            self._client.close()

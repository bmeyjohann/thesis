from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

DEFAULT_SAFETY_GAMEPAD_STATE_DIR = Path.home() / ".config" / "thesis" / "safetygym_gamepad"
DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH = DEFAULT_SAFETY_GAMEPAD_STATE_DIR / "mapping_profile.json"
DEFAULT_SAFETY_GAMEPAD_CACHE_PATH = DEFAULT_SAFETY_GAMEPAD_STATE_DIR / "last_endpoint.json"
DEFAULT_SAFETY_GAMEPAD_WEB_PORT = 8792
DEFAULT_SAFETY_GAMEPAD_PORT = 8793

GAMEPAD_CONTROL_MODE_OPTIONS = ("single_stick_mixed", "tank_dual_stick", "planar")
GAMEPAD_AXIS_OPTIONS = ("left_x", "left_y", "right_x", "right_y", "lt", "rt")
GAMEPAD_BUTTON_OPTIONS = (
    "a",
    "b",
    "x",
    "y",
    "lb",
    "rb",
    "back",
    "start",
    "guide",
    "left_stick",
    "right_stick",
    "dpad_up",
    "dpad_down",
    "dpad_left",
    "dpad_right",
)


@dataclass
class GamepadMappingConfig:
    device_index: int = 0
    control_mode: str = "single_stick_mixed"
    action_scale: float = 1.0
    wheel_command_limit: float = 2.0
    deadzone: float = 0.12
    require_gate_button: bool = False
    gate_button: str = "rb"
    throttle_axis: str = "left_y"
    steer_axis: str = "right_x"
    left_wheel_axis: str = "left_y"
    right_wheel_axis: str = "right_y"
    planar_forward_axis: str = "left_y"
    planar_turn_axis: str = "right_x"
    invert_throttle: bool = True
    invert_steer: bool = False
    invert_left_wheel: bool = True
    invert_right_wheel: bool = True
    invert_planar_forward: bool = True
    invert_planar_turn: bool = False


def infer_control_scheme(env_name: str) -> str:
    name = str(env_name).lower()
    if "car" in name:
        return "differential_wheels"
    return "planar_velocity"


def gamepad_mapping_config_to_dict(config: GamepadMappingConfig) -> dict[str, Any]:
    return asdict(config)


def discover_local_ip_addresses() -> list[str]:
    out = {"127.0.0.1"}
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, family=socket.AF_INET):
            ip = str(info[4][0])
            if ip:
                out.add(ip)
    except Exception:
        pass
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        out.add(str(sock.getsockname()[0]))
        sock.close()
    except Exception:
        pass
    return sorted(out)


def gamepad_mapping_config_from_payload(payload: dict[str, Any]) -> GamepadMappingConfig:
    config = GamepadMappingConfig()
    for field in config.__dataclass_fields__:
        if field not in payload:
            continue
        value = payload[field]
        current = getattr(config, field)
        if isinstance(current, bool):
            value = bool(value)
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        else:
            value = str(value)
        setattr(config, field, value)
    config.control_mode = (
        str(config.control_mode)
        if str(config.control_mode) in GAMEPAD_CONTROL_MODE_OPTIONS
        else "single_stick_mixed"
    )
    config.gate_button = (
        str(config.gate_button)
        if str(config.gate_button) in GAMEPAD_BUTTON_OPTIONS
        else "rb"
    )
    return config


def load_gamepad_mapping_config(path: Path | str = DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH) -> tuple[GamepadMappingConfig, bool]:
    source = Path(path).expanduser()
    if not source.exists():
        return GamepadMappingConfig(), False
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return GamepadMappingConfig(), False
    return gamepad_mapping_config_from_payload(payload), True


def save_gamepad_mapping_config(
    config: GamepadMappingConfig,
    path: Path | str = DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(gamepad_mapping_config_to_dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def apply_gamepad_mapping_profile(
    config: GamepadMappingConfig,
    path: Path | str = DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
) -> tuple[GamepadMappingConfig, bool]:
    loaded, ok = load_gamepad_mapping_config(path)
    if not ok:
        return config, False
    payload = gamepad_mapping_config_to_dict(config)
    payload.update(gamepad_mapping_config_to_dict(loaded))
    return gamepad_mapping_config_from_payload(payload), True


def save_cached_gamepad_endpoint(
    *,
    host: str,
    port: int,
    path: Path | str = DEFAULT_SAFETY_GAMEPAD_CACHE_PATH,
) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"host": str(host), "port": int(port), "saved_at": time.time()}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_cached_gamepad_endpoint(path: Path | str = DEFAULT_SAFETY_GAMEPAD_CACHE_PATH) -> tuple[dict[str, Any], bool]:
    source = Path(path).expanduser()
    if not source.exists():
        return {}, False
    try:
        return json.loads(source.read_text(encoding="utf-8")), True
    except Exception:
        return {}, False


def resolve_cached_gamepad_endpoint(
    *,
    host: str,
    port: int,
    cache_path: Path | str = DEFAULT_SAFETY_GAMEPAD_CACHE_PATH,
) -> tuple[str, int, bool]:
    chosen_host = str(host or "").strip()
    chosen_port = int(port) if int(port) > 0 else int(DEFAULT_SAFETY_GAMEPAD_PORT)
    if chosen_host:
        return chosen_host, chosen_port, False
    cached, ok = load_cached_gamepad_endpoint(cache_path)
    if ok and cached.get("host"):
        cached_port = int(cached.get("port", chosen_port))
        return str(cached["host"]), cached_port, True
    return "127.0.0.1", chosen_port, False


def axis_value_from_state(state: dict[str, Any], axis_name: str) -> float:
    axes = state.get("named_axes", {}) if isinstance(state, dict) else {}
    return float(axes.get(str(axis_name), 0.0))


def button_pressed_from_state(state: dict[str, Any], button_name: str) -> bool:
    buttons = state.get("buttons", {}) if isinstance(state, dict) else {}
    return bool(buttons.get(str(button_name), False))


def _apply_deadzone(value: float, deadzone: float) -> float:
    deadzone = float(max(0.0, min(0.95, deadzone)))
    v = float(value)
    if abs(v) <= deadzone:
        return 0.0
    scale = 1.0 / max(1e-6, 1.0 - deadzone)
    return float(np.clip(np.sign(v) * (abs(v) - deadzone) * scale, -1.0, 1.0))


def empty_gamepad_state() -> dict[str, Any]:
    return {
        "connected": False,
        "device_index": -1,
        "name": "",
        "num_axes": 0,
        "num_buttons": 0,
        "timestamp": time.time(),
        "named_axes": {name: 0.0 for name in GAMEPAD_AXIS_OPTIONS},
        "raw_axes": [],
        "buttons": {name: False for name in GAMEPAD_BUTTON_OPTIONS},
    }


def _signed_axis_from_state(state: dict[str, Any], axis_name: str, invert: bool, deadzone: float) -> float:
    value = _apply_deadzone(axis_value_from_state(state, axis_name), deadzone)
    if invert:
        value = -value
    return float(np.clip(value, -1.0, 1.0))


def preview_action_from_state(
    *,
    state: dict[str, Any],
    action_dim: int,
    control_scheme: str,
    config: GamepadMappingConfig,
) -> np.ndarray:
    action = np.zeros((int(action_dim),), dtype=np.float32)
    if not bool(state.get("connected", False)):
        return action
    if config.require_gate_button and not button_pressed_from_state(state, config.gate_button):
        return action

    s = float(max(0.0, config.action_scale))
    if str(control_scheme) == "differential_wheels" and int(action_dim) >= 2:
        if config.control_mode == "tank_dual_stick":
            left = s * _signed_axis_from_state(state, config.left_wheel_axis, config.invert_left_wheel, config.deadzone)
            right = s * _signed_axis_from_state(state, config.right_wheel_axis, config.invert_right_wheel, config.deadzone)
            action[0] = left
            action[1] = right
        else:
            throttle = s * _signed_axis_from_state(state, config.throttle_axis, config.invert_throttle, config.deadzone)
            steer = s * _signed_axis_from_state(state, config.steer_axis, config.invert_steer, config.deadzone)
            action[0] = float(throttle - steer)
            action[1] = float(throttle + steer)
        lim = float(max(0.0, config.wheel_command_limit))
        action[:2] = np.clip(action[:2], -lim, lim)
    else:
        forward = s * _signed_axis_from_state(
            state, config.planar_forward_axis, config.invert_planar_forward, config.deadzone
        )
        turn = s * _signed_axis_from_state(state, config.planar_turn_axis, config.invert_planar_turn, config.deadzone)
        if int(action_dim) >= 1:
            action[0] = forward
        if int(action_dim) >= 2:
            action[1] = turn
        if int(action_dim) > 2:
            action[2:] = 0.0
        action = np.clip(action, -1.0, 1.0)
    return action.astype(np.float32, copy=False)


class GamepadStateServer:
    def __init__(self, *, host: str = "0.0.0.0", port: int = DEFAULT_SAFETY_GAMEPAD_PORT):
        self.host = str(host)
        self.port = int(port)
        self._sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen()
        sock.settimeout(1.0)
        self._sock = sock
        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, name="gamepad-accept", daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                conn, _addr = self._sock.accept()
                conn.setblocking(True)
            except socket.timeout:
                continue
            except OSError:
                break
            with self._lock:
                self._clients.append(conn)

    def publish(self, sample: dict[str, Any]) -> None:
        line = (json.dumps(sample, sort_keys=True) + "\n").encode("utf-8")
        stale: list[socket.socket] = []
        with self._lock:
            clients = list(self._clients)
        for conn in clients:
            try:
                conn.sendall(line)
            except OSError:
                stale.append(conn)
        if stale:
            with self._lock:
                for conn in stale:
                    if conn in self._clients:
                        self._clients.remove(conn)
                    try:
                        conn.close()
                    except Exception:
                        pass

    def banner_text(self) -> str:
        addrs = ", ".join(f"{ip}:{self.port}" for ip in discover_local_ip_addresses())
        return f"SafetyGym gamepad publisher listening on {addrs}"

    def close(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.0)
            self._accept_thread = None
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for conn in clients:
            try:
                conn.close()
            except Exception:
                pass


class GamepadStateClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        reconnect_seconds: float = 2.0,
        cache_path: Path | str = DEFAULT_SAFETY_GAMEPAD_CACHE_PATH,
    ):
        self.host = str(host)
        self.port = int(port)
        self.reconnect_seconds = float(max(0.2, reconnect_seconds))
        self.cache_path = Path(cache_path).expanduser()
        self.connected = False
        self.last_error = ""
        self.last_peer = ""
        self.latest_state: dict[str, Any] = empty_gamepad_state()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="gamepad-client", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        while self._running:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            try:
                sock.connect((self.host, self.port))
                try:
                    save_cached_gamepad_endpoint(host=self.host, port=self.port, path=self.cache_path)
                except Exception:
                    pass
                self.connected = True
                self.last_error = ""
                peer = sock.getpeername()
                self.last_peer = f"{peer[0]}:{peer[1]}"
                reader = sock.makefile("r", encoding="utf-8")
                while self._running:
                    line = reader.readline()
                    if not line:
                        raise ConnectionError("connection closed")
                    payload = json.loads(line)
                    state = payload.get("state", payload)
                    if not isinstance(state, dict):
                        continue
                    self.latest_state = dict(state)
                    self.latest_state["connected"] = True
                    self.latest_state["_transport_peer"] = self.last_peer
            except Exception as exc:
                self.connected = False
                self.last_error = str(exc)
                time.sleep(self.reconnect_seconds)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        self.connected = False

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


class RemoteGamepadController:
    def __init__(
        self,
        action_dim: int,
        *,
        control_scheme: str,
        client: GamepadStateClient,
        config: GamepadMappingConfig | None = None,
        config_path: Path | str = DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
        hot_reload: bool = True,
    ):
        self.action_dim = int(action_dim)
        self.control_scheme = str(control_scheme)
        self.client = client
        self.config = config or GamepadMappingConfig()
        self.config_path = Path(config_path).expanduser()
        self.hot_reload = bool(hot_reload)
        self._config_mtime_ns: Optional[int] = None
        self.client.start()

    def _maybe_reload_config(self) -> None:
        if not self.hot_reload:
            return
        try:
            stat = self.config_path.stat()
        except FileNotFoundError:
            return
        mtime_ns = int(stat.st_mtime_ns)
        if self._config_mtime_ns == mtime_ns:
            return
        loaded, ok = load_gamepad_mapping_config(self.config_path)
        if ok:
            self.config = loaded
            self._config_mtime_ns = mtime_ns

    def read_state(self) -> dict[str, Any]:
        self._maybe_reload_config()
        state = dict(self.client.latest_state)
        if not self.client.connected:
            state["connected"] = False
        state["_transport_connected"] = bool(self.client.connected)
        state["_transport_error"] = str(self.client.last_error)
        state["_transport_peer"] = str(self.client.last_peer)
        return state

    def preview_action(self, state: Optional[dict[str, Any]] = None) -> np.ndarray:
        state = dict(state or self.read_state())
        return preview_action_from_state(
            state=state,
            action_dim=self.action_dim,
            control_scheme=self.control_scheme,
            config=self.config,
        )

    def get_action(self) -> np.ndarray:
        return self.preview_action()

    def live_payload(self) -> dict[str, Any]:
        state = self.read_state()
        preview = self.preview_action(state)
        return {
            "state": state,
            "preview_action": preview.tolist(),
            "config": gamepad_mapping_config_to_dict(self.config),
            "control_scheme": self.control_scheme,
            "transport": {
                "connected": bool(self.client.connected),
                "peer": str(self.client.last_peer),
                "error": str(self.client.last_error),
                "target": f"{self.client.host}:{self.client.port}",
            },
        }

    def close(self) -> None:
        self.client.close()


class PygameGamepadController:
    def __init__(
        self,
        action_dim: int,
        *,
        control_scheme: str,
        config: GamepadMappingConfig | None = None,
        config_path: Path | str = DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
        hot_reload: bool = True,
    ):
        self.action_dim = int(action_dim)
        self.control_scheme = str(control_scheme)
        self.config = config or GamepadMappingConfig()
        self.config_path = Path(config_path).expanduser()
        self.hot_reload = bool(hot_reload)
        self._config_mtime_ns: Optional[int] = None
        self._pygame = None
        self._joystick = None
        self._joystick_id: Optional[int] = None
        self._last_state: dict[str, Any] = self._empty_state()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "connected": False,
            "device_index": -1,
            "name": "",
            "num_axes": 0,
            "num_buttons": 0,
            "timestamp": time.time(),
            "named_axes": {name: 0.0 for name in GAMEPAD_AXIS_OPTIONS},
            "raw_axes": [],
            "buttons": {name: False for name in GAMEPAD_BUTTON_OPTIONS},
        }

    def _maybe_reload_config(self) -> None:
        if not self.hot_reload:
            return
        try:
            stat = self.config_path.stat()
        except FileNotFoundError:
            return
        mtime_ns = int(stat.st_mtime_ns)
        if self._config_mtime_ns == mtime_ns:
            return
        loaded, ok = load_gamepad_mapping_config(self.config_path)
        if ok:
            self.config = loaded
            self._config_mtime_ns = mtime_ns

    def _ensure_pygame(self) -> None:
        if self._pygame is not None:
            return
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        pygame.init()
        pygame.joystick.init()
        self._pygame = pygame

    def _ensure_joystick(self) -> None:
        self._ensure_pygame()
        pygame = self._pygame
        assert pygame is not None
        pygame.event.pump()
        wanted = int(max(0, self.config.device_index))
        count = int(pygame.joystick.get_count())
        if count <= 0:
            self._joystick = None
            self._joystick_id = None
            return
        wanted = min(wanted, count - 1)
        if self._joystick is not None and self._joystick_id == wanted:
            try:
                if self._joystick.get_init():
                    return
            except Exception:
                pass
        js = pygame.joystick.Joystick(wanted)
        js.init()
        self._joystick = js
        self._joystick_id = wanted

    def read_state(self) -> dict[str, Any]:
        self._maybe_reload_config()
        self._ensure_joystick()
        pygame = self._pygame
        if pygame is None or self._joystick is None:
            self._last_state = self._empty_state()
            return dict(self._last_state)
        try:
            pygame.event.pump()
        except Exception:
            pass
        js = self._joystick
        raw_axes = [float(js.get_axis(i)) for i in range(js.get_numaxes())]
        hat = js.get_hat(0) if js.get_numhats() > 0 else (0, 0)
        buttons = {
            "a": bool(js.get_numbuttons() > 0 and js.get_button(0)),
            "b": bool(js.get_numbuttons() > 1 and js.get_button(1)),
            "x": bool(js.get_numbuttons() > 2 and js.get_button(2)),
            "y": bool(js.get_numbuttons() > 3 and js.get_button(3)),
            "lb": bool(js.get_numbuttons() > 4 and js.get_button(4)),
            "rb": bool(js.get_numbuttons() > 5 and js.get_button(5)),
            "back": bool(js.get_numbuttons() > 6 and js.get_button(6)),
            "start": bool(js.get_numbuttons() > 7 and js.get_button(7)),
            "guide": bool(js.get_numbuttons() > 8 and js.get_button(8)),
            "left_stick": bool(js.get_numbuttons() > 9 and js.get_button(9)),
            "right_stick": bool(js.get_numbuttons() > 10 and js.get_button(10)),
            "dpad_up": bool(int(hat[1]) > 0),
            "dpad_down": bool(int(hat[1]) < 0),
            "dpad_left": bool(int(hat[0]) < 0),
            "dpad_right": bool(int(hat[0]) > 0),
        }
        lt = raw_axes[4] if len(raw_axes) > 4 else 0.0
        rt = raw_axes[5] if len(raw_axes) > 5 else 0.0
        named_axes = {
            "left_x": raw_axes[0] if len(raw_axes) > 0 else 0.0,
            "left_y": raw_axes[1] if len(raw_axes) > 1 else 0.0,
            "right_x": raw_axes[2] if len(raw_axes) > 2 else 0.0,
            "right_y": raw_axes[3] if len(raw_axes) > 3 else 0.0,
            "lt": 0.5 * (float(lt) + 1.0),
            "rt": 0.5 * (float(rt) + 1.0),
        }
        self._last_state = {
            "connected": True,
            "device_index": int(self._joystick_id if self._joystick_id is not None else -1),
            "name": str(js.get_name()),
            "num_axes": int(js.get_numaxes()),
            "num_buttons": int(js.get_numbuttons()),
            "timestamp": time.time(),
            "named_axes": named_axes,
            "raw_axes": raw_axes,
            "buttons": buttons,
        }
        return dict(self._last_state)

    def _signed_axis(self, state: dict[str, Any], axis_name: str, invert: bool) -> float:
        value = _apply_deadzone(axis_value_from_state(state, axis_name), self.config.deadzone)
        if invert:
            value = -value
        return float(np.clip(value, -1.0, 1.0))

    def preview_action(self, state: Optional[dict[str, Any]] = None) -> np.ndarray:
        state = dict(state or self.read_state())
        action = np.zeros((self.action_dim,), dtype=np.float32)
        if not bool(state.get("connected", False)):
            return action
        if self.config.require_gate_button and not button_pressed_from_state(state, self.config.gate_button):
            return action

        s = float(max(0.0, self.config.action_scale))
        if self.control_scheme == "differential_wheels" and self.action_dim >= 2:
            if self.config.control_mode == "tank_dual_stick":
                left = s * self._signed_axis(state, self.config.left_wheel_axis, self.config.invert_left_wheel)
                right = s * self._signed_axis(state, self.config.right_wheel_axis, self.config.invert_right_wheel)
                action[0] = left
                action[1] = right
            else:
                throttle = s * self._signed_axis(state, self.config.throttle_axis, self.config.invert_throttle)
                steer = s * self._signed_axis(state, self.config.steer_axis, self.config.invert_steer)
                action[0] = float(throttle - steer)
                action[1] = float(throttle + steer)
            lim = float(max(0.0, self.config.wheel_command_limit))
            action[:2] = np.clip(action[:2], -lim, lim)
        else:
            forward = s * self._signed_axis(state, self.config.planar_forward_axis, self.config.invert_planar_forward)
            turn = s * self._signed_axis(state, self.config.planar_turn_axis, self.config.invert_planar_turn)
            if self.action_dim >= 1:
                action[0] = forward
            if self.action_dim >= 2:
                action[1] = turn
            if self.action_dim > 2:
                action[2:] = 0.0
            action = np.clip(action, -1.0, 1.0)
        return action.astype(np.float32, copy=False)

    def get_action(self) -> np.ndarray:
        return self.preview_action()

    def live_payload(self) -> dict[str, Any]:
        state = self.read_state()
        preview = self.preview_action(state)
        return {
            "state": state,
            "preview_action": preview.tolist(),
            "config": gamepad_mapping_config_to_dict(self.config),
            "control_scheme": self.control_scheme,
            "transport": {
                "connected": bool(state.get("connected", False)),
                "peer": "",
                "error": "",
                "target": "local",
            },
        }

    def close(self) -> None:
        pygame = self._pygame
        if pygame is not None:
            try:
                if self._joystick is not None:
                    self._joystick.quit()
            except Exception:
                pass
            try:
                pygame.joystick.quit()
            except Exception:
                pass
            try:
                pygame.quit()
            except Exception:
                pass
        self._pygame = None
        self._joystick = None
        self._joystick_id = None

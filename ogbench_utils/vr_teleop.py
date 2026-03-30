from __future__ import annotations

import json
import math
import socket
import socketserver
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_VR_PORT = 8765
DEFAULT_VR_SERVE_HOST = "0.0.0.0"
DEFAULT_VR_STATE_DIR = Path.home() / ".config" / "thesis" / "vr"
LEGACY_VR_CACHE_PATH = Path("codex/vr_last_endpoint.json")
LEGACY_VR_MAPPING_PATH = Path("codex/vr_mapping_profile.json")
DEFAULT_VR_CACHE_PATH = DEFAULT_VR_STATE_DIR / "last_endpoint.json"
DEFAULT_VR_MAPPING_PATH = DEFAULT_VR_STATE_DIR / "mapping_profile.json"
DEFAULT_VR_MAPPING_WEB_PORT = 8790


def _normalize_vr_path(path: Path | str) -> Path:
    return Path(path).expanduser()


def _resolve_vr_runtime_path(path: Path | str, *, legacy_path: Optional[Path | str] = None) -> Path:
    target = _normalize_vr_path(path)
    legacy = _normalize_vr_path(legacy_path) if legacy_path is not None else None
    if target.exists() or legacy is None or not legacy.exists():
        return target
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
        return target
    except Exception:
        return legacy


OPENVR_BUTTON_ALIASES: dict[str, int] = {
    "system": 0,
    "application_menu": 1,
    "grip": 2,
    "dpad_left": 3,
    "dpad_up": 4,
    "dpad_right": 5,
    "dpad_down": 6,
    "a": 7,
    "proximity_sensor": 31,
    "axis0": 32,
    "axis1": 33,
    "axis2": 34,
    "axis3": 35,
    "axis4": 36,
}

VR_POSITION_SOURCE_OPTIONS: tuple[str, ...] = ("x", "-x", "y", "-y", "z", "-z", "none")
VR_ROTATION_SOURCE_OPTIONS: tuple[str, ...] = (
    "global_yaw",
    "-global_yaw",
    "global_pitch",
    "-global_pitch",
    "global_roll",
    "-global_roll",
    "none",
    # Backward-compatible aliases.
    "yaw",
    "-yaw",
    "pitch",
    "-pitch",
    "roll",
    "-roll",
)
VR_HAND_OPTIONS: tuple[str, ...] = ("left", "right")
VR_GATE_BUTTON_OPTIONS: tuple[str, ...] = (
    "grip",
    "application_menu",
    "system",
    "a",
    "dpad_left",
    "dpad_up",
    "dpad_right",
    "dpad_down",
    "axis0",
    "axis1",
    "axis2",
    "axis3",
    "axis4",
)
VR_OPTIONAL_BUTTON_OPTIONS: tuple[str, ...] = ("none",) + VR_GATE_BUTTON_OPTIONS
VR_GRIPPER_AXIS_OPTIONS: tuple[str, ...] = (
    "trigger",
    "trackpad_x",
    "trackpad_y",
    "joystick_x",
    "joystick_y",
    "0.x",
    "0.y",
    "1.x",
    "1.y",
    "2.x",
    "2.y",
    "3.x",
    "3.y",
    "4.x",
    "4.y",
)
VR_MOTION_CONTROL_OPTIONS: tuple[str, ...] = ("target_hold", "delta")
VR_GRIPPER_CONTROL_OPTIONS: tuple[str, ...] = ("absolute", "delta")
MANIP_ACTION_RANGE = (0.05, 0.05, 0.05, 0.3, 1.0)


def _np():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as exc:
        raise RuntimeError("numpy is required for VR action mapping") from exc


def discover_local_ipv4_addresses() -> list[str]:
    """Best-effort discovery of non-loopback IPv4 addresses."""
    candidates: set[str] = {"127.0.0.1"}
    hostnames = {socket.gethostname(), socket.getfqdn()}
    for name in hostnames:
        try:
            for item in socket.getaddrinfo(name, None, socket.AF_INET, socket.SOCK_STREAM):
                ip = str(item[4][0])
                if ip:
                    candidates.add(ip)
        except Exception:
            continue
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        candidates.add(str(probe.getsockname()[0]))
        probe.close()
    except Exception:
        pass
    return sorted(candidates, key=lambda ip: (ip.startswith("127."), ip))


def load_cached_endpoint(cache_path: Path | str = DEFAULT_VR_CACHE_PATH) -> Optional[dict[str, Any]]:
    path = _resolve_vr_runtime_path(cache_path, legacy_path=LEGACY_VR_CACHE_PATH)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    host = str(payload.get("host", "")).strip()
    try:
        port = int(payload.get("port", DEFAULT_VR_PORT))
    except Exception:
        port = DEFAULT_VR_PORT
    if not host:
        return None
    return {"host": host, "port": port}


def save_cached_endpoint(host: str, port: int, cache_path: Path | str = DEFAULT_VR_CACHE_PATH) -> Path:
    path = _normalize_vr_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"host": str(host), "port": int(port), "updated_at": float(time.time())}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def resolve_cached_endpoint(
    host: Optional[str],
    port: Optional[int],
    *,
    cache_path: Path | str = DEFAULT_VR_CACHE_PATH,
    default_host: str = "127.0.0.1",
    default_port: int = DEFAULT_VR_PORT,
) -> tuple[str, int, bool]:
    host_text = str(host or "").strip()
    port_value: Optional[int]
    try:
        port_value = int(port) if port is not None else None
    except Exception:
        port_value = None
    if port_value is not None and port_value <= 0:
        port_value = None
    if host_text:
        return host_text, int(port_value or default_port), False
    cached = load_cached_endpoint(cache_path=cache_path)
    if isinstance(cached, dict):
        return str(cached["host"]), int(port_value or cached["port"]), True
    return default_host, int(port_value or default_port), False


def cacheable_host(host: str) -> str:
    host_text = str(host).strip()
    if host_text and host_text not in {"0.0.0.0", "::"}:
        return host_text
    ips = discover_local_ipv4_addresses()
    for ip in ips:
        if not ip.startswith("127."):
            return ip
    return ips[0] if ips else "127.0.0.1"


def format_receiver_banner(host: str, port: int) -> str:
    lines = [
        "VR receiver listening.",
        f"  bind={host}:{port}",
        "  give one of these addresses to the sender:",
    ]
    for ip in discover_local_ipv4_addresses():
        lines.append(f"    {ip}:{port}")
    return "\n".join(lines)


def format_publisher_banner(host: str, port: int) -> str:
    lines = [
        "VR publisher listening.",
        f"  bind={host}:{port}",
        "  clients can connect to one of these addresses:",
    ]
    for ip in discover_local_ipv4_addresses():
        lines.append(f"    {ip}:{port}")
    return "\n".join(lines)


class _VRJSONTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _RawStateStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._sample: Optional[dict[str, Any]] = None
        self._last_client: Optional[str] = None
        self._last_received_wall_time = 0.0
        self._connected_clients = 0

    def note_connect(self, client: str) -> None:
        with self._lock:
            self._connected_clients += 1
            self._last_client = client

    def note_disconnect(self) -> None:
        with self._lock:
            self._connected_clients = max(0, self._connected_clients - 1)

    def update(self, client: str, sample: dict[str, Any]) -> None:
        with self._lock:
            self._sample = sample
            self._last_client = client
            self._last_received_wall_time = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected_clients": int(self._connected_clients),
                "last_client": self._last_client,
                "last_received_wall_time": float(self._last_received_wall_time),
                "latest_sample": self._sample,
            }


class _VRRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        client = f"{self.client_address[0]}:{self.client_address[1]}"
        self.server.state_store.note_connect(client)  # type: ignore[attr-defined]
        try:
            while True:
                raw = self.rfile.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    self.server.state_store.update(client, payload)  # type: ignore[attr-defined]
        finally:
            self.server.state_store.note_disconnect()  # type: ignore[attr-defined]


class VRRawStateServer:
    """Background JSON-over-TCP server that stores only the latest sample."""

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_VR_PORT):
        self.host = str(host)
        self.port = int(port)
        self._store = _RawStateStore()
        self._server: Optional[_VRJSONTCPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._server is not None:
            return
        server = _VRJSONTCPServer((self.host, self.port), _VRRequestHandler)
        server.state_store = self._store  # type: ignore[attr-defined]
        self._server = server
        self.host, self.port = server.server_address[:2]
        self._thread = threading.Thread(target=server.serve_forever, name="vr-raw-server", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None

    def latest_sample(self) -> Optional[dict[str, Any]]:
        return self.snapshot().get("latest_sample")

    def snapshot(self) -> dict[str, Any]:
        snap = self._store.snapshot()
        snap.update(
            {
                "mode": "listen",
                "host": self.host,
                "port": int(self.port),
                "receiver_addresses": discover_local_ipv4_addresses(),
            }
        )
        return snap

    def banner_text(self) -> str:
        return format_receiver_banner(self.host, self.port)


class VRRawStateClient:
    """Background JSON-over-TCP client that consumes a remote raw stream."""

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_VR_PORT,
        *,
        reconnect_seconds: float = 2.0,
        cache_path: Path | str = DEFAULT_VR_CACHE_PATH,
        save_cache: bool = True,
    ):
        self.host = str(host)
        self.port = int(port)
        self.reconnect_seconds = float(reconnect_seconds)
        self.cache_path = Path(cache_path)
        self.save_cache = bool(save_cache)
        self._store = _RawStateStore()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._last_error = ""

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="vr-raw-client", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            sock: Optional[socket.socket] = None
            client_name = f"{self.host}:{self.port}"
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
                sock.settimeout(1.0)
                self._connected = True
                self._last_error = ""
                if self.save_cache:
                    save_cached_endpoint(self.host, self.port, cache_path=self.cache_path)
                self._store.note_connect(client_name)
                file_obj = sock.makefile("rb")
                while not self._stop.is_set():
                    try:
                        raw = file_obj.readline()
                    except socket.timeout:
                        continue
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        self._store.update(client_name, payload)
            except OSError as exc:
                self._last_error = str(exc)
                self._connected = False
            finally:
                self._connected = False
                self._store.note_disconnect()
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
            if not self._stop.is_set():
                self._stop.wait(max(0.1, self.reconnect_seconds))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def latest_sample(self) -> Optional[dict[str, Any]]:
        return self.snapshot().get("latest_sample")

    def snapshot(self) -> dict[str, Any]:
        snap = self._store.snapshot()
        snap.update(
            {
                "mode": "connect",
                "host": self.host,
                "port": int(self.port),
                "connected": bool(self._connected),
                "last_error": self._last_error,
            }
        )
        return snap

    def banner_text(self) -> str:
        return f"VR receiver configured to connect to publisher at {self.host}:{self.port}"


class _PublishedSampleStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._payload: bytes = b""
        self._seq = -1

    def update(self, payload: bytes, seq: int) -> None:
        with self._cond:
            self._payload = payload
            self._seq = int(seq)
            self._cond.notify_all()

    def wait_for_next(self, last_seq: int, timeout: float = 1.0) -> tuple[int, bytes]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
            return self._seq, self._payload


class _PublisherRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        client = f"{self.client_address[0]}:{self.client_address[1]}"
        self.server.state_store.note_connect(client)  # type: ignore[attr-defined]
        last_seq = -1
        try:
            while True:
                seq, payload = self.server.publisher_store.wait_for_next(last_seq, timeout=1.0)  # type: ignore[attr-defined]
                if seq == last_seq or not payload:
                    continue
                self.wfile.write(payload)
                self.wfile.flush()
                last_seq = seq
        except Exception:
            pass
        finally:
            self.server.state_store.note_disconnect()  # type: ignore[attr-defined]


class VRPublisherServer:
    """Sample a local backend and stream raw JSON packets to any connected clients."""

    def __init__(
        self,
        backend: Any,
        *,
        host: str = "0.0.0.0",
        port: int = DEFAULT_VR_PORT,
        rate_hz: float = 60.0,
        cache_path: Path | str = DEFAULT_VR_CACHE_PATH,
        save_cache: bool = False,
    ):
        self.backend = backend
        self.host = str(host)
        self.port = int(port)
        self.rate_hz = float(rate_hz)
        self.cache_path = Path(cache_path)
        self.save_cache = bool(save_cache)
        self._server: Optional[_VRJSONTCPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._sampler_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._state = _RawStateStore()
        self._publisher_store = _PublishedSampleStore()
        self._latest_sample: Optional[dict[str, Any]] = None

    def start(self) -> None:
        if self._server is not None:
            return
        server = _VRJSONTCPServer((self.host, self.port), _PublisherRequestHandler)
        server.state_store = self._state  # type: ignore[attr-defined]
        server.publisher_store = self._publisher_store  # type: ignore[attr-defined]
        self._server = server
        self.host, self.port = server.server_address[:2]
        if self.save_cache:
            save_cached_endpoint(cacheable_host(self.host), self.port, cache_path=self.cache_path)
        self._thread = threading.Thread(target=server.serve_forever, name="vr-publisher-server", daemon=True)
        self._thread.start()
        self._sampler_thread = threading.Thread(target=self._run_sampler, name="vr-publisher-sampler", daemon=True)
        self._sampler_thread.start()

    def _run_sampler(self) -> None:
        period = 1.0 / max(1.0, self.rate_hz)
        seq = 0
        while not self._stop.is_set():
            sample = self.backend.sample(seq)
            self._latest_sample = sample if isinstance(sample, dict) else None
            payload = json.dumps(sample, separators=(",", ":")).encode("utf-8") + b"\n"
            self._publisher_store.update(payload, seq)
            seq += 1
            self._stop.wait(period)

    def close(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._sampler_thread is not None and self._sampler_thread.is_alive():
            self._sampler_thread.join(timeout=1.0)
        if hasattr(self.backend, "close"):
            self.backend.close()

    def latest_sample(self) -> Optional[dict[str, Any]]:
        return self._latest_sample

    def snapshot(self) -> dict[str, Any]:
        snap = self._state.snapshot()
        snap.update(
            {
                "mode": "publish",
                "host": self.host,
                "port": int(self.port),
                "publisher_addresses": discover_local_ipv4_addresses(),
                "latest_sample": self._latest_sample,
            }
        )
        return snap

    def banner_text(self) -> str:
        return format_publisher_banner(self.host, self.port)


def _optional_pygame():
    try:
        import pygame  # type: ignore

        return pygame
    except Exception:
        return None


def prompt_sender_target_console(default_host: str = "127.0.0.1", default_port: int = DEFAULT_VR_PORT) -> tuple[str, int]:
    print("VR sender target selection")
    print("  Press ENTER to accept defaults.")
    print(f"  default host: {default_host}")
    print(f"  default port: {default_port}")
    host = input("Receiver host: ").strip() or str(default_host)
    port_text = input("Receiver port: ").strip()
    port = int(port_text) if port_text else int(default_port)
    return host, port


def _prompt_sender_target_pygame(*, pygame, default_host: str, default_port: int) -> tuple[str, int]:
    pygame.init()
    screen = pygame.display.set_mode((680, 220))
    pygame.display.set_caption("VR Sender Target")
    font = pygame.font.SysFont("Arial", 22)
    small = pygame.font.SysFont("Arial", 18)
    host = str(default_host)
    port = str(default_port)
    field = "host"
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return default_host, int(default_port)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return default_host, int(default_port)
                if event.key == pygame.K_TAB:
                    field = "port" if field == "host" else "host"
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    pygame.quit()
                    host_value = host.strip() or default_host
                    port_value = int(port.strip() or str(default_port))
                    return host_value, port_value
                elif event.key == pygame.K_BACKSPACE:
                    if field == "host":
                        host = host[:-1]
                    else:
                        port = port[:-1]
                else:
                    ch = event.unicode
                    if field == "host":
                        if ch and ch.isprintable():
                            host += ch
                    else:
                        if ch.isdigit():
                            port += ch

        screen.fill((18, 18, 22))
        lines = [
            "Enter the receiver address shown on the laptop.",
            "TAB switches fields. ENTER confirms. ESC uses defaults.",
        ]
        y = 18
        for text in lines:
            screen.blit(small.render(text, True, (220, 220, 220)), (16, y))
            y += 24

        host_box = pygame.Rect(16, 86, 500, 42)
        port_box = pygame.Rect(532, 86, 132, 42)
        pygame.draw.rect(screen, (70, 70, 90), host_box, border_radius=6)
        pygame.draw.rect(screen, (70, 70, 90), port_box, border_radius=6)
        active_color = (210, 180, 70)
        inactive_color = (120, 120, 140)
        pygame.draw.rect(screen, active_color if field == "host" else inactive_color, host_box, 2, border_radius=6)
        pygame.draw.rect(screen, active_color if field == "port" else inactive_color, port_box, 2, border_radius=6)
        screen.blit(font.render(host or default_host, True, (235, 235, 235)), (24, 94))
        screen.blit(font.render(port or str(default_port), True, (235, 235, 235)), (540, 94))
        screen.blit(small.render("Host", True, (220, 220, 220)), (18, 62))
        screen.blit(small.render("Port", True, (220, 220, 220)), (534, 62))
        screen.blit(
            small.render(f"Defaults: {default_host}:{default_port}", True, (200, 200, 200)),
            (16, 162),
        )
        pygame.display.flip()
        clock.tick(30)


def prompt_sender_target(
    default_host: str = "127.0.0.1",
    default_port: int = DEFAULT_VR_PORT,
    *,
    ui_mode: str = "auto",
) -> tuple[str, int]:
    mode = str(ui_mode).strip().lower()
    pygame = _optional_pygame()
    if mode in {"pygame", "auto"} and pygame is not None:
        return _prompt_sender_target_pygame(
            pygame=pygame,
            default_host=default_host,
            default_port=default_port,
        )
    return prompt_sender_target_console(default_host=default_host, default_port=default_port)


def extract_controller_state(sample: Optional[dict[str, Any]], hand: str = "right") -> Optional[dict[str, Any]]:
    if not isinstance(sample, dict):
        return None
    devices = sample.get("devices")
    if isinstance(devices, dict):
        item = devices.get(hand)
        if isinstance(item, dict):
            return item
        for value in devices.values():
            if isinstance(value, dict) and str(value.get("role", "")).lower() == hand:
                return value
    return None


def _pressed_id_set(controller_state: dict[str, Any]) -> set[int]:
    ids = controller_state.get("pressed_button_ids")
    if not isinstance(ids, list):
        return set()
    out: set[int] = set()
    for value in ids:
        try:
            out.add(int(value))
        except Exception:
            continue
    return out


def controller_named_button_states(controller_state: Optional[dict[str, Any]]) -> dict[str, bool]:
    if not isinstance(controller_state, dict):
        return {name: False for name in VR_GATE_BUTTON_OPTIONS}
    named = controller_state.get("buttons")
    out: dict[str, bool] = {}
    for name in VR_GATE_BUTTON_OPTIONS:
        value = False
        if isinstance(named, dict) and name in named:
            value = bool(named.get(name, False))
        else:
            button_id = OPENVR_BUTTON_ALIASES.get(name, -1)
            value = button_id >= 0 and button_id in _pressed_id_set(controller_state)
        out[name] = bool(value)
    return out


def controller_named_pressed_buttons(controller_state: Optional[dict[str, Any]]) -> list[str]:
    states = controller_named_button_states(controller_state)
    return [name for name in VR_GATE_BUTTON_OPTIONS if bool(states.get(name, False))]


def controller_button_pressed(controller_state: Optional[dict[str, Any]], button: Optional[str]) -> bool:
    if not isinstance(controller_state, dict):
        return False
    key = str(button or "").strip().lower()
    if not key:
        return False
    named = controller_state.get("buttons")
    if isinstance(named, dict) and key in named:
        return bool(named.get(key))
    try:
        button_id = int(key)
    except Exception:
        button_id = OPENVR_BUTTON_ALIASES.get(key, -1)
    if button_id < 0:
        return False
    return button_id in _pressed_id_set(controller_state)


def _axis_component(controller_state: dict[str, Any], axis_idx: int, component: str) -> float:
    axes = controller_state.get("axes")
    if not isinstance(axes, list) or axis_idx < 0 or axis_idx >= len(axes):
        return 0.0
    axis = axes[axis_idx]
    if not isinstance(axis, dict):
        return 0.0
    try:
        return float(axis.get(component, 0.0))
    except Exception:
        return 0.0


def controller_axis_value(controller_state: Optional[dict[str, Any]], axis_name: str) -> float:
    if not isinstance(controller_state, dict):
        return 0.0
    key = str(axis_name or "").strip().lower()
    if key == "trigger":
        if "trigger_value" in controller_state:
            try:
                return float(controller_state["trigger_value"])
            except Exception:
                return 0.0
        return _axis_component(controller_state, 1, "x")
    if key == "trackpad_x":
        return _axis_component(controller_state, 0, "x")
    if key == "trackpad_y":
        return _axis_component(controller_state, 0, "y")
    if key == "joystick_x":
        return _axis_component(controller_state, 2, "x")
    if key == "joystick_y":
        return _axis_component(controller_state, 2, "y")
    if "." in key:
        axis_idx, component = key.split(".", 1)
        try:
            return _axis_component(controller_state, int(axis_idx), component)
        except Exception:
            return 0.0
    return 0.0


def quaternion_wxyz_to_yaw(quat_wxyz: Any) -> float:
    np = _np()
    try:
        q = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1)
        if q.size < 4:
            return 0.0
        w, x, y, z = [float(v) for v in q[:4]]
    except Exception:
        return 0.0
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def quaternion_wxyz_to_rpy(quat_wxyz: Any) -> tuple[float, float, float]:
    np = _np()
    try:
        q = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1)
        if q.size < 4:
            return 0.0, 0.0, 0.0
        w, x, y, z = [float(v) for v in q[:4]]
    except Exception:
        return 0.0, 0.0, 0.0
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return float(angle)


def _position_source_value(delta_pos: Any, source: str) -> float:
    key = str(source or "").strip().lower()
    if key == "none":
        return 0.0
    sign = -1.0 if key.startswith("-") else 1.0
    axis = key[1:] if key.startswith("-") else key
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis, None)
    if axis_idx is None:
        return 0.0
    try:
        return float(delta_pos[axis_idx]) * sign
    except Exception:
        return 0.0


def _rotation_source_value(rpy: tuple[float, float, float], source: str) -> float:
    key = str(source or "").strip().lower()
    if key == "none":
        return 0.0
    alias_map = {
        "yaw": "global_yaw",
        "-yaw": "-global_yaw",
        "pitch": "global_pitch",
        "-pitch": "-global_pitch",
        "roll": "global_roll",
        "-roll": "-global_roll",
    }
    key = alias_map.get(key, key)
    sign = -1.0 if key.startswith("-") else 1.0
    axis = key[1:] if key.startswith("-") else key
    axis_idx = {"global_roll": 0, "global_pitch": 1, "global_yaw": 2}.get(axis, None)
    if axis_idx is None:
        return 0.0
    try:
        return float(rpy[axis_idx]) * sign
    except Exception:
        return 0.0


@dataclass
class VRManipMappingConfig:
    hand: str = "right"
    require_gate: bool = False
    gate_button: str = "grip"
    gripper_mirror_toggle_button: str = "none"
    action_x_source: str = "x"
    action_y_source: str = "y"
    action_z_source: str = "z"
    rotation_source: str = "global_yaw"
    motion_control_mode: str = "target_hold"
    mirror_gripper_when_inactive: bool = False
    position_gain: float = 25.0
    position_response_gain: float = 1.5
    position_feedforward: float = 0.5
    yaw_gain: float = 2.5
    yaw_response_gain: float = 1.5
    yaw_feedforward: float = 0.5
    gripper_gain: float = 5.0
    trigger_axis: str = "trigger"
    gripper_control_mode: str = "absolute"
    binary_gripper: bool = False
    trigger_close_threshold: float = 0.6
    trigger_open_threshold: float = 0.2
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    invert_yaw: bool = False
    invert_gripper: bool = False


def vr_mapping_config_to_dict(config: VRManipMappingConfig) -> dict[str, Any]:
    return {
        "hand": str(config.hand),
        "require_gate": bool(config.require_gate),
        "gate_button": str(config.gate_button),
        "gripper_mirror_toggle_button": str(config.gripper_mirror_toggle_button),
        "action_x_source": str(config.action_x_source),
        "action_y_source": str(config.action_y_source),
        "action_z_source": str(config.action_z_source),
        "rotation_source": str(config.rotation_source),
        "motion_control_mode": str(config.motion_control_mode),
        "mirror_gripper_when_inactive": bool(config.mirror_gripper_when_inactive),
        "position_gain": float(config.position_gain),
        "position_response_gain": float(config.position_response_gain),
        "position_feedforward": float(config.position_feedforward),
        "yaw_gain": float(config.yaw_gain),
        "yaw_response_gain": float(config.yaw_response_gain),
        "yaw_feedforward": float(config.yaw_feedforward),
        "gripper_gain": float(config.gripper_gain),
        "trigger_axis": str(config.trigger_axis),
        "gripper_control_mode": str(config.gripper_control_mode),
        "binary_gripper": bool(config.binary_gripper),
        "trigger_close_threshold": float(config.trigger_close_threshold),
        "trigger_open_threshold": float(config.trigger_open_threshold),
        "invert_x": bool(config.invert_x),
        "invert_y": bool(config.invert_y),
        "invert_z": bool(config.invert_z),
        "invert_yaw": bool(config.invert_yaw),
        "invert_gripper": bool(config.invert_gripper),
    }


def _sanitize_choice(value: Any, options: tuple[str, ...], default: str) -> str:
    text = str(value or "").strip()
    return text if text in options else default


def _sanitize_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _sanitize_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def vr_mapping_config_from_payload(payload: Optional[dict[str, Any]]) -> VRManipMappingConfig:
    data = payload if isinstance(payload, dict) else {}
    defaults = VRManipMappingConfig()
    return VRManipMappingConfig(
        hand=_sanitize_choice(data.get("hand"), VR_HAND_OPTIONS, defaults.hand),
        require_gate=_sanitize_bool(data.get("require_gate"), defaults.require_gate),
        gate_button=str(data.get("gate_button", defaults.gate_button) or defaults.gate_button),
        gripper_mirror_toggle_button=_sanitize_choice(
            data.get("gripper_mirror_toggle_button"),
            VR_OPTIONAL_BUTTON_OPTIONS,
            defaults.gripper_mirror_toggle_button,
        ),
        action_x_source=_sanitize_choice(data.get("action_x_source"), VR_POSITION_SOURCE_OPTIONS, defaults.action_x_source),
        action_y_source=_sanitize_choice(data.get("action_y_source"), VR_POSITION_SOURCE_OPTIONS, defaults.action_y_source),
        action_z_source=_sanitize_choice(data.get("action_z_source"), VR_POSITION_SOURCE_OPTIONS, defaults.action_z_source),
        rotation_source=_sanitize_choice(data.get("rotation_source"), VR_ROTATION_SOURCE_OPTIONS, defaults.rotation_source),
        motion_control_mode=_sanitize_choice(
            data.get("motion_control_mode"),
            VR_MOTION_CONTROL_OPTIONS,
            defaults.motion_control_mode,
        ),
        mirror_gripper_when_inactive=_sanitize_bool(
            data.get("mirror_gripper_when_inactive"),
            defaults.mirror_gripper_when_inactive,
        ),
        position_gain=_sanitize_float(data.get("position_gain"), defaults.position_gain),
        position_response_gain=_sanitize_float(
            data.get("position_response_gain"),
            defaults.position_response_gain,
        ),
        position_feedforward=_sanitize_float(
            data.get("position_feedforward"),
            defaults.position_feedforward,
        ),
        yaw_gain=_sanitize_float(data.get("yaw_gain"), defaults.yaw_gain),
        yaw_response_gain=_sanitize_float(
            data.get("yaw_response_gain"),
            defaults.yaw_response_gain,
        ),
        yaw_feedforward=_sanitize_float(
            data.get("yaw_feedforward"),
            defaults.yaw_feedforward,
        ),
        gripper_gain=_sanitize_float(data.get("gripper_gain"), defaults.gripper_gain),
        trigger_axis=_sanitize_choice(data.get("trigger_axis"), VR_GRIPPER_AXIS_OPTIONS, defaults.trigger_axis),
        gripper_control_mode=_sanitize_choice(
            data.get("gripper_control_mode"),
            VR_GRIPPER_CONTROL_OPTIONS,
            defaults.gripper_control_mode,
        ),
        binary_gripper=_sanitize_bool(data.get("binary_gripper"), defaults.binary_gripper),
        trigger_close_threshold=_sanitize_float(data.get("trigger_close_threshold"), defaults.trigger_close_threshold),
        trigger_open_threshold=_sanitize_float(data.get("trigger_open_threshold"), defaults.trigger_open_threshold),
        invert_x=_sanitize_bool(data.get("invert_x"), defaults.invert_x),
        invert_y=_sanitize_bool(data.get("invert_y"), defaults.invert_y),
        invert_z=_sanitize_bool(data.get("invert_z"), defaults.invert_z),
        invert_yaw=_sanitize_bool(data.get("invert_yaw"), defaults.invert_yaw),
        invert_gripper=_sanitize_bool(data.get("invert_gripper"), defaults.invert_gripper),
    )


def load_vr_mapping_config(mapping_path: Path | str = DEFAULT_VR_MAPPING_PATH) -> Optional[VRManipMappingConfig]:
    path = _resolve_vr_runtime_path(mapping_path, legacy_path=LEGACY_VR_MAPPING_PATH)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return vr_mapping_config_from_payload(payload)


def save_vr_mapping_config(config: VRManipMappingConfig, mapping_path: Path | str = DEFAULT_VR_MAPPING_PATH) -> Path:
    path = _normalize_vr_path(mapping_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = vr_mapping_config_to_dict(config)
    payload["updated_at"] = float(time.time())
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def apply_vr_mapping_profile(
    base_config: VRManipMappingConfig,
    mapping_path: Path | str = DEFAULT_VR_MAPPING_PATH,
) -> tuple[VRManipMappingConfig, bool]:
    loaded = load_vr_mapping_config(mapping_path)
    if loaded is None:
        return base_config, False
    return loaded, True


class VRManipActionMapper:
    """Map raw VR controller state into 4D/5D manipulation actions."""

    def __init__(self, action_dim: int, config: Optional[VRManipMappingConfig] = None):
        self.action_dim = int(action_dim)
        self.config = config or VRManipMappingConfig()
        self._last_position: Optional[Any] = None
        self._last_rotation_value: Optional[float] = None
        self._last_trigger: Optional[float] = None
        self._motion_active_prev = False
        self._anchor_controller_position: Optional[Any] = None
        self._anchor_controller_rotation: Optional[float] = None
        self._anchor_trigger_value: Optional[float] = None
        self._anchor_robot_position: Optional[Any] = None
        self._anchor_robot_rotation: Optional[float] = None
        self._anchor_robot_gripper: Optional[float] = None
        self._target_robot_position: Optional[Any] = None
        self._target_robot_rotation: Optional[float] = None
        self._target_robot_gripper: Optional[float] = None
        self._gripper_mirror_enabled = bool(self.config.mirror_gripper_when_inactive)
        self._gripper_toggle_prev_pressed: Optional[bool] = None

    def reset(self, *, preserve_runtime_toggles: bool = True) -> None:
        self._last_position = None
        self._last_rotation_value = None
        self._last_trigger = None
        self._motion_active_prev = False
        self._anchor_controller_position = None
        self._anchor_controller_rotation = None
        self._anchor_trigger_value = None
        self._anchor_robot_position = None
        self._anchor_robot_rotation = None
        self._anchor_robot_gripper = None
        self._target_robot_position = None
        self._target_robot_rotation = None
        self._target_robot_gripper = None
        if not preserve_runtime_toggles:
            self._gripper_mirror_enabled = bool(self.config.mirror_gripper_when_inactive)
        self._gripper_toggle_prev_pressed = None

    def _extract_robot_state(self, robot_state: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        np = _np()
        if not isinstance(robot_state, dict):
            return None
        try:
            effector_pos = np.asarray(robot_state.get("proprio/effector_pos"), dtype=np.float32).reshape(3)
            effector_yaw = float(np.asarray(robot_state.get("proprio/effector_yaw"), dtype=np.float32).reshape(-1)[0])
            gripper_opening = float(np.asarray(robot_state.get("proprio/gripper_opening"), dtype=np.float32).reshape(-1)[0])
        except Exception:
            return None
        return {
            "effector_pos": effector_pos,
            "effector_yaw": effector_yaw,
            "gripper_opening": float(np.clip(gripper_opening, 0.0, 1.0)),
        }

    def map_sample(
        self,
        sample: Optional[dict[str, Any]],
        *,
        robot_state: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        np = _np()
        action_range = np.asarray(MANIP_ACTION_RANGE, dtype=np.float32).reshape(5)
        action = np.zeros((self.action_dim,), dtype=np.float32)
        has_yaw_channel = self.action_dim >= 5
        gripper_idx = 4 if self.action_dim >= 5 else (3 if self.action_dim == 4 else None)
        diag = {
            "connected": False,
            "tracked": False,
            "gate_pressed": False,
            "motion_active": False,
            "gripper_mirror_enabled": bool(self._gripper_mirror_enabled),
            "gripper_mirror_toggle_button": str(self.config.gripper_mirror_toggle_button),
            "trigger_value": 0.0,
            "hand": self.config.hand,
            "raw_delta_position": [0.0, 0.0, 0.0],
            "raw_delta_rotation": 0.0,
            "mapped_action": [0.0] * self.action_dim,
            "robot_state_available": False,
            "target_robot_position": [0.0, 0.0, 0.0],
            "target_robot_rotation": 0.0,
            "target_robot_gripper": 0.0,
            "motion_control_mode": self.config.motion_control_mode,
            "gripper_control_mode": self.config.gripper_control_mode,
            "position_response_gain": float(self.config.position_response_gain),
            "position_feedforward": float(self.config.position_feedforward),
            "yaw_response_gain": float(self.config.yaw_response_gain),
            "yaw_feedforward": float(self.config.yaw_feedforward),
            "tracking_position_error": [0.0, 0.0, 0.0],
            "tracking_rotation_error": 0.0,
        }
        controller = extract_controller_state(sample, hand=self.config.hand)
        if not isinstance(controller, dict):
            self.reset()
            return action, diag

        connected = bool(controller.get("connected", False))
        tracked = bool(controller.get("tracked", False))
        gate_pressed = controller_button_pressed(controller, self.config.gate_button)
        toggle_pressed = controller_button_pressed(controller, self.config.gripper_mirror_toggle_button)
        if self._gripper_toggle_prev_pressed is None:
            self._gripper_toggle_prev_pressed = bool(toggle_pressed)
        else:
            if bool(toggle_pressed) and (not bool(self._gripper_toggle_prev_pressed)):
                self._gripper_mirror_enabled = not bool(self._gripper_mirror_enabled)
            self._gripper_toggle_prev_pressed = bool(toggle_pressed)
        motion_active = gate_pressed or (not self.config.require_gate)
        diag.update(
            {
                "connected": connected,
                "tracked": tracked,
                "gate_pressed": gate_pressed,
                "motion_active": motion_active,
                "gripper_mirror_enabled": bool(self._gripper_mirror_enabled),
            }
        )
        if not connected or not tracked:
            self.reset()
            return action, diag

        pose = controller.get("pose")
        if not isinstance(pose, dict):
            self.reset()
            return action, diag
        try:
            position = np.asarray(pose.get("position", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
        except Exception:
            self.reset()
            return action, diag
        robot_pose = self._extract_robot_state(robot_state)
        diag["robot_state_available"] = bool(robot_pose is not None)
        rpy = quaternion_wxyz_to_rpy(pose.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]))
        rotation_value = _rotation_source_value(rpy, self.config.rotation_source)
        trigger_value = float(np.clip(controller_axis_value(controller, self.config.trigger_axis), 0.0, 1.0))
        if self.config.invert_gripper:
            trigger_value = 1.0 - trigger_value
        diag["trigger_value"] = trigger_value
        if self._last_position is not None:
            raw_delta_pos = position - self._last_position
            diag["raw_delta_position"] = [float(v) for v in raw_delta_pos.tolist()]
        else:
            raw_delta_pos = None
        if self._last_rotation_value is not None:
            raw_delta_rotation = _wrap_angle(rotation_value - float(self._last_rotation_value))
            diag["raw_delta_rotation"] = float(raw_delta_rotation)
        else:
            raw_delta_rotation = None

        motion_just_enabled = bool(motion_active and (not self._motion_active_prev))
        if motion_just_enabled and robot_pose is not None:
            self._anchor_controller_position = position.copy()
            self._anchor_controller_rotation = float(rotation_value)
            self._anchor_trigger_value = float(trigger_value)
            self._anchor_robot_position = robot_pose["effector_pos"].copy()
            self._anchor_robot_rotation = float(robot_pose["effector_yaw"])
            self._anchor_robot_gripper = float(robot_pose["gripper_opening"])
            self._target_robot_position = robot_pose["effector_pos"].copy()
            self._target_robot_rotation = float(robot_pose["effector_yaw"])
            self._target_robot_gripper = float(robot_pose["gripper_opening"])

        can_integrate_motion = (
            motion_active
            and self._last_position is not None
            and self._last_rotation_value is not None
            and (not motion_just_enabled)
        )
        if can_integrate_motion:
            delta_pos = raw_delta_pos
            delta_rotation = raw_delta_rotation
            mapped_dx = float(
                _position_source_value(delta_pos, self.config.action_x_source)
                * self.config.position_gain
                * (-1.0 if self.config.invert_x else 1.0)
            )
            mapped_dy = float(
                _position_source_value(delta_pos, self.config.action_y_source)
                * self.config.position_gain
                * (-1.0 if self.config.invert_y else 1.0)
            )
            mapped_dz = float(
                _position_source_value(delta_pos, self.config.action_z_source)
                * self.config.position_gain
                * (-1.0 if self.config.invert_z else 1.0)
            )
            mapped_drot = float(delta_rotation * self.config.yaw_gain * (-1.0 if self.config.invert_yaw else 1.0))
            if (
                str(self.config.motion_control_mode).strip().lower() == "target_hold"
                and robot_pose is not None
            ):
                if self._anchor_controller_position is None:
                    self._anchor_controller_position = position.copy()
                if self._anchor_robot_position is None:
                    self._anchor_robot_position = robot_pose["effector_pos"].copy()
                if self._anchor_controller_rotation is None:
                    self._anchor_controller_rotation = float(rotation_value)
                if self._anchor_robot_rotation is None:
                    self._anchor_robot_rotation = float(robot_pose["effector_yaw"])
                controller_offset = position - np.asarray(self._anchor_controller_position, dtype=np.float32).reshape(3)
                target_offset = np.asarray(
                    [
                        _position_source_value(controller_offset, self.config.action_x_source)
                        * self.config.position_gain
                        * (-1.0 if self.config.invert_x else 1.0),
                        _position_source_value(controller_offset, self.config.action_y_source)
                        * self.config.position_gain
                        * (-1.0 if self.config.invert_y else 1.0),
                        _position_source_value(controller_offset, self.config.action_z_source)
                        * self.config.position_gain
                        * (-1.0 if self.config.invert_z else 1.0),
                    ],
                    dtype=np.float32,
                )
                controller_rot_offset = _wrap_angle(float(rotation_value) - float(self._anchor_controller_rotation))
                target_rot_offset = controller_rot_offset * self.config.yaw_gain * (-1.0 if self.config.invert_yaw else 1.0)
                self._target_robot_position = (
                    np.asarray(self._anchor_robot_position, dtype=np.float32).reshape(3) + target_offset
                )
                self._target_robot_rotation = _wrap_angle(float(self._anchor_robot_rotation) + target_rot_offset)
                pos_error = np.asarray(self._target_robot_position, dtype=np.float32) - robot_pose["effector_pos"]
                rot_error = _wrap_angle(float(self._target_robot_rotation) - float(robot_pose["effector_yaw"]))
                diag["tracking_position_error"] = [float(v) for v in pos_error.tolist()]
                diag["tracking_rotation_error"] = float(rot_error)
                pos_response_gain = max(0.0, float(self.config.position_response_gain))
                pos_feedforward_gain = max(0.0, float(self.config.position_feedforward))
                yaw_response_gain = max(0.0, float(self.config.yaw_response_gain))
                yaw_feedforward_gain = max(0.0, float(self.config.yaw_feedforward))
                if self.action_dim >= 1:
                    action[0] = float(
                        (pos_error[0] / action_range[0]) * pos_response_gain
                        + (mapped_dx / action_range[0]) * pos_feedforward_gain
                    )
                if self.action_dim >= 2:
                    action[1] = float(
                        (pos_error[1] / action_range[1]) * pos_response_gain
                        + (mapped_dy / action_range[1]) * pos_feedforward_gain
                    )
                if self.action_dim >= 3:
                    action[2] = float(
                        (pos_error[2] / action_range[2]) * pos_response_gain
                        + (mapped_dz / action_range[2]) * pos_feedforward_gain
                    )
                if has_yaw_channel:
                    action[3] = float(
                        (rot_error / action_range[3]) * yaw_response_gain
                        + (mapped_drot / action_range[3]) * yaw_feedforward_gain
                    )
            else:
                if self.action_dim >= 1:
                    action[0] = mapped_dx
                if self.action_dim >= 2:
                    action[1] = mapped_dy
                if self.action_dim >= 3:
                    action[2] = mapped_dz
                if has_yaw_channel:
                    action[3] = mapped_drot

        can_integrate_gripper = motion_active or bool(self._gripper_mirror_enabled)
        if gripper_idx is not None and can_integrate_gripper:
            if self.config.binary_gripper:
                if trigger_value >= self.config.trigger_close_threshold:
                    action[gripper_idx] = 1.0
                elif trigger_value <= self.config.trigger_open_threshold:
                    action[gripper_idx] = -1.0
                else:
                    action[gripper_idx] = 0.0
            elif (
                str(self.config.gripper_control_mode).strip().lower() == "absolute"
                and robot_pose is not None
            ):
                # True absolute gripper control over the full actuator range while
                # preserving the original trigger direction used in this repo's VR setup.
                desired_opening = float(np.clip(float(trigger_value), 0.0, 1.0))
                self._target_robot_gripper = desired_opening
                current_opening = float(robot_pose["gripper_opening"])
                gripper_range = float(action_range[4]) if self.action_dim >= 5 else 1.0
                action[gripper_idx] = float(
                    ((desired_opening - current_opening) / gripper_range) * float(self.config.gripper_gain)
                )
            elif self._last_trigger is not None:
                action[gripper_idx] = float((trigger_value - self._last_trigger) * self.config.gripper_gain)

        self._last_position = position.copy()
        self._last_rotation_value = float(rotation_value)
        self._last_trigger = float(trigger_value)
        self._motion_active_prev = bool(motion_active)
        np.clip(action, -1.0, 1.0, out=action)
        diag["mapped_action"] = [float(v) for v in action.tolist()]
        diag["raw_delta_yaw"] = float(diag.get("raw_delta_rotation", 0.0))
        if self._target_robot_position is not None:
            diag["target_robot_position"] = [float(v) for v in np.asarray(self._target_robot_position).reshape(3).tolist()]
        if self._target_robot_rotation is not None:
            diag["target_robot_rotation"] = float(self._target_robot_rotation)
        if self._target_robot_gripper is not None:
            diag["target_robot_gripper"] = float(self._target_robot_gripper)
        return action, diag


class VRTeleopInterface:
    """Teleop interface compatible with InterventionWrapper(get_action)."""

    def __init__(
        self,
        server: Any,
        mapper: VRManipActionMapper,
        *,
        return_none_when_idle: bool = True,
        idle_threshold: float = 1e-6,
        mapping_path: Optional[Path | str] = None,
    ):
        self.server = server
        self.mapper = mapper
        self.return_none_when_idle = bool(return_none_when_idle)
        self.idle_threshold = float(idle_threshold)
        self._last_diag: dict[str, Any] = {}
        self._robot_state: Optional[dict[str, Any]] = None
        self._suppress_until_gate_release = False
        self.mapping_path = Path(mapping_path) if mapping_path is not None else None
        self._mapping_mtime = 0.0
        if self.mapping_path is not None:
            try:
                self._mapping_mtime = float(self.mapping_path.stat().st_mtime)
            except Exception:
                self._mapping_mtime = 0.0

    def _reload_mapping_if_changed(self) -> None:
        if self.mapping_path is None:
            return
        try:
            current_mtime = float(self.mapping_path.stat().st_mtime)
        except Exception:
            current_mtime = 0.0
        if current_mtime <= 0.0 or current_mtime == self._mapping_mtime:
            return
        loaded = load_vr_mapping_config(self.mapping_path)
        if loaded is None:
            self._mapping_mtime = current_mtime
            return
        loaded_dict = vr_mapping_config_to_dict(loaded)
        current_dict = vr_mapping_config_to_dict(self.mapper.config)
        if loaded_dict != current_dict:
            for key, value in loaded_dict.items():
                setattr(self.mapper.config, key, value)
            self.mapper.reset(preserve_runtime_toggles=False)
        self._mapping_mtime = current_mtime

    def get_action(self) -> Optional[np.ndarray]:
        np = _np()
        self._reload_mapping_if_changed()
        sample = self.server.latest_sample()
        if self._suppress_until_gate_release and bool(getattr(self.mapper.config, "require_gate", False)):
            controller = extract_controller_state(sample, hand=self.mapper.config.hand)
            gate_pressed = controller_button_pressed(controller, self.mapper.config.gate_button)
            if not gate_pressed:
                self._suppress_until_gate_release = False
            else:
                diag = {
                    "connected": bool(controller.get("connected", False)) if isinstance(controller, dict) else False,
                    "tracked": bool(controller.get("tracked", False)) if isinstance(controller, dict) else False,
                    "gate_pressed": True,
                    "motion_active": False,
                    "trigger_value": float(
                        np.clip(controller_axis_value(controller, self.mapper.config.trigger_axis), 0.0, 1.0)
                    ) if isinstance(controller, dict) else 0.0,
                    "hand": self.mapper.config.hand,
                    "mapped_action": [0.0] * self.mapper.action_dim,
                    "reset_gate_latched": True,
                }
                self._last_diag = diag
                zero_action = np.zeros((self.mapper.action_dim,), dtype=np.float32)
                if not self.return_none_when_idle:
                    return zero_action
                return None
        action, diag = self.mapper.map_sample(sample, robot_state=self._robot_state)
        diag["reset_gate_latched"] = False
        self._last_diag = diag
        if not self.return_none_when_idle:
            return action
        if np.linalg.norm(action) > self.idle_threshold:
            return action
        if bool(diag.get("motion_active", False)) and bool(diag.get("connected", False)):
            return action
        return None

    def get_latest_sample(self) -> Optional[dict[str, Any]]:
        return self.server.latest_sample()

    def get_last_diag(self) -> dict[str, Any]:
        return dict(self._last_diag)

    def update_robot_state(self, robot_state: Optional[dict[str, Any]]) -> None:
        self._robot_state = dict(robot_state) if isinstance(robot_state, dict) else None

    def reset(self) -> None:
        self.mapper.reset()
        self._suppress_until_gate_release = bool(getattr(self.mapper.config, "require_gate", False))


class VRStatusPanel:
    """Optional pygame status window for receiver-side address, stream state, and VR mapping debug."""

    def __init__(
        self,
        title: str = "VR Receiver",
        width: Optional[int] = None,
        height: Optional[int] = None,
        *,
        action_dim: int = 5,
        mapping_config: Optional[VRManipMappingConfig] = None,
        mapping_path: Path | str = DEFAULT_VR_MAPPING_PATH,
        show_topdown: bool = False,
    ):
        pygame = _optional_pygame()
        if pygame is None:
            raise RuntimeError("pygame is not installed")
        self._pygame = pygame
        self._show_topdown = bool(show_topdown)
        width = int(width or (1200 if self._show_topdown else 960))
        height = int(height or (860 if self._show_topdown else 640))
        pygame.init()
        self._screen = pygame.display.set_mode((int(width), int(height)))
        pygame.display.set_caption(title)
        self._font = pygame.font.SysFont("Arial", 18)
        self._small = pygame.font.SysFont("Arial", 16)
        self._snapshot: dict[str, Any] = {}
        self._history_by_hand: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
        self._last_history_seq: Optional[int] = None
        self._mapping_path = Path(mapping_path)
        self._action_dim = int(action_dim)
        loaded = load_vr_mapping_config(self._mapping_path)
        self._mapping_config = mapping_config or loaded or VRManipMappingConfig()
        self._preview_mapper = VRManipActionMapper(action_dim=self._action_dim, config=self._mapping_config)
        self._preview_action = [0.0] * self._action_dim
        self._preview_diag: dict[str, Any] = {}
        self._topdown_state: Optional[dict[str, Any]] = None
        try:
            self._mapping_mtime = float(self._mapping_path.stat().st_mtime)
        except Exception:
            self._mapping_mtime = 0.0

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._reload_mapping_if_changed()
        self._snapshot = dict(snapshot or {})
        latest = self._snapshot.get("latest_sample")
        if not isinstance(latest, dict):
            return
        try:
            seq = int(latest.get("seq", -1))
        except Exception:
            seq = -1
        if seq >= 0 and seq == self._last_history_seq:
            return
        self._last_history_seq = seq if seq >= 0 else None
        for hand in ("left", "right"):
            controller = extract_controller_state(latest, hand)
            pose_data = self._controller_pose_data(controller)
            if pose_data is None:
                continue
            history = self._history_by_hand[hand]
            history.append(pose_data)
            if len(history) > 90:
                del history[:-90]
        preview_action, preview_diag = self._preview_mapper.map_sample(latest)
        self._preview_action = [float(v) for v in preview_action.tolist()]
        self._preview_diag = dict(preview_diag)

    def get_mapping_config(self) -> VRManipMappingConfig:
        return self._mapping_config

    def set_topdown_state(self, state: Optional[dict[str, Any]]) -> None:
        self._topdown_state = dict(state or {}) if isinstance(state, dict) else None

    def _reload_mapping_if_changed(self) -> None:
        try:
            current_mtime = float(self._mapping_path.stat().st_mtime)
        except Exception:
            current_mtime = 0.0
        if current_mtime <= 0.0 or current_mtime == self._mapping_mtime:
            return
        loaded = load_vr_mapping_config(self._mapping_path)
        if loaded is None:
            self._mapping_mtime = current_mtime
            return
        loaded_dict = vr_mapping_config_to_dict(loaded)
        current_dict = vr_mapping_config_to_dict(self._mapping_config)
        if loaded_dict != current_dict:
            for key, value in loaded_dict.items():
                setattr(self._mapping_config, key, value)
            self._preview_mapper.reset(preserve_runtime_toggles=False)
        self._mapping_mtime = current_mtime

    def _mapping_fields(self) -> list[dict[str, Any]]:
        return [
            {"attr": "hand", "label": "Hand", "kind": "choice", "options": list(VR_HAND_OPTIONS)},
            {"attr": "action_x_source", "label": "Robot X", "kind": "choice", "options": list(VR_POSITION_SOURCE_OPTIONS)},
            {"attr": "action_y_source", "label": "Robot Y", "kind": "choice", "options": list(VR_POSITION_SOURCE_OPTIONS)},
            {"attr": "action_z_source", "label": "Robot Z", "kind": "choice", "options": list(VR_POSITION_SOURCE_OPTIONS)},
            {"attr": "gate_button", "label": "Gate/Intervene", "kind": "choice", "options": list(VR_GATE_BUTTON_OPTIONS)},
            {
                "attr": "gripper_mirror_toggle_button",
                "label": "Grip Mirror Toggle",
                "kind": "choice",
                "options": list(VR_OPTIONAL_BUTTON_OPTIONS),
            },
            {"attr": "require_gate", "label": "Require Gate", "kind": "bool"},
            {"attr": "position_gain", "label": "XYZ Gain", "kind": "float", "step": 5.0, "min_value": 1.0, "max_value": 200.0},
            {"attr": "yaw_gain", "label": "Yaw Gain", "kind": "float", "step": 0.5, "min_value": 0.1, "max_value": 20.0},
            {"attr": "gripper_gain", "label": "Gripper Gain", "kind": "float", "step": 1.0, "min_value": 0.1, "max_value": 50.0},
            {"attr": "trigger_axis", "label": "Gripper Axis", "kind": "choice", "options": list(VR_GRIPPER_AXIS_OPTIONS)},
            {"attr": "mirror_gripper_when_inactive", "label": "Mirror Grip Idle", "kind": "bool"},
            {"attr": "binary_gripper", "label": "Binary Gripper", "kind": "bool"},
            {"attr": "invert_yaw", "label": "Invert Yaw", "kind": "bool"},
        ]

    def _persist_mapping(self) -> None:
        save_vr_mapping_config(self._mapping_config, self._mapping_path)

    def _update_mapping_field(self, delta: int) -> None:
        fields = self._mapping_fields()
        spec = fields[self._mapping_field_index % len(fields)]
        attr = str(spec["attr"])
        current = getattr(self._mapping_config, attr)
        kind = str(spec["kind"])
        if kind == "choice":
            options = list(spec["options"])
            try:
                idx = options.index(str(current))
            except Exception:
                idx = 0
            setattr(self._mapping_config, attr, options[(idx + int(delta)) % len(options)])
        elif kind == "bool":
            setattr(self._mapping_config, attr, not bool(current))
        elif kind == "float":
            step = float(spec.get("step", 1.0))
            min_value = float(spec.get("min_value", 0.0))
            max_value = float(spec.get("max_value", 9999.0))
            value = float(current) + step * float(delta)
            setattr(self._mapping_config, attr, max(min_value, min(max_value, value)))
        self._preview_mapper.reset(preserve_runtime_toggles=False)
        self._persist_mapping()

    def _controller_pose_data(self, controller: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not isinstance(controller, dict):
            return None
        if not bool(controller.get("tracked", False)):
            return None
        pose = controller.get("pose")
        if not isinstance(pose, dict):
            return None
        try:
            position = [float(v) for v in pose.get("position", [0.0, 0.0, 0.0])[:3]]
        except Exception:
            return None
        yaw = quaternion_wxyz_to_yaw(pose.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]))
        return {"position": position, "yaw": float(yaw)}

    def _draw_meter(self, rect, value: float, *, label: str, min_value: float, max_value: float) -> None:
        pygame = self._pygame
        screen = self._screen
        small = self._small
        pygame.draw.rect(screen, (34, 34, 42), rect, border_radius=6)
        pygame.draw.rect(screen, (100, 100, 120), rect, 1, border_radius=6)
        clamped = max(min_value, min(max_value, float(value)))
        frac = 0.0 if max_value <= min_value else (clamped - min_value) / (max_value - min_value)
        fill_h = int((rect.height - 6) * frac)
        fill_rect = pygame.Rect(rect.x + 3, rect.bottom - 3 - fill_h, rect.width - 6, fill_h)
        pygame.draw.rect(screen, (84, 170, 120), fill_rect, border_radius=5)
        screen.blit(small.render(label, True, (230, 230, 230)), (rect.x, rect.y - 18))
        screen.blit(small.render(f"{clamped:.2f}", True, (230, 230, 230)), (rect.x - 2, rect.bottom + 2))

    def _draw_axis_pad(self, rect, *, label: str, x_value: float, y_value: float) -> None:
        pygame = self._pygame
        screen = self._screen
        small = self._small
        pygame.draw.rect(screen, (34, 34, 42), rect, border_radius=6)
        pygame.draw.rect(screen, (100, 100, 120), rect, 1, border_radius=6)
        cx = rect.centerx
        cy = rect.centery
        pygame.draw.line(screen, (72, 72, 84), (rect.x + 8, cy), (rect.right - 8, cy), 1)
        pygame.draw.line(screen, (72, 72, 84), (cx, rect.y + 8), (cx, rect.bottom - 8), 1)
        px = int(cx + max(-1.0, min(1.0, float(x_value))) * (rect.width * 0.35))
        py = int(cy - max(-1.0, min(1.0, float(y_value))) * (rect.height * 0.35))
        pygame.draw.circle(screen, (220, 190, 80), (px, py), 6)
        screen.blit(small.render(label, True, (230, 230, 230)), (rect.x, rect.y - 18))
        screen.blit(
            small.render(f"x={float(x_value):+.2f} y={float(y_value):+.2f}", True, (210, 210, 210)),
            (rect.x, rect.bottom + 2),
        )

    def _draw_yaw_gauge(self, rect, *, yaw: float) -> None:
        pygame = self._pygame
        screen = self._screen
        small = self._small
        pygame.draw.rect(screen, (34, 34, 42), rect, border_radius=6)
        pygame.draw.rect(screen, (100, 100, 120), rect, 1, border_radius=6)
        center = rect.center
        radius = max(16, min(rect.width, rect.height) // 2 - 12)
        pygame.draw.circle(screen, (90, 90, 108), center, radius, 1)
        pygame.draw.line(screen, (70, 70, 84), (center[0], rect.y + 12), (center[0], rect.bottom - 12), 1)
        pygame.draw.line(screen, (70, 70, 84), (rect.x + 12, center[1]), (rect.right - 12, center[1]), 1)
        tip = (
            int(center[0] + math.cos(float(yaw)) * radius * 0.85),
            int(center[1] - math.sin(float(yaw)) * radius * 0.85),
        )
        pygame.draw.line(screen, (220, 190, 80), center, tip, 3)
        pygame.draw.circle(screen, (220, 190, 80), center, 4)
        screen.blit(small.render("Yaw", True, (230, 230, 230)), (rect.x, rect.y - 18))
        screen.blit(small.render(f"{float(yaw):+.2f} rad", True, (210, 210, 210)), (rect.x, rect.bottom + 2))

    def _draw_position_plot(self, rect, *, hand: str, current_position: Optional[list[float]]) -> None:
        pygame = self._pygame
        screen = self._screen
        small = self._small
        pygame.draw.rect(screen, (34, 34, 42), rect, border_radius=6)
        pygame.draw.rect(screen, (100, 100, 120), rect, 1, border_radius=6)
        screen.blit(small.render("XY motion trail", True, (230, 230, 230)), (rect.x, rect.y - 18))
        cx = rect.centerx
        cy = rect.centery
        pygame.draw.line(screen, (72, 72, 84), (rect.x + 10, cy), (rect.right - 10, cy), 1)
        pygame.draw.line(screen, (72, 72, 84), (cx, rect.y + 10), (cx, rect.bottom - 10), 1)
        scale_m = 0.15
        points: list[tuple[int, int]] = []
        history = self._history_by_hand.get(hand, [])
        if current_position is None and history:
            current_position = history[-1]["position"]
        if current_position is not None:
            cur_x, cur_y = float(current_position[0]), float(current_position[1])
            for item in history:
                pos = item.get("position")
                if not isinstance(pos, list) or len(pos) < 2:
                    continue
                dx = (float(pos[0]) - cur_x) / scale_m
                dy = (float(pos[1]) - cur_y) / scale_m
                px = int(cx + dx * (rect.width * 0.42))
                py = int(cy - dy * (rect.height * 0.42))
                px = max(rect.x + 8, min(rect.right - 8, px))
                py = max(rect.y + 8, min(rect.bottom - 8, py))
                points.append((px, py))
        if len(points) >= 2:
            pygame.draw.lines(screen, (94, 154, 226), False, points, 2)
        if points:
            pygame.draw.circle(screen, (220, 190, 80), points[-1], 6)
        if current_position is None:
            screen.blit(small.render("no tracked pose", True, (190, 190, 190)), (rect.x + 12, rect.y + 12))
        else:
            screen.blit(
                small.render(
                    f"x={float(current_position[0]):+.3f} y={float(current_position[1]):+.3f} z={float(current_position[2]):+.3f}",
                    True,
                    (210, 210, 210),
                ),
                (rect.x + 10, rect.bottom + 2),
            )

    def _draw_z_strip(self, rect, *, hand: str, current_position: Optional[list[float]]) -> None:
        pygame = self._pygame
        screen = self._screen
        small = self._small
        pygame.draw.rect(screen, (34, 34, 42), rect, border_radius=6)
        pygame.draw.rect(screen, (100, 100, 120), rect, 1, border_radius=6)
        screen.blit(small.render("Z", True, (230, 230, 230)), (rect.x, rect.y - 18))
        cx = rect.centerx
        pygame.draw.line(screen, (72, 72, 84), (cx, rect.y + 8), (cx, rect.bottom - 8), 1)
        history = self._history_by_hand.get(hand, [])
        if current_position is None and history:
            current_position = history[-1]["position"]
        if current_position is None:
            screen.blit(small.render("no", True, (190, 190, 190)), (rect.x + 2, rect.centery - 8))
            return
        cur_z = float(current_position[2])
        scale_m = 0.10
        pts: list[tuple[int, int]] = []
        for item in history:
            pos = item.get("position")
            if not isinstance(pos, list) or len(pos) < 3:
                continue
            dz = (float(pos[2]) - cur_z) / scale_m
            py = int(rect.centery - dz * (rect.height * 0.42))
            py = max(rect.y + 8, min(rect.bottom - 8, py))
            pts.append((cx, py))
        if len(pts) >= 2:
            pygame.draw.lines(screen, (94, 154, 226), False, pts, 2)
        if pts:
            pygame.draw.circle(screen, (220, 190, 80), pts[-1], 5)
        screen.blit(small.render(f"{cur_z:+.3f}", True, (210, 210, 210)), (rect.x - 4, rect.bottom + 2))

    def _draw_controller_card(self, rect, *, label: str, hand: str, controller: Optional[dict[str, Any]]) -> None:
        pygame = self._pygame
        screen = self._screen
        small = self._small
        pygame.draw.rect(screen, (42, 42, 52), rect, border_radius=6)
        pygame.draw.rect(screen, (100, 100, 120), rect, 1, border_radius=6)

        trigger = controller_axis_value(controller, "trigger")
        joy_x = controller_axis_value(controller, "joystick_x")
        joy_y = controller_axis_value(controller, "joystick_y")
        pad_x = controller_axis_value(controller, "trackpad_x")
        pad_y = controller_axis_value(controller, "trackpad_y")
        buttons = controller_named_pressed_buttons(controller)
        pose = controller.get("pose") if isinstance(controller, dict) else None
        position = None
        yaw = 0.0
        if isinstance(pose, dict):
            try:
                position = [float(v) for v in pose.get("position", [0.0, 0.0, 0.0])[:3]]
            except Exception:
                position = None
            yaw = quaternion_wxyz_to_yaw(pose.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]))

        header_lines = [
            f"{label}: connected={int(bool(isinstance(controller, dict) and controller.get('connected', False)))} tracked={int(bool(isinstance(controller, dict) and controller.get('tracked', False)))}",
            f"role={controller.get('role', hand) if isinstance(controller, dict) else hand} device_index={controller.get('device_index', -1) if isinstance(controller, dict) else -1}",
            f"buttons={buttons or ['-']}",
        ]
        y_text = rect.y + 10
        for text in header_lines:
            screen.blit(small.render(text, True, (230, 230, 230)), (rect.x + 12, y_text))
            y_text += 22

        inner_x = rect.x + 12
        inner_y = rect.y + 86
        inner_w = rect.width - 24
        z_w = 28
        plot_w = min(196, max(156, inner_w // 2 - 16))
        gap = 12
        z_x = inner_x + plot_w + 6
        right_x = z_x + z_w + gap
        right_w = max(120, rect.right - 12 - right_x)
        yaw_w = min(88, max(74, right_w // 2 - 8))
        yaw_h = 88
        trigger_w = min(22, max(18, right_w - yaw_w - gap))
        pad_gap = 12
        pad_w = max(72, (right_w - pad_gap) // 2)
        joy_w = max(72, right_w - pad_gap - pad_w)

        plot_rect = pygame.Rect(inner_x, inner_y, plot_w, 148)
        z_rect = pygame.Rect(z_x, inner_y + 10, z_w, 128)
        yaw_rect = pygame.Rect(right_x, inner_y + 12, yaw_w, yaw_h)
        trigger_rect = pygame.Rect(yaw_rect.right + gap, inner_y + 12, trigger_w, yaw_h)
        pad_rect = pygame.Rect(right_x, inner_y + 168, pad_w, 80)
        joy_rect = pygame.Rect(pad_rect.right + pad_gap, inner_y + 168, joy_w, 80)

        self._draw_position_plot(plot_rect, hand=hand, current_position=position)
        self._draw_z_strip(z_rect, hand=hand, current_position=position)
        self._draw_yaw_gauge(yaw_rect, yaw=yaw)
        self._draw_meter(trigger_rect, trigger, label="Trigger", min_value=0.0, max_value=1.0)
        self._draw_axis_pad(pad_rect, label="Trackpad", x_value=pad_x, y_value=pad_y)
        self._draw_axis_pad(joy_rect, label="Joystick", x_value=joy_x, y_value=joy_y)

    def _draw_mapping_summary(self, rect) -> None:
        pygame = self._pygame
        screen = self._screen
        font = self._font
        small = self._small
        pygame.draw.rect(screen, (42, 42, 52), rect, border_radius=6)
        pygame.draw.rect(screen, (100, 100, 120), rect, 1, border_radius=6)
        screen.blit(font.render("VR Mapping Profile", True, (220, 190, 80)), (rect.x + 12, rect.y + 10))
        raw_delta = self._preview_diag.get("raw_delta_position", [0.0, 0.0, 0.0])
        mapped = self._preview_diag.get("mapped_action", self._preview_action)
        gate = int(bool(self._preview_diag.get("gate_pressed", False)))
        tracked = int(bool(self._preview_diag.get("tracked", False)))
        motion = int(bool(self._preview_diag.get("motion_active", False)))
        cfg = self._mapping_config
        mapped_str = ", ".join(f"{float(v):+.3f}" for v in mapped)
        lines = [
            f"Profile: {self._mapping_path} | Web UI: http://127.0.0.1:{DEFAULT_VR_MAPPING_WEB_PORT}",
            (
                f"Input raw hand={cfg.hand} tracked={tracked} gate={gate} motion={motion} "
                f"raw_dxyz=[{float(raw_delta[0]):+.4f}, {float(raw_delta[1]):+.4f}, {float(raw_delta[2]):+.4f}] "
                f"drot={float(self._preview_diag.get('raw_delta_rotation', self._preview_diag.get('raw_delta_yaw', 0.0))):+.4f}"
            ),
            (
                f"Map out: robot_x<-{cfg.action_x_source} robot_y<-{cfg.action_y_source} robot_z<-{cfg.action_z_source} rot<-{cfg.rotation_source} "
                f"gate_button={cfg.gate_button} require_gate={int(bool(cfg.require_gate))} "
                f"grip_mirror={int(bool(self._preview_diag.get('gripper_mirror_enabled', cfg.mirror_gripper_when_inactive)))} "
                f"mirror_toggle={cfg.gripper_mirror_toggle_button} gripper_axis={cfg.trigger_axis}"
            ),
            (
                f"Modes: motion={cfg.motion_control_mode} gripper={cfg.gripper_control_mode} invert_gripper={int(bool(cfg.invert_gripper))} "
                f"| Gains: xyz={float(cfg.position_gain):.2f} resp={float(cfg.position_response_gain):.2f} ff={float(cfg.position_feedforward):.2f} "
                f"yaw={float(cfg.yaw_gain):.2f} yaw_resp={float(cfg.yaw_response_gain):.2f} yaw_ff={float(cfg.yaw_feedforward):.2f} grip={float(cfg.gripper_gain):.2f} "
                f"| Mapped action=[{mapped_str}]"
            ),
        ]
        y = rect.y + 40
        for text in lines:
            screen.blit(small.render(text, True, (220, 220, 220)), (rect.x + 12, y))
            y += 20

    def _world_to_panel(self, rect, bounds: np.ndarray, xy: list[float]) -> tuple[int, int]:
        x_min = float(bounds[0, 0])
        y_min = float(bounds[0, 1])
        x_max = float(bounds[1, 0])
        y_max = float(bounds[1, 1])
        pad = 24
        inner_w = max(20.0, float(rect.width - 2 * pad))
        inner_h = max(20.0, float(rect.height - 2 * pad))
        sx = inner_w / max(1e-6, x_max - x_min)
        sy = inner_h / max(1e-6, y_max - y_min)
        scale = min(sx, sy)
        cx = float(rect.x + rect.width / 2.0)
        cy = float(rect.y + rect.height / 2.0)
        mx = (x_min + x_max) * 0.5
        my = (y_min + y_max) * 0.5
        px = int(round(cx + (float(xy[0]) - mx) * scale))
        py = int(round(cy - (float(xy[1]) - my) * scale))
        return px, py

    def _draw_topdown_view(self, rect) -> None:
        pygame = self._pygame
        screen = self._screen
        font = self._font
        small = self._small
        np = _np()
        pygame.draw.rect(screen, (42, 42, 52), rect, border_radius=6)
        pygame.draw.rect(screen, (100, 100, 120), rect, 1, border_radius=6)
        screen.blit(font.render("Manip XY Helper", True, (220, 190, 80)), (rect.x + 12, rect.y + 10))

        state = self._topdown_state if isinstance(self._topdown_state, dict) else {}
        available = bool(state.get("available", False))
        if not available:
            screen.blit(small.render("No manipulation top-down state yet.", True, (210, 210, 210)), (rect.x + 12, rect.y + 42))
            return

        try:
            bounds = np.asarray(state.get("bounds"), dtype=np.float32).reshape(2, 2)
        except Exception:
            bounds = np.asarray([[0.3, -0.3], [0.55, 0.3]], dtype=np.float32)
        plot_rect = pygame.Rect(rect.x + 12, rect.y + 42, min(360, rect.width - 220), rect.height - 56)
        pygame.draw.rect(screen, (30, 30, 38), plot_rect, border_radius=6)
        pygame.draw.rect(screen, (90, 90, 108), plot_rect, 1, border_radius=6)

        x_mid = (float(bounds[0, 0]) + float(bounds[1, 0])) * 0.5
        y_mid = (float(bounds[0, 1]) + float(bounds[1, 1])) * 0.5
        x0, y0 = self._world_to_panel(plot_rect, bounds, [float(bounds[0, 0]), y_mid])
        x1, y1 = self._world_to_panel(plot_rect, bounds, [float(bounds[1, 0]), y_mid])
        pygame.draw.line(screen, (68, 68, 80), (x0, y0), (x1, y1), 1)
        x0, y0 = self._world_to_panel(plot_rect, bounds, [x_mid, float(bounds[0, 1])])
        x1, y1 = self._world_to_panel(plot_rect, bounds, [x_mid, float(bounds[1, 1])])
        pygame.draw.line(screen, (68, 68, 80), (x0, y0), (x1, y1), 1)

        for target in list(state.get("targets", [])):
            xy = target.get("xy")
            if not isinstance(xy, list) or len(xy) < 2:
                continue
            px, py = self._world_to_panel(plot_rect, bounds, xy)
            color = (235, 166, 54) if bool(target.get("is_target", False)) else (110, 180, 255)
            pygame.draw.circle(screen, color, (px, py), 10, width=2)
            pygame.draw.circle(screen, color, (px, py), 2)

        for cube in list(state.get("cubes", [])):
            xy = cube.get("xy")
            if not isinstance(xy, list) or len(xy) < 2:
                continue
            px, py = self._world_to_panel(plot_rect, bounds, xy)
            color = (90, 220, 135) if bool(cube.get("is_target", False)) else (200, 90, 120)
            cube_rect = pygame.Rect(0, 0, 12, 12)
            cube_rect.center = (px, py)
            pygame.draw.rect(screen, color, cube_rect, border_radius=2)
            pygame.draw.rect(screen, (20, 20, 24), cube_rect, 1, border_radius=2)

        eff_xy = state.get("effector_xy")
        if isinstance(eff_xy, list) and len(eff_xy) >= 2:
            px, py = self._world_to_panel(plot_rect, bounds, eff_xy)
            pygame.draw.circle(screen, (245, 220, 90), (px, py), 7)
            pygame.draw.circle(screen, (20, 20, 24), (px, py), 7, width=1)
            active_target_xy = state.get("active_target_xy")
            if isinstance(active_target_xy, list) and len(active_target_xy) >= 2:
                tx, ty = self._world_to_panel(plot_rect, bounds, active_target_xy)
                pygame.draw.line(screen, (200, 200, 90), (px, py), (tx, ty), 1)

        legend_x = plot_rect.right + 18
        lines = [
            f"env={state.get('source_env', '-')}",
            f"target_block={int(state.get('target_block', -1))}",
            (
                f"eff_xy={eff_xy[0]:+.3f},{eff_xy[1]:+.3f} z={float(state.get('effector_z', 0.0)):+.3f}"
                if isinstance(eff_xy, list) and len(eff_xy) >= 2
                else "eff_xy=-"
            ),
            (
                f"target_xy={state['active_target_xy'][0]:+.3f},{state['active_target_xy'][1]:+.3f} z={float(state.get('active_target_z', 0.0)):+.3f}"
                if isinstance(state.get("active_target_xy"), list) and len(state.get("active_target_xy", [])) >= 2
                else "target_xy=-"
            ),
            f"cubes={len(list(state.get('cubes', [])))} targets={len(list(state.get('targets', [])))}",
            "Legend:",
            "yellow dot = effector",
            "green square = target cube",
            "red square = other cube",
            "orange ring = active target pose",
            "blue ring = other target pose",
        ]
        y = rect.y + 46
        for text in lines:
            screen.blit(small.render(text, True, (220, 220, 220)), (legend_x, y))
            y += 22

    def poll(self) -> tuple[bool, bool, bool, bool, float]:
        pygame = self._pygame
        prev_requested = False
        next_requested = False
        quit_requested = False
        advance_requested = False
        fps_delta = 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    quit_requested = True
                elif event.key == pygame.K_RIGHT:
                    next_requested = True
                elif event.key == pygame.K_LEFT:
                    prev_requested = True
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    advance_requested = True
                elif event.key == pygame.K_UP:
                    fps_delta += 2.0
                elif event.key == pygame.K_DOWN:
                    fps_delta -= 2.0
        return prev_requested, next_requested, quit_requested, advance_requested, fps_delta

    def draw(self) -> None:
        pygame = self._pygame
        screen = self._screen
        font = self._font
        snapshot = self._snapshot
        latest = snapshot.get("latest_sample") if isinstance(snapshot, dict) else None
        right = extract_controller_state(latest, "right")
        left = extract_controller_state(latest, "left")
        screen.fill((20, 20, 24))
        mode = str(snapshot.get("mode", "listen"))
        if mode == "connect":
            link_state = "CONNECTED" if snapshot.get("connected", False) else "DISCONNECTED (retrying)"
            lines = [
                f"Receiver target: {snapshot.get('host', '127.0.0.1')}:{snapshot.get('port', DEFAULT_VR_PORT)}",
                f"Link state: {link_state}",
                f"Last error: {snapshot.get('last_error', '') or '-'}",
                f"Last remote: {snapshot.get('last_client', '-') or '-'}",
            ]
        elif mode == "publish":
            lines = [
                f"Publisher bind: {snapshot.get('host', '0.0.0.0')}:{snapshot.get('port', DEFAULT_VR_PORT)}",
                "Clients can connect to:",
                ", ".join(f"{ip}:{snapshot.get('port', DEFAULT_VR_PORT)}" for ip in snapshot.get("publisher_addresses", []))
                or "127.0.0.1",
                (
                    f"Clients: {snapshot.get('connected_clients', 0)} | "
                    f"Last client: {snapshot.get('last_client', '-') or '-'}"
                ),
            ]
        else:
            lines = [
                f"Receiver bind: {snapshot.get('host', '0.0.0.0')}:{snapshot.get('port', DEFAULT_VR_PORT)}",
                "Send one of these addresses to the desktop:",
                ", ".join(f"{ip}:{snapshot.get('port', DEFAULT_VR_PORT)}" for ip in snapshot.get("receiver_addresses", []))
                or "127.0.0.1",
                (
                    f"Clients: {snapshot.get('connected_clients', 0)} | "
                    f"Last client: {snapshot.get('last_client', '-') or '-'}"
                ),
            ]
        y = 18
        for text in lines:
            screen.blit(font.render(text, True, (230, 230, 230)), (16, y))
            y += 28

        screen.blit(font.render("Latest stream sample", True, (220, 190, 80)), (16, 140))
        if self._show_topdown:
            self._draw_controller_card(pygame.Rect(16, 172, 576, 332), label="Left", hand="left", controller=left)
            self._draw_controller_card(pygame.Rect(608, 172, 576, 332), label="Right", hand="right", controller=right)
            self._draw_topdown_view(pygame.Rect(16, 516, 650, 320))
            self._draw_mapping_summary(pygame.Rect(682, 516, 502, 320))
        else:
            self._draw_controller_card(pygame.Rect(16, 172, 456, 332), label="Left", hand="left", controller=left)
            self._draw_controller_card(pygame.Rect(488, 172, 456, 332), label="Right", hand="right", controller=right)
            self._draw_mapping_summary(pygame.Rect(16, 516, 928, 96))
        pygame.display.flip()

    def close(self) -> None:
        self._pygame.quit()

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
DEFAULT_VR_CACHE_PATH = Path("codex/vr_last_endpoint.json")
DEFAULT_VR_SERVE_HOST = "0.0.0.0"


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
    path = Path(cache_path)
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
    path = Path(cache_path)
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


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return float(angle)


@dataclass
class VRManipMappingConfig:
    hand: str = "right"
    require_gate: bool = False
    gate_button: str = "grip"
    mirror_gripper_when_inactive: bool = False
    position_gain: float = 25.0
    yaw_gain: float = 2.5
    gripper_gain: float = 5.0
    trigger_axis: str = "trigger"
    binary_gripper: bool = False
    trigger_close_threshold: float = 0.6
    trigger_open_threshold: float = 0.2
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    invert_yaw: bool = False


class VRManipActionMapper:
    """Map raw VR controller state into 5D manipulation actions."""

    def __init__(self, action_dim: int, config: Optional[VRManipMappingConfig] = None):
        self.action_dim = int(action_dim)
        self.config = config or VRManipMappingConfig()
        self._last_position: Optional[Any] = None
        self._last_yaw: Optional[float] = None
        self._last_trigger: Optional[float] = None
        self._motion_active_prev = False

    def reset(self) -> None:
        self._last_position = None
        self._last_yaw = None
        self._last_trigger = None
        self._motion_active_prev = False

    def map_sample(self, sample: Optional[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
        np = _np()
        action = np.zeros((self.action_dim,), dtype=np.float32)
        diag = {
            "connected": False,
            "tracked": False,
            "gate_pressed": False,
            "motion_active": False,
            "trigger_value": 0.0,
            "hand": self.config.hand,
        }
        controller = extract_controller_state(sample, hand=self.config.hand)
        if not isinstance(controller, dict):
            self.reset()
            return action, diag

        connected = bool(controller.get("connected", False))
        tracked = bool(controller.get("tracked", False))
        gate_pressed = controller_button_pressed(controller, self.config.gate_button)
        motion_active = gate_pressed or (not self.config.require_gate)
        diag.update(
            {
                "connected": connected,
                "tracked": tracked,
                "gate_pressed": gate_pressed,
                "motion_active": motion_active,
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
        yaw = quaternion_wxyz_to_yaw(pose.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0]))
        trigger_value = float(np.clip(controller_axis_value(controller, self.config.trigger_axis), 0.0, 1.0))
        diag["trigger_value"] = trigger_value

        can_integrate_motion = (
            motion_active
            and self._motion_active_prev == motion_active
            and self._last_position is not None
            and self._last_yaw is not None
        )
        if can_integrate_motion:
            delta_pos = position - self._last_position
            delta_yaw = _wrap_angle(yaw - float(self._last_yaw))
            if self.action_dim >= 1:
                action[0] = float(delta_pos[0] * self.config.position_gain * (-1.0 if self.config.invert_x else 1.0))
            if self.action_dim >= 2:
                action[1] = float(delta_pos[1] * self.config.position_gain * (-1.0 if self.config.invert_y else 1.0))
            if self.action_dim >= 3:
                action[2] = float(delta_pos[2] * self.config.position_gain * (-1.0 if self.config.invert_z else 1.0))
            if self.action_dim >= 4:
                action[3] = float(delta_yaw * self.config.yaw_gain * (-1.0 if self.config.invert_yaw else 1.0))

        can_integrate_gripper = motion_active or self.config.mirror_gripper_when_inactive
        if self.action_dim >= 5 and can_integrate_gripper:
            if self.config.binary_gripper:
                if trigger_value >= self.config.trigger_close_threshold:
                    action[4] = 1.0
                elif trigger_value <= self.config.trigger_open_threshold:
                    action[4] = -1.0
                else:
                    action[4] = 0.0
            elif self._last_trigger is not None:
                action[4] = float((trigger_value - self._last_trigger) * self.config.gripper_gain)

        self._last_position = position.copy()
        self._last_yaw = float(yaw)
        self._last_trigger = float(trigger_value)
        self._motion_active_prev = bool(motion_active)
        np.clip(action, -1.0, 1.0, out=action)
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
    ):
        self.server = server
        self.mapper = mapper
        self.return_none_when_idle = bool(return_none_when_idle)
        self.idle_threshold = float(idle_threshold)
        self._last_diag: dict[str, Any] = {}

    def get_action(self) -> Optional[np.ndarray]:
        np = _np()
        action, diag = self.mapper.map_sample(self.server.latest_sample())
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

    def reset(self) -> None:
        self.mapper.reset()


class VRStatusPanel:
    """Optional pygame status window for receiver-side address and stream state."""

    def __init__(self, title: str = "VR Receiver", width: int = 960, height: int = 520):
        pygame = _optional_pygame()
        if pygame is None:
            raise RuntimeError("pygame is not installed")
        self._pygame = pygame
        pygame.init()
        self._screen = pygame.display.set_mode((int(width), int(height)))
        pygame.display.set_caption(title)
        self._font = pygame.font.SysFont("Arial", 18)
        self._small = pygame.font.SysFont("Arial", 16)
        self._snapshot: dict[str, Any] = {}
        self._history_by_hand: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
        self._last_history_seq: Optional[int] = None

    def set_snapshot(self, snapshot: dict[str, Any]) -> None:
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
        buttons = controller.get("pressed_button_ids", []) if isinstance(controller, dict) else []
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
            f"buttons={buttons}",
        ]
        y_text = rect.y + 10
        for text in header_lines:
            screen.blit(small.render(text, True, (230, 230, 230)), (rect.x + 12, y_text))
            y_text += 22

        inner_x = rect.x + 12
        inner_y = rect.y + 86
        inner_w = rect.width - 24
        plot_w = min(214, max(170, inner_w // 2))
        plot_h = 148
        gap = 12
        right_x = inner_x + plot_w + gap
        right_w = max(120, rect.right - 12 - right_x)
        yaw_w = min(88, max(74, right_w // 2 - 8))
        yaw_h = 88
        trigger_w = min(22, max(18, right_w - yaw_w - gap))
        pad_gap = 12
        pad_w = max(72, (right_w - pad_gap) // 2)
        joy_w = max(72, right_w - pad_gap - pad_w)

        plot_rect = pygame.Rect(inner_x, inner_y, plot_w, plot_h)
        yaw_rect = pygame.Rect(right_x, inner_y + 12, yaw_w, yaw_h)
        trigger_rect = pygame.Rect(yaw_rect.right + gap, inner_y + 12, trigger_w, yaw_h)
        pad_rect = pygame.Rect(right_x, inner_y + 168, pad_w, 80)
        joy_rect = pygame.Rect(pad_rect.right + pad_gap, inner_y + 168, joy_w, 80)

        self._draw_position_plot(plot_rect, hand=hand, current_position=position)
        self._draw_yaw_gauge(yaw_rect, yaw=yaw)
        self._draw_meter(trigger_rect, trigger, label="Trigger", min_value=0.0, max_value=1.0)
        self._draw_axis_pad(pad_rect, label="Trackpad", x_value=pad_x, y_value=pad_y)
        self._draw_axis_pad(joy_rect, label="Joystick", x_value=joy_x, y_value=joy_y)

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
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_PERIOD):
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
        small = self._small
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
        self._draw_controller_card(pygame.Rect(16, 172, 456, 340), label="Left", hand="left", controller=left)
        self._draw_controller_card(pygame.Rect(488, 172, 456, 340), label="Right", hand="right", controller=right)
        pygame.display.flip()

    def close(self) -> None:
        self._pygame.quit()

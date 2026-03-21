#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "ogbench_utils"))

from vr_teleop import (  # noqa: E402
    DEFAULT_VR_CACHE_PATH,
    DEFAULT_VR_PORT,
    OPENVR_BUTTON_ALIASES,
    VRPublisherServer,
    load_cached_endpoint,
    prompt_sender_target,
    resolve_cached_endpoint,
    save_cached_endpoint,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish raw VR controller state for thesis teleoperation.")
    p.add_argument("--mode", type=str, default="serve", choices=["serve", "push"])
    p.add_argument("--host", type=str, default="", help="Bind host in serve mode, receiver host in push mode.")
    p.add_argument("--port", type=int, default=DEFAULT_VR_PORT)
    p.add_argument("--backend", type=str, default="demo", choices=["demo", "openvr"])
    p.add_argument("--rate_hz", type=float, default=60.0)
    p.add_argument("--reconnect_seconds", type=float, default=2.0)
    p.add_argument("--ui", type=str, default="auto", choices=["auto", "console", "pygame"])
    p.add_argument("--print_every", type=int, default=120, help="Push mode: print every N packets. Serve mode: nonzero enables periodic status lines.")
    p.add_argument("--cache_path", type=str, default=str(DEFAULT_VR_CACHE_PATH))
    p.add_argument(
        "--save_cache",
        action="store_true",
        default=True,
        help="Remember the last endpoint in the cache file for receiver-side reuse.",
    )
    p.add_argument("--no_save_cache", dest="save_cache", action="store_false")
    return p.parse_args()


def _button_pressed(mask: int, button_id: int) -> bool:
    return bool(mask & (1 << max(0, int(button_id))))


def _pose_matrix_to_pose(m) -> dict[str, Any]:
    r00, r01, r02, tx = float(m[0][0]), float(m[0][1]), float(m[0][2]), float(m[0][3])
    r10, r11, r12, ty = float(m[1][0]), float(m[1][1]), float(m[1][2]), float(m[1][3])
    r20, r21, r22, tz = float(m[2][0]), float(m[2][1]), float(m[2][2]), float(m[2][3])
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s
    return {
        "position": [tx, ty, tz],
        "quaternion_wxyz": [w, x, y, z],
    }


class DemoBackend:
    def __init__(self):
        self._start = time.perf_counter()

    def close(self) -> None:
        return None

    def sample(self, seq: int) -> dict[str, Any]:
        t = time.perf_counter() - self._start
        trigger = 0.5 + 0.5 * math.sin(t * 0.8)
        gate_active = int((t % 6.0) < 3.0)
        pose = {
            "position": [
                0.02 * math.sin(t * 1.4),
                0.01 * math.cos(t * 1.7),
                0.015 * math.sin(t * 0.9),
            ],
            "quaternion_wxyz": [
                math.cos(0.5 * 0.2 * math.sin(t * 0.7)),
                0.0,
                0.0,
                math.sin(0.5 * 0.2 * math.sin(t * 0.7)),
            ],
        }
        right = {
            "role": "right",
            "connected": True,
            "tracked": True,
            "device_index": 1,
            "pose": pose,
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "axes": [
                {"x": 0.0, "y": 0.0},
                {"x": trigger, "y": 0.0},
                {"x": 0.0, "y": 0.0},
                {"x": 0.0, "y": 0.0},
                {"x": 0.0, "y": 0.0},
            ],
            "trigger_value": trigger,
            "pressed_button_ids": [OPENVR_BUTTON_ALIASES["grip"]] if gate_active else [],
            "buttons": {"grip": bool(gate_active)},
        }
        return {
            "version": 1,
            "backend": "demo",
            "seq": int(seq),
            "monotonic_time": float(time.perf_counter()),
            "devices": {"left": None, "right": right},
        }


class OpenVRBackend:
    def __init__(self):
        try:
            import openvr  # type: ignore
        except Exception as exc:
            raise RuntimeError("openvr import failed. Install it with `pip install openvr`.") from exc
        self._openvr = openvr
        self._system = openvr.init(openvr.VRApplication_Background)

    def close(self) -> None:
        self._openvr.shutdown()

    def sample(self, seq: int) -> dict[str, Any]:
        openvr = self._openvr
        poses = self._system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0.0,
            openvr.k_unMaxTrackedDeviceCount,
        )
        devices: dict[str, Any] = {"left": None, "right": None}
        for device_index in range(openvr.k_unMaxTrackedDeviceCount):
            try:
                device_class = self._system.getTrackedDeviceClass(device_index)
            except Exception:
                continue
            if device_class != openvr.TrackedDeviceClass_Controller:
                continue
            role_id = int(self._system.getControllerRoleForTrackedDeviceIndex(device_index))
            role = "left" if role_id == openvr.TrackedControllerRole_LeftHand else "right"
            pose_struct = poses[device_index]
            tracked = bool(getattr(pose_struct, "bPoseIsValid", False))
            pose = _pose_matrix_to_pose(pose_struct.mDeviceToAbsoluteTracking) if tracked else None
            ok, controller_state = self._system.getControllerState(device_index)
            pressed_mask = int(controller_state.ulButtonPressed) if ok else 0
            touched_mask = int(controller_state.ulButtonTouched) if ok else 0
            axes = []
            trigger_value = 0.0
            for axis_idx in range(len(controller_state.rAxis)):
                axis = controller_state.rAxis[axis_idx]
                axis_dict = {"x": float(axis.x), "y": float(axis.y)}
                axes.append(axis_dict)
                if axis_idx == 1:
                    trigger_value = float(axis.x)
            buttons = {
                name: _button_pressed(pressed_mask, button_id)
                for name, button_id in OPENVR_BUTTON_ALIASES.items()
            }
            devices[role] = {
                "role": role,
                "device_index": int(device_index),
                "connected": True,
                "tracked": tracked,
                "pose": pose,
                "linear_velocity": list(getattr(pose_struct, "vVelocity", [0.0, 0.0, 0.0])),
                "angular_velocity": list(getattr(pose_struct, "vAngularVelocity", [0.0, 0.0, 0.0])),
                "pressed_button_ids": [i for i in range(64) if _button_pressed(pressed_mask, i)],
                "touched_button_ids": [i for i in range(64) if _button_pressed(touched_mask, i)],
                "buttons": buttons,
                "axes": axes,
                "trigger_value": trigger_value,
            }
        return {
            "version": 1,
            "backend": "openvr",
            "seq": int(seq),
            "monotonic_time": float(time.perf_counter()),
            "devices": devices,
        }


def _connect_socket(host: str, port: int) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=5.0)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def _run_push_mode(args: argparse.Namespace, backend: Any) -> int:
    cache_path = Path(args.cache_path)
    host, port, from_cache = resolve_cached_endpoint(
        args.host,
        args.port,
        cache_path=cache_path,
        default_host="127.0.0.1",
        default_port=args.port,
    )
    if not str(args.host).strip() and not from_cache:
        host, port = prompt_sender_target(default_host=host, default_port=port, ui_mode=args.ui)
    if args.save_cache:
        save_cached_endpoint(host, port, cache_path=cache_path)
    send_period = 1.0 / max(1.0, float(args.rate_hz))
    print(f"vr sender mode=push backend={args.backend} target={host}:{port} rate_hz={float(args.rate_hz):.1f}")
    sock: Optional[socket.socket] = None
    sent = 0
    seq = 0
    try:
        while True:
            if sock is None:
                try:
                    sock = _connect_socket(host, port)
                    print("connected", flush=True)
                except OSError as exc:
                    print(f"connect failed: {exc}; retrying in {float(args.reconnect_seconds):.1f}s", flush=True)
                    time.sleep(max(0.1, float(args.reconnect_seconds)))
                    continue
            sample = backend.sample(seq)
            seq += 1
            try:
                payload = (json.dumps(sample, separators=(",", ":")) + "\n").encode("utf-8")
                sock.sendall(payload)
                sent += 1
                if args.print_every > 0 and (sent % int(args.print_every) == 0):
                    print(f"sent seq={sample['seq']} backend={sample['backend']}", flush=True)
            except OSError as exc:
                print(f"send failed: {exc}; reconnecting", flush=True)
                try:
                    sock.close()
                except Exception:
                    pass
                sock = None
            time.sleep(send_period)
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        backend.close()
    return 0


def _run_serve_mode(args: argparse.Namespace, backend: Any) -> int:
    cache_path = Path(args.cache_path)
    host = str(args.host).strip() or "0.0.0.0"
    publisher = VRPublisherServer(
        backend,
        host=host,
        port=int(args.port),
        rate_hz=float(args.rate_hz),
        cache_path=cache_path,
        save_cache=bool(args.save_cache),
    )
    publisher.start()
    print(publisher.banner_text(), flush=True)
    if args.save_cache:
        cached = load_cached_endpoint(cache_path=cache_path)
        if cached:
            print(f"cached endpoint: {cached['host']}:{cached['port']} ({cache_path})", flush=True)
    last_print = 0.0
    try:
        while True:
            snapshot = publisher.snapshot()
            now = time.time()
            if args.print_every > 0 and (now - last_print) >= 2.0:
                print(
                    "publisher "
                    f"clients={snapshot.get('connected_clients', 0)} "
                    f"last_client={snapshot.get('last_client', '-') or '-'}",
                    flush=True,
                )
                last_print = now
                time.sleep(0.2)
            else:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        publisher.close()
    return 0


def main() -> int:
    args = _parse_args()
    backend = DemoBackend() if args.backend == "demo" else OpenVRBackend()
    if args.mode == "push":
        return _run_push_mode(args, backend)
    return _run_serve_mode(args, backend)


if __name__ == "__main__":
    raise SystemExit(main())

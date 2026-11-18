#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetygym_utils.gamepad import (
    DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
    DEFAULT_SAFETY_GAMEPAD_PORT,
    GamepadMappingConfig,
    GamepadStateServer,
    PygameGamepadController,
    apply_gamepad_mapping_profile,
    empty_gamepad_state,
    infer_control_scheme,
)


class HotplugControllerSupervisor:
    """Keep recreating a local Pygame controller until a joystick is available."""

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        retry_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = factory
        self._retry_seconds = max(0.1, float(retry_seconds))
        self._clock = clock
        self._controller = None
        self._next_retry_at = 0.0
        self.restart_count = 0
        self.last_error = ""

    def _close_controller(self) -> None:
        controller, self._controller = self._controller, None
        if controller is not None:
            try:
                controller.close()
            except Exception:
                pass

    def _restart_controller(self, now: float) -> None:
        self._close_controller()
        try:
            self._controller = self._factory()
            self.restart_count += 1
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"controller initialization failed: {exc}"
            self._controller = None
        self._next_retry_at = now + self._retry_seconds

    def read_state(self) -> dict[str, Any]:
        now = self._clock()
        if self._controller is None:
            if now < self._next_retry_at:
                return empty_gamepad_state()
            self._restart_controller(now)
        if self._controller is None:
            return empty_gamepad_state()
        try:
            state = dict(self._controller.read_state())
        except Exception as exc:
            # SDL joystick handles become invalid immediately on unplug. Drop
            # the stale handle so the next retry reinitializes the subsystem.
            self.last_error = f"controller read failed: {exc}"
            self._close_controller()
            self._next_retry_at = now + self._retry_seconds
            return empty_gamepad_state()
        if not bool(state.get("connected", False)) and now >= self._next_retry_at:
            # Pygame/SDL can cache a zero-device enumeration. Reinitializing
            # periodically makes controllers connected after startup visible.
            self._restart_controller(now)
            if self._controller is not None:
                try:
                    state = dict(self._controller.read_state())
                except Exception as exc:
                    self.last_error = f"controller read failed: {exc}"
                    self._close_controller()
                    state = empty_gamepad_state()
        return state

    def close(self) -> None:
        self._close_controller()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Windows-side raw SafetyGym gamepad publisher")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--action_dim", type=int, default=2)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_SAFETY_GAMEPAD_PORT)
    p.add_argument("--sample_hz", type=float, default=60.0)
    p.add_argument("--debug_hz", type=float, default=2.0, help="Console status frequency; 0 disables periodic status.")
    p.add_argument("--debug_axis_threshold", type=float, default=0.03)
    p.add_argument("--gamepad_config_path", type=str, default=str(DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH))
    p.add_argument("--gamepad_use_saved_config", action="store_true", default=True)
    p.add_argument("--no_gamepad_use_saved_config", dest="gamepad_use_saved_config", action="store_false")
    p.add_argument("--gamepad_device_index", type=int, default=0)
    p.add_argument(
        "--hotplug_retry_seconds",
        type=float,
        default=1.0,
        help="Seconds between joystick subsystem re-enumeration attempts while disconnected.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    config = GamepadMappingConfig(device_index=int(args.gamepad_device_index))
    if bool(args.gamepad_use_saved_config):
        config, loaded = apply_gamepad_mapping_profile(config, args.gamepad_config_path)
        if loaded:
            print(f"loaded gamepad mapping profile from {args.gamepad_config_path}", flush=True)
    controller = HotplugControllerSupervisor(
        lambda: PygameGamepadController(
            action_dim=int(args.action_dim),
            control_scheme=infer_control_scheme(args.env_name),
            config=config,
            config_path=args.gamepad_config_path,
            hot_reload=True,
        ),
        retry_seconds=float(args.hotplug_retry_seconds),
    )
    server = GamepadStateServer(host=str(args.host), port=int(args.port))
    server.start()
    print(server.banner_text(), flush=True)
    interval = 1.0 / max(1.0, float(args.sample_hz))
    debug_interval = float("inf") if float(args.debug_hz) <= 0.0 else 1.0 / float(args.debug_hz)
    last_debug_at = 0.0
    previous_connected: bool | None = None
    seq = 0
    try:
        while True:
            state = controller.read_state()
            payload = {
                "seq": int(seq),
                "timestamp": time.time(),
                "state": state,
            }
            server.publish(payload)
            now = time.time()
            connected = bool(state.get("connected", False))
            if connected != previous_connected:
                if connected:
                    print(
                        "[gamepad-hotplug] connected "
                        f"device={state.get('device_index', -1)} name={state.get('name', '')!r} "
                        f"restarts={controller.restart_count}",
                        flush=True,
                    )
                else:
                    detail = f" error={controller.last_error!r}" if controller.last_error else ""
                    print(
                        "[gamepad-hotplug] disconnected; waiting for controller "
                        f"(retry={float(args.hotplug_retry_seconds):.1f}s){detail}",
                        flush=True,
                    )
                previous_connected = connected
            if now - last_debug_at >= debug_interval:
                named_axes = {
                    str(name): round(float(value), 3)
                    for name, value in dict(state.get("named_axes", {})).items()
                    if abs(float(value)) >= float(args.debug_axis_threshold)
                }
                print(
                    json.dumps(
                        {
                            "seq": int(seq),
                            "connected": bool(state.get("connected", False)),
                            "device_index": int(state.get("device_index", -1)),
                            "name": str(state.get("name", "")),
                            "named_axes": named_axes,
                            "buttons": [k for k, v in dict(state.get("buttons", {})).items() if v],
                            "hotplug_restarts": int(controller.restart_count),
                            "hotplug_error": str(controller.last_error),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                last_debug_at = now
            seq += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

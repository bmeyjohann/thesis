#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetygym_utils.controllers import build_human_controller
from safetygym_utils.env import resolve_control_scheme
from safetygym_utils.gamepad import (
    DEFAULT_SAFETY_GAMEPAD_CACHE_PATH,
    DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
    DEFAULT_SAFETY_GAMEPAD_PORT,
)
from safetygym_utils.gamepad_web import GamepadWebServer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Monitor SafetyGym gamepad input and edit mapping in a browser")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--action_dim", type=int, default=2)
    p.add_argument("--car_action_mode", type=str, default="raw_wheels", choices=["raw_wheels", "throttle_turn", "cardinal"])
    p.add_argument("--gamepad_mode", type=str, default="local", choices=["local", "connect"])
    p.add_argument("--gamepad_host", type=str, default="")
    p.add_argument("--gamepad_port", type=int, default=0)
    p.add_argument("--gamepad_cache_path", type=str, default=str(DEFAULT_SAFETY_GAMEPAD_CACHE_PATH))
    p.add_argument("--gamepad_reconnect_seconds", type=float, default=2.0)
    p.add_argument("--gamepad_config_path", type=str, default=str(DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH))
    p.add_argument("--gamepad_use_saved_config", action="store_true", default=True)
    p.add_argument("--no_gamepad_use_saved_config", dest="gamepad_use_saved_config", action="store_false")
    p.add_argument("--gamepad_device_index", type=int, default=0)
    p.add_argument("--web_port", type=int, default=8992)
    p.add_argument("--print_hz", type=float, default=2.0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    controller = build_human_controller(
        input_device="gamepad",
        action_dim=int(args.action_dim),
        env_name=args.env_name,
        action_scale=1.0,
        wheel_command_limit=2.0,
        overlay_fps_limit=0,
        overlay_draw_hz=0.0,
        gamepad_mode=str(args.gamepad_mode),
        gamepad_host=str(args.gamepad_host),
        gamepad_port=int(args.gamepad_port or DEFAULT_SAFETY_GAMEPAD_PORT),
        gamepad_cache_path=args.gamepad_cache_path,
        gamepad_reconnect_seconds=float(args.gamepad_reconnect_seconds),
        gamepad_config_path=args.gamepad_config_path,
        gamepad_use_saved_config=bool(args.gamepad_use_saved_config),
        gamepad_device_index=int(args.gamepad_device_index),
        control_scheme_override=resolve_control_scheme(args.env_name, car_action_mode=args.car_action_mode),
    )
    web = GamepadWebServer(controller=controller, config_path=args.gamepad_config_path, port=int(args.web_port))
    web.start()
    print(f"gamepad web ui listening at {web.url()}", flush=True)
    print(f"gamepad profile path: {args.gamepad_config_path}", flush=True)
    interval = 1.0 / max(0.2, float(args.print_hz))
    try:
        while True:
            live = controller.live_payload()
            state = live["state"]
            print(
                json.dumps(
                    {
                        "connected": bool(state.get("connected", False)),
                        "device_index": int(state.get("device_index", -1)),
                        "name": str(state.get("name", "")),
                        "buttons": [k for k, v in dict(state.get("buttons", {})).items() if v],
                        "named_axes": state.get("named_axes", {}),
                        "preview_action": live.get("preview_action", []),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        web.close()
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

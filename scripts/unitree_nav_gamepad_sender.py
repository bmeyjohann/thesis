#!/usr/bin/env python3
"""Windows-side raw gamepad publisher for Unitree navigation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safetygym_utils.gamepad import GamepadMappingConfig, GamepadStateServer, PygameGamepadController
from unitree_nav_gamepad import DEFAULT_UNITREE_GAMEPAD_PORT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Windows-side raw gamepad publisher for Unitree navigation")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_UNITREE_GAMEPAD_PORT)
    parser.add_argument("--sample-hz", type=float, default=60.0)
    parser.add_argument("--gamepad-device-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    controller = PygameGamepadController(
        action_dim=3,
        control_scheme="planar_velocity",
        config=GamepadMappingConfig(device_index=int(args.gamepad_device_index)),
        hot_reload=False,
    )
    server = GamepadStateServer(host=str(args.host), port=int(args.port))
    server.start()
    print(server.banner_text().replace("SafetyGym", "Unitree navigation"), flush=True)
    print("Mapping is configured in WSL by unitree_nav_gamepad.py; this process publishes raw gamepad state.", flush=True)
    interval = 1.0 / max(1.0, float(args.sample_hz))
    sequence = 0
    try:
        while True:
            state = controller.read_state()
            server.publish({"seq": sequence, "timestamp": time.time(), "state": state})
            if sequence % max(1, round(float(args.sample_hz))) == 0:
                print(
                    json.dumps(
                        {
                            "seq": sequence,
                            "connected": bool(state.get("connected", False)),
                            "name": str(state.get("name", "")),
                            "buttons": [key for key, value in dict(state.get("buttons", {})).items() if value],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            sequence += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())


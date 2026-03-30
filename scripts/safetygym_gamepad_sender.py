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

from safetygym_utils.gamepad import (
    DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
    DEFAULT_SAFETY_GAMEPAD_PORT,
    GamepadMappingConfig,
    GamepadStateServer,
    PygameGamepadController,
    apply_gamepad_mapping_profile,
    infer_control_scheme,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Windows-side raw SafetyGym gamepad publisher")
    p.add_argument("--env_name", type=str, default="SafetyCarGoal2-v0")
    p.add_argument("--action_dim", type=int, default=2)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_SAFETY_GAMEPAD_PORT)
    p.add_argument("--sample_hz", type=float, default=60.0)
    p.add_argument("--gamepad_config_path", type=str, default=str(DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH))
    p.add_argument("--gamepad_use_saved_config", action="store_true", default=True)
    p.add_argument("--no_gamepad_use_saved_config", dest="gamepad_use_saved_config", action="store_false")
    p.add_argument("--gamepad_device_index", type=int, default=0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    config = GamepadMappingConfig(device_index=int(args.gamepad_device_index))
    if bool(args.gamepad_use_saved_config):
        config, loaded = apply_gamepad_mapping_profile(config, args.gamepad_config_path)
        if loaded:
            print(f"loaded gamepad mapping profile from {args.gamepad_config_path}", flush=True)
    controller = PygameGamepadController(
        action_dim=int(args.action_dim),
        control_scheme=infer_control_scheme(args.env_name),
        config=config,
        config_path=args.gamepad_config_path,
        hot_reload=True,
    )
    server = GamepadStateServer(host=str(args.host), port=int(args.port))
    server.start()
    print(server.banner_text(), flush=True)
    interval = 1.0 / max(1.0, float(args.sample_hz))
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
            if seq % max(1, int(args.sample_hz)) == 0:
                print(
                    json.dumps(
                        {
                            "seq": int(seq),
                            "connected": bool(state.get("connected", False)),
                            "device_index": int(state.get("device_index", -1)),
                            "name": str(state.get("name", "")),
                            "buttons": [k for k, v in dict(state.get("buttons", {})).items() if v],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
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

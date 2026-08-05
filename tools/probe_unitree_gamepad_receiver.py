#!/usr/bin/env python3
"""Verify live Windows-to-WSL Unitree gamepad transport without Isaac."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unitree_nav_gamepad import UnitreeGamepadController


DEFAULT_ENDPOINT = Path("/mnt/c/Data/thesis/local/unitree_gamepad_endpoint.json")


def _endpoint(path: Path, host: str, port: int) -> tuple[str, int]:
    if host:
        return host, port
    if not path.exists():
        raise FileNotFoundError(
            f"Endpoint file not found: {path}. Start the Windows sender-only script first."
        )
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return str(payload["host"]), int(payload.get("port", port))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-file", type=Path, default=DEFAULT_ENDPOINT)
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--status-interval-s", type=float, default=0.5)
    args = parser.parse_args()

    host, port = _endpoint(args.endpoint_file, args.host, args.port)
    print(f"[probe] connecting to Windows gamepad publisher at {host}:{port}", flush=True)
    controller = UnitreeGamepadController(
        mode="connect",
        host=host,
        port=port,
        config_path="/tmp/unitree_gamepad_receiver_probe_config.json",
        reconnect_seconds=0.25,
        stale_timeout_s=0.5,
    )
    received = False
    deadline = time.monotonic() + max(0.5, float(args.duration_s))
    try:
        while time.monotonic() < deadline:
            sample = controller.sample()
            receiving = bool(sample.connected and not sample.stale)
            received = received or receiving
            axes = {
                str(name): round(float(value), 3)
                for name, value in dict(sample.state.get("named_axes", {})).items()
                if abs(float(value)) >= 0.02
            }
            buttons = [
                str(name)
                for name, value in dict(sample.state.get("buttons", {})).items()
                if bool(value)
            ]
            print(
                "[probe] "
                f"transport_connected={int(sample.transport.get('connected', False))} "
                f"receiving={int(receiving)} stale={int(sample.stale)} "
                f"age_s={sample.state_age_s:.3f} device={sample.state.get('name', '')!r} "
                f"gate={int(sample.gate_held)} axes={axes} buttons={buttons} "
                f"error={sample.transport.get('error', '') or '-'}",
                flush=True,
            )
            time.sleep(max(0.05, float(args.status_interval_s)))
    finally:
        controller.close()

    if not received:
        print("[probe] FAILED: no fresh controller packets were received", flush=True)
        return 1
    print("[probe] PASS: fresh Windows controller packets reached WSL", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

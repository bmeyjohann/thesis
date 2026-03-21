#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "ogbench_utils"))

from vr_teleop import (  # noqa: E402
    DEFAULT_VR_CACHE_PATH,
    DEFAULT_VR_PORT,
    VRRawStateClient,
    VRRawStateServer,
    VRStatusPanel,
    extract_controller_state,
    resolve_cached_endpoint,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect the raw VR stream from either a local listener or remote publisher.")
    p.add_argument("--mode", type=str, default="connect", choices=["connect", "listen"])
    p.add_argument("--host", type=str, default="")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--ui", type=str, default="auto", choices=["auto", "none", "pygame"])
    p.add_argument("--print_every", type=float, default=1.0, help="Console summary cadence in seconds.")
    p.add_argument("--cache_path", type=str, default=str(DEFAULT_VR_CACHE_PATH))
    p.add_argument("--reconnect_seconds", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cache_path = Path(args.cache_path)
    if args.mode == "listen":
        listen_port = int(args.port) if int(args.port) > 0 else DEFAULT_VR_PORT
        source = VRRawStateServer(host=str(args.host).strip() or "0.0.0.0", port=listen_port)
        source.start()
    else:
        host, port, from_cache = resolve_cached_endpoint(
            args.host,
            args.port,
            cache_path=cache_path,
            default_host="127.0.0.1",
            default_port=DEFAULT_VR_PORT,
        )
        source = VRRawStateClient(
            host=host,
            port=port,
            reconnect_seconds=float(args.reconnect_seconds),
            cache_path=cache_path,
            save_cache=True,
        )
        source.start()
        print(f"resolved endpoint: {host}:{port} (from_cache={int(from_cache)})", flush=True)
    print(source.banner_text(), flush=True)

    panel = None
    if args.ui in {"auto", "pygame"}:
        try:
            panel = VRStatusPanel(title="VR Receiver Monitor")
        except Exception:
            if args.ui == "pygame":
                raise
    last_print = 0.0
    try:
        while True:
            snapshot = source.snapshot()
            now = time.time()
            if panel is not None:
                panel.set_snapshot(snapshot)
                _, _, quit_requested, _, _ = panel.poll()
                panel.draw()
                if quit_requested:
                    break
            if now - last_print >= float(args.print_every):
                latest = snapshot.get("latest_sample")
                right = extract_controller_state(latest, "right")
                left = extract_controller_state(latest, "left")
                connected = bool(snapshot.get("connected", snapshot.get("connected_clients", 0)))
                state = "CONNECTED" if connected else "DISCONNECTED"
                print(
                    "monitor "
                    f"mode={snapshot.get('mode', args.mode)} "
                    f"state={state} "
                    f"connected={int(connected)} "
                    f"last_peer={snapshot.get('last_client', '-') or '-'} "
                    f"last_error={snapshot.get('last_error', '') or '-'} "
                    f"left_tracked={int(bool(left and left.get('tracked', False)))} "
                    f"right_tracked={int(bool(right and right.get('tracked', False)))}",
                    flush=True,
                )
                last_print = now
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if panel is not None:
            panel.close()
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

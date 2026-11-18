from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from safetygym_utils.gamepad import GamepadStateClient, GamepadStateServer


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class GamepadTransportHotplugTest(unittest.TestCase):
    def test_late_controller_connection_preserves_device_status(self) -> None:
        server = GamepadStateServer(host="127.0.0.1", port=0)
        server.start()
        assert server._sock is not None
        port = server._sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            client = GamepadStateClient(
                host="127.0.0.1",
                port=port,
                cache_path=Path(directory) / "endpoint.json",
            )
            try:
                client.start()
                self.assertTrue(wait_until(lambda: client.connected and bool(server._clients)))

                server.publish({
                    "state": {"connected": False, "name": "", "named_axes": {}},
                })
                self.assertTrue(wait_until(
                    lambda: "_transport_received_at" in client.latest_state
                ))
                self.assertTrue(client.connected)
                self.assertFalse(client.latest_state["connected"])

                server.publish({
                    "state": {
                        "connected": True,
                        "name": "late controller",
                        "named_axes": {"left_y": 0.5},
                    },
                })
                self.assertTrue(wait_until(
                    lambda: client.latest_state.get("name") == "late controller"
                ))
                self.assertTrue(client.latest_state["connected"])
            finally:
                server.close()
                client.close()

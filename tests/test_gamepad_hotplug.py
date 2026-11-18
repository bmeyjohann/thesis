from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
# This Windows checkout supplies the shared gamepad protocol module while the
# migrated canonical workspace is projected without all WSL-only source files.
WINDOWS_REPO = Path(r"C:\Data\thesis")
if WINDOWS_REPO.exists():
    sys.path.insert(0, str(WINDOWS_REPO))

spec = importlib.util.spec_from_file_location(
    "gamepad_sender_under_test",
    REPO / "scripts" / "safetygym_gamepad_sender.py",
)
assert spec is not None and spec.loader is not None
sender = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sender)
HotplugControllerSupervisor = sender.HotplugControllerSupervisor
ParentProcessWatch = sender.ParentProcessWatch


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeController:
    def __init__(self, reads) -> None:
        self.reads = list(reads)
        self.last_result = None
        self.closed = False

    def read_state(self):
        if self.reads:
            self.last_result = self.reads.pop(0)
        result = self.last_result
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.closed = True


class HotplugControllerSupervisorTest(unittest.TestCase):
    def test_detects_controller_connected_after_startup(self) -> None:
        clock = FakeClock()
        first = FakeController([{"connected": False}])
        second = FakeController([{"connected": True, "name": "late controller"}])
        controllers = iter([first, second])
        supervisor = HotplugControllerSupervisor(
            lambda: next(controllers), retry_seconds=1.0, clock=clock
        )

        self.assertFalse(supervisor.read_state()["connected"])
        clock.now = 0.5
        self.assertFalse(supervisor.read_state()["connected"])
        clock.now = 1.1
        self.assertTrue(supervisor.read_state()["connected"])
        self.assertTrue(first.closed)
        self.assertEqual(supervisor.restart_count, 2)


    def test_recovers_after_connected_controller_disconnects(self) -> None:
        clock = FakeClock()
        first = FakeController(
            [{"connected": True, "name": "controller"}, RuntimeError("device removed")]
        )
        second = FakeController([{"connected": True, "name": "reconnected"}])
        controllers = iter([first, second])
        supervisor = HotplugControllerSupervisor(
            lambda: next(controllers), retry_seconds=1.0, clock=clock
        )

        self.assertTrue(supervisor.read_state()["connected"])
        clock.now = 0.1
        self.assertFalse(supervisor.read_state()["connected"])
        self.assertTrue(first.closed)
        self.assertIn("device removed", supervisor.last_error)
        clock.now = 1.2
        self.assertTrue(supervisor.read_state()["connected"])
        self.assertEqual(supervisor.restart_count, 2)


class ParentProcessWatchTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows process handles only")
    def test_detects_original_parent_exit(self) -> None:
        parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        watch = None
        try:
            watch = ParentProcessWatch(parent.pid)
            self.assertTrue(watch.alive())
            parent.terminate()
            parent.wait(timeout=5)
            self.assertFalse(watch.alive())
        finally:
            if watch is not None:
                watch.close()
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()

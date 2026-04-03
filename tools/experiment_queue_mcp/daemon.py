from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.experiment_queue_mcp.server import build_queue
else:
    from .server import build_queue


LOGGER = logging.getLogger("experiment_queue_daemon")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Background worker daemon for the experiment queue.")
    parser.add_argument("--queue-root", required=True, help="Path to the queue root directory.")
    parser.add_argument("--workspace-root", required=True, help="Workspace root used for relative paths and source validation.")
    parser.add_argument("--default-cwd", required=True, help="Default working directory for jobs.")
    parser.add_argument(
        "--script-root",
        action="append",
        dest="script_roots",
        default=[],
        help="Allowed source roots for enqueue_script. Repeatable.",
    )
    parser.add_argument("--default-conda-env", default=None, help="Default conda env for queued jobs.")
    parser.add_argument("--conda-sh-path", default=None, help="Path to conda.sh for conda activation.")
    parser.add_argument("--shell-path", default="/bin/bash", help="Shell used to run scripts.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Worker poll interval in seconds.")
    parser.add_argument("--terminate-grace", type=float, default=10.0, help="Seconds to wait after SIGTERM before SIGKILL.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = build_arg_parser().parse_args(argv)
    queue = build_queue(args)
    if not queue.start():
        LOGGER.info("Experiment queue worker already active for %s; exiting daemon startup.", queue.queue_root)
        return 0

    LOGGER.info("Experiment queue daemon started for %s", queue.queue_root)
    stop_event = threading.Event()

    def _handle_signal(signum, _frame) -> None:
        LOGGER.info("Experiment queue daemon received signal %s; shutting down.", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while not stop_event.is_set():
            time.sleep(max(0.1, args.poll_interval))
        return 0
    finally:
        queue.stop(kill_running=False)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any
from typing import Literal

from pydantic import Field

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.experiment_queue_mcp.queue_engine import ExperimentQueue
else:
    from .queue_engine import ExperimentQueue

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError


LOGGER = logging.getLogger("experiment_queue_mcp")


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _resolve_path(queue: ExperimentQueue, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = queue.workspace_root / path
    return path.resolve()


def _job_summary(job: Any) -> dict[str, Any] | None:
    if job is None:
        return None
    data = _serialize(job)
    return {
        "job_id": data["job_id"],
        "name": data["name"],
        "status": data["status"],
        "submitted_at": data["submitted_at"],
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "exit_code": data.get("exit_code"),
        "has_owner": bool(data.get("owner_label") or data.get("owner_session_id")),
        "manual_drop": bool(data.get("manual_drop", False)),
        "note": data.get("note"),
    }


def _queue_status_summary(status: Any) -> dict[str, Any]:
    data = _serialize(status)
    return {
        "daemon_running": data["daemon_running"],
        "daemon_pid": data.get("daemon_pid"),
        "paused": data["paused"],
        "stop_after_current": data["stop_after_current"],
        "stop_now": data["stop_now"],
        "counts": data["counts"],
        "current_job": _job_summary(data.get("current_job")),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Queue-backed MCP server for experiment scripts.")
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


def build_queue(args: argparse.Namespace) -> ExperimentQueue:
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    return ExperimentQueue(
        queue_root=Path(args.queue_root).expanduser().resolve(),
        workspace_root=workspace_root,
        script_roots=[Path(path).expanduser().resolve() for path in (args.script_roots or [workspace_root / "scripts"])],
        default_cwd=Path(args.default_cwd).expanduser().resolve(),
        default_conda_env=args.default_conda_env,
        conda_sh_path=Path(args.conda_sh_path).expanduser().resolve() if args.conda_sh_path else None,
        shell_path=args.shell_path,
        poll_interval_s=args.poll_interval,
        terminate_grace_s=args.terminate_grace,
    )


def build_daemon_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("daemon.py")),
        "--queue-root",
        str(Path(args.queue_root).expanduser().resolve()),
        "--workspace-root",
        str(Path(args.workspace_root).expanduser().resolve()),
        "--default-cwd",
        str(Path(args.default_cwd).expanduser().resolve()),
        *[
            token
            for root in (args.script_roots or [])
            for token in ("--script-root", str(Path(root).expanduser().resolve()))
        ],
        *(
            ["--default-conda-env", str(args.default_conda_env)]
            if args.default_conda_env
            else []
        ),
        "--poll-interval",
        str(args.poll_interval),
        "--terminate-grace",
        str(args.terminate_grace),
        *(
            ["--conda-sh-path", str(Path(args.conda_sh_path).expanduser().resolve())]
            if args.conda_sh_path
            else []
        ),
    ]


def create_mcp_server(queue: ExperimentQueue, daemon_command: list[str]) -> FastMCP:
    mcp = FastMCP("experiment-queue", json_response=True, log_level="INFO")
    tool_names = [
        "prime_queue_session",
        "enqueue_script",
        "queue_status",
        "queue_debug_status",
        "list_jobs",
        "list_jobs_debug",
        "get_job",
        "get_job_debug",
        "read_job_log",
        "pause_queue",
        "resume_queue",
        "stop_after_current",
        "stop_now",
        "cancel_job",
    ]

    @mcp.tool()
    def prime_queue_session(
        list_limit: int = Field(default=5, ge=1, description="Number of recent jobs to include in the summary."),
        exercise_control_tools: bool = Field(
            default=True,
            description="If true, perform a reversible pause/resume and stop-after-current restore cycle.",
        ),
        exercise_enqueue_cancel: bool = Field(
            default=False,
            description="If true, enqueue and immediately cancel a temporary no-op job to verify that path too.",
        ),
    ) -> dict[str, Any]:
        """Warm up the queue MCP and optionally exercise safe control paths before an autonomous run."""
        try:
            queue.ensure_daemon_running(daemon_command)
            initial_status = queue.queue_status()
            actions: list[str] = []
            probe_job: dict[str, Any] | None = None

            if exercise_control_tools:
                queue.pause_queue()
                actions.append("pause_queue")
                queue.resume_queue()
                actions.append("resume_queue")
                queue.stop_after_current()
                actions.append("stop_after_current")
                if initial_status["paused"]:
                    queue.pause_queue()
                else:
                    queue.resume_queue()
                if initial_status["stop_after_current"]:
                    queue.stop_after_current()

            if exercise_enqueue_cancel:
                probe_dir = queue.workspace_root / ".experiment_queue_mcp"
                probe_dir.mkdir(parents=True, exist_ok=True)
                probe_script = probe_dir / "prime_queue_session_noop.sh"
                probe_script.write_text(
                    "#!/usr/bin/env bash\nset -euo pipefail\necho prime-queue-session-noop\n",
                    encoding="utf-8",
                )
                probe_script.chmod(0o700)
                queued_probe = queue.enqueue_script(
                    source_path=str(probe_script),
                    name="prime-queue-session-noop",
                    cwd=str(queue.default_cwd),
                    conda_env=queue.default_conda_env,
                )
                actions.append("enqueue_script")
                cancelled_probe = queue.cancel_job(queued_probe.job_id)
                actions.append("cancel_job")
                probe_job = _job_summary(cancelled_probe)

            return {
                "server_name": "experiment-queue",
                "available_tools": tool_names,
                "queue_status_before": _queue_status_summary(initial_status),
                "queue_status_after": _queue_status_summary(queue.queue_status()),
                "recent_jobs": [_job_summary(job) for job in queue.list_jobs(limit=list_limit)],
                "actions_exercised": actions,
                "probe_job": probe_job,
                "note": (
                    "This warms up the MCP server and can exercise reversible queue operations, "
                    "but it does not pre-authorize future Codex shell approvals outside the MCP."
                ),
            }
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def enqueue_script(
        source_path: str = Field(..., description="Absolute or workspace-relative path to a bash script."),
        args: list[str] = Field(default_factory=list, description="Optional positional arguments passed to the script."),
        name: str | None = Field(default=None, description="Optional display name override."),
        cwd: str | None = Field(default=None, description="Optional working directory for the job."),
        conda_env: str | None = Field(default=None, description="Optional conda env override."),
        owner_label: str | None = Field(default=None, description="Optional human-readable owner label for mutation guards."),
        owner_session_id: str | None = Field(default=None, description="Optional owning session identifier for mutation guards."),
    ) -> dict[str, Any]:
        """Copy a bash script into the queue and schedule it for execution."""
        try:
            result = queue.enqueue_script(
                source_path=str(_resolve_path(queue, source_path)),
                args=list(args),
                name=name,
                cwd=str(_resolve_path(queue, cwd)) if cwd else None,
                conda_env=conda_env,
                owner_label=owner_label,
                owner_session_id=owner_session_id,
            )
            queue.ensure_daemon_running(daemon_command)
            return _job_summary(result) or {}
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def queue_status() -> dict[str, Any]:
        """Return a compact queue summary for unattended monitoring."""
        try:
            return _queue_status_summary(queue.queue_status())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def queue_debug_status() -> dict[str, Any]:
        """Return the full queue status, including absolute paths, for manual debugging."""
        try:
            return _serialize(queue.queue_status())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def list_jobs(
        status: str | None = Field(default=None, description="Optional state filter."),
        limit: int | None = Field(default=None, ge=1, description="Maximum number of jobs to return."),
    ) -> list[dict[str, Any]]:
        """List compact job summaries, optionally filtered by state."""
        try:
            return [_job_summary(job) or {} for job in queue.list_jobs(status=status, limit=limit)]
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def list_jobs_debug(
        status: str | None = Field(default=None, description="Optional state filter."),
        limit: int | None = Field(default=None, ge=1, description="Maximum number of jobs to return."),
    ) -> list[dict[str, Any]]:
        """List full job metadata, including absolute paths, for manual debugging."""
        try:
            return _serialize(queue.list_jobs(status=status, limit=limit))
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def get_job(
        job_id: str = Field(..., description="Queue job ID."),
    ) -> dict[str, Any]:
        """Get a compact summary for one job."""
        try:
            return _job_summary(queue.get_job(job_id)) or {}
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def get_job_debug(
        job_id: str = Field(..., description="Queue job ID."),
    ) -> dict[str, Any]:
        """Get full job metadata, including absolute paths, for manual debugging."""
        try:
            return _serialize(queue.get_job(job_id))
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def read_job_log(
        job_id: str = Field(..., description="Queue job ID."),
        stream: Literal["stdout", "stderr"] = Field(default="stdout", description="Log stream to read."),
        lines: int = Field(default=100, ge=1, description="Number of trailing log lines to return."),
    ) -> dict[str, str]:
        """Tail stdout or stderr for a queued or completed job."""
        try:
            return {"text": queue.read_log(job_id, stream=stream, lines=lines)}
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def pause_queue() -> dict[str, Any]:
        """Pause dequeuing new jobs without stopping the current job."""
        try:
            return _serialize(queue.pause_queue())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def resume_queue() -> dict[str, Any]:
        """Resume dequeuing jobs and clear stop-after-current."""
        try:
            queue.ensure_daemon_running(daemon_command)
            return _serialize(queue.resume_queue())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def stop_after_current(
        requester_label: str | None = Field(default=None, description="Optional requester label for ownership checks."),
        requester_session_id: str | None = Field(default=None, description="Optional requester session id for ownership checks."),
        force: bool = Field(default=False, description="Override ownership checks for the active job."),
    ) -> dict[str, Any]:
        """Finish the active job and then stop starting new jobs."""
        try:
            queue.assert_current_job_control_allowed(
                requester_label=requester_label,
                requester_session_id=requester_session_id,
                force=force,
            )
            queue.ensure_daemon_running(daemon_command)
            return _serialize(queue.stop_after_current())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def stop_now(
        requester_label: str | None = Field(default=None, description="Optional requester label for ownership checks."),
        requester_session_id: str | None = Field(default=None, description="Optional requester session id for ownership checks."),
        force: bool = Field(default=False, description="Override ownership checks for the active job."),
    ) -> dict[str, Any]:
        """Terminate the active job and stop starting new jobs until resumed."""
        try:
            queue.assert_current_job_control_allowed(
                requester_label=requester_label,
                requester_session_id=requester_session_id,
                force=force,
            )
            queue.ensure_daemon_running(daemon_command)
            return _serialize(queue.stop_now())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def cancel_job(
        job_id: str = Field(..., description="Queue job ID."),
        requester_label: str | None = Field(default=None, description="Optional requester label for ownership checks."),
        requester_session_id: str | None = Field(default=None, description="Optional requester session id for ownership checks."),
        force: bool = Field(default=False, description="Override ownership checks for this job."),
    ) -> dict[str, Any]:
        """Cancel a queued job immediately or request stop for a running job."""
        try:
            return _job_summary(
                queue.cancel_job(
                    job_id,
                    requester_label=requester_label,
                    requester_session_id=requester_session_id,
                    force=force,
                )
            ) or {}
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    return mcp


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = build_arg_parser().parse_args(argv)
    queue = build_queue(args)
    daemon_command = build_daemon_command(args)
    server = create_mcp_server(queue, daemon_command)
    try:
        server.run(transport="stdio")
    finally:
        LOGGER.info("Shutting down experiment queue MCP control server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

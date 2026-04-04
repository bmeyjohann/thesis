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
    running_jobs = [_job_summary(job) for job in data.get("running_jobs", [])]
    return {
        "queue_root_name": Path(str(data.get("queue_root", ""))).name if data.get("queue_root") else None,
        "daemon_running": data["daemon_running"],
        "daemon_pid": data.get("daemon_pid"),
        "paused": data["paused"],
        "stop_after_current": data["stop_after_current"],
        "stop_now": data["stop_now"],
        "max_concurrent_jobs": data.get("max_concurrent_jobs", 1),
        "scheduler_mode": data.get("scheduler_mode"),
        "last_scheduled_owner": data.get("last_scheduled_owner"),
        "counts": data["counts"],
        "running_jobs": running_jobs,
        "running_count": len(running_jobs),
        "current_job": _job_summary(data.get("current_job")),
    }


def _daemon_status_summary(status: Any) -> dict[str, Any]:
    data = _serialize(status)
    summary = {
        "daemon_running": data["daemon_running"],
        "daemon_pid": data.get("daemon_pid"),
        "worker_lock_held": data.get("worker_lock_held"),
        "worker_heartbeat_fresh": data.get("worker_heartbeat_fresh"),
        "worker_heartbeat_age_s": data.get("worker_heartbeat_age_s"),
        "worker_lock_stale": data.get("worker_lock_stale"),
        "max_concurrent_jobs": data.get("max_concurrent_jobs"),
        "scheduler_mode": data.get("scheduler_mode"),
        "last_scheduled_owner": data.get("last_scheduled_owner"),
    }
    for key in ("started_now", "stopped_now", "restarted_now", "note"):
        if key in data:
            summary[key] = data[key]
    return summary


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
    parser.add_argument("--max-concurrent-jobs", type=int, default=2, help="Maximum number of jobs the daemon may run at once.")
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
        max_concurrent_jobs=args.max_concurrent_jobs,
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
        "--max-concurrent-jobs",
        str(args.max_concurrent_jobs),
        *(
            ["--conda-sh-path", str(Path(args.conda_sh_path).expanduser().resolve())]
            if args.conda_sh_path
            else []
        ),
    ]


def create_mcp_server(queue: ExperimentQueue, daemon_command: list[str]) -> FastMCP:
    mcp = FastMCP("experiment-queue", json_response=True, log_level="INFO")
    runtime: dict[str, Any] = {
        "queue": queue,
        "daemon_command": daemon_command,
    }

    def current_queue() -> ExperimentQueue:
        return runtime["queue"]

    def current_daemon_command() -> list[str]:
        return runtime["daemon_command"]

    unified_actions = [
        "help",
        "prime_queue_session",
        "enqueue_script",
        "set_queue_root",
        "daemon_status",
        "queue_status",
        "list_jobs",
        "get_job",
        "read_job_log",
        "pause_queue",
        "resume_queue",
        "shutdown_daemon",
        "restart_daemon",
        "set_max_concurrent_jobs",
        "stop_after_current",
        "stop_now",
        "cancel_job",
    ]
    tool_names = ["queue"]

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
            q = current_queue()
            q.ensure_daemon_running(current_daemon_command())
            initial_status = q.queue_status()
            actions: list[str] = []
            probe_job: dict[str, Any] | None = None

            if exercise_control_tools:
                q.pause_queue()
                actions.append("pause_queue")
                q.resume_queue()
                actions.append("resume_queue")
                q.stop_after_current()
                actions.append("stop_after_current")
                if initial_status["paused"]:
                    q.pause_queue()
                else:
                    q.resume_queue()
                if initial_status["stop_after_current"]:
                    q.stop_after_current()

            if exercise_enqueue_cancel:
                probe_dir = q.workspace_root / ".experiment_queue_mcp"
                probe_dir.mkdir(parents=True, exist_ok=True)
                probe_script = probe_dir / "prime_queue_session_noop.sh"
                probe_script.write_text(
                    "#!/usr/bin/env bash\nset -euo pipefail\necho prime-queue-session-noop\n",
                    encoding="utf-8",
                )
                probe_script.chmod(0o700)
                queued_probe = q.enqueue_script(
                    source_path=str(probe_script),
                    name="prime-queue-session-noop",
                    cwd=str(q.default_cwd),
                    conda_env=q.default_conda_env,
                )
                actions.append("enqueue_script")
                cancelled_probe = q.cancel_job(queued_probe.job_id)
                actions.append("cancel_job")
                probe_job = _job_summary(cancelled_probe)

            return {
                "server_name": "experiment-queue",
                "available_tools": tool_names,
                "queue_status_before": _queue_status_summary(initial_status),
                "queue_status_after": _queue_status_summary(q.queue_status()),
                "recent_jobs": [_job_summary(job) for job in q.list_jobs(limit=list_limit)],
                "actions_exercised": actions,
                "probe_job": probe_job,
                "note": (
                    "This warms up the MCP server and can exercise reversible queue operations, "
                    "but it does not pre-authorize future Codex shell approvals outside the MCP."
                ),
            }
        except Exception as exc:
            raise ToolError(str(exc)) from exc

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
            q = current_queue()
            result = q.enqueue_script(
                source_path=str(_resolve_path(q, source_path)),
                args=list(args),
                name=name,
                cwd=str(_resolve_path(q, cwd)) if cwd else None,
                conda_env=conda_env,
                owner_label=owner_label,
                owner_session_id=owner_session_id,
            )
            q.ensure_daemon_running(current_daemon_command())
            return _job_summary(result) or {}
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def set_queue_root(
        queue_root: str = Field(..., description="Absolute or workspace-relative path to the queue root directory to use for this MCP server."),
    ) -> dict[str, Any]:
        """Switch this MCP server to a different queue root under the current workspace."""
        try:
            q = current_queue()
            resolved_root = _resolve_path(q, queue_root)
            workspace_root = q.workspace_root
            try:
                resolved_root.relative_to(workspace_root)
            except ValueError as exc:
                raise ToolError(f"queue_root must stay under workspace_root: {workspace_root}") from exc

            old_status = q.queue_status()
            args = argparse.Namespace(
                queue_root=str(resolved_root),
                workspace_root=str(workspace_root),
                default_cwd=str(q.default_cwd),
                script_roots=[str(path) for path in q.script_roots],
                default_conda_env=q.default_conda_env,
                conda_sh_path=str(q.conda_sh_path) if q.conda_sh_path else None,
                shell_path=q.shell_path,
                poll_interval=q.poll_interval_s,
                terminate_grace=q.terminate_grace_s,
                max_concurrent_jobs=old_status.get("max_concurrent_jobs", 2),
            )
            runtime["queue"] = build_queue(args)
            runtime["daemon_command"] = build_daemon_command(args)
            new_status = current_queue().queue_status()
            return {
                "queue_root": str(resolved_root),
                "queue_root_name": resolved_root.name,
                "workspace_root": str(workspace_root),
                "max_concurrent_jobs": new_status.get("max_concurrent_jobs", 2),
                "counts": new_status.get("counts", {}),
                "note": "This switches only the active MCP server session. Other sessions keep their own queue root.",
            }
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def daemon_status() -> dict[str, Any]:
        """Return daemon health and stale-lock indicators without full path metadata."""
        try:
            return _daemon_status_summary(current_queue().daemon_status())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def queue_status() -> dict[str, Any]:
        """Return a compact queue summary for unattended monitoring."""
        try:
            return _queue_status_summary(current_queue().queue_status())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def queue_debug_status() -> dict[str, Any]:
        """Return the full queue status, including absolute paths, for manual debugging."""
        try:
            return _serialize(current_queue().queue_status())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def list_jobs(
        status: str | None = Field(default=None, description="Optional state filter."),
        limit: int | None = Field(default=None, ge=1, description="Maximum number of jobs to return."),
    ) -> list[dict[str, Any]]:
        """List compact job summaries, optionally filtered by state."""
        try:
            return [_job_summary(job) or {} for job in current_queue().list_jobs(status=status, limit=limit)]
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def list_jobs_debug(
        status: str | None = Field(default=None, description="Optional state filter."),
        limit: int | None = Field(default=None, ge=1, description="Maximum number of jobs to return."),
    ) -> list[dict[str, Any]]:
        """List full job metadata, including absolute paths, for manual debugging."""
        try:
            return _serialize(current_queue().list_jobs(status=status, limit=limit))
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def get_job(
        job_id: str = Field(..., description="Queue job ID."),
    ) -> dict[str, Any]:
        """Get a compact summary for one job."""
        try:
            return _job_summary(current_queue().get_job(job_id)) or {}
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def get_job_debug(
        job_id: str = Field(..., description="Queue job ID."),
    ) -> dict[str, Any]:
        """Get full job metadata, including absolute paths, for manual debugging."""
        try:
            return _serialize(current_queue().get_job(job_id))
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def read_job_log(
        job_id: str = Field(..., description="Queue job ID."),
        stream: Literal["stdout", "stderr"] = Field(default="stdout", description="Log stream to read."),
        lines: int = Field(default=100, ge=1, description="Number of trailing log lines to return."),
    ) -> dict[str, str]:
        """Tail stdout or stderr for a queued or completed job."""
        try:
            return {"text": current_queue().read_log(job_id, stream=stream, lines=lines)}
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def pause_queue() -> dict[str, Any]:
        """Pause dequeuing new jobs without stopping the current job."""
        try:
            return _serialize(current_queue().pause_queue())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def resume_queue() -> dict[str, Any]:
        """Resume dequeuing jobs and clear stop-after-current."""
        try:
            q = current_queue()
            q.ensure_daemon_running(current_daemon_command())
            return _serialize(q.resume_queue())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def set_max_concurrent_jobs(
        max_concurrent_jobs: int = Field(..., ge=1, description="Maximum number of queued jobs allowed to run simultaneously."),
    ) -> dict[str, Any]:
        """Set the queue-wide concurrency limit for simultaneously running jobs."""
        try:
            q = current_queue()
            q.ensure_daemon_running(current_daemon_command())
            return _serialize(q.set_max_concurrent_jobs(max_concurrent_jobs))
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def shutdown_daemon() -> dict[str, Any]:
        """Stop the background daemon explicitly and leave queued jobs untouched."""
        try:
            return _daemon_status_summary(current_queue().shutdown_daemon())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def restart_daemon() -> dict[str, Any]:
        """Restart the background daemon explicitly."""
        try:
            q = current_queue()
            return _daemon_status_summary(q.restart_daemon(current_daemon_command()))
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def stop_after_current(
        requester_label: str | None = Field(default=None, description="Optional requester label for ownership checks."),
        requester_session_id: str | None = Field(default=None, description="Optional requester session id for ownership checks."),
        force: bool = Field(default=False, description="Override ownership checks for the active job."),
    ) -> dict[str, Any]:
        """Finish the active job and then stop starting new jobs."""
        try:
            q = current_queue()
            q.assert_running_jobs_control_allowed(
                requester_label=requester_label,
                requester_session_id=requester_session_id,
                force=force,
            )
            q.ensure_daemon_running(current_daemon_command())
            return _serialize(q.stop_after_current())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def stop_now(
        requester_label: str | None = Field(default=None, description="Optional requester label for ownership checks."),
        requester_session_id: str | None = Field(default=None, description="Optional requester session id for ownership checks."),
        force: bool = Field(default=False, description="Override ownership checks for the active job."),
    ) -> dict[str, Any]:
        """Terminate the active job and stop starting new jobs until resumed."""
        try:
            q = current_queue()
            q.assert_running_jobs_control_allowed(
                requester_label=requester_label,
                requester_session_id=requester_session_id,
                force=force,
            )
            q.ensure_daemon_running(current_daemon_command())
            return _serialize(q.stop_now())
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    def cancel_job(
        job_id: str = Field(..., description="Queue job ID."),
        requester_label: str | None = Field(default=None, description="Optional requester label for ownership checks."),
        requester_session_id: str | None = Field(default=None, description="Optional requester session id for ownership checks."),
        force: bool = Field(default=False, description="Override ownership checks for this job."),
    ) -> dict[str, Any]:
        """Cancel a queued job immediately or request stop for a running job."""
        try:
            return _job_summary(
                current_queue().cancel_job(
                    job_id,
                    requester_label=requester_label,
                    requester_session_id=requester_session_id,
                    force=force,
                )
            ) or {}
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool(name="queue")
    def queue_tool(
        action: Literal[
            "help",
            "prime_queue_session",
            "enqueue_script",
            "set_queue_root",
            "daemon_status",
            "queue_status",
            "list_jobs",
            "get_job",
            "read_job_log",
            "pause_queue",
            "resume_queue",
            "shutdown_daemon",
            "restart_daemon",
            "set_max_concurrent_jobs",
            "stop_after_current",
            "stop_now",
            "cancel_job",
        ] = Field(..., description="Queue action to perform."),
        debug: bool = Field(
            default=False,
            description="For queue_status, list_jobs, and get_job, return the full debug payload instead of the compact summary.",
        ),
        source_path: str | None = Field(default=None, description="Absolute or workspace-relative path to a bash script."),
        queue_root: str | None = Field(default=None, description="Workspace-relative or absolute queue root for action=set_queue_root."),
        script_args: list[str] = Field(default_factory=list, description="Optional positional arguments passed to an enqueued script."),
        name: str | None = Field(default=None, description="Optional display name override for enqueue_script."),
        cwd: str | None = Field(default=None, description="Optional working directory for enqueue_script."),
        conda_env: str | None = Field(default=None, description="Optional conda env override for enqueue_script."),
        owner_label: str | None = Field(default=None, description="Optional human-readable owner label for enqueue_script."),
        owner_session_id: str | None = Field(default=None, description="Optional owning session identifier for enqueue_script."),
        job_id: str | None = Field(default=None, description="Queue job ID for get_job, read_job_log, or cancel_job."),
        status_filter: str | None = Field(default=None, description="Optional state filter for list_jobs."),
        limit: int | None = Field(default=None, ge=1, description="Maximum number of jobs to return for list_jobs."),
        stream: Literal["stdout", "stderr"] = Field(default="stdout", description="Log stream for read_job_log."),
        lines: int = Field(default=100, ge=1, description="Number of trailing lines for read_job_log."),
        max_concurrent_jobs: int | None = Field(default=None, ge=1, description="Concurrency limit for set_max_concurrent_jobs."),
        requester_label: str | None = Field(default=None, description="Optional requester label for ownership checks."),
        requester_session_id: str | None = Field(default=None, description="Optional requester session id for ownership checks."),
        force: bool = Field(default=False, description="Override ownership checks for stop/cancel actions."),
        list_limit: int = Field(default=5, ge=1, description="Number of recent jobs to include for prime_queue_session."),
        exercise_control_tools: bool = Field(
            default=True,
            description="For prime_queue_session, whether to exercise reversible control operations.",
        ),
        exercise_enqueue_cancel: bool = Field(
            default=False,
            description="For prime_queue_session, whether to enqueue and cancel a temporary no-op job.",
        ),
    ) -> dict[str, Any]:
        """Unified experiment queue tool. Use the `action` argument to select queue behavior."""
        try:
            if action == "help":
                return {
                    "tool_name": "queue",
                    "available_actions": unified_actions,
                    "debug_capable_actions": ["queue_status", "list_jobs", "get_job"],
                    "preferred_for_autonomy": True,
                    "note": (
                        "For autonomous runs, use this unified `queue` tool for all experiment-queue operations."
                    ),
                }

            if action == "prime_queue_session":
                result = prime_queue_session(
                    list_limit=list_limit,
                    exercise_control_tools=exercise_control_tools,
                    exercise_enqueue_cancel=exercise_enqueue_cancel,
                )
            elif action == "enqueue_script":
                if source_path is None:
                    raise ToolError("source_path is required for action=enqueue_script")
                result = enqueue_script(
                    source_path=source_path,
                    args=script_args,
                    name=name,
                    cwd=cwd,
                    conda_env=conda_env,
                    owner_label=owner_label,
                    owner_session_id=owner_session_id,
                )
            elif action == "set_queue_root":
                if queue_root is None:
                    raise ToolError("queue_root is required for action=set_queue_root")
                result = set_queue_root(queue_root=queue_root)
            elif action == "daemon_status":
                result = daemon_status()
            elif action == "queue_status":
                result = queue_debug_status() if debug else queue_status()
            elif action == "list_jobs":
                result = list_jobs_debug(status=status_filter, limit=limit) if debug else list_jobs(status=status_filter, limit=limit)
            elif action == "get_job":
                if job_id is None:
                    raise ToolError("job_id is required for action=get_job")
                result = get_job_debug(job_id=job_id) if debug else get_job(job_id=job_id)
            elif action == "read_job_log":
                if job_id is None:
                    raise ToolError("job_id is required for action=read_job_log")
                result = read_job_log(job_id=job_id, stream=stream, lines=lines)
            elif action == "pause_queue":
                result = pause_queue()
            elif action == "resume_queue":
                result = resume_queue()
            elif action == "shutdown_daemon":
                result = shutdown_daemon()
            elif action == "restart_daemon":
                result = restart_daemon()
            elif action == "set_max_concurrent_jobs":
                if max_concurrent_jobs is None:
                    raise ToolError("max_concurrent_jobs is required for action=set_max_concurrent_jobs")
                result = set_max_concurrent_jobs(max_concurrent_jobs=max_concurrent_jobs)
            elif action == "stop_after_current":
                result = stop_after_current(
                    requester_label=requester_label,
                    requester_session_id=requester_session_id,
                    force=force,
                )
            elif action == "stop_now":
                result = stop_now(
                    requester_label=requester_label,
                    requester_session_id=requester_session_id,
                    force=force,
                )
            elif action == "cancel_job":
                if job_id is None:
                    raise ToolError("job_id is required for action=cancel_job")
                result = cancel_job(
                    job_id=job_id,
                    requester_label=requester_label,
                    requester_session_id=requester_session_id,
                    force=force,
                )
            else:  # pragma: no cover - Literal already constrains this
                raise ToolError(f"Unsupported action: {action}")

            return {
                "action": action,
                "result": result,
            }
        except Exception as exc:
            if isinstance(exc, ToolError):
                raise
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

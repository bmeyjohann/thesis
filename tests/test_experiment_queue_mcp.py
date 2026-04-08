from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from tools.experiment_queue_mcp.queue_engine import ExperimentQueue
from tools.experiment_queue_mcp.queue_engine import JobRecord
from tools.experiment_queue_mcp.queue_engine import STATUS_FAILED
from tools.experiment_queue_mcp.queue_engine import STATUS_RUNNING


SERVER_PYTHON = "/usr/bin/python3"


def _write_message(stdin, payload: dict) -> None:
    stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
    stdin.flush()


def _read_message(stdout) -> dict:
    while True:
        line = stdout.readline()
        assert line, "unexpected EOF from MCP server"
        payload = json.loads(line.decode("utf-8"))
        if payload.get("id") is not None:
            return payload


def _call(proc: subprocess.Popen[bytes], request_id: int, method: str, params: dict | None = None) -> dict:
    _write_message(
        proc.stdin,
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}},
    )
    payload = _read_message(proc.stdout)
    result = payload.get("result")
    structured = result.get("structuredContent") if isinstance(result, dict) else None
    if isinstance(structured, dict) and "result" not in structured:
        result["structuredContent"] = {"result": structured, **structured}
    return payload


def test_mcp_queue_runs_script(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    scripts_dir.mkdir(parents=True)

    script_path = scripts_dir / "hello.sh"
    script_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho queue-ok\n", encoding="utf-8")
    script_path.chmod(0o755)

    server_path = Path(__file__).resolve().parents[1] / "tools" / "experiment_queue_mcp" / "server.py"
    proc = subprocess.Popen(
        [
            SERVER_PYTHON,
            str(server_path),
            "--queue-root",
            str(queue_root),
            "--workspace-root",
            str(workspace),
            "--default-cwd",
            str(workspace),
            "--script-root",
            str(scripts_dir),
            "--poll-interval",
            "0.05",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        init = _call(
            proc,
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        )
        assert init["result"]["serverInfo"]["name"] == "experiment-queue"

        tools = _call(proc, 2, "tools/list")
        tool_names = {tool["name"] for tool in tools["result"]["tools"]}
        assert tool_names == {"queue"}

        queue_status = _call(
            proc,
            3,
            "tools/call",
            {
                "name": "queue",
                "arguments": {
                    "action": "queue_status",
                },
            },
        )
        assert queue_status["result"]["structuredContent"]["result"]["queue_root_name"] == "experiment_queue"

        queued = _call(
            proc,
            4,
            "tools/call",
            {
                "name": "queue",
                "arguments": {
                    "action": "enqueue_script",
                    "source_path": str(script_path),
                },
            },
        )
        job_id = queued["result"]["structuredContent"]["result"]["job_id"]
        assert job_id

        deadline = time.time() + 10.0
        final_job = None
        while time.time() < deadline:
            result = _call(
                proc,
                5,
                "tools/call",
                {
                    "name": "queue",
                    "arguments": {
                        "action": "get_job",
                        "job_id": job_id,
                    },
                },
            )
            final_job = result["result"]["structuredContent"]["result"]
            if final_job["status"] == "finished":
                break
            time.sleep(0.05)

        assert final_job is not None
        assert final_job["status"] == "finished"
        log_result = _call(
            proc,
            6,
            "tools/call",
            {
                "name": "queue",
                "arguments": {
                    "action": "read_job_log",
                    "job_id": job_id,
                    "stream": "stdout",
                    "lines": 20,
                },
            },
        )
        assert "queue-ok" in log_result["result"]["structuredContent"]["result"]["text"]
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_running_job_survives_server_restart(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    scripts_dir.mkdir(parents=True)

    script_path = scripts_dir / "slow.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nsleep 1.0\necho survived\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    server_path = Path(__file__).resolve().parents[1] / "tools" / "experiment_queue_mcp" / "server.py"

    def _start_server() -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [
                SERVER_PYTHON,
                str(server_path),
                "--queue-root",
                str(queue_root),
                "--workspace-root",
                str(workspace),
                "--default-cwd",
                str(workspace),
                "--script-root",
                str(scripts_dir),
                "--poll-interval",
                "0.05",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    proc = _start_server()
    try:
        _call(
            proc,
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        )
        queued = _call(
            proc,
            2,
            "tools/call",
            {
                "name": "queue",
                "arguments": {
                    "action": "enqueue_script",
                    "source_path": str(script_path),
                },
            },
        )
        job_id = queued["result"]["structuredContent"]["result"]["job_id"]

        deadline = time.time() + 5.0
        while time.time() < deadline:
            result = _call(
                proc,
                3,
                "tools/call",
                {
                    "name": "queue",
                    "arguments": {
                        "action": "get_job",
                        "job_id": job_id,
                    },
                },
            )
            if result["result"]["structuredContent"]["result"]["status"] == "running":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("job never reached running state")

        proc.kill()
        proc.wait(timeout=5)

        proc = _start_server()
        _call(
            proc,
            4,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        )

        deadline = time.time() + 10.0
        final_job = None
        while time.time() < deadline:
            result = _call(
                proc,
                5,
                "tools/call",
                {
                    "name": "queue",
                    "arguments": {
                        "action": "get_job",
                        "job_id": job_id,
                    },
                },
            )
            final_job = result["result"]["structuredContent"]["result"]
            if final_job["status"] == "finished":
                break
            time.sleep(0.05)

        assert final_job is not None
        assert final_job["status"] == "finished"
        log_result = _call(
            proc,
            6,
            "tools/call",
            {
                "name": "queue",
                "arguments": {
                    "action": "read_job_log",
                    "job_id": job_id,
                    "stream": "stdout",
                    "lines": 20,
                },
            },
        )
        assert "survived" in log_result["result"]["structuredContent"]["result"]["text"]
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_queue_tool_can_switch_queue_root_within_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    alt_queue_root = workspace / "alt_queue"
    scripts_dir.mkdir(parents=True)

    server_path = Path(__file__).resolve().parents[1] / "tools" / "experiment_queue_mcp" / "server.py"
    proc = subprocess.Popen(
        [
            SERVER_PYTHON,
            str(server_path),
            "--queue-root",
            str(queue_root),
            "--workspace-root",
            str(workspace),
            "--default-cwd",
            str(workspace),
            "--script-root",
            str(scripts_dir),
            "--poll-interval",
            "0.05",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _call(
            proc,
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        )
        switched = _call(
            proc,
            2,
            "tools/call",
            {
                "name": "queue",
                "arguments": {
                    "action": "set_queue_root",
                    "queue_root": str(alt_queue_root),
                },
            },
        )
        payload = switched["result"]["structuredContent"]["result"]
        assert payload["queue_root"] == str(alt_queue_root)
        assert payload["queue_root_name"] == "alt_queue"
        assert payload["max_concurrent_jobs"] == 2

        status = _call(
            proc,
            3,
            "tools/call",
            {
                "name": "queue",
                "arguments": {
                    "action": "queue_status",
                    "debug": True,
                },
            },
        )
        assert status["result"]["structuredContent"]["result"]["queue_root"] == str(alt_queue_root)
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_two_servers_share_one_auto_started_daemon(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    scripts_dir.mkdir(parents=True)
    script_path = scripts_dir / "slow.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nsleep 1.0\necho shared-daemon\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    server_path = Path(__file__).resolve().parents[1] / "tools" / "experiment_queue_mcp" / "server.py"

    def _start_server() -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [
                SERVER_PYTHON,
                str(server_path),
                "--queue-root",
                str(queue_root),
                "--workspace-root",
                str(workspace),
                "--default-cwd",
                str(workspace),
                "--script-root",
                str(scripts_dir),
                "--poll-interval",
                "0.05",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    proc_a = _start_server()
    proc_b = None
    try:
        init_a = _call(
            proc_a,
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest-a", "version": "0"},
            },
        )
        proc_b = _start_server()
        init_b = _call(
            proc_b,
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest-b", "version": "0"},
            },
        )
        assert init_a["result"]["serverInfo"]["name"] == "experiment-queue"
        assert init_b["result"]["serverInfo"]["name"] == "experiment-queue"

        queued = _call(
            proc_a,
            2,
            "tools/call",
            {
                "name": "queue",
                "arguments": {"action": "enqueue_script", "source_path": str(script_path)},
            },
        )
        job_id = queued["result"]["structuredContent"]["result"]["job_id"]
        assert job_id

        deadline = time.time() + 5.0
        daemon_pid_a = daemon_pid_b = None
        while time.time() < deadline:
            status_a = _call(proc_a, 3, "tools/call", {"name": "queue", "arguments": {"action": "queue_status"}})
            status_b = _call(proc_b, 3, "tools/call", {"name": "queue", "arguments": {"action": "queue_status"}})
            daemon_pid_a = status_a["result"]["structuredContent"]["result"].get("daemon_pid")
            daemon_pid_b = status_b["result"]["structuredContent"]["result"].get("daemon_pid")
            if (
                status_a["result"]["structuredContent"]["result"]["daemon_running"] is True
                and status_b["result"]["structuredContent"]["result"]["daemon_running"] is True
                and daemon_pid_a is not None
                and daemon_pid_a == daemon_pid_b
            ):
                break
            time.sleep(0.1)
        else:
            raise AssertionError("two attached servers never converged on the same running daemon")
    finally:
        proc_a.kill()
        proc_a.wait(timeout=5)
        if proc_b is not None:
            proc_b.kill()
            proc_b.wait(timeout=5)


def test_daemon_persists_until_explicit_shutdown(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    scripts_dir.mkdir(parents=True)
    queue = ExperimentQueue(
        queue_root=queue_root,
        workspace_root=workspace,
        script_roots=[scripts_dir],
        default_cwd=workspace,
        poll_interval_s=0.05,
        max_concurrent_jobs=2,
    )
    daemon_command = [
        SERVER_PYTHON,
        str(Path(__file__).resolve().parents[1] / "tools" / "experiment_queue_mcp" / "daemon.py"),
        "--queue-root",
        str(queue_root),
        "--workspace-root",
        str(workspace),
        "--default-cwd",
        str(workspace),
        "--script-root",
        str(scripts_dir),
        "--poll-interval",
        "0.05",
        "--max-concurrent-jobs",
        "2",
    ]

    before = queue.daemon_status()
    assert before["daemon_running"] is False

    started = queue.restart_daemon(daemon_command, startup_wait_s=10.0)
    assert started["daemon_running"] is True
    assert started["restarted_now"] is True

    time.sleep(0.35)
    still_running = queue.daemon_status()
    assert still_running["daemon_running"] is True
    assert still_running["worker_heartbeat_fresh"] is True

    stopped = queue.shutdown_daemon(wait_s=10.0)
    assert stopped["daemon_running"] is False
    assert stopped["stopped_now"] is True

    after = queue.daemon_status()
    assert after["daemon_running"] is False


def test_queue_can_run_two_jobs_in_parallel_when_configured(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    scripts_dir.mkdir(parents=True)

    script_path = scripts_dir / "slow.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nsleep 1.0\necho parallel-ok\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    queue = ExperimentQueue(
        queue_root=queue_root,
        workspace_root=workspace,
        script_roots=[scripts_dir],
        default_cwd=workspace,
        poll_interval_s=0.05,
        max_concurrent_jobs=2,
    )
    queue.set_max_concurrent_jobs(2)
    daemon_command = [
        SERVER_PYTHON,
        str(Path(__file__).resolve().parents[1] / "tools" / "experiment_queue_mcp" / "daemon.py"),
        "--queue-root",
        str(queue_root),
        "--workspace-root",
        str(workspace),
        "--default-cwd",
        str(workspace),
        "--script-root",
        str(scripts_dir),
        "--poll-interval",
        "0.05",
        "--max-concurrent-jobs",
        "2",
    ]

    job_ids = [queue.enqueue_script(str(script_path)).job_id for _ in range(3)]
    queue.ensure_daemon_running(daemon_command, startup_wait_s=10.0)

    try:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            status = queue.queue_status()
            if status["max_concurrent_jobs"] == 2 and len(status.get("running_jobs", [])) == 2:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("queue never reached two concurrent running jobs")

        deadline = time.time() + 15.0
        final_statuses: dict[str, str] = {}
        while time.time() < deadline:
            final_statuses = {job_id: queue.get_job(job_id).status for job_id in job_ids}
            if all(status == "finished" for status in final_statuses.values()):
                break
            time.sleep(0.05)

        assert all(status == "finished" for status in final_statuses.values())
    finally:
        try:
            queue.shutdown_daemon(wait_s=10.0)
        except RuntimeError:
            pass


def test_cancelling_one_running_job_does_not_stop_another_running_job(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    scripts_dir.mkdir(parents=True)

    slow_a = scripts_dir / "slow-a.sh"
    slow_a.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nsleep 3.0\necho a-finished\n",
        encoding="utf-8",
    )
    slow_a.chmod(0o755)
    slow_b = scripts_dir / "slow-b.sh"
    slow_b.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nsleep 3.0\necho b-finished\n",
        encoding="utf-8",
    )
    slow_b.chmod(0o755)

    queue = ExperimentQueue(
        queue_root=queue_root,
        workspace_root=workspace,
        script_roots=[scripts_dir],
        default_cwd=workspace,
        poll_interval_s=0.05,
        max_concurrent_jobs=2,
    )
    queue.set_max_concurrent_jobs(2)
    daemon_command = [
        SERVER_PYTHON,
        str(Path(__file__).resolve().parents[1] / "tools" / "experiment_queue_mcp" / "daemon.py"),
        "--queue-root",
        str(queue_root),
        "--workspace-root",
        str(workspace),
        "--default-cwd",
        str(workspace),
        "--script-root",
        str(scripts_dir),
        "--poll-interval",
        "0.05",
        "--max-concurrent-jobs",
        "2",
    ]

    queued_a = queue.enqueue_script(str(slow_a))
    queued_b = queue.enqueue_script(str(slow_b))
    job_a = queued_a.job_id
    job_b = queued_b.job_id
    queue.ensure_daemon_running(daemon_command)

    try:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            status = queue.queue_status()
            if len(status.get("running_jobs", [])) == 2:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("jobs never reached parallel running state")

        queue.cancel_job(job_a)

        deadline = time.time() + 10.0
        final_a = final_b = None
        while time.time() < deadline:
            final_a = queue.get_job(job_a).status
            final_b = queue.get_job(job_b).status
            if final_a == "cancelled" and final_b == "finished":
                break
            time.sleep(0.05)

        assert final_a in {"cancelled", "finished"}
        assert final_b == "finished"
    finally:
        try:
            queue.shutdown_daemon()
        except RuntimeError:
            pass


def test_queue_uses_round_robin_by_owner_for_next_job(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    scripts_dir.mkdir(parents=True)

    slow = scripts_dir / "slow.sh"
    slow.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nsleep 0.3\necho rr\n",
        encoding="utf-8",
    )
    slow.chmod(0o755)

    queue = ExperimentQueue(
        queue_root=queue_root,
        workspace_root=workspace,
        script_roots=[scripts_dir],
        default_cwd=workspace,
        poll_interval_s=0.05,
        max_concurrent_jobs=1,
    )
    queue.set_max_concurrent_jobs(1)
    daemon_command = [
        SERVER_PYTHON,
        str(Path(__file__).resolve().parents[1] / "tools" / "experiment_queue_mcp" / "daemon.py"),
        "--queue-root",
        str(queue_root),
        "--workspace-root",
        str(workspace),
        "--default-cwd",
        str(workspace),
        "--script-root",
        str(scripts_dir),
        "--poll-interval",
        "0.05",
        "--max-concurrent-jobs",
        "1",
    ]

    queued_a1 = queue.enqueue_script(str(slow), owner_session_id="session-a", name="a1")
    queued_a2 = queue.enqueue_script(str(slow), owner_session_id="session-a", name="a2")
    queued_b1 = queue.enqueue_script(str(slow), owner_session_id="session-b", name="b1")
    job_ids = [queued_a1.job_id, queued_a2.job_id, queued_b1.job_id]
    queue.ensure_daemon_running(daemon_command)

    try:
        deadline = time.time() + 10.0
        jobs = {}
        while time.time() < deadline:
            jobs = {job_id: queue.get_job(job_id) for job_id in job_ids}
            if all(job.status == "finished" for job in jobs.values()):
                break
            time.sleep(0.05)

        assert all(job.status == "finished" for job in jobs.values())

        ordered = sorted(
            ((job.name, job.started_at) for job in jobs.values()),
            key=lambda item: item[1],
        )
        assert [name for name, _ in ordered] == ["a1", "b1", "a2"]
    finally:
        try:
            queue.shutdown_daemon(force=True)
        except RuntimeError:
            pass


def test_cancel_job_respects_owner_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    scripts_dir.mkdir(parents=True)
    script_path = scripts_dir / "hello.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nsleep 1.0\necho owner-guard\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    server_path = Path(__file__).resolve().parents[1] / "tools" / "experiment_queue_mcp" / "server.py"
    proc = subprocess.Popen(
        [
            SERVER_PYTHON,
            str(server_path),
            "--queue-root",
            str(queue_root),
            "--workspace-root",
            str(workspace),
            "--default-cwd",
            str(workspace),
            "--script-root",
            str(scripts_dir),
            "--poll-interval",
            "0.05",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _call(
            proc,
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        )
        queued = _call(
            proc,
            2,
            "tools/call",
            {
                "name": "queue",
                "arguments": {
                    "action": "enqueue_script",
                    "source_path": str(script_path),
                    "owner_label": "session-a",
                    "owner_session_id": "session-a-id",
                },
            },
        )
        job_id = queued["result"]["structuredContent"]["result"]["job_id"]
        denied = _call(
            proc,
            3,
            "tools/call",
            {
                "name": "queue",
                "arguments": {
                    "action": "cancel_job",
                    "job_id": job_id,
                    "requester_label": "session-b",
                    "requester_session_id": "session-b-id",
                },
            },
        )
        assert denied["result"]["isError"] is True
        assert "Job control denied" in denied["result"]["content"][0]["text"]

        allowed = _call(
            proc,
            4,
            "tools/call",
            {
                "name": "queue",
                "arguments": {
                    "action": "cancel_job",
                    "job_id": job_id,
                    "requester_label": "session-a",
                    "requester_session_id": "session-a-id",
                },
            },
        )
        allowed_status = allowed["result"]["structuredContent"]["result"]["status"]
        assert allowed_status in {"queued", "running", "cancelled"}

        deadline = time.time() + 5.0
        final_job = None
        while time.time() < deadline:
            result = _call(
                proc,
                5,
                "tools/call",
                {
                    "name": "queue",
                    "arguments": {
                        "action": "get_job",
                        "job_id": job_id,
                    },
                },
            )
            final_job = result["result"]["structuredContent"]["result"]
            if final_job["status"] == "cancelled":
                break
            time.sleep(0.05)

        assert final_job is not None
        assert final_job["status"] == "cancelled"
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_summary_tools_redact_paths_but_debug_tools_expose_them(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    scripts_dir.mkdir(parents=True)
    script_path = scripts_dir / "hello.sh"
    script_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho redaction-test\n", encoding="utf-8")
    script_path.chmod(0o755)

    server_path = Path(__file__).resolve().parents[1] / "tools" / "experiment_queue_mcp" / "server.py"
    proc = subprocess.Popen(
        [
            SERVER_PYTHON,
            str(server_path),
            "--queue-root",
            str(queue_root),
            "--workspace-root",
            str(workspace),
            "--default-cwd",
            str(workspace),
            "--script-root",
            str(scripts_dir),
            "--poll-interval",
            "0.05",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _call(
            proc,
            1,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        )

        status_summary = _call(proc, 2, "tools/call", {"name": "queue", "arguments": {"action": "queue_status"}})
        status_debug = _call(proc, 3, "tools/call", {"name": "queue", "arguments": {"action": "queue_status", "debug": True}})
        assert "queue_root" not in status_summary["result"]["structuredContent"]["result"]
        assert "workspace_root" not in status_summary["result"]["structuredContent"]["result"]
        assert "daemon_log_path" not in status_summary["result"]["structuredContent"]["result"]
        assert "worker_lock_path" not in status_summary["result"]["structuredContent"]["result"]
        assert status_debug["result"]["structuredContent"]["result"]["queue_root"] == str(queue_root)
        assert status_debug["result"]["structuredContent"]["result"]["workspace_root"] == str(workspace)

        queued = _call(
            proc,
            4,
            "tools/call",
            {"name": "queue", "arguments": {"action": "enqueue_script", "source_path": str(script_path)}},
        )
        job_id = queued["result"]["structuredContent"]["result"]["job_id"]
        assert "script_path" not in queued["result"]["structuredContent"]["result"]
        assert queued["result"]["structuredContent"]["result"]["name"] == "hello"

        summary_job = _call(
            proc,
            5,
            "tools/call",
            {"name": "queue", "arguments": {"action": "get_job", "job_id": job_id}},
        )
        debug_job = _call(
            proc,
            6,
            "tools/call",
            {"name": "queue", "arguments": {"action": "get_job", "job_id": job_id, "debug": True}},
        )
        assert "script_path" not in summary_job["result"]["structuredContent"]["result"]
        assert "cwd" not in summary_job["result"]["structuredContent"]["result"]
        assert debug_job["result"]["structuredContent"]["result"]["script_path"]
        assert debug_job["result"]["structuredContent"]["result"]["cwd"] == str(workspace)
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_daemon_status_ignores_stale_pid_text_when_lock_is_free(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    scripts_dir.mkdir(parents=True)

    queue = ExperimentQueue(
        queue_root=queue_root,
        workspace_root=workspace,
        script_roots=[scripts_dir],
        default_cwd=workspace,
    )

    queue.worker_lock_path.write_text("1\n", encoding="utf-8")
    status = queue.daemon_status()
    assert status["daemon_running"] is False
    assert status["daemon_pid"] is None


def test_daemon_status_ignores_lock_without_fresh_heartbeat(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    scripts_dir.mkdir(parents=True)

    queue = ExperimentQueue(
        queue_root=queue_root,
        workspace_root=workspace,
        script_roots=[scripts_dir],
        default_cwd=workspace,
        poll_interval_s=0.05,
    )

    holder = subprocess.Popen(
        [
            SERVER_PYTHON,
            "-c",
            (
                "import fcntl, sys, time; "
                "path, pid_text = sys.argv[1], sys.argv[2]; "
                "handle = open(path, 'a+', encoding='utf-8'); "
                "fcntl.flock(handle.fileno(), fcntl.LOCK_EX); "
                "handle.seek(0); "
                "handle.truncate(); "
                "handle.write(pid_text + '\\n'); "
                "handle.flush(); "
                "time.sleep(5)"
            ),
            str(queue.worker_lock_path),
            "999999",
        ],
    )
    try:
        time.sleep(0.2)
        status = queue.daemon_status()
        assert status["worker_lock_held"] is True
        assert status["worker_heartbeat_fresh"] is False
        assert status["worker_lock_stale"] is True
        assert status["daemon_running"] is False
        assert status["daemon_pid"] is None
    finally:
        holder.terminate()
        holder.wait(timeout=5)


def test_queue_status_recovers_orphaned_running_job_when_worker_is_unhealthy(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
    scripts_dir.mkdir(parents=True)

    queue = ExperimentQueue(
        queue_root=queue_root,
        workspace_root=workspace,
        script_roots=[scripts_dir],
        default_cwd=workspace,
        poll_interval_s=0.05,
    )

    running_script = queue.running_dir / "20260403-000000-deadbeef__orphan.sh"
    running_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho orphan\n", encoding="utf-8")
    running_script.chmod(0o755)

    job = JobRecord(
        job_id="20260403-000000-deadbeef",
        name="orphan",
        status=STATUS_RUNNING,
        script_path=str(running_script),
        submitted_at=time.time() - 30.0,
        started_at=time.time() - 30.0,
        cwd=str(workspace),
        pid=999999,
        stdout_log=str(queue.logs_dir / "20260403-000000-deadbeef.out"),
        stderr_log=str(queue.logs_dir / "20260403-000000-deadbeef.err"),
        exit_code_path=str(queue.results_dir / "20260403-000000-deadbeef.exitcode"),
    )
    queue._write_job_locked(job)

    status = queue.queue_status()
    recovered = queue.get_job(job.job_id)
    assert recovered.status == STATUS_FAILED
    assert Path(recovered.script_path).parent == queue.failed_dir
    assert "Recovered stale running job" in (recovered.note or "")
    assert status["current_job"] is None
    assert status["counts"].get(STATUS_FAILED) == 1

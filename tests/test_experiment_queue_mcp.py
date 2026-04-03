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
    return _read_message(proc.stdout)


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
        assert "prime_queue_session" in tool_names
        assert "enqueue_script" in tool_names
        assert "queue_status" in tool_names

        prime = _call(
            proc,
            3,
            "tools/call",
            {"name": "prime_queue_session", "arguments": {"exercise_control_tools": True}},
        )
        assert prime["result"]["structuredContent"]["server_name"] == "experiment-queue"

        queued = _call(
            proc,
            4,
            "tools/call",
            {"name": "enqueue_script", "arguments": {"source_path": str(script_path)}},
        )
        job_id = queued["result"]["structuredContent"]["job_id"]
        assert job_id

        deadline = time.time() + 10.0
        final_job = None
        while time.time() < deadline:
            result = _call(
                proc,
                5,
                "tools/call",
                {"name": "get_job", "arguments": {"job_id": job_id}},
            )
            final_job = result["result"]["structuredContent"]
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
                "name": "read_job_log",
                "arguments": {"job_id": job_id, "stream": "stdout", "lines": 20},
            },
        )
        assert "queue-ok" in log_result["result"]["structuredContent"]["text"]
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
            {"name": "enqueue_script", "arguments": {"source_path": str(script_path)}},
        )
        job_id = queued["result"]["structuredContent"]["job_id"]

        deadline = time.time() + 5.0
        while time.time() < deadline:
            result = _call(
                proc,
                3,
                "tools/call",
                {"name": "get_job", "arguments": {"job_id": job_id}},
            )
            if result["result"]["structuredContent"]["status"] == "running":
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
                {"name": "get_job", "arguments": {"job_id": job_id}},
            )
            final_job = result["result"]["structuredContent"]
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
                "name": "read_job_log",
                "arguments": {"job_id": job_id, "stream": "stdout", "lines": 20},
            },
        )
        assert "survived" in log_result["result"]["structuredContent"]["text"]
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
            {"name": "enqueue_script", "arguments": {"source_path": str(script_path)}},
        )
        job_id = queued["result"]["structuredContent"]["job_id"]
        assert job_id

        status_a = _call(proc_a, 3, "tools/call", {"name": "queue_status", "arguments": {}})
        status_b = _call(proc_b, 3, "tools/call", {"name": "queue_status", "arguments": {}})
        daemon_pid_a = status_a["result"]["structuredContent"]["daemon_pid"]
        daemon_pid_b = status_b["result"]["structuredContent"]["daemon_pid"]
        assert status_a["result"]["structuredContent"]["daemon_running"] is True
        assert status_b["result"]["structuredContent"]["daemon_running"] is True
        assert daemon_pid_a is not None
        assert daemon_pid_a == daemon_pid_b
    finally:
        proc_a.kill()
        proc_a.wait(timeout=5)
        if proc_b is not None:
            proc_b.kill()
            proc_b.wait(timeout=5)


def test_daemon_persists_until_explicit_shutdown_via_mcp(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scripts_dir = workspace / "scripts"
    queue_root = workspace / "experiment_queue"
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

        before = _call(proc, 2, "tools/call", {"name": "daemon_status", "arguments": {}})
        assert before["result"]["structuredContent"]["daemon_running"] is False

        started = _call(proc, 3, "tools/call", {"name": "restart_daemon", "arguments": {}})
        assert started["result"]["structuredContent"]["daemon_running"] is True
        assert started["result"]["structuredContent"]["restarted_now"] is True

        time.sleep(0.35)
        still_running = _call(proc, 4, "tools/call", {"name": "daemon_status", "arguments": {}})
        assert still_running["result"]["structuredContent"]["daemon_running"] is True
        assert still_running["result"]["structuredContent"]["worker_heartbeat_fresh"] is True

        stopped = _call(proc, 5, "tools/call", {"name": "shutdown_daemon", "arguments": {}})
        assert stopped["result"]["structuredContent"]["daemon_running"] is False
        assert stopped["result"]["structuredContent"]["stopped_now"] is True

        after = _call(proc, 6, "tools/call", {"name": "daemon_status", "arguments": {}})
        assert after["result"]["structuredContent"]["daemon_running"] is False
    finally:
        proc.kill()
        proc.wait(timeout=5)


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
                "name": "enqueue_script",
                "arguments": {
                    "source_path": str(script_path),
                    "owner_label": "session-a",
                    "owner_session_id": "session-a-id",
                },
            },
        )
        job_id = queued["result"]["structuredContent"]["job_id"]
        denied = _call(
            proc,
            3,
            "tools/call",
            {
                "name": "cancel_job",
                "arguments": {
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
                "name": "cancel_job",
                "arguments": {
                    "job_id": job_id,
                    "requester_label": "session-a",
                    "requester_session_id": "session-a-id",
                },
            },
        )
        allowed_status = allowed["result"]["structuredContent"]["status"]
        assert allowed_status in {"queued", "running", "cancelled"}

        deadline = time.time() + 5.0
        final_job = None
        while time.time() < deadline:
            result = _call(
                proc,
                5,
                "tools/call",
                {"name": "get_job", "arguments": {"job_id": job_id}},
            )
            final_job = result["result"]["structuredContent"]
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

        status_summary = _call(proc, 2, "tools/call", {"name": "queue_status", "arguments": {}})
        status_debug = _call(proc, 3, "tools/call", {"name": "queue_debug_status", "arguments": {}})
        assert "queue_root" not in status_summary["result"]["structuredContent"]
        assert "workspace_root" not in status_summary["result"]["structuredContent"]
        assert "daemon_log_path" not in status_summary["result"]["structuredContent"]
        assert "worker_lock_path" not in status_summary["result"]["structuredContent"]
        assert status_debug["result"]["structuredContent"]["queue_root"] == str(queue_root)
        assert status_debug["result"]["structuredContent"]["workspace_root"] == str(workspace)

        queued = _call(
            proc,
            4,
            "tools/call",
            {"name": "enqueue_script", "arguments": {"source_path": str(script_path)}},
        )
        job_id = queued["result"]["structuredContent"]["job_id"]
        assert "script_path" not in queued["result"]["structuredContent"]
        assert queued["result"]["structuredContent"]["name"] == "hello"

        summary_job = _call(
            proc,
            5,
            "tools/call",
            {"name": "get_job", "arguments": {"job_id": job_id}},
        )
        debug_job = _call(
            proc,
            6,
            "tools/call",
            {"name": "get_job_debug", "arguments": {"job_id": job_id}},
        )
        assert "script_path" not in summary_job["result"]["structuredContent"]
        assert "cwd" not in summary_job["result"]["structuredContent"]
        assert debug_job["result"]["structuredContent"]["script_path"]
        assert debug_job["result"]["structuredContent"]["cwd"] == str(workspace)
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

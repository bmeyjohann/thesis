from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_FINISHED = "finished"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
FINAL_STATUSES = {
    STATUS_FINISHED,
    STATUS_FAILED,
    STATUS_CANCELLED,
}


@dataclass
class JobRecord:
    job_id: str
    name: str
    status: str
    script_path: str
    submitted_at: float
    owner_label: str | None = None
    owner_session_id: str | None = None
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    conda_env: str | None = None
    source_path: str | None = None
    manual_drop: bool = False
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    pid: int | None = None
    pgid: int | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    exit_code_path: str | None = None
    note: str | None = None


@dataclass
class ControlState:
    paused: bool = False
    stop_after_current: bool = False
    stop_now: bool = False


@dataclass
class RecoveredProcessHandle:
    pid: int

    def poll(self) -> int | None:
        proc_path = Path("/proc") / str(self.pid)
        return None if proc_path.exists() else 1


class ExperimentQueue:
    def __init__(
        self,
        queue_root: Path,
        workspace_root: Path,
        script_roots: list[Path],
        default_cwd: Path,
        default_conda_env: str | None = None,
        conda_sh_path: Path | None = None,
        shell_path: str = "/bin/bash",
        poll_interval_s: float = 1.0,
        terminate_grace_s: float = 10.0,
    ) -> None:
        self.queue_root = queue_root.resolve()
        self.workspace_root = workspace_root.resolve()
        self.script_roots = [path.resolve() for path in script_roots]
        self.default_cwd = default_cwd.resolve()
        self.default_conda_env = default_conda_env
        self.conda_sh_path = conda_sh_path.resolve() if conda_sh_path else None
        self.shell_path = shell_path
        self.poll_interval_s = poll_interval_s
        self.terminate_grace_s = terminate_grace_s

        self.queued_dir = self.queue_root / "queued"
        self.running_dir = self.queue_root / "running"
        self.finished_dir = self.queue_root / "finished"
        self.failed_dir = self.queue_root / "failed"
        self.cancelled_dir = self.queue_root / "cancelled"
        self.logs_dir = self.queue_root / "logs"
        self.state_dir = self.queue_root / "state"
        self.jobs_dir = self.state_dir / "jobs"
        self.results_dir = self.state_dir / "results"
        self.control_path = self.state_dir / "control.json"
        self.worker_lock_path = self.state_dir / "worker.lock"
        self.daemon_log_path = self.logs_dir / "daemon.log"

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._current_process: subprocess.Popen[bytes] | RecoveredProcessHandle | None = None
        self._terminate_requested_at: float | None = None
        self._worker_lock_handle: Any | None = None

        self._ensure_layout()
        self._ensure_control_state()
        self._reconcile_jobs_locked()

    def _ensure_layout(self) -> None:
        for path in (
            self.queue_root,
            self.queued_dir,
            self.running_dir,
            self.finished_dir,
            self.failed_dir,
            self.cancelled_dir,
            self.logs_dir,
            self.state_dir,
            self.jobs_dir,
            self.results_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _ensure_control_state(self) -> None:
        if self.control_path.exists():
            return
        self._write_json(self.control_path, asdict(ControlState()))

    def start(self) -> bool:
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return True
            try:
                self._acquire_worker_lock_locked()
            except RuntimeError:
                return False
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="experiment-queue-worker",
                daemon=True,
            )
            self._worker_thread.start()
            return True

    def stop(self, *, kill_running: bool = False) -> None:
        self._stop_event.set()
        with self._lock:
            if kill_running and self._current_process:
                self._kill_current_process_locked(force=True)
        thread = self._worker_thread
        if thread:
            thread.join(timeout=5.0)
        with self._lock:
            self._release_worker_lock_locked()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[experiment-queue] worker error: {exc}", file=os.sys.stderr)
                time.sleep(self.poll_interval_s)

    def _tick(self) -> None:
        with self._lock:
            self._reconcile_jobs_locked()
            control = self._read_control_locked()
            running_job = self._get_running_job_locked()

            if running_job:
                self._poll_running_job_locked(running_job, control)
            else:
                if control.stop_now:
                    control.stop_now = False
                    self._write_control_locked(control)
                if control.stop_after_current:
                    return
                if control.paused:
                    return
                next_job = self._dequeue_next_job_locked()
                if next_job:
                    self._start_job_locked(next_job)
        time.sleep(self.poll_interval_s)

    def _poll_running_job_locked(
        self,
        job: JobRecord,
        control: ControlState,
    ) -> None:
        process = self._current_process
        if not process:
            process = self._recover_process_handle_locked(job)
            if not process:
                recorded_exit = self._read_recorded_exit_code_locked(job)
                if recorded_exit is not None:
                    status = STATUS_FINISHED if recorded_exit == 0 else STATUS_FAILED
                    self._finish_job_locked(job, status=status, exit_code=recorded_exit)
                    return
                if self._should_wait_for_exit_sidecar_locked(job):
                    return
                self._finish_job_locked(
                    job,
                    status=STATUS_FAILED,
                    exit_code=None,
                    note="Process handle missing while job was marked running.",
                )
                return

        script_path = Path(job.script_path)
        if control.stop_now or not script_path.exists():
            self._request_stop_locked(job, reason="stop_now" if control.stop_now else "running script removed")
            control.stop_now = False
            self._write_control_locked(control)

        return_code = process.poll()
        if return_code is None:
            if self._terminate_requested_at is not None:
                elapsed = time.time() - self._terminate_requested_at
                if elapsed >= self.terminate_grace_s:
                    self._kill_current_process_locked(force=True)
            return

        recorded_exit = self._read_recorded_exit_code_locked(job)
        if recorded_exit is not None:
            return_code = recorded_exit

        status = STATUS_FINISHED if return_code == 0 else STATUS_FAILED
        if self._terminate_requested_at is not None and return_code != 0:
            status = STATUS_CANCELLED
        self._finish_job_locked(job, status=status, exit_code=return_code)

    def _recover_process_handle_locked(self, job: JobRecord) -> subprocess.Popen[str] | None:
        if not job.pid:
            return None
        handle = RecoveredProcessHandle(pid=job.pid)
        if handle.poll() is not None:
            return None
        self._current_process = handle
        return handle

    def _request_stop_locked(self, job: JobRecord, reason: str) -> None:
        if self._terminate_requested_at is not None:
            return
        self._terminate_requested_at = time.time()
        job.note = reason
        self._write_job_locked(job)
        self._kill_current_process_locked(force=False)

    def _kill_current_process_locked(self, force: bool) -> None:
        process = self._current_process
        if not process:
            return
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[experiment-queue] failed to signal process: {exc}", file=os.sys.stderr)

    def enqueue_script(
        self,
        source_path: str,
        *,
        args: list[str] | None = None,
        name: str | None = None,
        cwd: str | None = None,
        conda_env: str | None = None,
        owner_label: str | None = None,
        owner_session_id: str | None = None,
    ) -> JobRecord:
        src = Path(source_path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Script not found: {src}")
        if not self._is_under_allowed_root(src):
            raise PermissionError(f"Script path is outside allowed roots: {src}")

        with self._lock:
            job_id = self._new_job_id()
            job_name = name or src.stem
            dest_name = self._canonical_script_name(job_id, src.name)
            dest = self.queued_dir / dest_name
            shutil.copy2(src, dest)
            dest.chmod(dest.stat().st_mode | 0o700)

            record = JobRecord(
                job_id=job_id,
                name=job_name,
                status=STATUS_QUEUED,
                script_path=str(dest),
                submitted_at=time.time(),
                owner_label=owner_label,
                owner_session_id=owner_session_id,
                args=list(args or []),
                cwd=str(Path(cwd).expanduser().resolve()) if cwd else str(self.default_cwd),
                conda_env=conda_env or self.default_conda_env,
                source_path=str(src),
                stdout_log=str(self.logs_dir / f"{job_id}.out"),
                stderr_log=str(self.logs_dir / f"{job_id}.err"),
                exit_code_path=str(self.results_dir / f"{job_id}.exitcode"),
            )
            self._write_job_locked(record)
            return record

    def daemon_status(self) -> dict[str, Any]:
        with self._lock:
            return self._daemon_status_locked()

    def ensure_daemon_running(
        self,
        daemon_command: list[str],
        *,
        startup_wait_s: float = 3.0,
    ) -> dict[str, Any]:
        with self._lock:
            status = self._daemon_status_locked()
            if status["daemon_running"]:
                status["started_now"] = False
                return status

            self.daemon_log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = self.daemon_log_path.open("ab")
            try:
                subprocess.Popen(
                    daemon_command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=log_handle,
                    cwd=str(self.workspace_root),
                    close_fds=True,
                    start_new_session=True,
                )
            finally:
                log_handle.close()

        deadline = time.time() + startup_wait_s
        last_status: dict[str, Any] | None = None
        while time.time() < deadline:
            last_status = self.daemon_status()
            if last_status["daemon_running"]:
                last_status["started_now"] = True
                return last_status
            time.sleep(0.1)

        status = last_status or self.daemon_status()
        status["started_now"] = False
        raise RuntimeError(f"Experiment queue daemon did not start for {self.queue_root}")

    def list_jobs(self, status: str | None = None, limit: int | None = None) -> list[JobRecord]:
        with self._lock:
            self._reconcile_jobs_locked()
            jobs = [self._read_job_file(path) for path in sorted(self.jobs_dir.glob("*.json"))]
            jobs.sort(key=lambda item: item.submitted_at, reverse=True)
            if status:
                jobs = [job for job in jobs if job.status == status]
            if limit is not None:
                jobs = jobs[:limit]
            return jobs

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            path = self.jobs_dir / f"{job_id}.json"
            if not path.exists():
                raise KeyError(f"Unknown job: {job_id}")
            return self._read_job_file(path)

    def read_log(self, job_id: str, stream: str = "stdout", lines: int = 100) -> str:
        job = self.get_job(job_id)
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be 'stdout' or 'stderr'")
        log_path = Path(job.stdout_log if stream == "stdout" else job.stderr_log or "")
        if not log_path.exists():
            return ""
        content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])

    def pause_queue(self) -> ControlState:
        with self._lock:
            control = self._read_control_locked()
            control.paused = True
            self._write_control_locked(control)
            return control

    def resume_queue(self) -> ControlState:
        with self._lock:
            control = self._read_control_locked()
            control.paused = False
            control.stop_after_current = False
            self._write_control_locked(control)
            return control

    def stop_after_current(self) -> ControlState:
        with self._lock:
            control = self._read_control_locked()
            control.stop_after_current = True
            self._write_control_locked(control)
            return control

    def stop_now(self) -> ControlState:
        with self._lock:
            control = self._read_control_locked()
            control.stop_now = True
            self._write_control_locked(control)
            return control

    def cancel_job(
        self,
        job_id: str,
        *,
        requester_label: str | None = None,
        requester_session_id: str | None = None,
        force: bool = False,
    ) -> JobRecord:
        with self._lock:
            job = self.get_job(job_id)
            self._assert_job_control_allowed(
                job,
                requester_label=requester_label,
                requester_session_id=requester_session_id,
                force=force,
            )
            if job.status == STATUS_QUEUED:
                script = Path(job.script_path)
                dest = self.cancelled_dir / script.name
                if script.exists():
                    shutil.move(str(script), dest)
                job.script_path = str(dest)
                job.status = STATUS_CANCELLED
                job.finished_at = time.time()
                job.note = "Cancelled while queued."
                self._write_job_locked(job)
                return job
            if job.status == STATUS_RUNNING:
                control = self._read_control_locked()
                control.stop_now = True
                self._write_control_locked(control)
                return job
            return job

    def assert_current_job_control_allowed(
        self,
        *,
        requester_label: str | None = None,
        requester_session_id: str | None = None,
        force: bool = False,
    ) -> JobRecord | None:
        with self._lock:
            job = self._get_running_job_locked()
            if job is None:
                return None
            self._assert_job_control_allowed(
                job,
                requester_label=requester_label,
                requester_session_id=requester_session_id,
                force=force,
            )
            return job

    def queue_status(self) -> dict[str, Any]:
        with self._lock:
            self._reconcile_jobs_locked()
            control = self._read_control_locked()
            jobs = self.list_jobs(limit=None)
            counts: dict[str, int] = {}
            for job in jobs:
                counts[job.status] = counts.get(job.status, 0) + 1
            current = self._get_running_job_locked()
            return {
                "queue_root": str(self.queue_root),
                "workspace_root": str(self.workspace_root),
                **self._daemon_status_locked(),
                "paused": control.paused,
                "stop_after_current": control.stop_after_current,
                "stop_now": control.stop_now,
                "counts": counts,
                "current_job": asdict(current) if current else None,
            }

    def has_pending_work(self) -> bool:
        with self._lock:
            self._reconcile_jobs_locked()
            current = self._get_running_job_locked()
            if current is not None:
                return True
            queued_jobs = [job for job in self.list_jobs_from_disk_locked() if job.status == STATUS_QUEUED]
            return bool(queued_jobs)

    def _dequeue_next_job_locked(self) -> JobRecord | None:
        queued_jobs = [job for job in self.list_jobs(limit=None) if job.status == STATUS_QUEUED]
        if not queued_jobs:
            return None
        queued_jobs.sort(
            key=lambda item: (
                Path(item.script_path).stat().st_mtime if Path(item.script_path).exists() else item.submitted_at,
                item.submitted_at,
            )
        )
        return queued_jobs[0]

    def _start_job_locked(self, job: JobRecord) -> None:
        src = Path(job.script_path)
        dest = self.running_dir / src.name
        if src != dest:
            shutil.move(str(src), dest)
        job.script_path = str(dest)
        job.status = STATUS_RUNNING
        job.started_at = time.time()
        job.finished_at = None
        job.exit_code = None
        job.note = None

        stdout_path = Path(job.stdout_log or self.logs_dir / f"{job.job_id}.out")
        stderr_path = Path(job.stderr_log or self.logs_dir / f"{job.job_id}.err")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        job.stdout_log = str(stdout_path)
        job.stderr_log = str(stderr_path)
        job.exit_code_path = job.exit_code_path or str(self.results_dir / f"{job.job_id}.exitcode")
        exit_code_path = Path(job.exit_code_path)
        if exit_code_path.exists():
            exit_code_path.unlink()

        command = self._build_job_command(job)
        stdout_handle = open(stdout_path, "ab")
        stderr_handle = open(stderr_path, "ab")
        process = subprocess.Popen(
            command,
            cwd=job.cwd or str(self.default_cwd),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=False,
            preexec_fn=os.setsid,
        )
        stdout_handle.close()
        stderr_handle.close()
        job.pid = process.pid
        job.pgid = os.getpgid(process.pid)
        self._current_process = process
        self._terminate_requested_at = None
        self._write_job_locked(job)

    def _build_job_command(self, job: JobRecord) -> list[str]:
        script = Path(job.script_path)
        args = " ".join(self._shell_quote(arg) for arg in job.args)
        script_text = self._shell_quote(str(script))
        exit_code_path = self._shell_quote(str(Path(job.exit_code_path or (self.results_dir / f"{job.job_id}.exitcode"))))
        run_script = f"bash {script_text}{(' ' + args) if args else ''}"
        run_with_exit_capture = (
            f"{run_script}; "
            f"code=$?; "
            f"printf '%s\\n' \"$code\" > {exit_code_path}; "
            f"exit \"$code\""
        )
        if job.conda_env:
            if not self.conda_sh_path:
                raise RuntimeError("conda_env requested but no conda.sh path configured")
            setup = (
                f"source {self._shell_quote(str(self.conda_sh_path))} && "
                f"conda activate {self._shell_quote(job.conda_env)} && "
            )
            return [self.shell_path, "-lc", setup + run_with_exit_capture]
        return [self.shell_path, "-lc", run_with_exit_capture]

    def _finish_job_locked(
        self,
        job: JobRecord,
        *,
        status: str,
        exit_code: int | None,
        note: str | None = None,
    ) -> None:
        src = Path(job.script_path)
        if status == STATUS_FINISHED:
            target_dir = self.finished_dir
        elif status == STATUS_CANCELLED:
            target_dir = self.cancelled_dir
        else:
            target_dir = self.failed_dir
        dest = target_dir / src.name
        if src.exists() and src != dest:
            shutil.move(str(src), dest)
        job.script_path = str(dest)
        job.status = status
        job.exit_code = exit_code
        job.finished_at = time.time()
        if note:
            job.note = note
        self._write_job_locked(job)
        self._current_process = None
        self._terminate_requested_at = None

    def _reconcile_jobs_locked(self) -> None:
        known_jobs = self._dedupe_jobs_locked()
        known_paths = {Path(job.script_path): job for job in known_jobs}
        for directory, status in (
            (self.queued_dir, STATUS_QUEUED),
            (self.running_dir, STATUS_RUNNING),
            (self.finished_dir, STATUS_FINISHED),
            (self.failed_dir, STATUS_FAILED),
            (self.cancelled_dir, STATUS_CANCELLED),
        ):
            for script in sorted(directory.iterdir()):
                if not script.is_file():
                    continue
                if script in known_paths:
                    job = known_paths[script]
                    if job.status != status:
                        job.status = status
                        self._write_job_locked(job)
                    continue
                record = JobRecord(
                    job_id=self._new_job_id(),
                    name=script.stem,
                    status=status,
                    script_path=str(script),
                    submitted_at=script.stat().st_mtime,
                    manual_drop=True,
                    cwd=str(self.default_cwd),
                    conda_env=self.default_conda_env,
                    stdout_log=str(self.logs_dir / f"{script.stem}.out"),
                    stderr_log=str(self.logs_dir / f"{script.stem}.err"),
                    exit_code_path=str(self.results_dir / f"{script.stem}.exitcode"),
                    note="Imported from filesystem queue.",
                )
                if status in FINAL_STATUSES:
                    record.finished_at = script.stat().st_mtime
                self._write_job_locked(record)

    def _dedupe_jobs_locked(self) -> list[JobRecord]:
        jobs = self.list_jobs_from_disk_locked()
        by_script_path: dict[str, list[JobRecord]] = {}
        for job in jobs:
            by_script_path.setdefault(job.script_path, []).append(job)

        survivors: list[JobRecord] = []
        for path, grouped_jobs in by_script_path.items():
            if len(grouped_jobs) == 1:
                survivors.append(grouped_jobs[0])
                continue

            grouped_jobs.sort(
                key=lambda job: (
                    0 if not job.manual_drop else 1,
                    0 if job.status == STATUS_RUNNING else 1,
                    -(job.started_at or job.submitted_at),
                    job.job_id,
                )
            )
            winner = grouped_jobs[0]
            survivors.append(winner)
            for duplicate in grouped_jobs[1:]:
                duplicate_path = self.jobs_dir / f"{duplicate.job_id}.json"
                if duplicate_path.exists():
                    duplicate_path.unlink()
        return survivors

    def list_jobs_from_disk_locked(self) -> list[JobRecord]:
        jobs = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            jobs.append(self._read_job_file(path))
        return jobs

    def _acquire_worker_lock_locked(self) -> None:
        if self._worker_lock_handle is not None:
            return
        handle = self.worker_lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(f"Another experiment queue worker is already active for {self.queue_root}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._worker_lock_handle = handle

    def _release_worker_lock_locked(self) -> None:
        handle = self._worker_lock_handle
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._worker_lock_handle = None

    def _daemon_status_locked(self) -> dict[str, Any]:
        pid = self._read_worker_pid_locked()
        running = pid is not None and self._pid_is_running(pid)
        return {
            "daemon_running": running,
            "daemon_pid": pid if running else None,
            "daemon_log_path": str(self.daemon_log_path),
            "worker_lock_path": str(self.worker_lock_path),
        }

    def _read_worker_pid_locked(self) -> int | None:
        if not self.worker_lock_path.exists():
            return None
        raw = self.worker_lock_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _pid_is_running(self, pid: int) -> bool:
        proc_path = Path("/proc") / str(pid)
        return proc_path.exists()

    def _assert_job_control_allowed(
        self,
        job: JobRecord,
        *,
        requester_label: str | None,
        requester_session_id: str | None,
        force: bool,
    ) -> None:
        if force:
            return
        if not job.owner_label and not job.owner_session_id:
            return
        if job.owner_session_id and requester_session_id and job.owner_session_id == requester_session_id:
            return
        if job.owner_label and requester_label and job.owner_label == requester_label:
            return
        owner_bits = []
        if job.owner_label:
            owner_bits.append(f"owner_label={job.owner_label}")
        if job.owner_session_id:
            owner_bits.append(f"owner_session_id={job.owner_session_id}")
        if not owner_bits:
            return
        requester_bits = []
        if requester_label:
            requester_bits.append(f"requester_label={requester_label}")
        if requester_session_id:
            requester_bits.append(f"requester_session_id={requester_session_id}")
        requester_text = ", ".join(requester_bits) if requester_bits else "requester unspecified"
        owner_text = ", ".join(owner_bits)
        raise PermissionError(
            f"Job control denied for {job.job_id}: {owner_text}; {requester_text}. "
            "Pass force=true to override."
        )

    def _get_running_job_locked(self) -> JobRecord | None:
        running_jobs = [job for job in self.list_jobs_from_disk_locked() if job.status == STATUS_RUNNING]
        running_jobs.sort(key=lambda item: item.started_at or item.submitted_at, reverse=True)
        return running_jobs[0] if running_jobs else None

    def _read_control_locked(self) -> ControlState:
        data = self._read_json(self.control_path)
        return ControlState(**data)

    def _write_control_locked(self, control: ControlState) -> None:
        self._write_json(self.control_path, asdict(control))

    def _write_job_locked(self, record: JobRecord) -> None:
        self._write_json(self.jobs_dir / f"{record.job_id}.json", asdict(record))

    def _read_job_file(self, path: Path) -> JobRecord:
        return JobRecord(**self._read_json(path))

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _new_job_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]

    def _read_recorded_exit_code_locked(self, job: JobRecord) -> int | None:
        if not job.exit_code_path:
            return None
        path = Path(job.exit_code_path)
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return int(raw)

    def _should_wait_for_exit_sidecar_locked(self, job: JobRecord) -> bool:
        started_at = job.started_at or time.time()
        elapsed = time.time() - started_at
        settle_window_s = max(2.0, self.poll_interval_s * 3.0)
        return elapsed <= settle_window_s

    def _canonical_script_name(self, job_id: str, original_name: str) -> str:
        suffix = "".join(Path(original_name).suffixes) or ".sh"
        stem = Path(original_name).stem
        safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in stem)
        safe_stem = safe_stem.strip("-_") or "job"
        return f"{job_id}__{safe_stem}{suffix}"

    def _is_under_allowed_root(self, path: Path) -> bool:
        allowed_roots = [self.workspace_root, *self.script_roots]
        for root in allowed_roots:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _shell_quote(self, value: str) -> str:
        return "'" + value.replace("'", "'\"'\"'") + "'"

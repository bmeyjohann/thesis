# Experiment Queue MCP

`tools/experiment_queue_mcp/server.py` is a FastMCP stdio control server for a persistent on-disk bash-script queue. It auto-starts a separate background daemon that actually consumes queued jobs.

## Queue layout

The default queue root is [`experiment_queue/`](/home/benjamin/thesis/experiment_queue) and contains:

- `queued/`
- `running/`
- `finished/`
- `failed/`
- `cancelled/`
- `logs/`
- `state/`

The server imports manually dropped scripts from `queued/`, but the preferred interface is the MCP `enqueue_script` tool.

## Start manually

Install the MCP SDK on the Python interpreter used by the server if it is not present already:

```bash
python3 -m pip install --user --break-system-packages "mcp[cli]>=1.17.0"
```

Then start the control server:

```bash
scripts/run_experiment_queue_mcp.sh
```

For debugging only, you can also start the daemon directly:

```bash
scripts/run_experiment_queue_daemon.sh
```

## Codex MCP config

Add this to `~/.codex/config.toml`:

```toml
[mcp_servers.experiment_queue]
command = "/usr/bin/python3"
args = [
  "/home/benjamin/thesis/tools/experiment_queue_mcp/server.py",
  "--queue-root", "/home/benjamin/thesis/experiment_queue",
  "--workspace-root", "/home/benjamin/thesis",
  "--default-cwd", "/home/benjamin/thesis",
  "--script-root", "/home/benjamin/thesis/scripts",
  "--script-root", "/home/benjamin/thesis",
  "--default-conda-env", "fasttd3",
  "--conda-sh-path", "/home/benjamin/miniconda3/etc/profile.d/conda.sh",
]
```

Do not rely on a repo-local `.codex/config.toml` entry like `bash scripts/run_experiment_queue_mcp.sh` for this server. Codex Desktop currently has an upstream bug where local MCP servers are not always launched in the workspace cwd, so relative launcher paths can fail during startup and surface as an MCP handshake error.

## Tool surface

- `prime_queue_session`
- `enqueue_script`
- `queue_status`
- `queue_debug_status`
- `list_jobs`
- `list_jobs_debug`
- `get_job`
- `get_job_debug`
- `read_job_log`
- `pause_queue`
- `resume_queue`
- `stop_after_current`
- `stop_now`
- `cancel_job`

## Warmup Tool

- `prime_queue_session` is a preflight helper for autonomous runs.
- It can:
  - report queue state and recent jobs
  - exercise reversible control operations (`pause_queue`, `resume_queue`, `stop_after_current`)
  - optionally enqueue and immediately cancel a temporary no-op job to verify the enqueue/cancel path
- It does **not** pre-authorize future Codex shell approvals outside this MCP server. Approval mode is still controlled by the surrounding Codex session.

## Execution model

- Jobs are bash scripts copied into the queue.
- The daemon runs exactly one job at a time.
- The MCP server is control-only. It does not own the long-lived worker loop.
- Mutating queue tools (`enqueue_script`, `resume_queue`, `stop_after_current`, `stop_now`, `cancel_job`, and the optional probe path in `prime_queue_session`) check whether the daemon is running and start it if needed.
- Read-only queue tools (`queue_status`, `list_jobs`, `get_job`, `read_job_log`, `pause_queue`) inspect persisted queue state without waking the daemon.
- The default monitoring tools (`queue_status`, `list_jobs`, `get_job`) intentionally return compact summaries without absolute paths so unattended autonomous sessions are less likely to trip client-side approval heuristics.
- The explicit debug tools (`queue_debug_status`, `list_jobs_debug`, `get_job_debug`) expose full path-rich metadata and are intended for manual diagnosis when a human is present.
- The daemon exits automatically once the queue is empty and no job is active for a short idle grace period.
- Multiple MCP sessions can attach to the same queue root because they all talk to the same daemon-backed queue state.
- Running jobs can survive an MCP control-server restart because the daemon is independent of MCP client lifetime.
- Daemon liveness is inferred from the actual queue worker file lock, not only from a stored PID, so cross-namespace PID mismatches do not prevent auto-start from waking the worker.
- Removing the current script from `running/` requests cancellation.
- `stop_now` sends `SIGTERM` to the job process group and escalates to `SIGKILL` after the configured grace period.
- `stop_after_current` lets the active job finish, then stops dequeuing new jobs until resumed.

## Ownership metadata

- `enqueue_script` accepts optional `owner_label` and `owner_session_id` fields.
- `cancel_job`, `stop_now`, and `stop_after_current` accept optional `requester_label`, `requester_session_id`, and `force`.
- If a job has ownership metadata, matching owners may mutate it normally.
- A different session can still inspect the job, but it must pass `force=true` to cancel or stop it.
- Jobs without ownership metadata remain mutable by any session, which is useful for ad hoc manual runs.

## Approval guidance

- Codex can auto-approve custom MCP tools via config, but app-side approval behavior is still heuristic and can be sensitive to path-heavy payloads.
- If you want unattended autonomous work, prefer the summary tools plus `read_job_log` and `wandb` for normal monitoring.
- Treat the debug tools as manual-only. They are the ones most likely to trigger an extra approval because they expose absolute paths and richer execution metadata.
- Put any per-tool MCP approval overrides in the active user config that your Codex app actually loads, not only in a repo-local config, if you need the app to honor them reliably.

## Trust model

This server is intentionally an escape hatch for autonomous experiments. Anything placed into the queue is executed outside Codex's normal shell sandbox. Keep the queue root and allowed source roots scoped to scripts you are willing to run unsandboxed.

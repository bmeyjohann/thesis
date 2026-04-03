# Codex Multi-Machine Setup

This repo carries the shared Codex workflow pieces that are safe to commit:

- portable project config in [`.codex/config.toml`](/home/benjamin/thesis/.codex/config.toml)
- repo-local autonomous research skill in [`.agents/skills/autonomous-research/SKILL.md`](/home/benjamin/thesis/.agents/skills/autonomous-research/SKILL.md)
- repo-owned MCP wrapper scripts in [`scripts/run_wandb_mcp.sh`](/home/benjamin/thesis/scripts/run_wandb_mcp.sh), [`scripts/run_experiment_queue_mcp.sh`](/home/benjamin/thesis/scripts/run_experiment_queue_mcp.sh), and [`scripts/run_zotero_mcp.sh`](/home/benjamin/thesis/scripts/run_zotero_mcp.sh)

Important:

- local stdio MCP launchers do **not** live in the repo config anymore
- define `wandb`, `experiment_queue`, and `zotero` in `~/.codex/config.toml` on each machine with absolute paths
- this avoids a Codex Desktop startup bug where relative `bash scripts/...` MCP commands can fail if the app does not launch them from the repo cwd

## What Stays Machine-Local

Do **not** commit these:

- `WANDB_API_KEY`
- Codex auth/session history under `~/.codex`
- machine-specific path overrides
- queue runtime state under `experiment_queue/`

Recommended machine-local setup:

- keep secrets in `~/.codex/config.toml` or normal shell environment variables
- keep absolute MCP command paths in `~/.codex/config.toml`
- if the clone path, conda path, or `uvx` path differ on another machine, update those absolute paths there instead of changing the repo config

## Mirror These Local Codex Settings

Mirror the following into `~/.codex/config.toml` on the other machine. Replace paths if the clone or local tools live somewhere else. Do **not** add API keys to the repo copy of this document.

```toml
sandbox_mode = "workspace-write"

[mcp_servers.wandb]
command = "/home/benjamin/.local/bin/uvx"
args = ["--from", "git+https://github.com/wandb/wandb-mcp-server", "wandb_mcp_server"]
startup_timeout_sec = 30.0

[mcp_servers.wandb.tools.query_wandb_tool]
approval_mode = "approve"

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
  "--poll-interval", "1.0",
  "--terminate-grace", "10.0",
]
startup_timeout_sec = 30.0

[mcp_servers.experiment_queue.tools.prime_queue_session]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.daemon_status]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.queue_status]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.list_jobs]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.enqueue_script]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.get_job]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.read_job_log]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.pause_queue]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.shutdown_daemon]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.restart_daemon]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.stop_after_current]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.stop_now]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.cancel_job]
approval_mode = "auto"

[mcp_servers.experiment_queue.tools.resume_queue]
approval_mode = "auto"

[mcp_servers.linear]
url = "https://mcp.linear.app/mcp"

[mcp_servers.zotero]
command = "powershell.exe"
args = ["-NoProfile", "-Command", "uvx --from git+https://github.com/54yyyu/zotero-mcp.git zotero-mcp"]

[mcp_servers.zotero.env]
ZOTERO_LOCAL = "true"

[sandbox_workspace_write]
network_access = true
exclude_slash_tmp = false
writable_roots = [
  "/dev/dxg",
  "/dev/shm",
  "/home/benjamin/.nv",
  "/home/benjamin/.triton",
  "/home/benjamin/.cache/torch",
  "/home/benjamin/.cache/torch_extensions",
  "/home/benjamin/.cache/triton",
]
```

Notes on adapting that snippet on another machine:

- if the repo is cloned somewhere other than `/home/benjamin/thesis`, change every thesis path consistently
- if `uvx` is not at `/home/benjamin/.local/bin/uvx`, point `mcp_servers.wandb.command` at the real binary
- if conda lives elsewhere, update `--conda-sh-path`
- if the GPU device/cache layout differs, adjust `writable_roots` accordingly
- keep `WANDB_API_KEY` out of this shared file; put it in the machine-local `~/.codex/config.toml` or shell env

## Other Computer Bootstrap

### Assumptions

The target machine should have:

- the repo cloned
- Codex Desktop installed
- CUDA working if this machine should run GPU experiments
- the `fasttd3` conda env available
- `uvx` available, either on PATH or via `UVX_BIN`
- `powershell.exe` callable from WSL if you want the same Zotero setup

If the queue should run training jobs, also make sure:

- the repo is opened from the actual clone path you want Codex to use

### One-Time Machine Setup

On the other machine, inside the cloned repo:

```bash
git pull \
&& command -v uvx
```

If `uvx` is not on PATH, either install it or export an explicit path:

```bash
export UVX_BIN="/full/path/to/uvx"
```

If the machine uses a different conda location than the current laptop, update the `--conda-sh-path` value in `~/.codex/config.toml` accordingly.

If the queue should use a different conda env name, update `--default-conda-env` in `~/.codex/config.toml`.

For WANDB, use either of these machine-local options:

1. Keep `WANDB_API_KEY` in `~/.codex/config.toml`
2. Or export it in the shell before starting Codex:

```bash
export WANDB_API_KEY="..."
```

### Sanity Checks Before Opening Codex

Verify the absolute local MCP commands can start.

```bash
/home/benjamin/.local/bin/uvx \
  --from git+https://github.com/wandb/wandb-mcp-server \
  wandb_mcp_server --help || true
```

```bash
/usr/bin/python3 \
  /home/benjamin/thesis/tools/experiment_queue_mcp/server.py \
  --queue-root /home/benjamin/thesis/experiment_queue \
  --workspace-root /home/benjamin/thesis \
  --default-cwd /home/benjamin/thesis \
  --script-root /home/benjamin/thesis/scripts \
  --script-root /home/benjamin/thesis \
  --default-conda-env fasttd3 \
  --conda-sh-path /home/benjamin/miniconda3/etc/profile.d/conda.sh
```

The queue server should start and wait on stdio. Stop it with `Ctrl+C`.

If you want to verify the queue daemon path directly:

```bash
bash scripts/run_experiment_queue_daemon.sh
```

It should start, notice no work, and exit after the idle timeout.

### Start Codex

Open Codex Desktop in the repo root so the project config and repo skill are loaded, but rely on `~/.codex/config.toml` for the local stdio MCP wiring.

Then start a fresh session and verify:

- `wandb` MCP is available
- `experiment_queue` MCP is available
- `daemon_status`, `shutdown_daemon`, and `restart_daemon` are visible on `experiment_queue`
- `linear` MCP is available
- `zotero` MCP is available
- the repo-local `autonomous-research` skill is visible

### Prompt To Hand To Codex On The Other Machine

If you want the other Codex instance to do the verification itself, give it a prompt like:

```text
Verify the shared Codex setup for this repo. Check that the wandb, experiment_queue, linear, and zotero MCPs are available, that the autonomous-research skill is visible, and that the experiment queue can do a tiny enqueue/read/cancel smoke test without running a real long experiment. If anything is misconfigured, fix only the machine-local setup and explain what you changed.
```

## Chat Sessions

Do not try to sync Codex chat/session history through Git.

Reasons:

- session state is local app state, not repo state
- it can contain machine-specific paths, approvals, cached MCP/session details, and other brittle local context
- concurrent edits from two machines would be awkward and conflict-prone

Recommended approach:

- start clean sessions on the other machine
- rely on shared repo files, WANDB, and tracked handoff notes for continuity
- if needed, keep a repo-tracked handoff markdown file that summarizes:
  - current research question
  - best current hypothesis
  - latest useful runs
  - next planned experiments

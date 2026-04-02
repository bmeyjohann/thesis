# Codex Multi-Machine Setup

This repo now carries the shared Codex wiring that is safe to commit:

- project MCP config in [`.codex/config.toml`](/home/benjamin/thesis/.codex/config.toml)
- repo-local autonomous research skill in [`.agents/skills/autonomous-research/SKILL.md`](/home/benjamin/thesis/.agents/skills/autonomous-research/SKILL.md)
- repo-owned MCP wrapper scripts in [`scripts/run_wandb_mcp.sh`](/home/benjamin/thesis/scripts/run_wandb_mcp.sh), [`scripts/run_experiment_queue_mcp.sh`](/home/benjamin/thesis/scripts/run_experiment_queue_mcp.sh), and [`scripts/run_zotero_mcp.sh`](/home/benjamin/thesis/scripts/run_zotero_mcp.sh)

## What Stays Machine-Local

Do **not** commit these:

- `WANDB_API_KEY`
- Codex auth/session history under `~/.codex`
- machine-specific path overrides
- queue runtime state under `experiment_queue/`

Recommended machine-local setup:

- keep secrets in `~/.codex/config.toml` or normal shell environment variables
- keep machine-specific overrides in environment variables, for example:
  - `EXPERIMENT_QUEUE_CONDA_SH`
  - `EXPERIMENT_QUEUE_CONDA_ENV`
  - `UVX_BIN`

## Other Computer Bootstrap

### Assumptions

The target machine should have:

- the repo cloned
- Codex Desktop installed
- CUDA working if this machine should run GPU experiments
- the `fasttd3` conda env available
- `uvx` available, either on PATH or via `UVX_BIN`

If the queue should run training jobs, also make sure:

- `EXPERIMENT_QUEUE_CONDA_SH` points at the local `conda.sh` if it differs from the default
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

If the machine uses a different conda location than the current laptop:

```bash
export EXPERIMENT_QUEUE_CONDA_SH="/full/path/to/miniconda3/etc/profile.d/conda.sh"
```

If the queue should use a different conda env name:

```bash
export EXPERIMENT_QUEUE_CONDA_ENV="fasttd3"
```

For WANDB, use either of these machine-local options:

1. Keep `WANDB_API_KEY` in `~/.codex/config.toml`
2. Or export it in the shell before starting Codex:

```bash
export WANDB_API_KEY="..."
```

### Sanity Checks Before Opening Codex

Verify the local wrappers can start:

```bash
bash scripts/run_wandb_mcp.sh --help || true
```

```bash
bash scripts/run_experiment_queue_mcp.sh
```

The queue server should start and wait on stdio. Stop it with `Ctrl+C`.

If you want to verify the queue daemon path directly:

```bash
bash scripts/run_experiment_queue_daemon.sh
```

It should start, notice no work, and exit after the idle timeout.

### Start Codex

Open Codex Desktop in the repo root so the project config in [`.codex/config.toml`](/home/benjamin/thesis/.codex/config.toml) is loaded.

Then start a fresh session and verify:

- `wandb` MCP is available
- `experiment_queue` MCP is available
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

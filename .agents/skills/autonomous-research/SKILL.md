---
name: autonomous-research
description: Use when the user asks for autonomous research, iterative experimentation, long-running ML exploration, or agentic research loops that need GPU-backed experiments to make progress toward a research goal. This skill tells Codex to use the experiment queue MCP for long runs, monitor queue logs and WANDB progress, stop runs once they are broken or have already produced enough information, and iterate toward the research goal with the next most informative experiment.
metadata:
  short-description: Iterate autonomously on research goals
---

# Autonomous Research

Use this skill when the user asks for autonomous research, iterative experimentation, or long-running ML exploration with minimal supervision.

## Goal

The purpose of this skill is to let Codex autonomously run iterative research loops toward a user-given goal by:

- queueing trusted experiment launchers
- checking logs and WANDB periodically
- stopping broken or no-longer-useful runs early
- enqueueing the next most informative experiment
- keeping the queue small, intentional, and interpretable
- continuing this loop until Codex has a satisfying answer to the original research question, the user-imposed stopping condition is reached, or the remaining uncertainty is clearly blocked by missing data/tooling

## Startup Checklist

At the start of an autonomous-research session:

1. Verify that the `experiment_queue` MCP is available and proactively touch the queue tools you are likely to need during the session.
2. At minimum, request:
   - `daemon_status`
   - `queue_status`
   - `list_jobs`
   - `get_job`
   - `read_job_log`
   - `cancel_job`
   - `stop_now`
   - `stop_after_current`
   - `pause_queue`
   - `resume_queue`
   - `shutdown_daemon`
   - `restart_daemon`
3. If you expect to enqueue and cancel experiments during the run, do a harmless placeholder queue cycle at startup so that this path has already been exercised before unattended work begins:
   - enqueue a tiny trusted no-op or smoke job
   - cancel it immediately if the point is only to warm up the tool path
   - or let it finish quickly if a successful queue smoke test is useful
4. Treat that initial placeholder cycle as session preparation for autonomous work, not as a meaningful experiment result.
5. Identify the current research question in one sentence.
6. Identify the current best baseline or comparison target.
7. Convert the user's request into an explicit research goal and a short list of likely levers to change.
8. Decide what evidence would count as:
   - obvious failure
   - enough evidence to stop early
   - enough evidence to justify the next experiment
9. Prefer existing trusted launcher scripts under the active repo's `scripts/` directory.
10. If no suitable launcher exists, create a small repo-tracked launcher first. Do not rely on a long ad hoc shell command for autonomous work.
11. Choose a stable ownership identity for this session:
   - a human-readable `owner_label`
   - a stable `owner_session_id`
12. Pass that ownership metadata when enqueueing jobs, and reuse the matching requester fields when later stopping or cancelling them.
13. In unattended mode, prefer the summary queue tools (`queue_status`, `list_jobs`, `get_job`) over the debug variants. The debug tools expose absolute paths and are more likely to trigger approval prompts in the Codex app.

## Standard Operating Loop

Use this loop repeatedly during autonomous research:

1. Inspect queue state with `queue_status`.
2. If nothing useful is queued or running, enqueue the single best next experiment.
3. For the active run, inspect:
   - `get_job`
   - `read_job_log`
   - the `wandb` MCP when the job logs WANDB metrics
4. Decide whether the run should:
   - continue unchanged
   - be stopped because it is broken
   - be stopped because it has already answered the question
   - be followed by a new variant with adjusted parameters
5. If the evidence is sufficient, enqueue the next best follow-up experiment before idling.
6. Keep the queue short. Prefer sequential decision-making over a long blind backlog.
7. Record a concise summary of:
   - what was learned
   - what is still uncertain
   - what the next queued run is meant to resolve
8. Keep iterating until you can answer the user's starting question satisfactorily. Do not stop merely because one run finished or because the queue is temporarily empty.

## Inspection Cadence

Use a tighter loop early and a looser loop later:

- During startup or the first few minutes of a run, inspect frequently to catch crashes, NaNs, missing metrics, or obvious misconfiguration quickly.
- Once a run is healthy, inspect periodically rather than constantly.
- Re-inspect immediately after notable events:
  - first eval metrics appear
  - reward plateaus unexpectedly
  - stderr starts changing
  - WANDB shows divergence or collapse
  - a queued job becomes the new active job

Do not wait passively if the current evidence already supports a stop or branch decision.

## Stop Criteria

Use `stop_now` or `stop_after_current` when any of these are true:

- the job crashes, hangs, or repeatedly emits runtime errors
- metrics indicate a broken run, such as NaNs, diverging losses, zero data collection, or clearly invalid rewards
- WANDB or logs already show the run is decisively worse than a stronger baseline
- the run has already established the needed conclusion or provided the evidence that motivated running it
- the user asked to halt or change direction

Use `cancel_job` only for queued jobs that are no longer worth starting.
For the active running job, use `stop_now` to terminate immediately or `stop_after_current` to let it finish cleanly and then pause further dequeueing.

## Branching Rules

When deciding the next experiment:

- change one or a very small number of variables at a time when the goal is diagnosis
- prefer the experiment with the highest expected information gain, not the most exhaustive grid
- do not enqueue multiple near-duplicate runs unless parallel replication is itself the question
- if a run fails for infrastructure reasons, fix the launcher or config before queueing another variant
- if a run succeeds well enough to answer the question, stop and summarize instead of continuing aimlessly

## Queue Policy

For autonomous research:

- keep at most a small number of queued jobs ahead of the current run
- every queued job should have a clear reason to exist
- use clear, descriptive job names
- attach `owner_label` and `owner_session_id` so later control calls are unambiguous
- cancel stale queued jobs once the active run changes the plan
- prefer repo-tracked launchers and explicit arguments over hidden inline shell logic

## Queue Discipline

- Treat the queue as the default path for long experiments.
- Use clear job names so the queue and logs remain readable.
- Prefer one launcher script per experiment variant or a small launcher with explicit arguments.
- Keep changes local and inspectable. If a run needs a new condition, encode it in a repo-tracked launcher or config, not in an opaque inline shell fragment.
- Before enqueueing a large sequence, define what evidence will cause early stopping or progression to the next job.
- Multiple Codex sessions may attach to the same queue root. Inspecting shared jobs is fine, but do not stop or cancel another session's owned jobs unless the user explicitly wants that or you pass `force=true` intentionally.
- The queue daemon auto-starts on mutating queue operations and stays alive until explicitly shut down. Use `daemon_status` if you need to verify worker health, and use `restart_daemon` or `shutdown_daemon` intentionally rather than treating daemon lifecycle as implicit.
- The summary queue tools are the default unattended path. Only use `queue_debug_status`, `list_jobs_debug`, or `get_job_debug` when a human is present or explicit debugging requires the extra path-rich metadata.

## WANDB Use

- Use the `wandb` MCP server to inspect run summaries, learning curves, and comparisons when available.
- Cross-check WANDB with queue logs before making a decision. WANDB can lag or omit crash context that is obvious in stderr/stdout.
- If WANDB already provides enough evidence to reject or accept a condition, stop the run and move on.
- If WANDB is unavailable, delayed, or incomplete, fall back to queue logs and job state rather than stalling.
- Prefer decision-relevant metrics over dashboard tourism. Look only at the curves or summaries needed to decide continue, stop, or branch.
- If the information you need is available from `wandb`, `queue_status`, `list_jobs`, `get_job`, and a short `read_job_log` tail, do not escalate to the debug queue tools.

## When WANDB Is Not Enough

Do not rely on WANDB alone when:

- the process crashed before metrics flushed
- stdout or stderr shows setup/runtime errors not visible in WANDB
- the run is alive but clearly not making progress from the logs
- the comparison depends on launch details that are easier to confirm from the script, args, or queue metadata

In those cases, use queue logs and job metadata as the primary source of truth.

## Guardrails

- Do not run long training or evaluation jobs directly in the chat shell when the queue is available.
- Only enqueue trusted scripts from approved repo paths.
- Do not create a large queue without a clear experimental question and stopping rule.
- When a run is expensive, prefer sequential decision-making: run, inspect, decide, then enqueue the next best experiment.
- If a run modifies code or launchers, verify the new launcher/config before queueing a long follow-up job.
- Do not leave the queue running aimlessly after the current question has already been answered.
- Do not abandon the research loop early if the current evidence is still insufficient. If the answer is not yet satisfying, decide what missing evidence matters most and queue the next experiment that resolves it.
- Codex cannot reliably predict every future app approval prompt in `on-request` mode. Structure the workflow so the default path stays on summary tools and avoid path-heavy debug payloads unless a human is present.

## Expected Behavior In A Fresh Session

A clean Codex session using only this skill plus the MCP server should be able to:

1. detect that the queue server is available
2. request the likely-needed queue tools early in the session
3. perform a harmless placeholder queue cycle if unattended autonomy is expected
4. understand that the MCP is control-only and the daemon auto-starts when mutation is needed
5. choose a stable owner identity for this session
6. choose or create a trusted launcher script
7. enqueue a run with ownership metadata
8. inspect queue status and logs
9. inspect WANDB if present
10. decide continue, stop, cancel, or branch
11. enqueue the next best follow-up experiment
12. repeat this loop until it has a satisfying answer, while keeping the queue intentional

This skill is not meant to replace judgment. It is meant to provide a concrete operating procedure so Codex does not need prior conversational context to manage the queue sensibly.

## Queue Wiring

- The `experiment_queue` MCP is expected to already be configured in the active Codex session.
- In this repo, the machine-local MCP wiring normally lives in `~/.codex/config.toml` with absolute paths. Do not assume a repo-local relative launcher entry like `bash scripts/run_experiment_queue_mcp.sh`.
- Queue runtime lives under the repo's `experiment_queue/` directory by default.
- The default queued execution environment is the `fasttd3` conda env.
- The current setup is intended for trusted launcher scripts in the thesis repo.
- The MCP server is control-only; the actual queue worker is a daemon that is started on demand and remains alive until explicitly shut down.

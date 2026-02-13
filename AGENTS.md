# Repository Guidelines

## General Guidelines

- Whenever I need to tell you to do something in a different way or am dissatisfied with your work, append concise info on what you made wrong and how to do it correctly to AGENTS.md. Also save a short note in codex/[appropriate_naming_for_info].md.
- Save other relevant info that should be preserved in codex/[appropriate_naming_for_info].md
- Always inform me if you save info in a file.
- Subagent permission policy: subagents must not execute commands that require escalated permissions. If escalation is needed (e.g., SSH/cluster operations), subagents should return the required command/context and the main agent executes it with explicit user approval.
- STT clarification: whenever the user says or writes `1B`, interpret it as `WANDB` every time. This mixup is caused by the user's speech-to-text tool.
- Correction noted: for cube state-observation variants, use FastSAC (`train_fast_sac_ogbench.py`) rather than DrQ-v2 scripts.
- Correction noted: for intervention behavior, do not use intervention-probability decay scheduling; keep interventions deterministic by teacher-student deviation/tolerance rules.

# Knowledge Management
- The Linear team corresponding to this workspace is named Thesis, issues have the format THE-XXX
- Use this info when requested to access Linear

## Project Structure & Module Organization
- `train_*.py`: Entry points for training (RSL‑RL, PointMaze variants).
- `eval_interactive.py`: Load a trained model and render episodes.
- `config/`: YAML configs (e.g., `ogbench_config.yaml`).
- `rsl_rl/`, `fasttd3/`, `ogbench/`, `IsaacLab/`: Vendor/submodules used by training.
- `scripts/`: Cluster helpers (copy logs, SSH, allocation).
- `sc_venv_template/`: Virtual gridworld environment setup for HPC (Isaac Sim and Isaac Lab use an apptainer container instead of a virtual environment).
- `logs/`, `models/`, `wandb/`: Outputs, artifacts, offline tracking.

## Build, Test, and Development Commands
- To run/test anything locally, ALWAYS activate the local virtual env first: `conda activate fasttd3`
- See `README.md`
- Starting jobs on the cluster with automatic account detection works like this but is reserved for the human Use this info to create appropriate sbatch files and submit them: `bash submit_job.sh [experiment_name].sbatch`
- Logs: tail `logs/<name>_<JOBID>.log` for artifacts.
- If possible, ALWAYS run local smoke tests before telling me something is working and done.
- Always at least attempt a local smoke test before cluster submission to avoid queue/time waste.
- Assume local testing is CPU-only: validate startup/initialization and short-step execution, but keep smoke tests lightweight (no computationally intensive long training loops).

## Coding Style & Naming Conventions
- Python: PEP 8, 4 spaces, max line length 120.
- Names: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.
- Scripts: executable with shebang; keep flags explicit and self-documenting.

## Testing Guidelines
- Framework: no unit test suite at root; add minimal smoke tests when changing training/eval.
- Quick check: run a short job and verify logs and model saved.

## Commit & Pull Request Guidelines
- Commits: imperative, concise, scoped (e.g., “Adjust SLURM script for booster queue”).
- Include why + scope when touching `run_experiment.sbatch`/`submit_job.sh` and configs.
- PRs: clear description, reproducible command(s), affected configs/paths, and sample logs/metrics (screenshots or snippets).
- Link related issues; note breaking changes and migration steps.

## Security & Configuration Tips
- WANDB runs offline by default as compute node on cluster does not have internet access; set `WANDB_MODE=run` to sync online.
- SLURM account replacement uses `.slurm_account`; review before submitting.
- Keep large artifacts out of Git; rely on `logs/`, `models/`, and external storage.

## Cluster Access Notes
- The cluster project workspace path is `/p/project1/hai_1074/meyjohann1/thesis` and mirrors the local repo structure (`logs/`, `models/`, `wandb/`, `run_*.sbatch`, `train_*.py`).
- For repeated non-interactive cluster access after one MFA login, use SSH connection multiplexing (`ControlMaster`/`ControlPersist`) with a control socket under `~/.ssh/cm/`.
- Prefer `scripts/cluster_readonly.sh` for cluster read operations (logs/wandb inspection). It enforces a read-only command allowlist and should be the default path for non-mutating cluster triage.
- Typical flow:
  - Start master once: `ssh -MNf juwels-booster.fz-juelich.de` (requires key passphrase + TOTP once).
  - Verify reuse: `ssh -O check juwels-booster.fz-juelich.de`.
  - Reuse for commands/log triage without re-prompt while master is alive: `ssh juwels-booster.fz-juelich.de 'cd /p/project1/hai_1074/meyjohann1/thesis && ...'`.
  - Close when done: `ssh -O exit juwels-booster.fz-juelich.de`.
- Cluster submission workflow (non-interactive):
  - Keep local + cluster repos in sync first (use local commit/push, then `git pullall` on cluster repo).
  - Use `bash submit_job.sh -y <script>.sbatch` to accept defaults without interactive prompts.
  - `-y` requires SLURM account auto-detection to work (or `SLURM_ACCOUNT`/`.slurm_account` to be set), otherwise it exits with a clear error.
  - After submit, verify queued/running jobs with `squeue -u "$USER"` or `squeue -j <JOBID>`.
- For log access, prefer direct paths in the cluster project dir, e.g.:
  - `/p/project1/hai_1074/meyjohann1/thesis/logs/drqv2_*.out`
  - `/p/project1/hai_1074/meyjohann1/thesis/logs/<run_name>_<JOBID>.log`

## W&B Debugging Workflow
- Canonical term is `wandb` (voice-to-text may transcribe this as `1B`; interpret `1B` as `wandb`).
- Cluster pipeline expectation:
  - Training jobs write offline runs to local `wandb/` on cluster (compute nodes have no internet).
  - `submit_job.sh` launches background sync on login node (`WANDB_MODE=online wandb sync --sync-all`) to upload those offline runs.
- Standard triage order:
  1. Inspect Slurm and per-run logs first (`logs/*.out`, `logs/*.err`, `logs/<run>_<JOBID>.log`).
  2. Inspect offline `wandb/` folders for the same job IDs (`offline-run-*` presence/timestamps).
  3. Check online W&B project/runs for corresponding uploaded runs.
- Interpretation:
  - No offline `wandb` run folder usually means failure happened before/during wandb initialization.
  - Offline folder exists but no online run usually means post-run sync failure (upload stage), not training-stage logging failure.

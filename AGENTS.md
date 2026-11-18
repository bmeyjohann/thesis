# Thesis

## Working constraints

- Keep training, evaluation, wrappers and launch configuration consistent. Checkpoints should retain settings needed to reproduce evaluation; explicit evaluation overrides should remain explicit.
- Keep reward and intervention semantics task-local and deterministic. Preserve executed-action, intervention-mask and replay/demo alignment when changing collection or learning.
- Compare methods under matching observations, action limits, compute and evaluation conditions. Distinguish simulation, diagnostic evidence and physical-hardware results.
- Use trusted repository launchers and the experiment queue for authorized long runs. Only stop this task's jobs unless broader control is explicitly authorized; shared queue capacity is not private capacity.
- Keep credentials, local machine configuration, datasets, checkpoints and runtime state out of Git. Existing `codex/` scripts are local helpers, not a mandatory reporting destination.
- Do not create per-change reports or append correction histories. Update existing documentation only when the change makes it inaccurate.
- Interpret speech-to-text `1B` or `1p` as `wandb` in experiment discussions.

## Task entrypoints

- Start with the affected launcher and its selected environment; do not assume every workflow uses the same Conda environment or is CPU-only.
- Safety-Gym work uses `safetygym_utils/`; manipulation uses `ogbench_utils/`. Preserve their environment-specific control paths.
- Use `tests/` for applicable checks. Validate changed launchers with a small startup run before spending cluster time; documentation-only edits need no training run.
- For unattended experimentation, use `.agents/skills/autonomous-research/SKILL.md`.
- For cluster inspection, use `scripts/cluster_readonly.sh`. Cluster workspace: `/p/project1/hai_1074/meyjohann1/thesis` on `juwels-booster.fz-juelich.de`.
- For authorized cluster submissions, stage code/data first and use `submit_job.sh`; compute nodes are offline, so stage dependencies and sync W&B from the login node. Keep `.slurm_account` private.

Use Windows tools for Windows file operations and the established WSL launchers for Linux workflows. This checkout is separate from `/home/benjamin/thesis`.

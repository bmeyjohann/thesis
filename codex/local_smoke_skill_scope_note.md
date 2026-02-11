# Local Smoke Skill Scope (2026-02-11)

- Updated `local-smoke-test-policy` scope to trigger only for **pre-submission** actions.
- It now applies when preparing to launch new/changed cluster training runs (`.sbatch`, launch flags, `submit_job.sh`, `sbatch`).
- It now explicitly does **not** apply to cluster diagnostics/post-mortem tasks (SSH log triage, wandb/offline sync checks, failure investigation of already launched jobs).

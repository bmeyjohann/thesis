# Cluster Submit Workflow (Non-Interactive)

Purpose: reliably submit cluster jobs without prompt blocking, then verify queue state.

## Preconditions
- SSH control master is active (`ssh -O check juwels-booster.fz-juelich.de`).
- Local repo changes are committed and pushed.
- Cluster repo is updated via `git pull all` in `/p/project1/hai_1074/meyjohann1/thesis`.

## Submit command
- Use:
  - `bash submit_job.sh -y <run_script>.sbatch`
- Behavior of `-y`:
  - Accepts default answers for submit prompts.
  - Keeps wandb sync enabled with default directory/interval.
  - Requires SLURM account auto-detection (or pre-set `SLURM_ACCOUNT`/`.slurm_account`).

## Verify
- List own jobs:
  - `squeue -u "$USER"`
- Check specific IDs:
  - `squeue -j <JOBID1>,<JOBID2>,...`

## Recommended order for batch starts
1. Pull latest code on cluster (`git pull all`).
2. Submit all target scripts with `submit_job.sh -y`.
3. Immediately verify with `squeue`.
4. Then monitor logs in `logs/` and wandb sync logs (`logs/wandb_sync_<JOBID>.log`).

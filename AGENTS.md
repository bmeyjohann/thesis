# Repository Guidelines

## Project Structure & Module Organization
- `train_*.py`: Entry points for training (RSL‑RL, PointMaze variants).
- `eval_interactive.py`: Load a trained model and render episodes.
- `config/`: YAML configs (e.g., `ogbench_config.yaml`).
- `rsl_rl/`, `fasttd3/`, `ogbench/`: Vendor/submodules used by training.
- `scripts/`: Cluster helpers (copy logs, SSH, allocation).
- `sc_venv_template/`: Virtual environment setup for HPC.
- `logs/`, `models/`, `wandb/`: Outputs, artifacts, offline tracking.

## Build, Test, and Development Commands
- Env: `source sc_venv_template/activate.sh` (sets modules, venv, `PYTHONPATH`).
- Local fallback env: `conda activate fasttd3` (preferred on developer machines).
- Local run: `python train_rsl_rl_integrated.py --env_name pointmaze-medium-v0`.
- Evaluate: `python eval_interactive.py --model_path models/<file>.pt --env_name pointmaze-medium-v0`.
- SLURM: `bash submit_job.sh run_experiment.sbatch` (auto-detects `--account`).
- Logs: tail `logs/<name>_<JOBID>.log` and `logs/rsl_rl/<experiment>/` for artifacts.

## Coding Style & Naming Conventions
- Python: PEP 8, 4 spaces, max line length 120.
- Format/lint: use pre-commit in submodules
  - `cd rsl_rl && pre-commit run -a` (Black, isort, Flake8); same idea for `fasttd3`.
- Names: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.
- Scripts: executable with shebang; keep flags explicit and self-documenting.

## Testing Guidelines
- Framework: no unit test suite at root; add minimal smoke tests when changing training/eval.
- Quick check: run a short job `--total_timesteps 20000 --num_envs 8` and verify logs and model save.
- Consistency: ensure `eval_interactive.py` loads the produced checkpoint without edits.

## Commit & Pull Request Guidelines
- Commits: imperative, concise, scoped (e.g., “Adjust SLURM script for booster queue”).
- Include why + scope when touching `run_experiment.sbatch`/`submit_job.sh` and configs.
- PRs: clear description, reproducible command(s), affected configs/paths, and sample logs/metrics (screenshots or snippets).
- Link related issues; note breaking changes and migration steps.

## Security & Configuration Tips
- WANDB runs offline by default; set `WANDB_MODE=run` to sync online.
- SLURM account replacement uses `.slurm_account`; review before submitting.
- Keep large artifacts out of Git; rely on `logs/`, `models/`, and external storage.

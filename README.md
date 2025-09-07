# OGBench + RSL-RL Experiments

This repository contains reinforcement learning experiments on OGBench environments, using an integrated RSL‑RL PPO training pipeline with HPC-friendly tooling (SLURM, offline WandB).

- Quick start (local): `source sc_venv_template/activate.sh` then `python train_rsl_rl_integrated.py --env_name pointmaze-medium-v0`
- Evaluate: `python eval_interactive.py --model_path models/<file>.pt --env_name pointmaze-medium-v0`
- Submit on SLURM: `bash submit_job.sh run_experiment.sbatch`

See Repository Guidelines for structure, style, and workflows:
- `AGENTS.md`

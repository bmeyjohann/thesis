# Repository for paper "Implicitly Learning Traversability from Minimal/Requested Human Intervention"

## Abstract

Navigating challenging terrains without continuous human oversight is critical for deploying legged robots in real-world scenarios. This thesis aims to develop a method for rapidly and cost-effectively teaching robots to traverse unknown environments by leveraging dense human interventions as off-policy corrections. Initially, a naive navigation policy is trained using only the Euclidean distance to the goal as a reward in an environment with varying terrain types. Human corrections are logged whenever the policy chooses poor or unsafe routes. These corrections are then integrated as off-policy data to shape the policy's implicit understanding of terrain traversability, without relying on explicit maps. Later, the agent should decide by itself when human input is needed, with the goal of minimizing intervention frequency and need for continuous human oversight over time. We will evaluate how effectively the policy learns to avoid risky terrain, how quickly it can reduce the need for human oversight, and how well its uncertainty estimates align with actual hazards. While initial experiments will be conducted in a 2D gridworld, this research ultimately aims to inform future applications in quadrupedal legged robots navigating real, unstructured environments.

## Quick Start for gridworld experiments

- Remote training: `source sc_venv_template/activate.sh` then `python train_rsl_rl_integrated.py --env_name [pointmaze-medium-v0, pointmaze-danger-{wall,lethal,sticky,floor}-v0]`
- Local training: `conda activate fasttd3` then `python train_fast_sac_ogbench.py --env_name [pointmaze-medium-v0, pointmaze-danger-{wall,lethal,sticky,floor}-v0]`
- Evaluate: `python eval_interactive.py --model_path models/<file>.pt --env_name [pointmaze-medium-v0, pointmaze-danger-{wall,lethal,sticky,floor}-v0]`
- Submit on SLURM: `bash submit_job.sh [experiment_name].sbatch`

## Apptainer for Isaac Sim experiments

- Create separate directory for apptainer: `mkdir -p ../apptainer/{cache,tmp,images}`
- Use created cache and tmp dirs (must be absolute paths): `export APPTAINER_TMPDIR="[absolute path to ../apptainer/tmp]" && export APPTAINER_CACHE="[absolute path to ../apptainer/cache]"`
- Get image using: `apptainer pull ../apptainer/images/isaacsim-5.0.0_base.sif docker://nvcr.io/nvidia/isaac-sim:5.0.0`
- Copy image before applying any modifications: `cp ../apptainer/images/isaacsim-5.0.0_base.sif ../apptainer/images/isaacsim-5.0.0.sif`
- Drop into container: `apptainer shell --nv --cleanenv --writable --containall --fakeroot --bind $PWD:/workspace ../apptainer/images/isaacsim-5.0.0.sif`
- Python must be executed using `/isaac-sim/python.sh`

---

See Repository Guidelines for structure, style, and workflows:
- `AGENTS.md`

---

## SLURM commands

- Start a session: `salloc --partition=booster --account=<your_account> --gres=gpu:1 --time=00:30:00`
- Enter session: `srun --pty bash -l`